"""Dependency-free course tests plus CPU MuJoCo regression tests."""

import ast
import copy
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
import tempfile
import tomllib
import unittest

import torch

from diablo_training.contract import (
    ACTION_DTYPE,
    ACTION_SHAPE,
    BLIND_SEEDS,
    OBSERVATION_DTYPE,
    OBSERVATION_SHAPE,
    RESET_BASE_HEIGHT_M as CONTRACT_RESET_BASE_HEIGHT_M,
    SCREEN_SEEDS,
    SELECTION_PROBE_SEED,
    SELECTION_VERIFY_SEEDS,
    TARGET_BASE_HEIGHT_M as CONTRACT_BASE_HEIGHT_M,
    TARGET_STANDING_HEIGHT_M as CONTRACT_STANDING_HEIGHT_M,
    VERIFY_SEEDS,
    checkpoint_interval,
    gate_failures,
    gate_score,
    interface_contract,
    load_course,
    metric_name,
    profile_command,
    reward_defaults_for_stage,
    select_policy_best,
    verify_interface_contract,
    verify_run_contract,
    verify_warm_start_contract,
    stage_training_contract,
)

try:
    import mujoco
    import numpy as np

    from diablo_training.control import decode_action, radial_feedforward
    from diablo_training.control import wheel_command_targets
    from diablo_training.env import Config, DiabloEnv, command_profile
    from diablo_training.model import (
        DEFAULT_Q,
        DOCUMENTED_MODEL_MASS_KG,
        HIP_DAMPING_GAIN,
        HIP_POSITION_GAIN,
        JOINTS,
        KNEE_DAMPING_GAIN,
        KNEE_POSITION_GAIN,
        NO_LOAD_SPEED,
        RESET_BASE_HEIGHT_M,
        RESET_CLEARANCE_M,
        STALL_TORQUE,
        TARGET_BASE_HEIGHT_M,
        TARGET_STANDING_HEIGHT_M,
        TARGET_WHOLE_HEIGHT_M,
        closure_errors,
    )
    from diablo_training.validation import validate_model

    DEPENDENCIES_AVAILABLE = True
except ModuleNotFoundError:
    DEPENDENCIES_AVAILABLE = False


COMPOSITE_METRICS = {
    "pass_rate": 1.0,
    "duration_fraction": 1.0,
    "speed_mae_mps": 0.1,
    "yaw_mae_rad_s": 0.08,
    "upright": 0.99,
    "mean_height_error_m": 0.01,
    "height_deviation_m": 0.02,
    "wheel_slip_mps": 0.1,
    "backslide_mps": 0.05,
    "wheel_center_offset_m": 0.005,
    "straight_wheel_speed_difference_mps": 0.02,
    "forward_progress_m": 47.5,
    "obstacle_passes": 3,
    "finished": True,
    "mean_positive_power_w": 80.0,
}


def seal_contract(contract: dict) -> None:
    unsigned = {key: value for key, value in contract.items() if key != "sha256"}
    contract["sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True).encode()
    ).hexdigest()


class ContractTests(unittest.TestCase):
    def test_checkpoint_interval_is_ten_percent(self):
        self.assertEqual(checkpoint_interval(102), 10)
        self.assertEqual(checkpoint_interval(82), 8)
        self.assertEqual(checkpoint_interval(143), 14)
        self.assertEqual(checkpoint_interval(1), 1)
        with self.assertRaises(ValueError):
            checkpoint_interval(0)

    def test_gpu_reset_initializes_passive_closed_loop_joints(self):
        source_path = (
            Path(__file__).resolve().parents[1]
            / "diablo_training"
            / "mjlab_tasks.py"
        )
        source = source_path.read_text(encoding="utf-8")
        self.assertIn("joint_pos=_initial_joint_positions()", source)
        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_initial_joint_positions"
        )
        namespace = {
            "JOINTS": (
                "left_hip_motor",
                "left_knee_motor",
                "left_wheel_motor",
                "right_hip_motor",
                "right_knee_motor",
                "right_wheel_motor",
            ),
            "DEFAULT_Q": (
                0.969324,
                -1.938616,
                0.0,
                0.969324,
                -1.938616,
                0.0,
            ),
        }
        exec(
            compile(
                ast.Module(body=[function], type_ignores=[]),
                str(source_path),
                "exec",
            ),
            namespace,
        )
        positions = namespace["_initial_joint_positions"]()
        self.assertEqual(len(positions), 10)
        self.assertAlmostEqual(positions["left_return_passive"], -1.938616)
        self.assertAlmostEqual(positions["left_branch_passive"], 1.938616)
        self.assertAlmostEqual(positions["right_return_passive"], -1.938616)
        self.assertAlmostEqual(positions["right_branch_passive"], 1.938616)
        self.assertIn(
            '"reset_passive_joints": EventTermCfg(\n'
            '            func=_reset_passive_joints,',
            source,
        )
        self.assertLess(
            source.index('"reset_joints": EventTermCfg('),
            source.index('"reset_passive_joints": EventTermCfg('),
        )
        self.assertIn("robot.write_joint_state_to_sim(", source)
        self.assertIn('pattern=r"^diablo_base_link$"', source)
        self.assertIn('name="leg_ground_contact"', source)
        self.assertIn('"leg_ground": TerminationTermCfg(', source)
        self.assertIn('geom_names_expr=(r"^.*_collision_\\d+$",)', source)
        self.assertIn(
            "/ KNEE_POSITION_GAIN",
            source,
        )
        self.assertIn('float(w.get("zero_motion", 0.0)) * zero_motion_cost', source)
        self.assertIn(
            'float(w.get("straight_heading", 0.0))',
            source,
        )
        self.assertIn("def _huber(error: torch.Tensor, tolerance: float)", source)
        self.assertIn("ratio - 0.5", source)
        self.assertIn(
            "speed_scale = torch.clamp(2.0 * cmd[:, 0].abs(), min=2.0 * zero_tol)",
            source,
        )
        self.assertIn("- speed_error.abs() / speed_scale", source)
        self.assertIn("directional_speed = direction * vx", source)
        self.assertIn(
            "cmd[:, 0].abs() - directional_speed",
            source,
        )
        self.assertIn(
            "torch.clamp(1.0 - underspeed_error / speed_scale, min=0.0)",
            source,
        )
        self.assertNotIn("torch.abs(cmd[:, 0]) - torch.abs(vx)", source)
        self.assertIn("acceleration_cost = torch.where(", source)
        self.assertIn("self._huber(accel - commanded_accel, accel_tol)", source)
        self.assertIn(
            '- float(w.get("acceleration", 0.0)) * acceleration_cost',
            source,
        )
        self.assertIn(
            "torch.where(straight, wheel_difference, torch.zeros_like(wheel_difference))",
            source,
        )
        self.assertIn("stopping & stop_metric_active", source)
        self.assertIn("wheel_command_targets(", source)
        self.assertIn("wheel_residual_scale_rad_s", source)
        self.assertIn("wheel_yaw_feedforward_gain", source)
        self.assertIn('get_command("course")', source)
        self.assertIn("residual_scale_rad_s=self.cfg.wheel_residual_scale_rad_s", source)
        self.assertIn("underspeed_cost = torch.where(", source)
        self.assertIn("overspeed_error = torch.clamp(", source)
        self.assertIn("overspeed_cost = torch.where(", source)
        self.assertIn(
            "cmd[:, 0].abs() * (1.0 + overspeed_margin_fraction)",
            source,
        )
        self.assertIn("standing_wheel_sync_cost = torch.where(", source)
        self.assertIn("standing_wheel_action_cost = torch.where(", source)
        self.assertIn("startup_wheel_action_cost = torch.where(", source)
        self.assertIn("standing_wheel_target_cap_rad_s", source)
        self.assertIn("zero_wheel_after_hold_s", source)
        self.assertIn("self._stand_steps", source)
        self.assertIn(
            "round(self.cfg.zero_wheel_after_hold_s / PHYSICS_DT)",
            source,
        )
        self.assertIn("iterations=80", source)
        self.assertIn("wheel_target.clamp(-cap, cap)", source)
        self.assertIn("self._wheel_slip_peak_ratio", source)
        self.assertIn("self._wheel_sync_peak_ratio", source)
        self.assertIn("self._stop_speed_peak_ratio", source)
        self.assertIn("torch.clamp(self._wheel_slip_peak_ratio - 1.0", source)
        self.assertIn("torch.clamp(self._wheel_sync_peak_ratio - 1.0", source)
        self.assertIn("torch.clamp(self._stop_speed_peak_ratio - 1.0", source)
        self.assertIn("settled_wheel_action_cost = torch.where(", source)
        self.assertIn("settled_drift_cost = torch.where(", source)
        self.assertIn("roll_cost = torch.square(gravity_xy[:, 1])", source)
        self.assertIn(
            '- float(w.get("underspeed_cost", 0.0)) * underspeed_cost',
            source,
        )
        self.assertIn(
            '- float(w.get("overspeed", 0.0)) * overspeed_cost',
            source,
        )
        self.assertIn("Metrics/overspeed_cost", source)
        self.assertIn(
            '- float(w.get("standing_wheel_sync", 0.0))',
            source,
        )
        self.assertIn(
            '- float(w.get("settled_wheel_action", 0.0))',
            source,
        )
        self.assertIn('- float(w.get("settled_drift", 0.0))', source)
        self.assertIn('- float(w.get("roll", 0.0))', source)
        self.assertIn("torch.log1p(torch.square(zero_motion_error))", source)
        self.assertNotIn("torch.clamp(zero_motion_error, max=1.0)", source)
        self.assertIn("self._huber(hold_displacement, drift_tol)", source)
        self.assertIn(
            "stop_speed_current_cost = torch.where(\n            stopping,",
            source,
        )
        self.assertIn(
            "wheel_sync_current_cost = torch.where(\n            straight,",
            source,
        )
        self.assertIn(
            'float(p.get("fall", 0.0)) / reward_scale',
            source,
        )
        self.assertIn("self._env.cfg.scale_rewards_by_dt", source)
        self.assertIn("wheel_alignment_cost", source)
        self.assertIn("heading_tolerance_rad", source)
        self.assertIn("straight_heading_tolerance_rad", source)
        self.assertIn("Metrics/straight_heading_error_rad", source)
        self.assertIn("Metrics/cross_track_cost", source)
        self.assertIn("Metrics/wheel_slip_gate_cost", source)
        self.assertIn("Metrics/stop_speed_gate_cost", source)
        self.assertIn("Metrics/wheel_sync_gate_cost", source)
        self.assertIn("Metrics/wheel_chatter_cost", source)
        self.assertIn('f"Metrics/{profile_name}/{metric_name}_mean"', source)
        self.assertIn('f"Metrics/{profile_name}/{metric_name}_worst"', source)
        self.assertIn("Metrics/left_wheel_target_rad_s", source)
        self.assertIn("Metrics/wheel_actual_difference_rad_s", source)
        self.assertIn("Metrics/hold_drift_after_0.5s_m", source)
        self.assertIn("Metrics/settled_mean_height_error_m", source)
        self.assertIn("Metrics/settled_height_deviation_m", source)
        self.assertIn("Metrics/settled_wheel_slip_mps", source)
        self.assertIn(
            "Metrics/settled_straight_wheel_speed_difference_mps",
            source,
        )
        self.assertIn(
            "Metrics/mean_settled_straight_wheel_speed_difference_mps",
            source,
        )
        self.assertIn("reward_components = {", source)
        self.assertIn('"Reward/total": reward.mean()', source)
        self.assertIn('"evaluation_grace_s"', source)
        self.assertIn("center_clearance_m", source)
        self.assertIn("trace_zero_action_stance", source)
        self.assertIn('"training_contract": (', source)
        reward_keys = {
            key
            for stage in load_course().stages.values()
            for key in stage.get("reward_weights", {})
        }
        for key in reward_keys:
            self.assertIn(f'w.get("{key}"', source)
        self.assertIn(
            "Metrics/straight_heading_error_{profile_name}_deg",
            source,
        )
        self.assertIn('"cruise_speed_mae_mps"', source)
        self.assertIn('"left_wheel_target_rad_s"', source)
        self.assertIn("command_term.target_heading", source)
        self.assertIn("command_observation = torch.stack(", source)
        self.assertIn("HEADING_ERROR_OBS_SCALE", source)
        self.assertIn('"reset_heading_range_deg"', source)
        self.assertIn('stage.get("observation_noise", 0.02)', source)
        self.assertIn('schedule="adaptive"', source)
        self.assertIn("POLICY_CLASS", source)
        self.assertEqual(source.count("class_name=POLICY_CLASS"), 2)
        self.assertIn('"init_std": 0.3', source)
        self.assertIn("COURSE.speed_ppo", source)
        self.assertIn("runner_cls=XingtianOnPolicyRunner", source)
        self.assertIn('"obs": {0: "batch"}', source)
        self.assertIn("def create_actor_initial_checkpoint(", source)
        self.assertNotIn("load_actor_bundle(source)", source)
        self.assertIn('"critic": False', source)
        self.assertIn("hold_settle_time_s", source)
        self.assertIn(
            "self.defaults = reward_defaults_for_stage(COURSE, self.stage_name)",
            source,
        )
        self.assertIn("termination_tilt_deg", source)
        self.assertIn("drift_cost", source)
        self.assertIn("stop_quality", source)
        self.assertIn("settled_hold_drift_m", source)
        self.assertIn("settled_heading_drift", source)
        self.assertIn("_heading_drift_settled_max", source)
        self.assertIn("heading_drift_cost = torch.where(", source)
        self.assertIn("settled_heading_drift_weight", source)
        self.assertIn(
            "math.radians(float(heading_drift_gate_deg)) * 0.75",
            source,
        )
        self.assertIn("self._huber(settled_heading_drift, heading_drift_tol)", source)
        self.assertIn(
            "command_changed = torch.abs(cmd[:, 0] - self._last_cmd) > 1e-4",
            source,
        )
        self.assertIn("steady_slip_active = (", source)
        self.assertIn("steady_context = moving & straight & evaluation_active", source)
        self.assertIn("command_changed | ~steady_context", source)
        self.assertIn(
            "& (self._command_hold_steps >= settle_steps)",
            source,
        )
        self.assertIn("self._steady_wheel_slip_sum", source)
        self.assertIn("self._steady_wheel_slip_steps", source)
        self.assertIn(
            "self._steady_wheel_slip_sum,\n"
            "            self._steady_wheel_slip_steps,",
            source,
        )
        self.assertIn("Metrics/mean_steady_wheel_slip_mps", source)
        self.assertIn("max_transient_wheel_slip_mps", source)
        self.assertIn('"final_speed_mps"', source)
        self.assertIn(
            '- float(w.get("drift", 0.0)) * drift_cost',
            source,
        )
        self.assertIn(
            '+ float(w.get("zero_speed", 0.0)) * stop_quality',
            source,
        )
        self.assertIn("path_lateral_velocity = (", source)
        self.assertIn(
            '- float(p.get("lateral", 0.0))\n'
            "                * torch.square(path_lateral_velocity)",
            source,
        )
        self.assertIn("Metrics/path_lateral_velocity_mps", source)
        self.assertIn("def _path_tracking_state(", source)
        self.assertIn(
            "-delta_xy[:, 0] * sin_heading + delta_xy[:, 1] * cos_heading",
            source,
        )
        record_source = (
            Path(__file__).resolve().parents[1] / "record_headless.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"world_heading_deg"', record_source)
        self.assertIn('"left_wheel_target_rad_s"', record_source)
        self.assertIn('"hold_drift_m"', record_source)
        self.assertIn('"hold_drift_after_0.5s_m"', record_source)
        self.assertIn('bad_orientation = bool(', record_source)
        self.assertNotIn("tilt_deg >= 60.0", record_source)
        self.assertEqual(source.count("func=_height_scan"), 1)
        privileged_function = next(
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.FunctionDef) and node.name == "_privileged"
        )
        privileged_source = ast.get_source_segment(source, privileged_function)
        self.assertIn(
            "cross_track, path_lateral_velocity = _path_tracking_state(env)",
            privileged_source,
        )
        self.assertIn("velocity_w[:, 1:2]", privileged_source)
        self.assertIn("path_lateral_velocity", privileged_source)
        self.assertIn("robot.data.heading_w - command_term.target_heading", source)
        self.assertIn("if frame.shape[-1] != PROPRIO_FRAME_DIM", source)
        self.assertIn("validate_policy_io(obs, action)", source)
        self.assertIn("_raw[:, self._leg_channels]", source)
        self.assertIn("_raw[:, self._wheel_channels]", source)
        self.assertNotIn("_raw[:, self._leg_ids]", source)
        self.assertNotIn("_raw[:, self._wheel_ids]", source)

    def test_native_rsl_checkpoint_can_warm_start_actor(self):
        from diablo_training.actor_import import TARGET_KEYS, _native_actor_state

        class Tensor:
            def detach(self):
                return self

            def clone(self):
                return self

        legacy = {
            "model_state_dict": {
                f"actor.{key}": Tensor()
                for key in TARGET_KEYS
            }
        }
        current = {
            "actor_state_dict": {key: Tensor() for key in TARGET_KEYS}
        }
        for checkpoint in (legacy, current):
            actor_state = _native_actor_state(checkpoint)
            self.assertEqual(set(actor_state), set(TARGET_KEYS))

    def test_stage_training_contract_ignores_unrelated_stage_changes(self):
        course = load_course()
        original = stage_training_contract(course, "flat_motion_010")
        stages = copy.deepcopy(course.stages)
        stages["flat_motion_020"]["reward_weights"]["speed"] += 0.5
        stages["flat_motion_010"]["gate"]["min_pass_rate"] = 0.5
        changed = replace(course, stages=stages)
        self.assertEqual(
            original,
            stage_training_contract(changed, "flat_motion_010"),
        )
        self.assertNotEqual(
            stage_training_contract(course, "flat_motion_020"),
            stage_training_contract(changed, "flat_motion_020"),
        )
        stages["flat_motion_010"]["reward_weights"]["speed"] += 0.25
        training_changed = replace(course, stages=stages)
        self.assertNotEqual(
            original,
            stage_training_contract(training_changed, "flat_motion_010"),
        )

    def test_warm_start_allows_target_height_change(self):
        contract = interface_contract(load_course())
        previous = copy.deepcopy(contract)
        previous["target_base_height_m"] = 0.36
        seal_contract(previous)
        verify_warm_start_contract(previous)
        previous["action_order"] = list(reversed(previous["action_order"]))
        seal_contract(previous)
        with self.assertRaisesRegex(ValueError, "action_order"):
            verify_warm_start_contract(previous)

    def test_warm_start_rejects_legacy_policy_architecture(self):
        contract = interface_contract(load_course())
        previous = copy.deepcopy(contract)
        previous["policy_architecture"] = "legacy_mlp_v1"
        seal_contract(previous)
        with self.assertRaisesRegex(ValueError, "policy_architecture"):
            verify_warm_start_contract(previous)

    def test_robot_physics_uses_original_main_branch_cad_baseline(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "diablo_training" / "model.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "    [0.969324, -1.938616, 0.0, 0.969324, -1.938616, 0.0]",
            source,
        )
        self.assertIn("DOCUMENTED_MODEL_MASS_KG = 21.019506", source)
        self.assertIn("HIP_POSITION_GAIN = 15.0", source)
        self.assertIn("KNEE_POSITION_GAIN = 18.0", source)
        self.assertIn("HIP_DAMPING_GAIN = 0.3", source)
        self.assertIn("KNEE_DAMPING_GAIN = 0.2", source)
        self.assertIn("TARGET_BASE_HEIGHT_M = 0.350", source)
        self.assertIn("RESET_CLEARANCE_M = 0.005", source)
        self.assertIn("TARGET_STANDING_HEIGHT_M = 0.477859", source)
        self.assertNotIn("_scale_mass_properties", source)
        self.assertNotIn("_PUBLISHED_INERTIAL_ORIGINS", source)

    def test_policy_io_contract_is_float32_125_to_float32_6(self):
        self.assertEqual(OBSERVATION_DTYPE, "float32")
        self.assertEqual(OBSERVATION_SHAPE, (125,))
        self.assertEqual(ACTION_DTYPE, "float32")
        self.assertEqual(ACTION_SHAPE, (6,))

    def test_ppo_json_only_contains_consumed_parameters(self):
        root = Path(__file__).resolve().parents[1] / "diablo_training"
        removed = {"n_steps", "batch_size", "policy_kwargs"}
        for filename in ("curriculum.json", "flat_motion_speed_stages.json"):
            source = json.loads((root / filename).read_text(encoding="utf-8"))
            self.assertFalse(removed & set(source["ppo"]), filename)
        speed_source = json.loads(
            (root / "flat_motion_speed_stages.json").read_text(encoding="utf-8")
        )
        self.assertEqual(load_course().speed_ppo, speed_source["ppo"])

    def test_course_order_and_seed_sets(self):
        course = load_course()
        self.assertEqual(len(course.order), 20)
        self.assertEqual(course.order[0], "flat_balance")
        self.assertEqual(course.order[1], "flat_motion_010")
        self.assertEqual(course.order[2], "flat_motion_020")
        self.assertEqual(course.order[3], "flat_motion_030")
        self.assertEqual(course.order[4], "flat_motion_040")
        self.assertEqual(course.order[-1], "composite")
        balance = course.stage("flat_balance")
        motion_010 = course.stage("flat_motion_010")
        motion_020 = course.stage("flat_motion_020")
        motion_030 = course.stage("flat_motion_030")
        motion_040 = course.stage("flat_motion_040")
        motion_120 = course.stage("flat_motion_120")
        self.assertEqual(course.reward_penalties["fall"], 40.0)
        self.assertEqual(
            course.reward_defaults["settled_heading_drift_weight"],
            0.75,
        )
        self.assertNotIn("pitch_weight", course.reward_defaults)
        self.assertEqual(balance["budget"], 40000000)
        self.assertEqual(balance["reset_heading_range_deg"], [-5.0, 5.0])
        self.assertEqual(balance["observation_noise"], 0.005)
        self.assertEqual(balance["hold_settle_time_s"], 1.0)
        self.assertEqual(
            reward_defaults_for_stage(course, "flat_balance")["hold_settle_time_s"],
            1.0,
        )
        self.assertEqual(
            reward_defaults_for_stage(course, "flat_balance")[
                "evaluation_grace_s"
            ],
            1.0,
        )
        self.assertEqual(
            reward_defaults_for_stage(course, "flat_motion")[
                "evaluation_grace_s"
            ],
            0.5,
        )
        self.assertEqual(balance["termination_tilt_deg"], 20.0)
        self.assertEqual(balance["overrides"]["fall"], 20.0)
        self.assertEqual(balance["ppo"]["learning_rate"], 0.0001)
        self.assertEqual(balance["ppo"]["n_epochs"], 3)
        self.assertEqual(balance["ppo"]["target_kl"], 0.03)
        self.assertEqual(balance["control"]["wheel_speed_target_max_rad_s"], 4.0)
        self.assertEqual(balance["reward_weights"]["drift"], 2.0)
        self.assertEqual(balance["reward_weights"]["upright"], 3.0)
        self.assertEqual(balance["reward_weights"]["straight_heading"], 2.0)
        self.assertGreater(balance["reward_weights"]["zero_motion"], 0.0)
        self.assertEqual(balance["reward_defaults"]["yaw_tolerance_base_rad_s"], 0.02)
        self.assertEqual(
            balance["reward_defaults"]["zero_speed_tolerance_mps"],
            0.02,
        )
        self.assertEqual(balance["reward_defaults"]["heading_tolerance_rad"], 0.05236)
        self.assertEqual(balance["gate"]["max_final_heading_error_deg"], 3.0)
        self.assertEqual(balance["gate"]["max_hold_drift_m"], 0.05)
        self.assertNotIn("max_settled_hold_drift_m", balance["gate"])
        self.assertEqual(metric_name("max_hold_drift_m"), "hold_drift_m")
        self.assertEqual(balance["gate"]["max_stop_speed_mps"], 0.02)
        self.assertEqual(balance["gate"]["max_settled_wheel_slip_mps"], 0.08)
        self.assertNotIn("max_wheel_slip_mps", balance["gate"])
        self.assertNotIn("mean_height_tolerance_m", balance["gate"])
        self.assertEqual(
            balance["gate"]["max_settled_mean_height_error_m"],
            0.02,
        )
        self.assertEqual(
            balance["gate"]["max_settled_height_deviation_m"],
            0.04,
        )
        self.assertEqual(
            balance["gate"][
                "max_mean_settled_straight_wheel_speed_difference_mps"
            ],
            0.04,
        )
        self.assertNotIn(
            "max_settled_straight_wheel_speed_difference_mps",
            balance["gate"],
        )
        self.assertEqual(
            balance["gate"]["max_heading_drift_after_0.5s_deg"],
            3.0,
        )
        self.assertEqual(
            metric_name("max_settled_mean_height_error_m"),
            "settled_mean_height_error_m",
        )
        self.assertEqual(
            balance["control"]["wheel_target_slew_rad_s2"],
            100.0,
        )
        self.assertGreater(balance["overrides"]["wheel_alignment"], 1.0)
        self.assertEqual(motion_010["budget"], 6000000)
        self.assertEqual(motion_010["limits"]["forward"], 0.1)
        self.assertEqual(motion_010["reset_heading_range_deg"], [-5.0, 5.0])
        self.assertEqual(motion_010["observation_noise"], 0.005)
        self.assertEqual(motion_010["termination_tilt_deg"], 12.0)
        self.assertEqual(motion_010["hold_settle_time_s"], 0.5)
        self.assertEqual(motion_010["overrides"]["fall"], 40.0)
        self.assertEqual(motion_010["ppo"]["learning_rate"], 0.0003)
        self.assertEqual(motion_010["ppo"]["n_epochs"], 4)
        self.assertEqual(motion_010["ppo"]["ent_coef"], 0.006)
        self.assertEqual(motion_010["ppo"]["target_kl"], 0.015)
        self.assertEqual(
            motion_010["control"]["wheel_speed_target_max_rad_s"],
            2.0,
        )
        self.assertEqual(motion_010["reward_weights"]["pitch"], 2.2)
        self.assertEqual(motion_010["reward_weights"]["upright"], 2.2)
        self.assertEqual(motion_010["reward_weights"]["drift"], 1.5)
        self.assertEqual(motion_010["reward_weights"]["yaw"], 1.2)
        self.assertEqual(motion_010["overrides"]["action_rate"], 0.04)
        self.assertEqual(motion_010["overrides"]["wheel_alignment"], 1.2)
        self.assertEqual(
            motion_010["reward_defaults"]["yaw_tolerance_base_rad_s"],
            0.03,
        )
        self.assertEqual(
            motion_010["control"]["wheel_target_slew_rad_s2"],
            4.0,
        )
        self.assertEqual(
            motion_010["profiles"][0]["points"][2][1],
            0.1,
        )
        self.assertNotIn("pitch_weight", motion_010["reward_defaults"])
        self.assertEqual(motion_010["gate"]["max_backslide_mps"], 0.06)
        self.assertEqual(motion_010["gate"]["max_hold_drift_after_0.5s_m"], 0.02)
        self.assertEqual(
            motion_010["gate"]["max_heading_drift_after_0.5s_deg"],
            2.0,
        )
        self.assertEqual(motion_010["gate"]["max_final_speed_mps"], 0.02)
        self.assertEqual(motion_010["gate"]["max_speed_mae_mps"], 0.03)
        self.assertEqual(motion_010["gate"]["max_yaw_mae_rad_s"], 0.08)
        self.assertEqual(motion_010["gate"]["max_cross_track_error_m"], 0.05)
        self.assertEqual(
            motion_010["gate"]["max_mean_steady_wheel_slip_mps"],
            0.02,
        )
        self.assertEqual(
            motion_010["gate"]["max_transient_wheel_slip_mps"],
            0.05,
        )
        self.assertNotIn(
            "max_settled_wheel_slip_fraction_of_peak",
            motion_010["gate"],
        )
        self.assertEqual(
            motion_010["gate"][
                "max_settled_straight_wheel_speed_difference_mps"
            ],
            0.04,
        )
        self.assertNotIn("max_hold_drift_m", motion_010["gate"])
        self.assertEqual(
            metric_name("max_hold_drift_after_0.5s_m"),
            "hold_drift_after_0.5s_m",
        )
        self.assertEqual(motion_020["termination_tilt_deg"], 12.0)
        self.assertEqual(motion_020["hold_settle_time_s"], 0.5)
        self.assertEqual(motion_020["overrides"]["fall"], 40.0)
        self.assertEqual(motion_020["overrides"]["action_rate"], 0.04)
        self.assertEqual(motion_020["overrides"]["wheel_alignment"], 1.2)
        self.assertEqual(motion_020["ppo"]["learning_rate"], 0.0003)
        self.assertEqual(motion_020["ppo"]["n_epochs"], 4)
        self.assertEqual(motion_020["ppo"]["ent_coef"], 0.006)
        self.assertEqual(motion_020["ppo"]["target_kl"], 0.015)
        self.assertNotIn("pitch_weight", motion_020["reward_defaults"])
        self.assertEqual(motion_020["reward_weights"]["speed"], 3.0)
        self.assertEqual(motion_020["reward_weights"]["underspeed"], 1.2)
        self.assertEqual(motion_020["reward_weights"]["underspeed_cost"], 0.6)
        self.assertEqual(motion_020["reward_weights"]["backslide"], 0.3)
        self.assertEqual(motion_020["reward_weights"]["yaw"], 1.2)
        self.assertEqual(motion_020["reward_weights"]["straight_heading"], 1.2)
        self.assertEqual(motion_020["reward_weights"]["cross_track"], 1.2)
        self.assertEqual(motion_020["reward_weights"]["upright"], 2.0)
        self.assertEqual(motion_020["reward_weights"]["pitch"], 2.0)
        self.assertEqual(motion_020["reward_weights"]["drift"], 1.5)
        self.assertEqual(motion_020["reward_weights"]["wheel_slip_gate"], 1.0)
        self.assertEqual(motion_020["reward_weights"]["stop_speed_gate"], 0.5)
        self.assertEqual(motion_020["reward_weights"]["wheel_sync_gate"], 1.0)
        self.assertEqual(motion_020["reward_weights"]["wheel_chatter"], 0.75)
        self.assertEqual(motion_040["termination_tilt_deg"], 12.0)
        self.assertEqual(motion_040["hold_settle_time_s"], 0.5)
        self.assertEqual(motion_120["termination_tilt_deg"], 18.0)
        self.assertEqual(motion_120["hold_settle_time_s"], 0.5)
        self.assertEqual(motion_120["ppo"]["learning_rate"], 0.00015)
        self.assertEqual(motion_120["ppo"]["ent_coef"], 0.0045)
        self.assertNotIn("actor_activation", motion_120["ppo"])
        self.assertNotIn("actor_obs_normalization", motion_120["ppo"])
        self.assertEqual(motion_120["reward_weights"]["speed"], 3.7)
        self.assertEqual(motion_120["reward_weights"]["underspeed"], 1.1)
        self.assertEqual(motion_120["reward_weights"]["underspeed_cost"], 0.5)
        self.assertEqual(motion_120["overrides"]["fall"], 30.0)
        self.assertEqual(motion_120["reward_weights"]["straight_heading"], 0.75)
        self.assertEqual(motion_120["reward_weights"]["cross_track"], 0.75)
        self.assertEqual(motion_120["gate"]["max_backslide_mps"], 0.22)
        self.assertEqual(motion_120["gate"]["max_hold_drift_after_0.5s_m"], 0.032)
        self.assertNotIn("max_hold_drift_m", motion_120["gate"])
        motion_100 = course.stage("flat_motion_100")
        self.assertEqual(motion_100["termination_tilt_deg"], 18.0)
        self.assertEqual(motion_100["reward_weights"]["underspeed"], 1.0)
        self.assertEqual(motion_100["reward_weights"]["underspeed_cost"], 0.4)
        self.assertEqual(motion_100["reward_defaults"]["underspeed_cost_limit"], 1.0)
        self.assertEqual(motion_100["reward_weights"]["speed"], 3.6)
        self.assertEqual(motion_100["reward_weights"]["upright"], 1.1)
        self.assertEqual(motion_100["reward_weights"]["pitch"], 0.9)
        self.assertEqual(motion_100["overrides"]["fall"], 30.0)
        self.assertEqual(motion_100["ppo"]["ent_coef"], 0.005)
        self.assertNotIn("actor_activation", motion_100["ppo"])
        self.assertNotIn("actor_obs_normalization", motion_100["ppo"])
        self.assertEqual(motion_030["limits"]["forward"], 0.3)
        self.assertEqual(motion_030["termination_tilt_deg"], 12.0)
        self.assertEqual(motion_030["hold_settle_time_s"], 0.5)
        self.assertEqual(motion_030["reward_weights"]["speed"], 3.2)
        self.assertEqual(motion_030["reward_weights"]["underspeed"], 1.1)
        self.assertEqual(motion_030["reward_weights"]["underspeed_cost"], 0.55)
        self.assertEqual(motion_030["reward_weights"]["standing_wheel_sync"], 1.0)
        self.assertEqual(motion_030["reward_weights"]["standing_wheel_action"], 0.8)
        self.assertEqual(motion_030["reward_weights"]["startup_wheel_action"], 4.0)
        self.assertEqual(motion_030["reward_weights"]["settled_wheel_action"], 1.0)
        self.assertEqual(motion_030["reward_weights"]["settled_drift"], 2.0)
        self.assertEqual(motion_030["reward_weights"]["roll"], 0.5)
        self.assertEqual(motion_030["reward_weights"]["angular_xy"], 0.2)
        self.assertEqual(motion_030["reward_weights"]["upright"], 2.0)
        self.assertEqual(motion_030["reward_weights"]["pitch"], 2.0)
        self.assertEqual(motion_030["reward_weights"]["stop_speed_gate"], 0.5)
        self.assertEqual(motion_030["reward_weights"]["zero_speed"], 1.5)
        self.assertEqual(motion_030["reward_weights"]["braking"], 0.3)
        self.assertEqual(motion_030["reward_defaults"]["startup_window_s"], 0.2)
        self.assertEqual(
            motion_030["control"]["wheel_speed_target_max_rad_s"],
            6.0,
        )
        self.assertEqual(
            motion_030["control"]["wheel_velocity_feedforward_gain"],
            1.0,
        )
        self.assertEqual(
            motion_030["control"]["standing_wheel_target_cap_rad_s"],
            0.2,
        )
        self.assertEqual(
            motion_030["control"]["zero_wheel_after_hold_s"],
            0.5,
        )
        self.assertEqual(
            motion_030["control"]["wheel_target_slew_rad_s2"],
            10.0,
        )
        self.assertEqual(motion_030["gate"]["max_speed_mae_mps"], 0.06)
        self.assertEqual(motion_030["gate"]["min_upright"], 0.93)
        self.assertEqual(motion_030["gate"]["max_stop_speed_mps"], 0.017)
        self.assertEqual(
            motion_030["gate"][
                "max_settled_straight_wheel_speed_difference_mps"
            ],
            0.06,
        )
        self.assertEqual(motion_040["reward_weights"]["speed"], 3.3)
        self.assertEqual(motion_040["reward_weights"]["underspeed"], 1.0)
        self.assertEqual(motion_040["reward_weights"]["underspeed_cost"], 0.5)
        self.assertEqual(motion_040["reward_weights"]["standing_wheel_sync"], 1.0)
        self.assertEqual(motion_040["reward_weights"]["standing_wheel_action"], 0.8)
        self.assertEqual(motion_040["reward_weights"]["startup_wheel_action"], 4.0)
        self.assertEqual(motion_040["reward_weights"]["settled_wheel_action"], 1.0)
        self.assertEqual(motion_040["reward_weights"]["settled_drift"], 2.0)
        self.assertEqual(motion_040["reward_weights"]["roll"], 0.5)
        self.assertEqual(motion_040["reward_weights"]["angular_xy"], 0.2)
        self.assertEqual(motion_040["reward_weights"]["stop_speed_gate"], 0.5)
        self.assertEqual(motion_040["reward_weights"]["zero_speed"], 1.5)
        self.assertEqual(motion_040["reward_weights"]["braking"], 0.3)
        self.assertEqual(motion_040["reward_defaults"]["startup_window_s"], 0.2)
        self.assertEqual(
            motion_040["control"]["wheel_velocity_feedforward_gain"],
            1.0,
        )
        self.assertEqual(
            motion_040["control"]["zero_wheel_after_hold_s"],
            0.5,
        )
        self.assertEqual(motion_040["reward_weights"]["upright"], 1.9)
        self.assertEqual(motion_040["reward_weights"]["pitch"], 1.9)
        self.assertEqual(motion_040["reward_weights"]["drift"], 1.5)
        self.assertEqual(motion_040["overrides"]["fall"], 40.0)
        self.assertEqual(motion_040["overrides"]["action_rate"], 0.04)
        self.assertEqual(motion_040["overrides"]["wheel_alignment"], 1.2)
        self.assertEqual(motion_040["ppo"]["learning_rate"], 0.00025)
        self.assertEqual(motion_040["ppo"]["n_epochs"], 4)
        self.assertEqual(motion_040["ppo"]["ent_coef"], 0.006)
        self.assertEqual(
            motion_040["control"]["wheel_speed_target_max_rad_s"],
            8.0,
        )
        self.assertEqual(
            motion_040["control"]["wheel_target_slew_rad_s2"],
            12.0,
        )
        self.assertEqual(motion_040["gate"]["max_speed_mae_fraction_of_peak"], 0.25)
        self.assertEqual(motion_040["gate"]["min_upright"], 0.92)
        self.assertEqual(motion_040["gate"]["max_yaw_mae_rad_s"], 0.08)
        self.assertEqual(motion_040["gate"]["max_backslide_mps"], 0.12)
        self.assertEqual(
            motion_040["gate"]["max_transient_wheel_slip_mps"],
            0.12,
        )
        self.assertEqual(
            motion_040["gate"][
                "max_settled_straight_wheel_speed_difference_mps"
            ],
            0.07,
        )
        self.assertNotIn("max_hold_drift_m", motion_040["gate"])
        self.assertEqual(
            motion_020["reward_defaults"]["yaw_tolerance_base_rad_s"],
            0.035,
        )
        self.assertEqual(
            motion_020["reward_defaults"]["straight_heading_tolerance_rad"],
            0.05236,
        )
        self.assertEqual(
            motion_020["reward_defaults"]["gate_penalty_ema_decay"],
            0.98,
        )
        self.assertEqual(motion_020["gate"]["max_backslide_mps"], 0.08)
        self.assertEqual(
            motion_020["gate"]["max_cruise_speed_mae_mps"],
            0.03,
        )
        self.assertEqual(
            motion_020["gate"]["min_cruise_speed_fraction"],
            0.85,
        )
        self.assertEqual(motion_020["gate"]["max_yaw_mae_rad_s"], 0.08)
        self.assertEqual(motion_020["gate"]["max_final_speed_mps"], 0.025)
        self.assertEqual(motion_020["gate"]["max_hold_drift_after_0.5s_m"], 0.02)
        self.assertNotIn("max_hold_drift_m", motion_020["gate"])
        self.assertEqual(
            motion_020["control"]["wheel_speed_target_max_rad_s"],
            4.0,
        )
        self.assertEqual(
            motion_020["control"]["wheel_target_slew_rad_s2"],
            8.0,
        )
        self.assertEqual(SCREEN_SEEDS, (2001, 2002, 2003))
        self.assertEqual(VERIFY_SEEDS, (2004, 2005, 2006, 2007, 2008))
        self.assertEqual(SELECTION_PROBE_SEED, 2001)
        self.assertEqual(
            SELECTION_VERIFY_SEEDS,
            (2002, 2003, 2004, 2005, 2006),
        )
        self.assertEqual(len(SELECTION_VERIFY_SEEDS), 5)
        self.assertNotIn(SELECTION_PROBE_SEED, SELECTION_VERIFY_SEEDS)
        self.assertFalse(set(SCREEN_SEEDS) & set(VERIFY_SEEDS))
        self.assertFalse(set(SCREEN_SEEDS + VERIFY_SEEDS) & set(BLIND_SEEDS))
        contract = interface_contract(course)
        verify_interface_contract(contract)
        self.assertEqual(
            set(contract["source_sha256"]),
            {
                "model.py",
                "control.py",
                "policies.py",
                "mjlab_tasks.py",
                "upstream.lock.json",
                "curriculum.json",
                "flat_motion_speed_stages.json",
            },
        )
        tampered = dict(contract)
        tampered["action_shape"] = 7
        with self.assertRaises(ValueError):
            verify_interface_contract(tampered)
        code_changed = copy.deepcopy(contract)
        code_changed["source_sha256"]["mjlab_tasks.py"] = "changed"
        code_changed["screen_seeds"] = [9001]
        seal_contract(code_changed)
        verify_interface_contract(code_changed)
        with self.assertRaisesRegex(ValueError, "Legacy checkpoint"):
            verify_run_contract(
                {"contract": code_changed, "stage": "flat_balance"},
                "flat_balance",
                require_current_sources=True,
            )

        incompatible = copy.deepcopy(contract)
        incompatible["action_order"] = list(reversed(incompatible["action_order"]))
        seal_contract(incompatible)
        with self.assertRaisesRegex(ValueError, "action_order"):
            verify_interface_contract(incompatible)
        self.assertEqual(contract["observation_shape"], 125)
        self.assertEqual(contract["version"], 9)
        self.assertEqual(
            contract["policy_architecture"],
            "temporal_residual_v1",
        )
        self.assertEqual(contract["observation_dtype"], "float32")
        self.assertEqual(
            contract["observation_breakdown"],
            {"proprio_history": 125},
        )
        self.assertEqual(contract["action_shape"], 6)
        self.assertEqual(contract["action_dtype"], "float32")
        self.assertEqual(contract["target_base_height_m"], 0.350)
        self.assertEqual(contract["target_standing_height_m"], 0.477859)
        self.assertEqual(contract["reset_base_height_m"], 0.355)
        self.assertEqual(contract["target_whole_height_m"], 0.491)
        self.assertIs(
            verify_run_contract(
                {"contract": contract, "stage": "flat_balance"},
                "flat_balance",
            ),
            contract,
        )
        with self.assertRaises(ValueError):
            verify_run_contract(contract, "flat_balance")
        with self.assertRaises(ValueError):
            verify_run_contract(
                {"contract": contract, "stage": "flat_motion"},
                "flat_balance",
            )

    def test_motion_speed_curriculum_is_smooth_and_complete(self):
        course = load_course()
        names = tuple(
            name
            for name in course.order
            if name.startswith("flat_motion_") and name != "flat_motion"
        )
        stages = [course.stage(name) for name in names]
        expected_gate = {
            "min_pass_rate",
            "min_duration_fraction",
            "max_speed_mae_mps",
            "max_speed_mae_fraction_of_peak",
            "max_cruise_speed_mae_mps",
            "min_cruise_speed_fraction",
            "min_planned_peak_progress_fraction",
            "max_backslide_mps",
            "max_yaw_mae_rad_s",
            "min_upright",
            "max_mean_abs_pitch_deg",
            "max_pitch_deg",
            "max_mean_abs_steady_pitch_deg",
            "max_mean_abs_roll_deg",
            "max_roll_deg",
            "max_final_heading_error_deg",
            "max_heading_drift_after_0.5s_deg",
            "max_cross_track_error_m",
            "max_settled_mean_height_error_m",
            "max_settled_height_deviation_m",
            "max_mean_steady_wheel_slip_mps",
            "max_transient_wheel_slip_mps",
            "max_stop_speed_mps",
            "max_stop_speed_fraction_of_peak",
            "max_final_speed_mps",
            "max_hold_drift_after_0.5s_m",
            "max_wheel_center_offset_m",
            "max_settled_straight_wheel_speed_difference_mps",
        }
        expected_weights = {
            "speed",
            "underspeed",
            "underspeed_cost",
            "overspeed",
            "yaw",
            "straight_heading",
            "cross_track",
            "wheel_slip_gate",
            "stop_speed_gate",
            "wheel_sync_gate",
            "wheel_chatter",
            "height",
            "upright",
            "pitch",
            "roll",
            "stationary",
            "drift",
            "zero_speed",
            "acceleration",
            "wheel_slip",
            "in_place",
            "vertical_velocity",
            "angular_xy",
            "braking",
            "backslide",
            "over_rated",
            "power",
            "standing_wheel_sync",
            "standing_wheel_action",
            "startup_wheel_action",
            "settled_wheel_action",
            "settled_drift",
        }
        for stage in stages:
            self.assertEqual(set(stage["gate"]), expected_gate)
            self.assertEqual(set(stage["reward_weights"]), expected_weights)
            self.assertEqual(
                stage["control"].get("wheel_velocity_feedforward_gain", 1.0),
                1.0,
            )
            self.assertEqual(
                stage["control"]["standing_wheel_target_cap_rad_s"],
                0.2,
            )
            self.assertEqual(
                stage["control"]["zero_wheel_after_hold_s"],
                0.5,
            )
        self.assertEqual(
            reward_defaults_for_stage(course, names[0])[
                "overspeed_margin_fraction"
            ],
            0.05,
        )

        def nondecreasing(path):
            values = []
            for stage in stages:
                value = stage
                for key in path:
                    value = value[key]
                values.append(float(value))
            self.assertEqual(values, sorted(values))

        def nonincreasing(path):
            values = []
            for stage in stages:
                value = stage
                for key in path:
                    value = value[key]
                values.append(float(value))
            self.assertEqual(values, sorted(values, reverse=True))

        for key in (
            "max_speed_mae_mps",
            "max_cruise_speed_mae_mps",
            "max_backslide_mps",
            "max_yaw_mae_rad_s",
            "max_mean_abs_pitch_deg",
            "max_mean_abs_roll_deg",
            "max_final_heading_error_deg",
            "max_heading_drift_after_0.5s_deg",
            "max_cross_track_error_m",
            "max_settled_mean_height_error_m",
            "max_mean_steady_wheel_slip_mps",
            "max_stop_speed_mps",
            "max_final_speed_mps",
            "max_hold_drift_after_0.5s_m",
            "max_wheel_center_offset_m",
            "max_settled_straight_wheel_speed_difference_mps",
        ):
            nondecreasing(("gate", key))
        nonincreasing(("gate", "min_upright"))
        nondecreasing(("termination_tilt_deg",))
        nondecreasing(("control", "wheel_speed_target_max_rad_s"))
        nondecreasing(("control", "wheel_target_slew_rad_s2"))
        nondecreasing(("overrides", "action_rate"))
        nonincreasing(("overrides", "fall"))
        nonincreasing(("overrides", "wheel_alignment"))
        for key in (
            "speed_tolerance_base_mps",
            "yaw_tolerance_base_rad_s",
            "height_tolerance_m",
            "pitch_tolerance_rad",
            "wheel_slip_tolerance_mps",
            "acceleration_tolerance_mps2",
            "zero_speed_tolerance_mps",
            "drift_tolerance_m",
            "wheel_sync_tolerance_mps",
        ):
            nondecreasing(("reward_defaults", key))
        for key in ("speed",):
            nondecreasing(("reward_weights", key))
        for stage in stages:
            self.assertGreater(stage["reward_weights"]["underspeed"], 0.0)
            self.assertGreater(
                stage["reward_weights"]["underspeed_cost"],
                0.0,
            )
            self.assertEqual(stage["reward_weights"]["overspeed"], 0.8)
        self.assertEqual(
            course.stage("flat_motion")["reward_weights"]["overspeed"],
            0.8,
        )
        for key in (
            "height",
            "upright",
            "pitch",
            "straight_heading",
            "cross_track",
        ):
            nonincreasing(("reward_weights", key))

        effective_ppo = []
        for stage in stages:
            ppo = dict(course.speed_ppo)
            ppo.update(stage["ppo"])
            effective_ppo.append(ppo)
            self.assertEqual(ppo["n_epochs"], 4)
            self.assertEqual(ppo["gamma"], 0.99)
            self.assertEqual(ppo["gae_lambda"], 0.95)
            self.assertEqual(ppo["max_grad_norm"], 0.8)
            self.assertEqual(ppo["clip_range"], 0.2)
            self.assertEqual(ppo["vf_coef"], 0.5)
        self.assertEqual(
            [float(ppo["learning_rate"]) for ppo in effective_ppo],
            sorted(
                (float(ppo["learning_rate"]) for ppo in effective_ppo),
                reverse=True,
            ),
        )
        self.assertEqual(
            [float(ppo["ent_coef"]) for ppo in effective_ppo],
            sorted(
                (float(ppo["ent_coef"]) for ppo in effective_ppo),
                reverse=True,
            ),
        )

    def test_speed_stage_yaw_and_wheel_slip_calibration(self):
        course = load_course()
        expected_slip = {
            "flat_motion_010": (0.08, 0.03, 0.02, 0.05),
            "flat_motion_020": (0.08, 0.035, 0.025, 0.08),
            "flat_motion_030": (0.08, 0.04, 0.03, 0.10),
            "flat_motion_040": (0.08, 0.05, 0.035, 0.12),
            "flat_motion_060": (0.09, 0.06, 0.04, 0.14),
            "flat_motion_080": (0.09, 0.07, 0.045, 0.16),
            "flat_motion_100": (0.10, 0.08, 0.05, 0.18),
            "flat_motion_120": (0.10, 0.09, 0.05, 0.20),
            "flat_motion_140": (0.10, 0.10, 0.055, 0.22),
            "flat_motion_160": (0.11, 0.11, 0.06, 0.24),
            "flat_motion_180": (0.11, 0.12, 0.06, 0.26),
            "flat_motion_200": (0.12, 0.12, 0.065, 0.28),
        }
        for name, (yaw_gate, yaw_tol, mean_slip, max_slip) in expected_slip.items():
            stage = course.stage(name)
            self.assertEqual(stage["gate"]["max_yaw_mae_rad_s"], yaw_gate, name)
            self.assertEqual(
                stage["reward_defaults"]["yaw_tolerance_base_rad_s"],
                yaw_tol,
                name,
            )
            self.assertEqual(
                stage["gate"]["max_mean_steady_wheel_slip_mps"],
                mean_slip,
                name,
            )
            self.assertEqual(
                stage["gate"]["max_transient_wheel_slip_mps"],
                max_slip,
                name,
            )
            self.assertEqual(
                stage["reward_defaults"]["wheel_slip_tolerance_mps"],
                mean_slip,
                name,
            )
            self.assertNotIn(
                "max_settled_wheel_slip_fraction_of_peak",
                stage["gate"],
                name,
            )
        self.assertEqual(
            course.stage("flat_motion_010")["gate"][
                "max_heading_drift_after_0.5s_deg"
            ],
            2.0,
        )
        self.assertEqual(
            course.stage("flat_balance")["gate"]["max_yaw_mae_rad_s"],
            0.02,
        )
        self.assertEqual(
            course.stage("rough")["gate"]["max_yaw_mae_rad_s"],
            0.3,
        )

    def test_speed_stage_pitch_and_upright_calibration(self):
        course = load_course()
        expected = {
            "flat_motion_010": (3.0, 0.93),
            "flat_motion_020": (3.0, 0.93),
            "flat_motion_030": (3.0, 0.93),
            "flat_motion_040": (3.25, 0.92),
            "flat_motion_060": (3.25, 0.91),
            "flat_motion_080": (3.5, 0.90),
            "flat_motion_100": (3.5, 0.90),
            "flat_motion_120": (3.75, 0.89),
            "flat_motion_140": (3.75, 0.88),
            "flat_motion_160": (4.0, 0.87),
            "flat_motion_180": (4.0, 0.86),
            "flat_motion_200": (4.25, 0.85),
        }
        for name, (mean_pitch_deg, min_upright) in expected.items():
            stage = course.stage(name)
            self.assertEqual(
                stage["gate"]["max_mean_abs_pitch_deg"],
                mean_pitch_deg,
                name,
            )
            self.assertEqual(stage["gate"]["max_pitch_deg"], 10.0, name)
            self.assertEqual(
                stage["gate"]["max_mean_abs_steady_pitch_deg"],
                mean_pitch_deg,
                name,
            )
            self.assertEqual(stage["gate"]["min_upright"], min_upright, name)
            self.assertAlmostEqual(
                math.degrees(
                    stage["reward_defaults"]["pitch_tolerance_rad"]
                ),
                mean_pitch_deg,
                places=3,
            )

    def test_every_gate_has_an_evaluable_metric_name(self):
        course = load_course()
        for name in course.order:
            for gate_name in course.stage(name)["gate"]:
                self.assertTrue(metric_name(gate_name))
        reward_source = ast.parse(
            (Path(__file__).resolve().parents[1] / "diablo_training" / "mjlab_tasks.py")
            .read_text(encoding="utf-8")
        )
        metric_keys: set[str] = set()
        for node in ast.walk(reward_source):
            if isinstance(node, ast.FunctionDef) and node.name == "metrics":
                for child in ast.walk(node):
                    if isinstance(child, ast.Return) and isinstance(child.value, ast.Dict):
                        metric_keys = {
                            key.value
                            for key in child.value.keys
                            if isinstance(key, ast.Constant)
                        }
        required = {
            metric_name(gate_name)
            for name in course.order
            for gate_name in course.stage(name)["gate"]
        }
        self.assertEqual(required - metric_keys, set())
        for name in course.order:
            stage = course.stage(name)
            if stage["terrain"]["kind"] != "flat":
                continue
            self.assertIn(
                "max_settled_mean_height_error_m",
                stage["gate"],
            )
            self.assertIn(
                "max_settled_height_deviation_m",
                stage["gate"],
            )
            if name == "flat_balance":
                self.assertIn(
                    "max_mean_settled_straight_wheel_speed_difference_mps",
                    stage["gate"],
                )
            else:
                self.assertIn(
                    "max_settled_straight_wheel_speed_difference_mps",
                    stage["gate"],
                )
            for legacy in (
                "mean_height_tolerance_m",
                "max_height_deviation_m",
                "max_wheel_slip_mps",
                "max_wheel_slip_fraction_of_peak",
                "max_straight_wheel_speed_difference_mps",
            ):
                self.assertNotIn(legacy, stage["gate"])

    def test_gate_scoring_and_best_selection(self):
        gate = load_course().stage("composite")["gate"]
        self.assertEqual(gate_failures(COMPOSITE_METRICS, gate), [])
        score = gate_score(COMPOSITE_METRICS, gate)
        self.assertTrue(score["passed"])
        self.assertGreater(score["score"], 0.0)
        candidates = [
            {
                "checkpoint": "a.pt",
                "iteration": 10,
                "passed": True,
                "score": 80.0,
                "worst_seed_score": 70.0,
                "minimum_margin": 0.2,
                "mean_speed_mae_mps": 0.1,
                "mean_yaw_mae_rad_s": 0.1,
            },
            {
                "checkpoint": "b.pt",
                "iteration": 20,
                "passed": True,
                "score": 90.0,
                "worst_seed_score": 60.0,
                "minimum_margin": 0.1,
                "mean_speed_mae_mps": 0.2,
                "mean_yaw_mae_rad_s": 0.2,
            },
        ]
        self.assertEqual(select_policy_best(candidates)["checkpoint"], "b.pt")
        with self.assertRaises(ValueError):
            select_policy_best([{**candidates[0], "passed": False}])

    def test_flat_gate_requires_settled_metrics(self):
        gate = load_course().stage("flat_balance")["gate"]
        metrics = {
            metric_name(key): (1.0 if key.startswith("min_") else 0.01)
            for key in gate
        }
        self.assertEqual(gate_failures(metrics, gate), [])
        score = gate_score(metrics, gate)
        self.assertTrue(score["passed"])
        metrics.pop("settled_mean_height_error_m")
        metrics["mean_height_error_m"] = 0.001
        failures = gate_failures(metrics, gate)
        self.assertTrue(
            any("settled_mean_height_error_m" in failure for failure in failures)
        )

    def test_gate_reports_mean_and_transient_wheel_slip(self):
        gate = {
            "max_mean_steady_wheel_slip_mps": 0.02,
            "max_transient_wheel_slip_mps": 0.05,
        }
        metrics = {
            "mean_steady_wheel_slip_mps": 0.03,
            "transient_wheel_slip_mps": 0.06,
        }
        failures = gate_failures(metrics, gate)
        self.assertTrue(
            any("mean_steady_wheel_slip_mps" in failure for failure in failures)
        )
        self.assertTrue(
            any("transient_wheel_slip_mps" in failure for failure in failures)
        )

    def test_cruise_speed_gate_is_directional_and_evaluable(self):
        gate = {
            "max_cruise_speed_mae_mps": 0.02,
            "min_cruise_speed_fraction": 0.9,
        }
        passing = {
            "cruise_speed_mae_mps": 0.015,
            "cruise_speed_fraction": 0.95,
        }
        self.assertEqual(gate_failures(passing, gate), [])
        self.assertTrue(gate_score(passing, gate)["passed"])
        failing = {
            "cruise_speed_mae_mps": 0.03,
            "cruise_speed_fraction": 0.8,
        }
        self.assertEqual(len(gate_failures(failing, gate)), 2)
        self.assertFalse(gate_score(failing, gate)["passed"])

    def test_bestmodel_selection_probes_one_seed_then_verifies_five(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "diablo_training" / "mjlab_tasks.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("[SELECTION_PROBE_SEED]", source)
        self.assertIn("fail_fast=True", source)
        self.assertIn('"early_stopped": early_stopped', source)
        self.assertIn('"completed_runs": len(runs)', source)
        self.assertIn('if not probe["passed"]:', source)
        self.assertIn("SELECTION_VERIFY_SEEDS", source)
        self.assertIn('output_dir = run_dir / "bestmodel"', source)
        self.assertIn("path.unlink(missing_ok=True)", source)
        self.assertIn('(output_dir / "selection.json").write_text(', source)
        self.assertIn(
            "No checkpoint passed the one-seed probe; verification was not run",
            source,
        )

        selector = (root / "select_best_current.py").read_text(encoding="utf-8")
        self.assertIn("select_policy_best(", selector)
        cli_source = (root / "diablo_training" / "cli.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('add_argument("--fail-fast", action="store_true")', cli_source)
        self.assertIn('"--warm-start"', cli_source)
        self.assertIn("warm_start=args.warm_start", cli_source)
        self.assertIn('"--transfer"', cli_source)
        self.assertIn('"--initial-std"', cli_source)
        self.assertIn('"--critic-calibration-steps"', cli_source)
        self.assertIn("load_warm_start_contract(", cli_source)
        task_source = (
            root / "diablo_training" / "mjlab_tasks.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"transfer": transfer', task_source)
        self.assertIn("critic_calibration_steps", task_source)
        self.assertIn("actor_parameters", task_source)

    def test_resume_requires_current_interface_contract(self):
        from diablo_training.cli import _resume_args

        with tempfile.TemporaryDirectory() as temp:
            log_root = Path(temp) / "logs" / "rsl_rl"
            run_dir = log_root / "diablo_xingtian_course" / "run"
            run_dir.mkdir(parents=True)
            checkpoint = run_dir / "model_1.pt"
            checkpoint.write_bytes(b"checkpoint")
            with self.assertRaisesRegex(ValueError, "contract.json"):
                _resume_args(checkpoint, log_root, "flat_balance")
            contract = interface_contract(load_course())
            training = stage_training_contract(load_course(), "flat_balance")
            stale = copy.deepcopy(training)
            stale["sha256"] = "stale"
            (run_dir / "contract.json").write_text(
                json.dumps(
                    {
                        "contract": contract,
                        "stage": "flat_balance",
                        "training_contract": stale,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "training contract differs"):
                _resume_args(checkpoint, log_root, "flat_balance")
            (run_dir / "contract.json").write_text(
                json.dumps(
                    {
                        "contract": contract,
                        "stage": "flat_balance",
                        "training_contract": training,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                _resume_args(checkpoint, log_root, "flat_balance"),
                [
                    "--agent.resume",
                    "True",
                    "--agent.load-run",
                    "run",
                    "--agent.load-checkpoint",
                    "model_1.pt",
                ],
            )

    def test_course_profile_interpolation(self):
        course = load_course()
        forward = course.stage("flat_motion_020")["profiles"][0]
        self.assertEqual(profile_command(forward, 2.0), (0.0, 0.0))
        self.assertEqual(profile_command(forward, 6.0), (0.2, 0.0))
        self.assertEqual(profile_command(forward, 99.0), (0.0, 0.0))

    def test_terrain_requirements(self):
        course = load_course()
        self.assertEqual(course.stage("slope")["terrain"]["grade_deg"], 30.0)
        self.assertEqual(course.stage("rough")["terrain"]["max_grade_deg"], 30.0)
        obstacle = course.stage("roller_obstacle")["terrain"]
        self.assertEqual(obstacle["min_height_m"], 0.02)
        self.assertEqual(obstacle["max_height_m"], 0.05)
        self.assertEqual(set(obstacle["types"]), {"box", "cylinder", "capsule"})
        composite = course.stage("composite")["terrain"]
        self.assertEqual(composite["max_grade_deg"], 30.0)
        self.assertEqual(composite["length_m"], 47.5)

    def test_gpu_training_dependency_contract(self):
        root = Path(__file__).resolve().parents[1]
        project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        dependencies = " ".join(project["project"]["dependencies"]).lower()
        self.assertNotIn("baselines", dependencies)
        self.assertIn("mjlab==1.6.0", dependencies)


@unittest.skipUnless(
    DEPENDENCIES_AVAILABLE, "CPU MuJoCo/Gymnasium dependencies are not installed"
)
class WheelMappingTests(unittest.TestCase):
    def test_residual_wheel_target_mapping_is_symmetric(self):
        commands = torch.tensor(
            [
                [0.2, 0.0, 0.35],
                [-0.2, 0.0, 0.35],
                [0.0, 1.0, 0.35],
                [0.0, -1.0, 0.35],
            ],
            dtype=torch.float32,
        )
        zero = torch.zeros((4, 2), dtype=torch.float32)
        targets = wheel_command_targets(
            commands,
            zero,
            radius=0.1,
            half_track=0.18755,
            feedforward_gain=1.0,
            yaw_feedforward_gain=1.0,
            residual_scale_rad_s=4.0,
        )
        torch.testing.assert_close(
            targets[:2],
            torch.tensor([[-2.0, -2.0], [2.0, 2.0]]),
            atol=1e-5,
            rtol=0.0,
        )
        torch.testing.assert_close(
            targets[2:],
            torch.tensor(
                [
                    [1.8755, -1.8755],
                    [-1.8755, 1.8755],
                ]
            ),
            atol=1e-5,
            rtol=0.0,
        )
        residual = torch.tensor(
            [[0.25, 0.0], [-0.25, 0.0]], dtype=torch.float32
        )
        shifted = wheel_command_targets(
            commands[:2],
            residual,
            radius=0.1,
            half_track=0.18755,
            feedforward_gain=1.0,
            yaw_feedforward_gain=1.0,
            residual_scale_rad_s=4.0,
        )
        torch.testing.assert_close(
            shifted,
            torch.tensor([[-3.0, -2.0], [3.0, 2.0]]),
            atol=1e-5,
            rtol=0.0,
        )


@unittest.skipUnless(
    DEPENDENCIES_AVAILABLE, "CPU MuJoCo/Gymnasium dependencies are not installed"
)
class EnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = DiabloEnv()

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_sensor_actuator_and_height_contract(self):
        env = self.env
        observation, _ = env.reset(seed=9)
        self.assertEqual(env.model.nu, 6)
        self.assertEqual(observation.shape, (125,))
        self.assertEqual(observation.dtype, np.float32)
        self.assertEqual(env.observation_space.dtype, np.float32)
        self.assertAlmostEqual(float(observation[108]), 0.0)
        self.assertEqual(env.action_space.shape, (6,))
        self.assertEqual(env.action_space.dtype, np.float32)
        self.assertTrue(env.observation_space.contains(observation))
        names = [env.model.joint(int(j)).name for j in env.model.actuator_trnid[:, 0]]
        self.assertEqual(names, list(JOINTS))
        self.assertEqual(env.data.ncon, 0, "Initial pose must not penetrate the ground")
        self.assertAlmostEqual(np.linalg.norm(observation[-22:-19]), 1.0, places=5)
        self.assertAlmostEqual(TARGET_BASE_HEIGHT_M, 0.350)
        self.assertEqual(CONTRACT_BASE_HEIGHT_M, TARGET_BASE_HEIGHT_M)
        self.assertAlmostEqual(TARGET_STANDING_HEIGHT_M, 0.477859)
        self.assertAlmostEqual(
            CONTRACT_STANDING_HEIGHT_M,
            TARGET_STANDING_HEIGHT_M,
        )
        self.assertAlmostEqual(RESET_CLEARANCE_M, 0.005)
        self.assertAlmostEqual(RESET_BASE_HEIGHT_M, 0.355)
        self.assertAlmostEqual(
            CONTRACT_RESET_BASE_HEIGHT_M,
            RESET_BASE_HEIGHT_M,
        )
        self.assertAlmostEqual(TARGET_WHOLE_HEIGHT_M, 0.491)
        self.assertAlmostEqual(
            env.data.xpos[env.base_id, 2], RESET_BASE_HEIGHT_M, places=4
        )
        wheel_bottom = float(
            env.data.xpos[env.model.body("left_wheel").id, 2] - 0.1
        )
        self.assertAlmostEqual(wheel_bottom, 0.005, places=4)
        np.testing.assert_allclose(
            DEFAULT_Q,
            [0.969324, -1.938616, 0.0, 0.969324, -1.938616, 0.0],
        )
        self.assertAlmostEqual(
            float(env.model.body_mass.sum()), DOCUMENTED_MODEL_MASS_KG
        )
        self.assertEqual(HIP_POSITION_GAIN, 15.0)
        self.assertEqual(KNEE_POSITION_GAIN, 18.0)
        self.assertEqual(HIP_DAMPING_GAIN, 0.3)
        self.assertEqual(KNEE_DAMPING_GAIN, 0.2)
        visual = env.model.geom_group == 1
        self.assertTrue(np.all(env.model.geom_contype[visual] == 0))
        self.assertTrue(np.all(env.model.geom_conaffinity[visual] == 0))

    def test_healthy_stance_geometry_and_joint_margins(self):
        env = self.env
        env.reset(seed=12)
        env.data.qpos[2] = TARGET_BASE_HEIGHT_M
        mujoco.mj_forward(env.model, env.data)

        wheel_ids = [
            env.model.body("left_wheel").id,
            env.model.body("right_wheel").id,
        ]
        wheel_centres = env.data.xpos[wheel_ids]
        wheel_bottom = float((wheel_centres[:, 2] - 0.1).mean())
        self.assertAlmostEqual(wheel_bottom, 0.0, places=5)

        base_geom = env.model.geom("diablo_base_link_visual_1")
        mesh_id = int(base_geom.dataid[0])
        start = int(env.model.mesh_vertadr[mesh_id])
        count = int(env.model.mesh_vertnum[mesh_id])
        rotation = env.data.geom_xmat[base_geom.id].reshape(3, 3)
        vertices = (
            env.model.mesh_vert[start : start + count] @ rotation.T
            + env.data.geom_xpos[base_geom.id]
        )
        self.assertAlmostEqual(
            float(vertices[:, 2].max() - wheel_bottom),
            TARGET_STANDING_HEIGHT_M,
            places=5,
        )

        centre_of_mass = env.data.subtree_com[env.base_id]
        wheel_x = float(wheel_centres[:, 0].mean())
        self.assertLess(abs(float(centre_of_mass[0]) - wheel_x), 0.002)
        self.assertLess(max(closure_errors(env.model, env.data)), 1e-6)

        contact_geoms = {
            (env.model.geom(contact.geom1).name, env.model.geom(contact.geom2).name)
            for contact in env.data.contact[: env.data.ncon]
        }
        self.assertTrue(contact_geoms)
        self.assertTrue(
            all("floor" in pair and "wheel" in " ".join(pair) for pair in contact_geoms)
        )

        for name in ("left_hip_motor", "right_hip_motor"):
            joint_id = env.model.joint(name).id
            lower, upper = env.model.jnt_range[joint_id]
            position = env.data.qpos[env.model.jnt_qposadr[joint_id]]
            midpoint = 0.5 * (lower + upper)
            soft_lower = midpoint - 0.475 * (upper - lower)
            soft_upper = midpoint + 0.475 * (upper - lower)
            self.assertGreaterEqual(np.degrees(soft_upper - position), 25.0)
            self.assertGreaterEqual(np.degrees(position - soft_lower), 25.0)

        for name in ("left_knee_motor", "right_knee_motor"):
            joint_id = env.model.joint(name).id
            lower, _ = env.model.jnt_range[joint_id]
            position = env.data.qpos[env.model.jnt_qposadr[joint_id]]
            self.assertGreaterEqual(np.degrees(position - lower), 45.0)

        collision_ids = [
            geom_id
            for geom_id in range(env.model.ngeom)
            if env.model.geom_bodyid[geom_id] != 0
            and env.model.geom_group[geom_id] == 0
        ]
        fromto = np.zeros(6)
        clearances = []
        for index, first in enumerate(collision_ids):
            first_body = int(env.model.geom_bodyid[first])
            for second in collision_ids[index + 1 :]:
                second_body = int(env.model.geom_bodyid[second])
                if first_body == second_body:
                    continue
                if (
                    env.model.body_parentid[first_body] == second_body
                    or env.model.body_parentid[second_body] == first_body
                ):
                    continue
                body_names = {
                    env.model.body(first_body).name,
                    env.model.body(second_body).name,
                }
                if any(
                    f"{side}_shank_link" in body_names
                    and f"{side}_parallel_link" in body_names
                    for side in ("left", "right")
                ):
                    continue
                clearances.append(
                    float(
                        mujoco.mj_geomDistance(
                            env.model,
                            env.data,
                            first,
                            second,
                            0.1,
                            fromto,
                        )
                    )
                )
        self.assertGreater(min(clearances), 0.02)
        env.reset(seed=12)

    def test_actual_physics_step_and_reset_reproducibility(self):
        env = self.env
        before, _ = env.reset(seed=10)
        action = np.array([0.1, -0.1, 0.2, 0.1, -0.1, 0.2])
        after, _, _, _, info = env.step(action)
        self.assertAlmostEqual(env.data.time, 0.01)
        self.assertTrue(np.isfinite(after).all())
        self.assertLessEqual(info["max_torque_nm"], STALL_TORQUE)
        np.testing.assert_allclose(after[-6:], action, atol=1e-7)
        again, _ = env.reset(seed=10)
        np.testing.assert_array_equal(before, again)

    def test_invalid_action_cannot_advance_physics(self):
        self.env.reset(seed=1)
        with self.assertRaises(ValueError):
            self.env.step(np.full(6, np.nan))
        self.assertEqual(self.env.data.time, 0.0)

    def test_cpu_command_profiles_and_config_validation(self):
        self.assertEqual(command_profile(5, 0, 0.5), 0.0)
        for task in (1, 2, 3):
            speeds = [command_profile(t, task, 0.5) for t in np.linspace(0, 20, 201)]
            self.assertGreater(max(speeds), 0.1)
            self.assertLessEqual(max(speeds), 0.5)
            self.assertEqual(command_profile(19, task, 0.5), 0.0)
        with self.assertRaises(ValueError):
            Config(max_speed=np.nan)
        with self.assertRaises(ValueError):
            Config(episode_seconds=0.0)

    def test_published_mechanism_and_virtual_work(self):
        result = validate_model()
        self.assertEqual(result["checks"], "passed")
        self.assertAlmostEqual(
            result["model"]["mass_kg"], DOCUMENTED_MODEL_MASS_KG
        )

    def test_velocity_body_axes(self):
        env = self.env
        env.reset(seed=1)
        velocity = [0.25, -0.13, 0.07]
        env.data.qvel[:3] = velocity
        mujoco.mj_forward(env.model, env.data)
        np.testing.assert_allclose(env._body_velocity(), velocity, atol=1e-12)
        env.data.qpos[3:7] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
        mujoco.mj_forward(env.model, env.data)
        np.testing.assert_allclose(
            env._body_velocity(), [-0.13, -0.25, 0.07], atol=1e-12
        )
        self.assertAlmostEqual(float(env._frame()[8]), 0.5 * np.pi * 5.0, places=5)

    def test_zero_action_holds_standing_pose(self):
        env = DiabloEnv(Config(episode_seconds=0.25))
        try:
            env.reset(seed=11)
            for _ in range(20):
                _, _, terminated, truncated, info = env.step(np.zeros(6))
                self.assertFalse(terminated)
                self.assertFalse(truncated)
                self.assertFalse(info["body_ground_contact"])
            self.assertGreater(float(env.data.xpos[env.base_id, 2]), 0.23)
        finally:
            env.close()

    def test_motor_envelope_and_wheel_sign(self):
        self.assertLess(float(radial_feedforward(DEFAULT_Q)[1]), 0.0)
        _, tau = decode_action(np.ones(6), DEFAULT_Q, np.zeros(6), "torque")
        np.testing.assert_array_equal(tau, [17, 17, -17, 17, 17, -17])
        _, fast = decode_action(
            np.ones(6), DEFAULT_Q, np.ones(6) * NO_LOAD_SPEED, "torque"
        )
        np.testing.assert_array_equal(fast, [0, 0, -17, 0, 0, -17])
        _, reverse = decode_action(
            -np.ones(6), DEFAULT_Q, -np.ones(6) * NO_LOAD_SPEED, "torque"
        )
        np.testing.assert_array_equal(reverse, [0, 0, 17, 0, 0, 17])

    def test_lifecycle_and_fault_latch(self):
        env = DiabloEnv(Config(episode_seconds=0.01))
        try:
            with self.assertRaises(RuntimeError):
                env.step(np.zeros(6))
            env.reset(seed=1)
            _, _, _, truncated, _ = env.step(np.zeros(6))
            self.assertTrue(truncated)
            with self.assertRaises(RuntimeError):
                env.step(np.zeros(6))
            self.assertTrue(
                env.model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_AUTORESET
            )
            env.reset(seed=1)
            with self.assertRaises(FloatingPointError):
                env._fault("test fault latch")
            with self.assertRaises(RuntimeError):
                env.reset()
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
