"""Dependency-free course tests plus CPU MuJoCo regression tests."""

import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import tomllib
import unittest

from diablo_training.contract import (
    BLIND_SEEDS,
    SCREEN_SEEDS,
    SELECTION_PROBE_SEED,
    SELECTION_VERIFY_SEEDS,
    VERIFY_SEEDS,
    checkpoint_interval,
    gate_failures,
    gate_score,
    interface_contract,
    load_course,
    metric_name,
    profile_command,
    select_policy_best,
    verify_interface_contract,
    verify_run_contract,
)

try:
    import mujoco
    import numpy as np

    from diablo_training.control import decode_action, radial_feedforward
    from diablo_training.env import Config, DiabloEnv, command_profile
    from diablo_training.model import (
        DEFAULT_Q,
        JOINTS,
        NO_LOAD_SPEED,
        OFFICIAL_MASS_KG,
        STALL_TORQUE,
        TARGET_BASE_HEIGHT_M,
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
            "DEFAULT_Q": (0.5, -1.0, 0.0, 0.5, -1.0, 0.0),
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
        self.assertEqual(positions["left_return_passive"], -1.0)
        self.assertEqual(positions["left_branch_passive"], 1.0)
        self.assertEqual(positions["right_return_passive"], -1.0)
        self.assertEqual(positions["right_branch_passive"], 1.0)
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
            '- float(w.get("straight_heading", 0.0)) * straight_heading_cost',
            source,
        )
        self.assertIn("def _huber(error: torch.Tensor, tolerance: float)", source)
        self.assertIn("ratio - 0.5", source)
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
        self.assertIn(
            "Metrics/straight_heading_error_{profile_name}_deg",
            source,
        )
        self.assertIn("command_term.target_heading", source)
        self.assertIn("command_observation = torch.stack(", source)
        self.assertIn("HEADING_ERROR_OBS_SCALE", source)
        self.assertIn('"reset_heading_range_deg"', source)
        self.assertIn('stage.get("observation_noise", 0.02)', source)
        self.assertIn('schedule="adaptive"', source)
        self.assertIn("hold_settle_time_s", source)
        self.assertIn("termination_tilt_deg", source)
        self.assertIn("drift_cost", source)
        self.assertIn("stop_quality", source)
        self.assertIn("settled_hold_drift_m", source)
        self.assertIn(
            '- float(w.get("drift", 0.0)) * drift_cost',
            source,
        )
        self.assertIn(
            '+ float(w.get("zero_speed", 0.0)) * stop_quality',
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
        self.assertIn("robot.data.heading_w - command_term.target_heading", source)
        self.assertIn("if frame.shape[-1] != PROPRIO_FRAME_DIM", source)
        self.assertIn("validate_policy_io(obs, action)", source)
        self.assertIn("_raw[:, self._leg_channels]", source)
        self.assertIn("_raw[:, self._wheel_channels]", source)
        self.assertNotIn("_raw[:, self._leg_ids]", source)
        self.assertNotIn("_raw[:, self._wheel_ids]", source)

    def test_course_order_and_seed_sets(self):
        course = load_course()
        self.assertEqual(len(course.order), 18)
        self.assertEqual(course.order[0], "flat_balance")
        self.assertEqual(course.order[-1], "composite")
        balance = course.stage("flat_balance")
        motion_020 = course.stage("flat_motion_020")
        self.assertEqual(balance["budget"], 40000000)
        self.assertEqual(balance["reset_heading_range_deg"], [-5.0, 5.0])
        self.assertEqual(balance["observation_noise"], 0.005)
        self.assertEqual(balance["hold_settle_time_s"], 1.0)
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
        self.assertEqual(balance["reward_defaults"]["yaw_tolerance_base_rad_s"], 0.05)
        self.assertEqual(
            balance["reward_defaults"]["zero_speed_tolerance_mps"],
            0.02,
        )
        self.assertEqual(balance["reward_defaults"]["heading_tolerance_rad"], 0.05236)
        self.assertEqual(balance["gate"]["max_final_heading_error_deg"], 3.0)
        self.assertEqual(balance["gate"]["max_settled_hold_drift_m"], 0.05)
        self.assertEqual(balance["gate"]["max_stop_speed_mps"], 0.02)
        self.assertEqual(balance["gate"]["max_wheel_slip_mps"], 0.08)
        self.assertEqual(
            balance["control"]["wheel_target_slew_rad_s2"],
            100.0,
        )
        self.assertGreater(balance["overrides"]["wheel_alignment"], 1.0)
        self.assertEqual(motion_020["reward_weights"]["straight_heading"], 1.0)
        self.assertEqual(motion_020["reward_weights"]["cross_track"], 1.0)
        self.assertEqual(motion_020["reward_weights"]["wheel_slip_gate"], 1.0)
        self.assertEqual(motion_020["reward_weights"]["stop_speed_gate"], 0.5)
        self.assertEqual(motion_020["reward_weights"]["wheel_sync_gate"], 1.5)
        self.assertEqual(motion_020["reward_weights"]["wheel_chatter"], 1.0)
        self.assertEqual(
            motion_020["reward_defaults"]["yaw_tolerance_base_rad_s"],
            0.05,
        )
        self.assertEqual(
            motion_020["reward_defaults"]["straight_heading_tolerance_rad"],
            0.05236,
        )
        self.assertEqual(
            motion_020["reward_defaults"]["gate_penalty_ema_decay"],
            0.98,
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

        incompatible = copy.deepcopy(contract)
        incompatible["action_order"] = list(reversed(incompatible["action_order"]))
        seal_contract(incompatible)
        with self.assertRaisesRegex(ValueError, "action_order"):
            verify_interface_contract(incompatible)
        self.assertEqual(contract["observation_shape"], 125)
        self.assertEqual(contract["version"], 7)
        self.assertEqual(contract["observation_dtype"], "float32")
        self.assertEqual(
            contract["observation_breakdown"],
            {"proprio_history": 125},
        )
        self.assertEqual(contract["action_shape"], 6)
        self.assertEqual(contract["action_dtype"], "float32")
        self.assertEqual(contract["target_base_height_m"], 0.270943)
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

    def test_bestmodel_selection_probes_one_seed_then_verifies_five(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "diablo_training" / "mjlab_tasks.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("[SELECTION_PROBE_SEED]", source)
        self.assertIn('if not probe["passed"]:', source)
        self.assertIn("SELECTION_VERIFY_SEEDS", source)
        self.assertIn('output_dir = run_dir / "bestmodel"', source)
        self.assertIn("path.unlink(missing_ok=True)", source)
        self.assertIn('(output_dir / "selection.json").write_text(', source)

        selector = (root / "select_best_current.py").read_text(encoding="utf-8")
        self.assertIn("select_policy_best(", selector)

    def test_resume_requires_current_interface_contract(self):
        from diablo_training.cli import _resume_args

        with tempfile.TemporaryDirectory() as temp:
            log_root = Path(temp) / "logs" / "rsl_rl"
            run_dir = log_root / "diablo_xingtian_course" / "run"
            run_dir.mkdir(parents=True)
            checkpoint = run_dir / "model_1.pt"
            checkpoint.write_bytes(b"checkpoint")
            with self.assertRaisesRegex(ValueError, "contract.json"):
                _resume_args(checkpoint, log_root)
            contract = interface_contract(load_course())
            (run_dir / "contract.json").write_text(
                json.dumps(
                    {"contract": contract, "stage": "flat_balance"},
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                _resume_args(checkpoint, log_root),
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
        self.assertEqual(course.stage("slope")["terrain"]["grade_deg"], 35.0)
        self.assertEqual(course.stage("rough")["terrain"]["max_grade_deg"], 35.0)
        obstacle = course.stage("roller_obstacle")["terrain"]
        self.assertEqual(obstacle["min_height_m"], 0.02)
        self.assertEqual(obstacle["max_height_m"], 0.05)
        self.assertEqual(set(obstacle["types"]), {"box", "cylinder", "capsule"})
        composite = course.stage("composite")["terrain"]
        self.assertEqual(composite["max_grade_deg"], 35.0)
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
        self.assertAlmostEqual(TARGET_BASE_HEIGHT_M, 0.270943)
        self.assertAlmostEqual(
            env.data.xpos[env.base_id, 2], TARGET_BASE_HEIGHT_M, places=4
        )
        self.assertAlmostEqual(float(env.model.body_mass.sum()), OFFICIAL_MASS_KG)
        visual = env.model.geom_group == 1
        self.assertTrue(np.all(env.model.geom_contype[visual] == 0))
        self.assertTrue(np.all(env.model.geom_conaffinity[visual] == 0))

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
        self.assertAlmostEqual(result["model"]["mass_kg"], OFFICIAL_MASS_KG)

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
