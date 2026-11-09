from __future__ import annotations

import json
from pathlib import Path
import random
import shutil
import tempfile
import unittest
from unittest import mock

from diablo_training import contract
from diablo_training.adaptive_grid import AdaptiveGrid
from diablo_training.contract import (
    gate_score,
    load_course,
    resolved_profile_gate,
    resolved_task_gate,
    stage_profiles,
    task_id,
)


class AdaptiveGridTests(unittest.TestCase):
    def test_grid_respects_speed_yaw_and_radius_limits(self) -> None:
        course = load_course()
        grid = AdaptiveGrid(course.stage("flat_adaptive_speed")["adaptive_grid"])
        self.assertTrue(grid.cells)
        for cell in grid.cells.values():
            if cell.mode == "straight":
                self.assertGreaterEqual(abs(cell.vx), 0.2)
            if cell.mode == "in_place_turn":
                self.assertGreaterEqual(abs(cell.wz), 0.4)
            if cell.mode == "moving_turn":
                radius = abs(cell.vx / cell.wz)
                self.assertGreaterEqual(radius, 0.666667 - 1e-6)
                self.assertLessEqual(radius, 10.0 + 1e-6)
                self.assertLessEqual(abs(cell.wz), 1.2 + 1e-6)
            if cell.mode == "moving_turn":
                self.assertLessEqual(abs(cell.vx_bounds[1]), 0.9 + 1e-6)

    def test_seed_cells_are_mirrored_and_protected(self) -> None:
        course = load_course()
        grid = AdaptiveGrid(course.stage("flat_adaptive_speed")["adaptive_grid"])
        self.assertIn((4, 0), grid.protected)
        self.assertIn((-4, 0), grid.protected)
        self.assertIn((3, 4), grid.protected)
        self.assertIn((-3, -4), grid.protected)
        self.assertIn((0, 4), grid.protected)
        self.assertIn((0, -4), grid.protected)
        self.assertNotIn((0, 6), grid.protected)

    def test_sampling_enforces_task_and_direction_quotas(self) -> None:
        course = load_course()
        grid = AdaptiveGrid(course.stage("flat_adaptive_speed")["adaptive_grid"])
        cells = [grid.cells[key] for key in grid.sample_cell_keys(4000)]
        counts = {
            mode: sum(cell.mode == mode for cell in cells)
            for mode in grid.MODE_ORDER
        }
        self.assertEqual(
            counts,
            {
                "balance": 800,
                "straight": 1200,
                "in_place_turn": 1000,
                "moving_turn": 1000,
            },
        )
        for mode, axis in (
            ("straight", "vx_index"),
            ("in_place_turn", "wz_index"),
            ("moving_turn", "wz_index"),
        ):
            positive = sum(
                cell.mode == mode and getattr(cell, axis) > 0 for cell in cells
            )
            negative = sum(
                cell.mode == mode and getattr(cell, axis) < 0 for cell in cells
            )
            self.assertEqual(positive, negative)
        forward = sum(
            cell.mode == "moving_turn" and cell.vx_index > 0 for cell in cells
        )
        backward = sum(
            cell.mode == "moving_turn" and cell.vx_index < 0 for cell in cells
        )
        self.assertEqual(forward, backward)

    def test_single_env_resets_also_follow_sampling_quotas(self) -> None:
        course = load_course()
        grid = AdaptiveGrid(course.stage("flat_adaptive_speed")["adaptive_grid"])
        cells = [
            grid.cells[grid.sample_cell_keys(1)[0]]
            for _ in range(400)
        ]
        counts = {
            mode: sum(cell.mode == mode for cell in cells)
            for mode in grid.MODE_ORDER
        }
        self.assertEqual(
            counts,
            {
                "balance": 80,
                "straight": 120,
                "in_place_turn": 100,
                "moving_turn": 100,
            },
        )
        for mode, axis in (
            ("straight", "vx_index"),
            ("in_place_turn", "wz_index"),
            ("moving_turn", "wz_index"),
        ):
            positive = sum(
                cell.mode == mode and getattr(cell, axis) > 0 for cell in cells
            )
            negative = sum(
                cell.mode == mode and getattr(cell, axis) < 0 for cell in cells
            )
            self.assertEqual(positive, negative)

    def test_sampling_size_stays_exact_when_active_grid_changes(self) -> None:
        course = load_course()
        grid = AdaptiveGrid(course.stage("flat_adaptive_speed")["adaptive_grid"])
        keys = sorted(grid.cells)
        rng = random.Random(17)
        for _ in range(2000):
            if rng.random() < 0.1:
                key = rng.choice(keys)
                if rng.random() < 0.5:
                    grid.active.add(key)
                else:
                    grid.active.discard(key)
            count = rng.randint(1, 64)
            self.assertEqual(len(grid.sample_cell_keys(count)), count)

    def test_task_gates_are_not_shared_blindly(self) -> None:
        course = load_course()
        profiles = stage_profiles(course.stage("flat_adaptive_speed"))
        by_name = {profile["name"]: profile for profile in profiles}
        balance = resolved_profile_gate(
            course, "flat_adaptive_speed", by_name["balance_hold"]
        )
        straight = resolved_profile_gate(
            course, "flat_adaptive_speed", by_name["straight_forward_review"]
        )
        in_place = resolved_profile_gate(
            course, "flat_adaptive_speed", by_name["in_place_left_review"]
        )
        moving = resolved_profile_gate(
            course, "flat_adaptive_speed", by_name["moving_forward_review"]
        )
        moving_stage = resolved_profile_gate(
            course,
            "flat_turn_moving_120",
            course.stage("flat_turn_moving_120")["profiles"][0],
        )
        self.assertNotIn("min_forward_progress_m", balance)
        self.assertNotIn("max_cross_track_error_m", balance)
        self.assertIn("min_forward_progress_m", straight)
        self.assertNotIn("min_forward_progress_m", in_place)
        self.assertIn("max_in_place_displacement_m", in_place)
        self.assertIn("min_turn_radius_m", moving)
        self.assertIn("max_turn_radius_m", moving)
        self.assertIn("max_s_turn_path_error_m", moving)
        self.assertIn("max_segment_turn_angle_error_rad", moving)
        self.assertEqual(moving["max_wheel_center_offset_m"], 0.045)
        self.assertNotIn("max_cross_track_error_m", moving)
        health_gate = course.stage("flat_adaptive_speed")["adaptive_grid"][
            "health_gate"
        ]
        self.assertEqual(health_gate["min_duration_fraction"], 0.98)
        self.assertEqual(health_gate["min_upright"], 0.87)
        self.assertEqual(health_gate["max_pitch_deg"], 10.0)
        self.assertNotIn("max_wheel_center_offset_m", health_gate)
        self.assertNotIn("max_speed_mae_mps", health_gate)
        self.assertNotIn("max_yaw_mae_rad_s", health_gate)
        self.assertNotIn("max_cross_track_error_m", moving_stage)
        self.assertIn("max_s_turn_path_error_m", moving_stage)
        self.assertEqual(
            resolved_task_gate(course, "flat_adaptive_speed", "moving_turn"),
            moving,
        )
        legacy_balance = resolved_profile_gate(
            course,
            "flat_balance",
            course.stage("flat_balance")["profiles"][0],
        )
        self.assertNotIn("min_forward_progress_m", legacy_balance)
        self.assertNotIn("max_cross_track_error_m", legacy_balance)
        turning = course.stage("flat_turning")
        by_name = {profile["name"]: profile for profile in turning["profiles"]}
        left = resolved_profile_gate(
            course, "flat_turning", by_name["turn_left"]
        )
        forward = resolved_profile_gate(
            course, "flat_turning", by_name["turn_forward"]
        )
        self.assertNotIn("min_forward_progress_m", left)
        self.assertIn("min_turn_radius_m", forward)
        score = gate_score(
            {"turn_radius_min_m": 1.0, "turn_radius_max_m": 2.0},
            {"min_turn_radius_m": 0.666667, "max_turn_radius_m": 10.0},
        )
        self.assertTrue(score["passed"])

    def test_success_expands_and_failure_deactivates_optional_cell(self) -> None:
        course = load_course()
        grid = AdaptiveGrid(course.stage("flat_adaptive_speed")["adaptive_grid"])
        start = (4, 0)
        before = set(grid.active)
        grid.record([(start, True, True)] * 128)
        self.assertEqual(set(grid.active), before)
        for key in grid.protected:
            if grid.cells[key].mode == "straight":
                grid.last_health[key] = 1.0
                grid.health_pass_streak[key] = 2
        grid.record([(start, True, True)] * 128)
        self.assertGreater(len(grid.active), len(before))

        optional = next(
            key for key in grid.active if key not in grid.protected
        )
        grid.record([(optional, False, False)] * 128)
        self.assertNotIn(optional, grid.active)

    def test_protected_health_requires_two_consecutive_windows(self) -> None:
        course = load_course()
        grid = AdaptiveGrid(course.stage("flat_adaptive_speed")["adaptive_grid"])
        start = (4, 0)
        grid.protected = {start}
        grid.active = {start}
        grid.record([(start, True, True)] * 64)
        self.assertFalse(grid._protected_health_ready("straight"))
        grid.record([(start, True, True)] * 64)
        self.assertTrue(grid._protected_health_ready("straight"))

    def test_expansion_uses_eighty_percent_success_window(self) -> None:
        course = load_course()
        grid = AdaptiveGrid(course.stage("flat_adaptive_speed")["adaptive_grid"])
        start = (4, 0)
        grid.protected = {start}
        grid.active = {start}
        window = [True] * 52 + [False] * 12
        grid.record([(start, success, success) for success in window] * 2)
        self.assertGreater(grid.expansions, 0)

        strict = AdaptiveGrid(
            course.stage("flat_adaptive_speed")["adaptive_grid"]
        )
        strict.protected = {start}
        strict.active = {start}
        below_threshold = [True] * 48 + [False] * 16
        strict.record(
            [(start, success, success) for success in below_threshold] * 2
        )
        self.assertEqual(strict.expansions, 0)

    def test_vector_eval_captures_then_resets_finished_envs(self) -> None:
        root = Path(__file__).resolve().parents[1] / "diablo_training"
        source = (
            root / "evaluation.py"
        ).read_text(encoding="utf-8")
        task_source = (root / "mjlab_tasks.py").read_text(encoding="utf-8")
        cli_source = (root / "cli.py").read_text(encoding="utf-8")
        capture = source.index("metrics = reward_term.metrics()")
        done_now = source.index("reset_ids = done_now.nonzero().flatten()")
        reset = source.index("env.reset(env_ids=reset_ids)")
        merge = source.index("obs[key][reset_ids] =")
        self.assertLess(capture, reset)
        self.assertLess(capture, done_now)
        self.assertLess(done_now, reset)
        self.assertLess(reset, merge)
        self.assertIn("for key in obs.keys():", source)
        self.assertIn('getattr(env, "_manual_reset_pending", None)', source)
        self.assertNotIn(
            "if len(reset_ids) and not bool(done_envs.all().item()):",
            source,
        )
        self.assertNotIn("inference_mode()", source)
        self.assertNotIn("inference_mode()", task_source)
        self.assertNotIn("def _run_episode(", task_source)
        self.assertIn(
            "include_training_profiles=include_training_profiles",
            source,
        )
        check_script = (
            root.parent / "scripts" / "check_training.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--eval-mode fast --eval-envs 4", check_script)
        self.assertIn('select_parser.add_argument("--stage"', cli_source)

    def test_adaptive_stability_and_terminal_reward_contract(self) -> None:
        root = Path(__file__).resolve().parents[1] / "diablo_training"
        stage = load_course().stage("flat_adaptive_speed")
        self.assertEqual(stage["adaptive_grid"]["expand_success_rate"], 0.8)
        self.assertNotIn(
            "min_pass_rate", stage["adaptive_grid"]["health_gate"]
        )
        self.assertNotIn(
            "max_wheel_center_offset_m",
            stage["adaptive_grid"]["health_gate"],
        )
        self.assertEqual(stage["ppo"]["learning_rate"], 0.00005)
        self.assertEqual(stage["ppo"]["max_learning_rate"], 0.0001)
        self.assertEqual(stage["ppo"]["std_min"], 0.05)
        self.assertEqual(stage["ppo"]["std_max"], 0.35)
        self.assertEqual(stage["ppo"]["ent_coef"], 0.001)
        self.assertEqual(stage["ppo"]["target_kl"], 0.01)
        self.assertEqual(stage["reward_weights"]["profile_completion"], 4.0)
        self.assertEqual(stage["reward_weights"]["quality_completion"], 16.0)
        self.assertEqual(stage["reward_weights"]["s_turn_path"], 0.8)
        self.assertEqual(
            stage["reward_weights"]["profile_completion"]
            + stage["reward_weights"]["quality_completion"],
            20.0,
        )

        source = (root / "mjlab_tasks.py").read_text(encoding="utf-8")
        self.assertIn("SELECTION_MIN_EVAL_ENVS = 512", source)
        self.assertIn(
            "eval_envs = max(int(eval_envs), SELECTION_MIN_EVAL_ENVS)",
            source,
        )
        self.assertIn("self._bind_log_dir_to_env()", source)
        self.assertIn('setattr(target, "log_dir", str(log_dir))', source)
        self.assertIn(
            'getattr(self._env, "log_dir", None) or getattr(',
            source,
        )
        self.assertIn("path.parent.mkdir(parents=True, exist_ok=True)", source)
        self.assertIn(
            "self._stop_after_motion |= (\n"
            "            self._last_cmd.abs() >= self.stop_after_motion_arm_speed",
            source,
        )
        self.assertIn("self._completion_awarded", source)
        self.assertIn('reward_components["profile_completion"]', source)
        self.assertIn('reward_components["quality_completion"]', source)
        self.assertIn("turning_path_error", source)
        self.assertIn("straight_path_error", source)
        self.assertIn("gates=reward_term.health_gates", source)
        self.assertIn("success = active & health", source)
        self.assertIn("quality_score = self.gate_quality(self.metrics())", source)
        self.assertIn(
            "completion_now.float()\n"
            "            * quality_score",
            source,
        )
        self.assertIn('moving_task = self.task_mask("moving_turn").float()', source)
        self.assertIn('straight_task = self.task_mask("straight").float()', source)
        self.assertIn(
            "torch.where(\n"
            "                moving_task.bool(),\n"
            "                segment_turn_angle_error,\n"
            "                turn_angle_error,\n"
            "            )",
            source,
        )
        self.assertIn(
            "* torch.square(path_lateral_velocity)\n"
            "                * straight_task",
            source,
        )
        self.assertNotIn("flat_adaptive_speed must warm-start from baseline", source)
        cli_source = (root.parent / "diablo_training" / "cli.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "flat_adaptive_speed requires --warm-start with a compatible",
            cli_source,
        )
        task_source = (root / "mjlab_tasks.py").read_text(encoding="utf-8")
        self.assertIn(
            "architecture for alternate geometry",
            task_source,
        )
        self.assertIn(
            "speed_limit=self.cfg.wheel_speed_target_max_rad_s",
            task_source,
        )
        self.assertIn(
            'stage_yaw_limit = self.stage.get("limits", {}).get("yaw")',
            task_source,
        )
        self.assertNotIn(
            'if self._grid is not None or "adaptive_grid" in self.stage:',
            task_source,
        )
        self.assertIn("health &= (duration >= 0.98) & ~fell", source)

    def test_effective_profiles_use_central_timing(self) -> None:
        course = load_course()
        balance = course.stage("flat_balance")["profiles"][0]["points"]
        self.assertEqual(balance[-1][0], 20.0)

        straight = course.stage("flat_motion_020")["profiles"][0]["points"]
        self.assertEqual(straight[2][0] - straight[1][0], 4.0)
        self.assertEqual(straight[4][0] - straight[3][0], 4.0)

        in_place = course.stage("flat_turn_in_place_040")["profiles"][0][
            "points"
        ]
        self.assertEqual(in_place[2][0] - in_place[1][0], 2.0)
        self.assertEqual(in_place[3][0] - in_place[2][0], 10.0)
        self.assertEqual(in_place[4][0] - in_place[3][0], 2.0)

        moving = course.stage("flat_turn_moving_120")["profiles"][0]["points"]
        self.assertEqual(moving[2][0] - moving[1][0], 4.0)
        self.assertEqual(moving[3][0] - moving[2][0], 2.0)
        self.assertAlmostEqual(max(abs(point[1]) for point in moving), 0.9)
        self.assertAlmostEqual(max(abs(point[2]) for point in moving), 1.2)
        self.assertEqual(
            course.stage("flat_turn_moving_120")["control"][
                "wheel_speed_target_max_rad_s"
            ],
            16.0,
        )
        self.assertAlmostEqual(
            max(
                abs(point[1])
                for profile in course.stage("flat_turn_moving_080")["profiles"]
                for point in profile["points"]
            ),
            0.6,
        )
        self.assertEqual(
            course.stage("flat_turn_moving_080")["control"].get(
                "wheel_half_track_m", 0.24125
            ),
            0.24125,
        )
        self.assertEqual(
            course.stage("flat_turn_moving_080")["control"].get(
                "wheel_common_residual_limit_fraction"
            ),
            0.2,
        )
        for stage in course.stages.values():
            self.assertAlmostEqual(
                stage.get("control", {}).get("wheel_half_track_m", 0.24125),
                0.24125,
            )
        self.assertGreaterEqual(
            0.9 / 1.2,
            course.objectives["safety"]["min_turn_radius_m"],
        )
        adaptive_profiles = {
            profile["name"]: profile
            for profile in stage_profiles(
                course.stage("flat_adaptive_speed"),
                profile_set="all",
            )
        }
        self.assertEqual(
            adaptive_profiles["in_place_left_review"]["profile_set"],
            "target",
        )

        start_stop = course.stage("flat_start_stop")["profiles"][0]["points"]
        self.assertAlmostEqual(start_stop[2][0] - start_stop[1][0], 4.0)
        self.assertAlmostEqual(start_stop[4][0] - start_stop[3][0], 4.0)

    def test_adaptive_profiles_reference_single_objective_source(self) -> None:
        root = Path(__file__).resolve().parents[1] / "diablo_training"
        source = json.loads(
            (root / "flat_adaptive_speed_stages.json").read_text(encoding="utf-8")
        )
        text = json.dumps(source)
        self.assertIn("${target.straight_speed_max_mps}", text)
        self.assertIn("${target.moving_turn_speed_max_mps}", text)
        self.assertIn("${review.straight_speed_max_mps}", text)
        self.assertIn("${target.moving_turn_yaw_max_rad_s}", text)
        required = stage_profiles(
            load_course().stage("flat_adaptive_speed")
        )
        self.assertFalse(
            any(
                profile.get("profile_set") == "diagnostic"
                for profile in required
            )
        )
        self.assertNotIn(
            "flat_motion_120",
            load_course().order,
        )
        self.assertIn("flat_motion_100", load_course().order)
        self.assertAlmostEqual(
            max(
                abs(float(point[1]))
                for profile in load_course().stage("flat_motion")["profiles"]
                for point in profile["points"]
            ),
            1.0,
        )
        self.assertEqual(
            task_id("flat_adaptive_speed"),
            "Mjlab-Xingtian-Flat-AdaptiveSpeed",
        )

    def test_changing_target_speed_updates_profiles_and_grid(self) -> None:
        root = Path(__file__).resolve().parents[1] / "diablo_training"
        with tempfile.TemporaryDirectory() as temp:
            temp_root = Path(temp)
            for name in (
                "curriculum.json",
                "flat_motion_speed_stages.json",
                "flat_turning_speed_stages.json",
                "flat_adaptive_speed_stages.json",
                "training_objectives.json",
            ):
                shutil.copy2(root / name, temp_root / name)
            objectives_path = temp_root / "training_objectives.json"
            objectives = json.loads(objectives_path.read_text(encoding="utf-8"))
            objectives["target"]["straight_speed_max_mps"] = 1.2
            objectives["review"]["straight_speed_max_mps"] = 0.6
            objectives_path.write_text(
                json.dumps(objectives), encoding="utf-8"
            )
            with (
                mock.patch.object(
                    contract, "CURRICULUM_PATH", temp_root / "curriculum.json"
                ),
                mock.patch.object(
                    contract,
                    "SPEED_STAGES_PATH",
                    temp_root / "flat_motion_speed_stages.json",
                ),
                mock.patch.object(
                    contract,
                    "TURN_STAGES_PATH",
                    temp_root / "flat_turning_speed_stages.json",
                ),
                mock.patch.object(
                    contract,
                    "ADAPTIVE_STAGES_PATH",
                    temp_root / "flat_adaptive_speed_stages.json",
                ),
                mock.patch.object(
                    contract,
                    "OBJECTIVES_PATH",
                    temp_root / "training_objectives.json",
                ),
            ):
                course = load_course()
                stage = course.stage("flat_adaptive_speed")
                target = next(
                    profile
                    for profile in stage["profiles"]
                    if profile["name"] == "straight_forward_target"
                )
                self.assertEqual(target["points"][2][1], 1.2)
                moving_target = next(
                    profile
                    for profile in stage["profiles"]
                    if profile["name"] == "moving_forward_speed_diagnostic"
                )
                self.assertEqual(moving_target["points"][2][1], 0.9)
                review = next(
                    profile
                    for profile in stage["profiles"]
                    if profile["name"] == "straight_forward_review"
                )
                self.assertEqual(review["points"][2][1], 0.6)
                grid = AdaptiveGrid(stage["adaptive_grid"])
                self.assertAlmostEqual(
                    max(abs(cell.vx) for cell in grid.cells.values()), 1.2
                )
                self.assertIn((4, 0), grid.protected)
                self.assertIn("flat_motion_120", course.order)


if __name__ == "__main__":
    unittest.main()
