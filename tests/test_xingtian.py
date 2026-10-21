"""Pure task-component tests independent of MuJoCo assets and CUDA."""

import unittest
from unittest import mock

import mujoco
import numpy as np

from diablo_training.xingtian import (
    CommandConfig,
    CommandScheduler,
    LocomotionReward,
    TERRAIN_LEVELS,
    body_height_ladder,
    fractal_heightfield,
    sample_obstacles,
    wheel_lift_matrix,
)
from diablo_training.xingtian_env import XingtianStageConfig
from diablo_training.xingtian_terrain import (
    build_terrain_profile,
    make_terrain_model,
    profile_height,
)


class XingtianReferenceTests(unittest.TestCase):
    def test_command_scheduler_is_bounded_and_deterministic(self):
        config = CommandConfig(
            max_forward_speed=1.0,
            max_backward_speed=0.8,
            max_yaw_rate=1.5,
        )
        first = CommandScheduler(config, np.random.default_rng(7))
        second = CommandScheduler(config, np.random.default_rng(7))
        for time in np.linspace(0.0, 20.0, 100):
            one = first.command_at(float(time))
            two = second.command_at(float(time))
            np.testing.assert_allclose(one, two)
            self.assertLessEqual(abs(one[0]), 1.0)
            self.assertLessEqual(abs(one[1]), 1.5)
            self.assertGreaterEqual(one[2], config.min_height)
            self.assertLessEqual(one[2], config.max_height)

    def test_heightfield_reproducible_and_bounded(self):
        level = TERRAIN_LEVELS[3]
        a = fractal_heightfield(level, seed=11, length_m=4, width_m=2)
        b = fractal_heightfield(level, seed=11, length_m=4, width_m=2)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.isfinite(a).all())
        self.assertLessEqual(float(np.max(np.abs(a))), level.peak_height + 1e-6)
        obstacles = sample_obstacles(level, seed=11)
        self.assertTrue(all(item.height_m <= level.obstacle_height for item in obstacles))

    def test_reward_is_finite_for_normal_and_terminal_transition(self):
        reward = LocomotionReward()
        transition = {
            "command": np.array([0.5, 0.0, 0.25]),
            "velocity": np.zeros(3),
            "gyro": np.zeros(3),
            "torque": np.zeros(6),
            "upright": 1.0,
            "height_error": 0.02,
            "terminated": False,
        }
        value = reward.evaluate(
            transition, np.zeros(6), np.zeros(6), dt=0.01
        ).total
        self.assertTrue(np.isfinite(value))
        transition["terminated"] = True
        terminal = reward.evaluate(
            transition, np.zeros(6), np.zeros(6), dt=0.01
        ).total
        self.assertLess(terminal, value)

    def test_wheel_lift_matrix_covers_both_sides(self):
        heights = body_height_ladder(0.16, 0.34)
        matrix = wheel_lift_matrix(heights)
        self.assertEqual(len(matrix), 2 * len(heights) * 8 * 4)
        self.assertEqual({item["side"] for item in matrix}, {"left", "right"})

    def test_stage_configs_and_terrain_profiles(self):
        for stage, level in (("B", 1), ("C", 1), ("D", 4)):
            config = XingtianStageConfig(
                stage=stage,
                terrain_level=level,
                terrain_seed=17,
                obstacle_height_m=0.03 if stage in {"C", "D"} else 0.0,
            )
            profile = build_terrain_profile(
                stage=config.stage,
                level_index=config.terrain_level,
                seed=config.terrain_seed,
                track_length_m=config.track_length_m,
                track_width_m=config.track_width_m,
                obstacle_height_m=config.obstacle_height_m or None,
            )
            self.assertTrue(np.isfinite(profile.heightfield).all())
            self.assertEqual(profile.heightfield.ndim, 2)
            self.assertTrue(np.isfinite(profile_height(profile, 10.0, 0.0)))
            if stage in {"C", "D"}:
                self.assertTrue(profile.obstacles)
                self.assertTrue(all(item["height"] <= 0.03 for item in profile.obstacles))

    def test_terrain_model_contains_hfield_obstacles_and_safety_floor(self):
        spec = mujoco.MjSpec()
        body = spec.worldbody.add_body(name="dummy_robot")
        body.add_freejoint(name="floating_base")
        body.add_geom(
            name="dummy_body",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.1, 0, 0],
        )
        profile = build_terrain_profile(
            stage="D",
            level_index=2,
            seed=9,
            track_length_m=20,
            track_width_m=4,
            obstacle_height_m=0.03,
        )
        with mock.patch(
            "diablo_training.xingtian_terrain._robot_spec", return_value=spec
        ):
            model = make_terrain_model(profile)
        names = {model.geom(i).name for i in range(model.ngeom)}
        self.assertIn("terrain_surface", names)
        self.assertIn("floor", names)
        self.assertIn("lane_wall_left", names)
        self.assertIn("lane_wall_right", names)
        self.assertTrue(any(name.startswith("obstacle_") for name in names))

    def test_mujoco_hfield_orientation_matches_height_query(self):
        spec = mujoco.MjSpec()
        body = spec.worldbody.add_body(name="dummy_robot")
        body.add_freejoint(name="floating_base")
        body.add_geom(
            name="dummy_body",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.1, 0, 0],
        )
        profile = build_terrain_profile(
            stage="B",
            level_index=2,
            seed=13,
            track_length_m=20,
            track_width_m=4,
        )
        with mock.patch(
            "diablo_training.xingtian_terrain._robot_spec", return_value=spec
        ):
            model = make_terrain_model(profile)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        for x, y in ((14.0, -1.0), (15.0, 0.0), (16.0, 1.0)):
            distance = mujoco.mj_ray(
                model,
                data,
                np.array([x, y, 2.0]),
                np.array([0.0, 0.0, -1.0]),
                None,
                1,
                -1,
                None,
            )
            self.assertAlmostEqual(
                2.0 - distance,
                profile_height(profile, x, y),
                places=4,
            )


if __name__ == "__main__":
    unittest.main()
