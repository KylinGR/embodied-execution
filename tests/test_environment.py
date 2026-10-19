"""Physics/interface checks that do not require a trained policy or a GPU."""

import unittest

import numpy as np

from diablo_training.env import Config, DiabloEnv, command_profile
from diablo_training.model import JOINTS


class EnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = DiabloEnv()

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_sensor_and_actuator_contract(self):
        env = self.env
        observation, _ = env.reset(seed=9)
        self.assertEqual(env.model.nu, 6)
        self.assertEqual(observation.shape, (125,))
        self.assertTrue(env.observation_space.contains(observation))
        names = [env.model.joint(int(j)).name for j in env.model.actuator_trnid[:, 0]]
        self.assertEqual(names, list(JOINTS))
        self.assertEqual(env.data.ncon, 0, "Initial pose must not penetrate the ground")
        self.assertAlmostEqual(np.linalg.norm(observation[-22:-19]), 1.0, places=5)

    def test_actual_physics_step_and_reset_reproducibility(self):
        env = self.env
        before, _ = env.reset(seed=10)
        action = np.array([0.1, -0.1, 0.2, 0.1, -0.1, 0.2])
        after, _, _, _, info = env.step(action)
        self.assertAlmostEqual(env.data.time, 0.01)
        self.assertTrue(np.isfinite(after).all())
        self.assertLessEqual(info["max_torque_nm"], 27.0)
        np.testing.assert_allclose(after[-6:], action, atol=1e-7)
        again, _ = env.reset(seed=10)
        np.testing.assert_array_equal(before, again)

    def test_invalid_action_cannot_advance_physics(self):
        self.env.reset(seed=1)
        with self.assertRaises(ValueError):
            self.env.step(np.full(6, np.nan))
        self.assertEqual(self.env.data.time, 0.0)

    def test_courses_reach_motion_and_then_hold(self):
        self.assertEqual(command_profile(5, 0, 0.5), 0.0)
        for task in (1, 2, 3):
            speeds = [command_profile(t, task, 0.5) for t in np.linspace(0, 20, 201)]
            self.assertGreater(max(speeds), 0.1)
            self.assertLessEqual(max(speeds), 0.5)
            self.assertEqual(command_profile(19, task, 0.5), 0.0)

    def test_unsafe_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            Config(max_speed=2.0)


if __name__ == "__main__":
    unittest.main()
