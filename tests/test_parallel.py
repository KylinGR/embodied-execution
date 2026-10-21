"""CPU vectorisation and device-selection checks."""

from pathlib import Path
import sys
import tempfile
import unittest

import gymnasium as gym
import numpy as np

from diablo_training.parallel import make_vector_env, resolve_device


class _CountingEnv(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)

    def __init__(self, _config=None):
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        return (
            np.full(3, self.steps / 10, dtype=np.float32),
            float(action[0]),
            False,
            self.steps >= 2,
            {},
        )


class ParallelEnvironmentTests(unittest.TestCase):
    def test_auto_device_resolves_to_real_torch_device(self):
        device = resolve_device("auto")
        self.assertIn(device, {"cpu", "cuda"})

    def test_dummy_vector_env_batches_physics(self):
        with tempfile.TemporaryDirectory() as directory:
            env, kind = make_vector_env(
                _CountingEnv,
                None,
                n_envs=2,
                kind="dummy",
                monitor_path=str(Path(directory) / "episodes.csv"),
            )
            try:
                env.seed(4)
                observations = env.reset()
                self.assertEqual(kind, "dummy")
                self.assertEqual(observations.shape, (2, 3))
                observations, rewards, dones, infos = env.step(
                    np.zeros((2, 1), dtype=np.float32)
                )
                self.assertEqual(observations.shape, (2, 3))
                self.assertEqual(rewards.shape, (2,))
                self.assertEqual(dones.shape, (2,))
                self.assertEqual(len(infos), 2)
            finally:
                env.close()

    @unittest.skipIf(sys.platform == "win32", "SubprocVecEnv target is Linux")
    def test_subprocess_vector_env_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            env, kind = make_vector_env(
                _CountingEnv,
                None,
                n_envs=2,
                kind="subproc",
                monitor_path=str(Path(directory) / "episodes.csv"),
            )
            try:
                env.seed(5)
                self.assertEqual(env.reset().shape, (2, 3))
                self.assertEqual(kind, "subproc")
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()
