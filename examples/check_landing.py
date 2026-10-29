"""Exercise landing reset geometry, airborne success prevention and finite rollouts."""
import mujoco
import numpy as np
from stable_baselines3.common.env_checker import check_env
from diablo_training.env import Config
from landing_task import LandingEnv

env = LandingEnv(Config(episode_seconds=4.))
try:
    check_env(env, warn=True)
    for height in (.02, .035, .05):
        for seed in range(3):
            obs, _ = env.reset(seed=seed, options={"drop_height": height})
            initial_z = env.data.qpos[2]
            assert env.data.ncon == 0, "Drop must start airborne"
            assert not env.success
            assert abs(env.data.xpos[env.base_id, 2] - env.reference_height - height) < 1e-8
            for _ in range(400):
                obs, reward, done, truncated, info = env.step(np.zeros(6))
                assert np.isfinite(obs).all() and np.isfinite(reward)
                assert not info["landing_success"] or info["landed"]
                if not info["landed"]:
                    assert info["stable_seconds"] == 0.
                if done or truncated:
                    break
    for height in (0., .1, float("nan")):
        try:
            env.reset(options={"drop_height": height})
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid drop height accepted")
    print("Landing checks passed: Gymnasium, airborne resets, nine physical rollouts, invalid heights")
finally:
    env.close()
