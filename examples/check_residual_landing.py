"""Check controller ablation, bounded residuals and unchanged physics limits."""
import numpy as np
import argparse
from stable_baselines3.common.env_checker import check_env
from diablo_training.env import Config
from landing_task import LandingEnv
from landing_residual import ResidualLandingEnv, RESIDUAL_SCALE

parser = argparse.ArgumentParser()
parser.add_argument("--cushion", action="store_true")
parser.add_argument("--hold", action="store_true")
args = parser.parse_args()
if args.hold:
    from landing_hold import HoldLandingEnv
    env = HoldLandingEnv(Config(episode_seconds=4.))
elif args.cushion:
    from landing_cushion import CushionLandingEnv
    env = CushionLandingEnv(Config(episode_seconds=4.))
else:
    env = ResidualLandingEnv(Config(episode_seconds=4.))
reference = LandingEnv(Config(episode_seconds=4.))
try:
    check_env(env, warn=True)
    env.reset(seed=7, options={"drop_height": .02})
    reference.reset(seed=7, options={"drop_height": .02})
    for _ in range(400):
        feedback = env.feedback_action()
        _, _, done, truncated, info = env.step(np.zeros(6))
        reference.step(feedback)
        np.testing.assert_allclose(env.data.qpos, reference.data.qpos, atol=1e-10)
        np.testing.assert_allclose(env.data.qvel, reference.data.qvel, atol=1e-10)
        if done or truncated:
            break
    env.reset(seed=7)
    feedback = env.feedback_action()
    _, reward, _, _, info = env.step(np.ones(6)*10.)
    assert np.isfinite(reward)
    assert np.all(np.abs(np.asarray(info["applied_action"])-feedback) <= RESIDUAL_SCALE+1e-10)
    assert np.max(np.abs(env.data.ctrl)) <= 17.+1e-10
    for action in (np.zeros(5), np.ones(6)*np.nan):
        try:
            env.step(action)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid residual accepted")
    print("Passed: Gymnasium, zero-residual physical equivalence, bounded residual, torque limits, invalid actions")
finally:
    env.close()
    reference.close()
