"""Regression checks for explicit observation state and versioned compatibility."""
import numpy as np
import argparse
from stable_baselines3.common.env_checker import check_env
from diablo_training.env import Config,DiabloEnv
from diablo_training.cli import load_policy
from diablo_training.contract import policy_contract,require_compatible
from stable_landing_v3 import StableLandingEnv,WideStableLandingEnv
from qualify_landing import contact_state

parser=argparse.ArgumentParser();parser.add_argument("--fast",action="store_true");args=parser.parse_args()
if args.fast:
    from stable_landing_fast_v3 import FastStableLandingEnv,WideFastStableLandingEnv
    cls,other=WideFastStableLandingEnv,FastStableLandingEnv
else:cls,other=StableLandingEnv,WideStableLandingEnv
env=cls(Config());reference=cls(Config());wide=other(Config())
try:
    check_env(env,warn=True)
    obs,_=env.reset(seed=42,options={"drop_height":.02})
    reference.reset(seed=42,options={"drop_height":.02})
    assert obs.shape==(143,) and np.isfinite(obs).all()
    assert not contact_state(env)[0],"Drop must start airborne"
    for _ in range(2000):
        base=reference.feedback_action()
        obs,reward,done,truncated,info=env.step(np.zeros(6))
        DiabloEnv.step(reference,base)
        np.testing.assert_allclose(env.data.qpos,reference.data.qpos,atol=1e-10)
        np.testing.assert_allclose(env.data.qvel,reference.data.qvel,atol=1e-10)
        assert np.isfinite(obs).all() and np.isfinite(reward)
        assert np.max(np.abs(env.data.ctrl))<=17.+1e-9
        if done or truncated:break
    assert truncated and not done and contact_state(env)[0]
    contract=policy_contract(env)
    assert contract["interface_version"]==3 and contract["observation"]["shape"]==[143]
    assert "task_hierarchy" in contract["implementation_sha256"]
    try:require_compatible({"contract":contract},wide)
    except ValueError:pass
    else:raise AssertionError("Different curriculum classes silently share a contract")
    policy,_,_=load_policy("runs/local-gpu-20261021/policy.zip")
    assert policy.observation_space.shape==(125,),"Legacy v2 loading changed"
    print("Passed: Gymnasium, 143-D observation, 20-second zero-residual physical equivalence, torque limits, v3 class separation, legacy v2 load")
finally:
    env.close();reference.close();wide.close()
