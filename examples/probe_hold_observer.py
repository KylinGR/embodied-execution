"""Compare an observer ablation at development heights, including long hold."""
import json
import numpy as np
from diablo_training.env import Config
from landing_gyro_hold import GyroHoldLandingEnv

env=GyroHoldLandingEnv(Config(episode_seconds=20.))
try:
    for height in (.0184,.0196,.0204,.0216,.03,.05):
        env.reset(seed=42,options={"drop_height":height})
        stable=0.
        for i in range(2000):
            _,_,done,truncated,info=env.step(np.zeros(6))
            stable=max(stable,info["stable_seconds"])
            if done or truncated: break
        print(json.dumps({"height":height,"seconds":info["sim_seconds"],"fell":bool(done),
                          "max_stable":stable,"stable_at_end":info["stable_seconds"],"upright":info["upright"],"vx":info["forward_mps"]}),flush=True)
finally:
    env.close()
