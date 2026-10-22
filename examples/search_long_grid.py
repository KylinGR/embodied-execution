"""Bounded outer-loop diagnostic; keep the existing IMU observer."""
import itertools
import json
from pathlib import Path
import numpy as np
from diablo_training.env import Config
from landing_hold import HoldLandingEnv

env=HoldLandingEnv(Config(episode_seconds=20.))
results=[]
try:
    for kd,kv,kx,bias in itertools.product((30.,8.,2.),(-1.,0.,1.),(0.,-.5,.5),(-.3,-.15,-.45)):
        env.gains=np.array([7.3,kd,kv,kx,bias])
        env.reset(seed=42,options={"drop_height":.02})
        cost=stable=0.
        for i in range(2000):
            _,_,done,truncated,info=env.step(np.zeros(6))
            cost+=info["forward_mps"]**2
            stable=max(stable,info["stable_seconds"])
            if done or truncated: break
        success=not done and info["stable_seconds"]>=1.
        score=(100 if success else 0)+(20 if not done else 0)+info["sim_seconds"]+2*info["stable_seconds"]-.01*cost
        r={"gains":env.gains.tolist(),"damping":env.extra_damping.tolist(),"score":score,
           "seconds":info["sim_seconds"],"stable_at_end":info["stable_seconds"],"max_stable":stable,"fell":bool(done)}
        results.append(r)
        if len(results)==1 or score>=max(x["score"] for x in results[:-1]):
            print(json.dumps(r),flush=True)
        if success: break
    results.sort(key=lambda r:-r["score"])
    Path("runs/landing-long-grid-20261022.json").write_text(json.dumps(results,indent=2))
    print("BEST",json.dumps(results[0]),flush=True)
finally:
    env.close()
