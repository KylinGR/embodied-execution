"""Long-horizon feedback development; reject controllers that only delay falling."""
import itertools
import json
from pathlib import Path
import numpy as np
from diablo_training.env import Config
from landing_cushion import CushionLandingEnv, FEEDBACK_GAINS, EXTRA_DAMPING

env=CushionLandingEnv(Config(episode_seconds=10.))
rng=np.random.default_rng(20261025)
results=[]

def run(p):
    env.gains=np.array(p[:5])
    env.extra_damping=np.clip(p[5:],0,4)
    env.reset(seed=42,options={"drop_height":.02})
    cost=0.
    for i in range(1000):
        _,_,done,truncated,info=env.step(np.zeros(6))
        cost+=info["forward_mps"]**2
        if done or truncated: break
    success=not done and info["stable_seconds"]>=1.
    score=(100 if success else 0)+(20 if not done else 0)+info["sim_seconds"]+2*info["stable_seconds"]-.01*cost
    record={"parameters":list(map(float,np.r_[env.gains,env.extra_damping])),"score":float(score),
            "success":bool(success),"seconds":info["sim_seconds"],"stable_at_end":info["stable_seconds"],"fell":bool(done)}
    results.append(record)
    return record

try:
    for sp,sv,sx in itertools.product((1,-1),repeat=3):
        p=np.r_[FEEDBACK_GAINS,EXTRA_DAMPING]
        p[[0,2,3]]*=[sp,sv,sx]
        print(json.dumps(run(p)),flush=True)
    for i in range(100):
        best=max(results,key=lambda r:r["score"])
        if i%5==0:
            p=np.array([rng.uniform(-15,15),rng.uniform(1,35),rng.uniform(-8,8),rng.uniform(-4,4),rng.uniform(-.3,.3),rng.uniform(0,.4),rng.uniform(0,.2)])
        else:
            scale=.6 if i<50 else .2
            p=np.asarray(best["parameters"])+rng.normal(size=7)*np.array([4,4,2,1,.05,.15,.1])*scale
        result=run(p)
        if result["score"]>best["score"]:
            print(json.dumps(result),flush=True)
        if result["success"] and i>=10: break
    results.sort(key=lambda r:-r["score"])
    Path("runs/landing-hold-search-20261022.json").write_text(json.dumps(results,indent=2))
    print("BEST",json.dumps(results[0]),flush=True)
finally:
    env.close()
