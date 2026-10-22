"""Tune on three development heights, separate from the 30 evaluation fixtures."""
import json
from pathlib import Path
import numpy as np
from diablo_training.env import Config
from landing_task import LandingEnv
from landing_residual import GAINS

env=LandingEnv(Config(episode_seconds=4.))
rng=np.random.default_rng(20261023)
results=[]

def run(p):
    episodes=[]
    for height in (.018, .03, .055):
        env.reset(seed=42, options={"drop_height":height})
        stable, cost=0.,0.
        for i in range(400):
            pitch=np.arctan2(env.gravity_estimate[0],-env.gravity_estimate[2])
            gyro=env.data.sensor("gyro").data[1]
            v=-np.mean(env.data.qvel[env.vi[[2,5]]])*.1
            x=-np.mean(env.data.qpos[env.qi[[2,5]]])*.1
            a=np.zeros(6)
            a[[2,5]]=np.clip(np.dot(p,[pitch,gyro,v,x,1]),-1,1)
            _,_,done,truncated,info=env.step(a)
            stable=max(stable,info["stable_seconds"])
            cost+=info["forward_mps"]**2
            if done or truncated:
                break
        score=(20 if not done and stable>=1 else 0)+(8 if not done else 0)+info["sim_seconds"]+2*stable-.01*cost
        episodes.append({"height":height,"score":score,"stable":stable,"fell":bool(done)})
    score=min(e["score"] for e in episodes)+np.mean([e["score"] for e in episodes])
    result={"parameters":list(map(float,p)),"score":float(score),"episodes":episodes}
    results.append(result)
    return result

try:
    print(json.dumps(run(GAINS)),flush=True)
    for i in range(90):
        best=max(results,key=lambda r:r["score"])
        scale=.6 if i<50 else .2
        p=np.asarray(best["parameters"])+rng.normal(size=5)*[3*scale,3*scale,scale,scale,.05*scale]
        result=run(p)
        if result["score"]>best["score"]:
            print(json.dumps(result),flush=True)
    results.sort(key=lambda r:-r["score"])
    Path("runs/landing-feedback-robust-search-20261022.json").write_text(json.dumps(results,indent=2))
    print("BEST",json.dumps(results[0]),flush=True)
finally:
    env.close()
