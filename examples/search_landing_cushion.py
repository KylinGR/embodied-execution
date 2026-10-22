"""Bounded low-drop controller development before PPO is allowed to modify it."""
import json
import argparse
from pathlib import Path
import numpy as np
from diablo_training.env import Config
from landing_cushion import CushionLandingEnv, FEEDBACK_GAINS

parser=argparse.ArgumentParser()
parser.add_argument("--long",action="store_true")
args=parser.parse_args()
if args.long:
    from landing_hold import HoldLandingEnv
    env=HoldLandingEnv(Config(episode_seconds=10.))
else:
    env=CushionLandingEnv(Config(episode_seconds=4.))
rng=np.random.default_rng(20261024)
results=[]

def run(p):
    env.gains=np.array(p[:5])
    env.extra_damping=np.clip(p[5:],0,4)
    episodes=[]
    for height in (.018,.02,.022):
        env.reset(seed=42,options={"drop_height":height})
        stable=cost=0.
        for i in range(round(env.config.episode_seconds/.01)):
            _,_,done,truncated,info=env.step(np.zeros(6))
            stable=max(stable,info["stable_seconds"])
            cost+=info["forward_mps"]**2
            if done or truncated: break
        success=not done and stable>=1
        score=(40 if success else 0)+(8 if not done else 0)+info["sim_seconds"]+stable-.01*cost
        episodes.append({"height":height,"success":success,"score":score,"stable":stable,"fell":bool(done)})
    record={"parameters":list(map(float,np.r_[env.gains,env.extra_damping])),
            "score":float(min(e["score"] for e in episodes)+np.mean([e["score"] for e in episodes])),"episodes":episodes}
    results.append(record)
    return record

try:
    if args.long:
        print(json.dumps(run(np.r_[env.gains,env.extra_damping])),flush=True)
    else:
        for damping in (0.,.2,.5,1.,2.,4.):
            result=run(np.r_[FEEDBACK_GAINS,damping,damping])
            print(json.dumps(result),flush=True)
    for i in range(100):
        best=max(results,key=lambda r:r["score"])
        scales=[1.,1.,.3,.2,.02,.05,.05] if args.long else [2.,2.,.7,.5,.04,.3,.3]
        p=np.asarray(best["parameters"])+rng.normal(size=7)*scales
        result=run(p)
        if result["score"]>best["score"]:
            print(json.dumps(result),flush=True)
        if all(e["success"] for e in result["episodes"]):
            break
    results.sort(key=lambda r:-r["score"])
    path="runs/landing-hold-robust-search-20261022.json" if args.long else "runs/landing-cushion-search-20261022.json"
    Path(path).write_text(json.dumps(results,indent=2))
    print("BEST",json.dumps(results[0]),flush=True)
finally:
    env.close()
