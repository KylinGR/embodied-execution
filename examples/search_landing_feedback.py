"""Bounded development search for sensor feedback; no PPO claim or held-out tuning."""
import json
from pathlib import Path
import numpy as np
from diablo_training.env import Config
from landing_task import LandingEnv

env = LandingEnv(Config(episode_seconds=4.))
rng = np.random.default_rng(20261022)
results = []

def run(params):
    kp, kd, kv, kx, bias = params
    env.reset(seed=42, options={"drop_height": .02})
    maxstable = 0.
    speedcost = 0.
    for i in range(400):
        pitch = np.arctan2(env.gravity_estimate[0], -env.gravity_estimate[2])
        gyro = env.data.sensor("gyro").data[1]
        velocity = -np.mean(env.data.qvel[env.vi[[2, 5]]]) * .1
        position = -np.mean(env.data.qpos[env.qi[[2, 5]]]) * .1
        a = np.zeros(6)
        a[[2, 5]] = np.clip(kp*pitch + kd*gyro + kv*velocity + kx*position + bias, -1, 1)
        _, _, done, truncated, info = env.step(a)
        maxstable = max(maxstable, info["stable_seconds"])
        speedcost += info["forward_mps"]**2
        if done or truncated:
            break
    score = info["sim_seconds"] + 5*maxstable - .01*speedcost
    record = {"parameters": list(map(float,params)), "score": score, "seconds": info["sim_seconds"],
              "max_stable": maxstable,"fell": bool(done), "speed_cost":speedcost}
    results.append(record)
    return score

try:
    seeds = [[1,16,2,.2,0], [2,16,2,.2,0], [4,16,2,.2,0],
             [4,1,0,0,0], [8,2,0,0,0], [-4,-1,0,0,0], [0,0,0,0,0]]
    for p in seeds:
        run(p)
    for iteration in range(180):
        best = max(results, key=lambda x:x["score"])
        if iteration % 4 == 0:
            p = [rng.uniform(-12,12), rng.uniform(-4,20), rng.uniform(-5,5),rng.uniform(-2,2),rng.uniform(-.2,.2)]
        else:
            scale = .7 if iteration < 90 else .2
            p = np.asarray(best["parameters"]) + rng.normal(size=5)*[3*scale,3*scale,scale,scale,.05*scale]
        score = run(p)
        if score > best["score"]:
            print(json.dumps(results[-1]), flush=True)
    path=Path("runs/landing-feedback-search-20261022.json")
    path.write_text(json.dumps(sorted(results,key=lambda x:-x["score"]),indent=2))
    print("BEST",json.dumps(max(results,key=lambda x:x["score"])),flush=True)
finally:
    env.close()
