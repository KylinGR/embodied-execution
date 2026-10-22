"""Truth is logged for offline diagnosis only, never used for control."""
import json
from pathlib import Path
import numpy as np
from diablo_training.env import Config
from landing_hold import HoldLandingEnv

env=HoldLandingEnv(Config(episode_seconds=20.))
rows=[]
try:
    env.reset(seed=42,options={"drop_height":.02})
    for i in range(2000):
        _,_,done,truncated,info=env.step(np.zeros(6))
        if i%50==0 or done or truncated:
            rotation=env.data.xmat[env.base_id].reshape(3,3)
            pitch=np.arctan2(env.gravity_estimate[0],-env.gravity_estimate[2])
            true_pitch=np.arctan2(-rotation[2,0],rotation[2,2])
            row={"t":info["sim_seconds"],"pitch_est":float(pitch),"pitch_true":float(true_pitch),
                 "gyro":env.data.sensor("gyro").data.tolist(),"vx":info["forward_mps"],
                 "wheel_v":float(-np.mean(env.data.qvel[env.vi[[2,5]]])*.1),
                 "wheel_x":float(-np.mean(env.data.qpos[env.qi[[2,5]]])*.1),
                 "body_xy":info["position_xy_m"],"leg_q":env.data.qpos[env.qi[[0,1,3,4]]].tolist(),
                 "action":info["applied_action"],"stable_seconds":info["stable_seconds"]}
            rows.append(row)
        if done or truncated: break
    Path("runs/long-balance-diagnosis-20261022.json").write_text(json.dumps(rows,indent=2))
    for row in rows:
        print(json.dumps(row),flush=True)
finally:
    env.close()
