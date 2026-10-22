"""v2 qualification: terminal dual-wheel stability and bounded drift, no latch."""
import argparse
import gzip
import hashlib
import json
from collections import deque
from pathlib import Path
import mujoco
import numpy as np
from diablo_training.cli import load_policy
from diablo_training.contract import environment_class,policy_contract
from diablo_training.env import Config


def contact_state(env):
    wheels=set();other=False;force=0.
    for i,c in enumerate(env.data.contact):
        if env.floor_id not in (c.geom1,c.geom2):continue
        geom=c.geom2 if c.geom1==env.floor_id else c.geom1
        body=env.model.body(env.model.geom_bodyid[geom]).name
        wrench=np.zeros(6);mujoco.mj_contactForce(env.model,env.data,i,wrench)
        if wrench[0]>1.:
            force+=float(np.linalg.norm(wrench[:3]))
            if body in ("left_wheel","right_wheel"):wheels.add(body)
            else:other=True
    return len(wheels)==2,other,force


def main():
    p=argparse.ArgumentParser()
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument("--model",type=Path)
    source.add_argument("--environment")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seconds",type=float,default=20.)
    p.add_argument("--heights",type=float,nargs="+",default=[.019,.02,.021])
    p.add_argument("--jitter",type=float,default=.001)
    p.add_argument("--seed-start",type=int,default=6500)
    p.add_argument("--seeds",type=int,default=10)
    p.add_argument("--video",action="store_true")
    args=p.parse_args()
    if args.seconds<20 or args.seeds<1 or args.jitter<0:raise ValueError("Qualification requires at least 20 seconds")
    if args.model:
        policy,config,resolved=load_policy(args.model)
        if config.episode_seconds!=args.seconds:raise ValueError("Keep saved episode duration")
        name=resolved["environment_class"]
    else:
        policy=None;config=Config(episode_seconds=args.seconds);name=args.environment
    env=environment_class(name)(config)
    starting_contract=policy_contract(env)
    starting_qualification_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.mkdir(parents=True,exist_ok=False)
    video=None
    if args.video:
        import imageio.v2 as imageio
        video=imageio.get_writer(args.output/"qualification.mp4",fps=25)
    reports=[]
    try:
        with gzip.open(args.output/"trace.jsonl.gz","wt") as trace:
            for nominal in args.heights:
                for seed in range(args.seed_start,args.seed_start+args.seeds):
                    height=float(nominal+np.random.default_rng(seed).uniform(-args.jitter,args.jitter))
                    obs,_=env.reset(seed=seed,options={"drop_height":height})
                    origin=env.data.xpos[env.base_id,:2].copy();recent=deque(maxlen=500)
                    recent_speed=deque(maxlen=500);recent_gyro=deque(maxlen=500)
                    torque_peak=0.;saturation_sum=0.;residual_sum=0.
                    streak=0.;other_ever=False;max_drift=0.;max_force=0.;landed=False
                    for step in range(round(args.seconds/.01)):
                        action=np.zeros(6) if policy is None else policy.predict(obs,deterministic=True)[0]
                        obs,_,done,truncated,info=env.step(action)
                        dual,other,force=contact_state(env);other_ever|=other;landed|=dual
                        xy=env.data.xpos[env.base_id,:2].copy();recent.append(xy)
                        max_drift=max(max_drift,float(np.linalg.norm(xy-origin)));max_force=max(max_force,force)
                        velocity=env._body_velocity();gyro=env.data.sensor("gyro").data
                        recent_speed.append(float(np.linalg.norm(velocity[:2])));recent_gyro.append(float(np.linalg.norm(gyro)))
                        torque_peak=max(torque_peak,info["max_torque_nm"]);saturation_sum+=info["saturation_fraction"]
                        residual_sum+=float(np.linalg.norm(info.get("residual_action",np.zeros(6))))
                        stable=dual and not other and not done and info["upright"]>=.95 and np.linalg.norm(velocity[:2])<=.15 and np.linalg.norm(gyro)<=1.
                        streak=streak+.01 if stable else 0.
                        trace.write(json.dumps({"seed":seed,"height_m":height,**info,"dual_wheel_contact":dual,
                                                "other_ground_contact":other,"strict_stable_seconds":streak,
                                                "horizontal_velocity":velocity[:2].tolist(),"gyro":gyro.tolist()})+"\n")
                        if video is not None and seed==args.seed_start and step%4==0:
                            from PIL import Image,ImageDraw
                            frame=Image.fromarray(env.render());draw=ImageDraw.Draw(frame)
                            draw.rectangle((0,0,640,48),fill="black")
                            draw.text((8,8),f"DROP {height:.4f}m | t={info['sim_seconds']:.2f}s | terminal stable={streak:.2f}s",fill="white")
                            draw.text((8,28),f"dual={dual} | drift={max_drift:.3f}m | zero PPO={policy is None}",fill="white")
                            video.append_data(np.asarray(frame))
                        if done or truncated:break
                    window=np.asarray(recent)
                    # Bounding-box diagonal conservatively bounds any two-point displacement.
                    window_drift=float(np.linalg.norm(np.ptp(window,axis=0)))
                    passed=bool(truncated and not done and landed and not other_ever and streak>=3.-1e-9 and max_drift<=.2 and window_drift<=.05)
                    report={"height_m":height,"seed":seed,"passed":passed,"fell":bool(done),
                            "duration_s":info["sim_seconds"],"terminal_stable_seconds":streak,
                            "max_drift_m":max_drift,"last_5s_drift_m":window_drift,
                            "last_5s_max_horizontal_speed_mps":max(recent_speed),
                            "last_5s_max_angular_speed_rps":max(recent_gyro),
                            "peak_torque_nm":torque_peak,"mean_saturation_fraction":saturation_sum/(step+1),
                            "mean_residual_action_norm":residual_sum/(step+1),
                            "other_ground_contact":bool(other_ever),"peak_sampled_force_n":max_force}
                    reports.append(report);print(json.dumps(report),flush=True)
        if policy_contract(env)["sha256"]!=starting_contract["sha256"] or hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=starting_qualification_sha:
            raise RuntimeError("Source changed during qualification; do not accept this run")
        result={"metric_version":"landing-v2-terminal-dual-contact-drift","environment":name,
                "contract":starting_contract,
                "qualification_source_sha256":starting_qualification_sha,
                "source_unchanged_during_run":True,
                "model":None if args.model is None else str(args.model),"seconds":args.seconds,
                "episodes":reports,"pass_rate":sum(r["passed"] for r in reports)/len(reports),
                "qualified":all(r["passed"] for r in reports),
                "requirements":{"terminal_stable_seconds":3.,"max_drift_m":.2,"last_5s_drift_m":.05,
                                "both_wheels":True,"non_wheel_support_allowed":False,"min_contact_normal_n":1.,
                                "last_5s_drift_definition":"horizontal bounding-box diagonal", "force_sampling_hz":100},
                "scope":"Free-drop simulation only; no active takeoff, platform departure, sim-to-real or training-seed robustness claim"}
        (args.output/"result.json").write_text(json.dumps(result,indent=2))
    finally:
        if video is not None:video.close()
        env.close()


if __name__=="__main__":main()
