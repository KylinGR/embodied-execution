"""Full episode single jump evaluation; metrics computed from all physics steps."""
import argparse,gzip,hashlib,json
from pathlib import Path
import numpy as np
from diablo_training.env import Config
from diablo_training.cli import load_policy
from diablo_training.contract import environment_class,policy_contract
from single_jump_v1 import SingleJumpV1Env as SingleJumpEnv
from jump_metrics import sample,summarize

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--model',type=Path)
    p.add_argument('--seeds',type=int,default=30);p.add_argument('--seed-start',type=int,default=21000);p.add_argument('--video',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    if a.model: policy,config,resolved=load_policy(a.model);env=environment_class(resolved['environment_class'])(config)
    else:policy=None;env=SingleJumpEnv(Config(episode_seconds=20.))
    contract=policy_contract(env); files=[Path(__file__),Path(__file__).with_name('jump_metrics.py')]
    hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    video=None;reports=[]
    if a.video:
        import imageio.v2 as imageio
        video=imageio.get_writer(a.output/'qualification.mp4',fps=25)
    try:
        with gzip.open(a.output/'trace.jsonl.gz','wt') as stream:
            for seed in range(a.seed_start,a.seed_start+a.seeds):
                obs,_=env.reset(seed=seed);rows=[sample(env)]
                def record():
                    rows.append({**sample(env), 'qpos':env.data.qpos.tolist(),
                                 'qvel':env.data.qvel.tolist(),'motor_torque_nm':env.data.ctrl.tolist(),
                                 'residual_action':env.residual.tolist(),
                                 'feedback_action':record.feedback.tolist(),
                                 'applied_action':record.applied.tolist()})
                record.feedback=np.zeros(6);record.applied=np.zeros(6)
                env.physics_observer=record
                for step in range(2000):
                    action=np.zeros(6) if policy is None else policy.predict(obs,deterministic=True)[0]
                    record.feedback=env.feedback_action().copy()
                    record.applied=np.clip(record.feedback+np.clip(action,-1,1)*[.03,.03,.02,.03,.03,.02],-1,1)
                    obs,_,done,tr,info=env.step(action)
                    if video is not None and seed<a.seed_start+3 and step%4==0:video.append_data(env.render())
                    if done or tr:break
                env.physics_observer=None
                report={**summarize(rows),'seed':seed,'fell':bool(done),'truncated':bool(tr),'pulse':list(env.episode_pulse)}
                report['passed']=bool(report['passed'] and not done and tr)
                reports.append(report)
                for row in rows:stream.write(json.dumps({'seed':seed,**row})+'\n')
                print(json.dumps(report),flush=True)
        unchanged=contract['sha256']==policy_contract(env)['sha256'] and hashes=={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
        result=dict(episodes=reports,qualified=bool(len(reports)==30 and all(r['passed'] for r in reports) and unchanged),contract=contract,source_hashes=hashes,source_unchanged_during_run=unchanged,model=str(a.model),metric_version='single-jump-v1')
        (a.output/'result.json').write_text(json.dumps(result,indent=2))
    finally:
        if video is not None:video.close()
        env.close()
if __name__=='__main__':main()
