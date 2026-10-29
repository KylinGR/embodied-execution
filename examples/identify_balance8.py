"""Eight-state, three-input local sensor identification with physical LQR tests."""
import itertools
import json
from pathlib import Path
import numpy as np
from diablo_training.env import Config
from identified_balance8 import IdentifiedLanding8Env
from qualify_landing import contact_state

out=Path("runs/balance-identification8-20261022");out.mkdir(exist_ok=False)
env=IdentifiedLanding8Env(Config(episode_seconds=20.));rng=np.random.default_rng(20261028)
X=[];U=[];Y=[];E=[]
try:
    initial=env.gains.copy()
    for episode in range(24):
        env.gains=initial.copy();env.gains[4]+=rng.uniform(-.04,.04)
        env.reset(seed=episode,options={"drop_height":rng.uniform(.018,.022)})
        for step in range(450):
            state=env.sensor_state8();residual=np.zeros(6)
            if step%5==0:excitation=rng.uniform(-1,1,3)
            residual[[2,5]]=excitation[0];residual[[1,4]]=.5*excitation[1];residual[[0,3]]=.5*excitation[2]
            _,_,done,truncated,info=env.step(residual)
            if step>=100 and not done and info["saturation_fraction"]<.01:
                X.append(state);U.append(np.asarray(info["applied_action"])[[2,1,0]])
                Y.append(env.sensor_state8());E.append(episode)
            if done or truncated:break
    X,U,Y,E=map(np.asarray,(X,U,Y,E));F=np.c_[X,U,np.ones(len(X))];train=E<17
    coef=np.linalg.lstsq(F[train],(Y-X)[train],rcond=1e-9)[0]
    A=np.eye(8)+coef[:8].T;B=coef[8:11].T;c=coef[11]
    normalized=(F[train,:-1]-F[train,:-1].mean(axis=0))/np.maximum(F[train,:-1].std(axis=0),1e-9)
    rmse=np.sqrt(np.mean(((Y-X)[~train]-F[~train]@coef)**2,axis=0));multi={}
    for horizon in (20,100):
        errors=[]
        for episode in range(17,24):
            ids=np.flatnonzero(E==episode)
            for start in range(0,len(ids)-horizon,horizon):
                state=X[ids[start]].copy()
                for index in ids[start:start+horizon]:state=A@state+B@U[index]+c
                errors.append(state-Y[ids[start+horizon-1]])
        multi[str(horizon)]=np.sqrt(np.mean(np.square(errors),axis=0)).tolist()
    base=np.zeros(8);base[[4,6]]=X[train][:,[4,6]].mean(axis=0)
    solution=np.linalg.lstsq(np.c_[A[:,0]-np.eye(8)[:,0],B],-(A-np.eye(8))@base-c,rcond=None)[0]
    eq=base.copy();eq[0]=solution[0];ueq=solution[1:]
    meta={"A":A.tolist(),"B":B.tolist(),"c":c.tolist(),"equilibrium":eq.tolist(),"equilibrium_action":ueq.tolist(),
          "samples":len(X),"rank":int(np.linalg.matrix_rank(normalized)),"condition":float(np.linalg.cond(normalized)),
          "one_step_delta_rmse":rmse.tolist(),"open_loop_multistep_rmse":multi}
    (out/"identification.json").write_text(json.dumps(meta,indent=2));np.savez_compressed(out/"transitions.npz",X=X,U=U,Y=Y,episode=E)
    print(json.dumps(meta),flush=True);results=[]
    for qt,rw,rleg in itertools.product((20.,200.),(.1,1.),(1.,10.,100.)):
        Q=np.diag([qt,2.,20.,5.,20.,1.,20.,1.]);R=np.diag([rw,rleg,rleg]);P=Q.copy()
        for _ in range(10000):
            K=np.linalg.solve(R+B.T@P@B,B.T@P@A);nxt=Q+A.T@P@A-A.T@P@B@K
            if np.max(np.abs(nxt-P))<1e-7:P=nxt;break
            P=nxt
        K=np.linalg.solve(R+B.T@P@B,B.T@P@A)
        env.gain8=K;env.eq8=eq;env.u8=ueq;env.gains=initial.copy();reports=[]
        for height in (.0184,.0204,.0216):
            env.reset(seed=42,options={"drop_height":height});origin=env.data.xpos[env.base_id,:2].copy()
            streak=0.;drift=0.;other_ever=False
            for step in range(2000):
                _,_,done,truncated,info=env.step(np.zeros(6));dual,other,_=contact_state(env)
                other_ever|=other;drift=max(drift,float(np.linalg.norm(env.data.xpos[env.base_id,:2]-origin)))
                stable=dual and not other and not done and info["upright"]>=.95 and np.linalg.norm(env._body_velocity()[:2])<=.15 and np.linalg.norm(env.data.sensor("gyro").data)<=1
                streak=streak+.01 if stable else 0.
                if done or truncated:break
            reports.append({"height":height,"seconds":info["sim_seconds"],"fell":bool(done),"terminal_stable":streak,"max_drift":drift,"other_contact":bool(other_ever)})
        score=min(r["seconds"]+min(r["terminal_stable"],10)-5*r["max_drift"] for r in reports)
        result={"Q":np.diag(Q).tolist(),"R":np.diag(R).tolist(),"K":K.tolist(),"episodes":reports,"score":score,
                "closed_loop_radius":float(max(abs(np.linalg.eigvals(A-B@K))))}
        results.append(result);print(json.dumps(result),flush=True)
    results.sort(key=lambda r:-r["score"]);(out/"controllers.json").write_text(json.dumps(results,indent=2))
finally:env.close()
