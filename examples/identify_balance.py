"""Fit local dynamics from sensor transitions; design and physically test LQR."""
import itertools
import json
from pathlib import Path
import numpy as np
from diablo_training.env import Config
from identified_balance import IdentifiedLandingEnv

out=Path("runs/balance-identification-20261022")
out.mkdir(exist_ok=False)
env=IdentifiedLandingEnv(Config(episode_seconds=20.))
rng=np.random.default_rng(20261026)
X,U,Y,E=[],[],[],[]
try:
    initial=env.gains.copy()
    for episode in range(20):
        env.gains=initial.copy()
        env.gains[4]+=rng.uniform(-.04,.04)
        env.reset(seed=episode,options={"drop_height":rng.uniform(.018,.022)})
        for step in range(350):
            state=env.sensor_state()
            residual=np.zeros(6)
            if step%5==0:
                excitation=rng.uniform(-1,1)
            residual[[2,5]]=excitation
            _,_,done,truncated,info=env.step(residual)
            if step>=100 and not done:
                X.append(state);U.append(info["applied_action"][2]);Y.append(env.sensor_state());E.append(episode)
            if done or truncated:break
    X,U,Y,E=np.array(X),np.array(U),np.array(Y),np.array(E)
    F=np.c_[X,U,np.ones(len(X))]
    train=E<14
    coefficients=np.linalg.lstsq(F[train],(Y-X)[train],rcond=1e-9)[0]
    A=np.eye(4)+coefficients[:4].T
    B=coefficients[4].reshape(4,1)
    c=coefficients[5]
    rmse=np.sqrt(np.mean(((Y-X)[~train]-F[~train]@coefficients)**2,axis=0))
    theta,u=np.linalg.lstsq(np.c_[A[:,0]-np.eye(4)[:,0],B[:,0]],-c,rcond=None)[0]
    eq=np.array([theta,0,0,0])
    identification={"A":A.tolist(),"B":B.tolist(),"c":c.tolist(),"equilibrium":eq.tolist(),
                    "equilibrium_action":float(u),"validation_delta_rmse":rmse.tolist(),
                    "samples":len(X),"features_rank":int(np.linalg.matrix_rank(F[train]))}
    (out/"identification.json").write_text(json.dumps(identification,indent=2))
    np.savez_compressed(out/"transitions.npz",X=X,U=U,Y=Y,episode=E)
    print(json.dumps(identification),flush=True)
    results=[]
    for qtheta,qvelocity,r in itertools.product((20.,100.,400.),(1.,10.,50.),(.1,1.)):
        Q=np.diag([qtheta,1.,qvelocity,1.])
        P=Q.copy()
        for _ in range(10000):
            K=np.linalg.solve(np.array([[r]])+B.T@P@B,B.T@P@A)
            updated=Q+A.T@P@A-A.T@P@B@K
            if np.max(np.abs(updated-P))<1e-7:
                P=updated;break
            P=updated
        K=np.linalg.solve(np.array([[r]])+B.T@P@B,B.T@P@A).flatten()
        env.lqr_gain=K;env.equilibrium=eq;env.equilibrium_action=float(u)
        episodes=[]
        for height in (.0184,.0204,.0216):
            env.reset(seed=42,options={"drop_height":height})
            stable=0.
            for step in range(2000):
                _,_,done,truncated,info=env.step(np.zeros(6))
                stable=max(stable,info["stable_seconds"])
                if done or truncated:break
            episodes.append({"height":height,"seconds":info["sim_seconds"],"fell":bool(done),
                             "stable_at_end":info["stable_seconds"],"max_stable":stable})
        score=min(e["seconds"]+e["stable_at_end"] for e in episodes)
        result={"Q":np.diag(Q).tolist(),"R":r,"K":K.tolist(),"episodes":episodes,"score":score,
                "closed_loop_radius":float(max(abs(np.linalg.eigvals(A-B@K.reshape(1,4)))))}
        results.append(result)
        print(json.dumps(result),flush=True)
    results.sort(key=lambda r:-r["score"])
    (out/"controllers.json").write_text(json.dumps(results,indent=2))
finally:
    env.close()
