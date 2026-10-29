"""Frozen sensor LQR with explicit controller state for residual PPO.

Only IMU, encoders and reset clock enter control/observations. The attitude
observer assumes unbiased simulation gyro and a known upright reset. Truth
contact/velocity are used by reward/qualification only. Not a hardware policy.
"""
import mujoco
import numpy as np
from gymnasium import spaces
from diablo_training.env import DiabloEnv
from diablo_training.model import PHYSICS_DT

K=np.array([
 [-21.20889785333613,-5.084522356181832,-8.727299568299761,-2.3492386885378496,3.6064249995733424,.7678150414192323,23.59839076120673,3.645203592272883],
 [-.18728255515480433,-.03986158830296949,-.22680258201376766,-.07434574231493724,.05253441082003245,.099219323717618,.16373724303159307,.06629533403461538],
 [.9609454625651083,.1715353887938214,.10065232788991696,.031192176346154656,-.05239688949225034,.011017726683578075,-.642872470339228,-.008855334118191633]])
EQ=np.array([.022820013701161577,0,0,0,.08035284925455095,0,-5.624853122284397e-6,0])
UEQ=np.array([-.0006574374352064613,-.005579084502527975,-7.980866082064669e-6])
EARLY_GAINS=np.array([7.2891376284685325,30.203843227318135,.010403357460483487,1.4234305379392216,-.31098988251019777])


class StableLandingEnv(DiabloEnv):
    contract_version=3
    training_drop_range=(.018,.022)
    contract_observation={"shape":[143],"base_frames":5,"base_frame_width":25,
                          "extras":["8 scaled IMU/encoder controller states","6 next feedback actions","clock blend","3 accelerometer channels in g"],
                          "order":"125 history then 8 states then 6 actions then 1 blend then 3 accelerometer"}

    def __init__(self,config=None,render_mode=None):
        super().__init__(config,render_mode)
        if self.config.action_mode!="position_velocity":raise ValueError("Frozen controller requires position_velocity")
        self.observation_space=spaces.Box(-100.,100.,shape=(143,),dtype=np.float32)
        self.residual=np.zeros(6);self.previous_residual=np.zeros(6)

    def _update_gravity(self):
        self.gravity_estimate-=np.cross(self.data.sensor("gyro").data,self.gravity_estimate)*PHYSICS_DT
        self.gravity_estimate/=max(np.linalg.norm(self.gravity_estimate),1e-8)

    def get_command(self,sim_seconds):return np.array([0.,0.,self.reference_height])

    def sensor_state(self):
        return np.array([np.arctan2(self.gravity_estimate[0],-self.gravity_estimate[2]),
                         self.data.sensor("gyro").data[1],
                         -np.mean(self.data.qvel[self.vi[[2,5]]])*.1,
                         -np.mean(self.data.qpos[self.qi[[2,5]]])*.1,
                         np.mean(self.data.qpos[self.qi[[1,4]]])+1.,np.mean(self.data.qvel[self.vi[[1,4]]]),
                         np.mean(self.data.qpos[self.qi[[0,3]]])-.5,np.mean(self.data.qvel[self.vi[[0,3]]])])

    def phase(self):return float(np.clip((self.steps*.01-1.)/.5,0,1))

    def feedback_action(self):
        state=self.sensor_state();action=np.zeros(6)
        action[[2,5]]=np.clip(EARLY_GAINS@np.r_[state[:4],1.],-1,1)
        for index,stiffness,damping in ((0,15.,.25052162446158793),(3,15.,.25052162446158793),
                                       (1,18.,.2028829899438376),(4,18.,.2028829899438376)):
            action[index]=np.clip(-2*damping*self.data.qvel[self.vi[index]]/stiffness,-.8,.8)
        u=UEQ-K@(state-EQ);blend=self.phase()
        for indices,value,limit in (([2,5],u[0],1.),([1,4],u[1],.3),([0,3],u[2],.3)):
            action[indices]=(1-blend)*action[indices]+blend*np.clip(value,-limit,limit)
        return action

    def augment(self,obs):
        scaled=self.sensor_state()*[10.,.25,5.,1.,5.,.05,5.,.05]
        return np.clip(np.r_[obs,scaled,self.feedback_action(),self.phase(),
                             self.data.sensor("accelerometer").data/9.81],-100,100).astype(np.float32)

    def reset(self,*,seed=None,options=None):
        obs,info=super().reset(seed=seed,options=options)
        self.drop_height=float((options or {}).get("drop_height",self.np_random.uniform(*self.training_drop_range)))
        if not np.isfinite(self.drop_height) or not .01<=self.drop_height<=.08:raise ValueError("Drop height outside supported simulation range")
        self.data.qpos[2]+=self.drop_height;mujoco.mj_forward(self.model,self.data)
        self.residual[:]=0;self.previous_residual[:]=0
        self.origin=self.data.xpos[self.base_id,:2].copy()
        self.history[:]=self._frame()
        return self.augment(self.history.flatten().copy()),{**info,"drop_height_m":self.drop_height}

    def reward_for_transition(self,transition,action,previous_action):
        v=transition["velocity"][:2];gyro=transition["gyro"]
        drift=self.data.xpos[self.base_id,:2]-self.origin
        return float(2*np.exp(-np.square(v).sum()/.01)+2*np.exp(-np.square(gyro).sum()/.1)
                     +2*np.exp(-max(0.,1-transition["upright"])/.01)+np.exp(-np.square(drift).sum()/.01)
                     -np.square(self.residual).sum()-.1*np.square(self.residual-self.previous_residual).sum()
                     -(100. if transition["terminated"] else 0.))

    def step(self,action):
        action=np.asarray(action,dtype=float)
        if action.shape!=(6,) or not np.isfinite(action).all():raise ValueError("Expected six finite residual actions")
        self.residual=np.clip(action,-1,1);feedback=self.feedback_action()
        applied=np.clip(feedback+self.residual*[.03,.03,.02,.03,.03,.02],-1,1)
        obs,reward,done,truncated,info=super().step(applied)
        self.previous_residual=self.residual.copy()
        info.update(drop_height_m=self.drop_height,residual_action=self.residual.tolist(),
                    feedback_action=feedback.tolist(),applied_action=applied.tolist(),controller_phase=self.phase())
        return self.augment(obs),reward,done,truncated,info


class WideStableLandingEnv(StableLandingEnv):
    training_drop_range=(.02,.05)
