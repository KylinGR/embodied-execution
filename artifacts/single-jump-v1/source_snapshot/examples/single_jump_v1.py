"""Single-jump sensor LQR plus bounded torque residual; independent v3 contract.

No ground-truth contact, height or world velocity enters feedback or actor.
Known upright reset and unbiased simulated gyro remain assumptions.
"""
import mujoco
import numpy as np
from gymnasium import spaces
from diablo_training.control import decode_action
from diablo_training.env import Config
from stable_landing_v3 import StableLandingEnv
from single_jump_torque import TorqueSingleJumpEnv

class SingleJumpV1Env(TorqueSingleJumpEnv):
    pulse=(.4,-.84,.4,.2)
    contract_observation={'shape':[148],'base':'143 sensor LQR observation',
                          'extras':['normalized episode clock','four sampled pulse parameters'],
                          'order':'143 then clock then pulse'}
    def __init__(self,config=None,render_mode=None):
        super().__init__(config,render_mode)
        self.observation_space=spaces.Box(-100.,100.,shape=(148,),dtype=np.float32)
        self.episode_pulse=np.array(self.pulse)
    def reset(self,*,seed=None,options=None):
        # Parent ground reset uses neither extra elevation nor initial velocity.
        obs,info=super().reset(seed=seed,options=options)
        self.episode_pulse=np.array(self.pulse,dtype=float)
        # Independent, bounded command perturbations; qualification reports each.
        self.episode_pulse[:2]+=self.np_random.uniform(-.001,.001,2)
        return self.augment(self.history.flatten().copy()),info
    def _update_gravity(self):
        # Identical physics refresh with and without external trace recording.
        StableLandingEnv._update_gravity(self)
        mujoco.mj_forward(self.model,self.data)
        if self.physics_observer is not None:self.physics_observer()
    def augment(self,obs):
        base=StableLandingEnv.augment(self,obs)
        return np.r_[base,self.steps*.01/20.,self.episode_pulse].astype(np.float32)
    def feedback_action(self):
        position=StableLandingEnv.feedback_action(self)
        q=self.data.qpos[self.qi];dq=self.data.qvel[self.vi]
        requested,_=decode_action(position,q,dq,'position_velocity')
        t=self.steps*.01
        if 1.<=t<2.:requested[[0,3]]-=7.5*(q[[1,4]]+1.)
        action=requested/17./np.array([1,1,-1,1,1,-1])
        crouch,extension,duration,thrust=self.episode_pulse
        if 1.<=t<1.+duration:action[[1,4]]+=crouch
        if 1.+duration<=t<1.+duration+thrust:action[[1,4]]=extension
        return np.clip(action,-1,1)
    def reward_for_transition(self,transition,action,previous_action):
        # Truth is used only here and by external qualification.
        t=self.steps*.01
        drift=self.data.xpos[self.base_id,:2]-self.origin
        hold=not 1.35<t<2.
        velocity=transition['velocity'];gyro=transition['gyro']
        return float(3*transition['upright']+2*np.exp(-np.square(drift).sum()/.02)
                     +(2*np.exp(-np.square(velocity[:2]).sum()/.01)
                       +2*np.exp(-np.square(gyro).sum()/.1) if hold else 0.)
                     -.2*np.square(self.residual).sum()
                     -.05*np.square(self.residual-self.previous_residual).sum()
                     -(100. if transition['terminated'] else 0.))
