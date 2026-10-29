"""Direct motor torque jump pulse with the same physical torque-speed bounds."""
import numpy as np
from diablo_training.control import decode_action
from diablo_training.env import DiabloEnv,Config
from single_jump import SingleJumpEnv

class RadialTorqueSingleJumpEnv(SingleJumpEnv):
    pulse=(.5,-.9,.3,.15)
    def __init__(self,config=None,render_mode=None):
        c=config or Config();super().__init__(Config(episode_seconds=c.episode_seconds,max_speed=c.max_speed,observation_noise=c.observation_noise),render_mode)
        self.config.action_mode='torque'
    def feedback_action(self):
        # Evaluate the stable position controller without the jump offsets.
        from stable_landing_v3 import StableLandingEnv
        position=StableLandingEnv.feedback_action(self)
        requested,_=decode_action(position,self.data.qpos[self.qi],self.data.qvel[self.vi],'position_velocity')
        if 1.<=self.steps*.01<2.:
            requested[[0,3]]-=7.5*(self.data.qpos[self.qi[[1,4]]]+1.)
        action=requested/17./np.array([1,1,-1,1,1,-1])
        t=self.steps*.01;crouch,extension,duration,thrust=self.pulse
        if 1.<=t<1.+duration:action[[1,4]]+=crouch
        if 1.+duration<=t<1.+duration+thrust:action[[1,4]]=extension
        return np.clip(action,-1,1)
