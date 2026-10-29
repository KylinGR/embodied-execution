"""Single clock-commanded jump; sensor LQR, no truth in actor/control."""
import numpy as np
import mujoco
from diablo_training.env import DiabloEnv
from stable_landing_v3 import StableLandingEnv

class SingleJumpEnv(StableLandingEnv):
    pulse=(0.,0.,.15,.15) # knee crouch, knee extension, crouch duration, thrust duration
    physics_observer=None
    def _update_gravity(self):
        super()._update_gravity()
        if self.physics_observer is not None:
            mujoco.mj_forward(self.model,self.data)
            self.physics_observer()
    def phase(self): return float(np.clip((self.steps*.01-.12)/.18,0,1))
    def reset(self,*,seed=None,options=None):
        self.residual[:]=0; self.previous_residual[:]=0
        obs,info=DiabloEnv.reset(self,seed=seed,options=options)
        # Remove the 5mm release gap: this task starts on the ground, at rest.
        self.data.qpos[2]-=.005; mujoco.mj_forward(self.model,self.data)
        self.episode_pulse=self.pulse
        self.origin=self.data.xpos[self.base_id,:2].copy(); self.drop_height=0.
        self.history[:]=self._frame()
        return self.augment(self.history.flatten().copy()),info
    def feedback_action(self):
        action=super().feedback_action(); t=self.steps*.01
        crouch,extension,duration,thrust=self.pulse
        if 1.<=t<1.+duration: action[[1,4]]+=crouch
        if 1.+duration<=t<1.+duration+thrust: action[[1,4]]+=extension
        return np.clip(action,-1,1)
