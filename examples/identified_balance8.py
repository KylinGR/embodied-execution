"""Complete symmetric encoder/IMU state: pitch, wheels, knees and hips."""
import numpy as np
from identified_balance6 import IdentifiedLanding6Env


class IdentifiedLanding8Env(IdentifiedLanding6Env):
    def __init__(self,config=None,render_mode=None):
        super().__init__(config,render_mode)
        self.gain8=None;self.eq8=np.zeros(8);self.u8=np.zeros(3)

    def sensor_state8(self):
        return np.r_[self.sensor_state6(),np.mean(self.data.qpos[self.qi[[0,3]]])-.5,
                     np.mean(self.data.qvel[self.vi[[0,3]]])]

    def feedback_action(self):
        action=super().feedback_action()
        if self.gain8 is not None:
            u=self.u8-self.gain8@(self.sensor_state8()-self.eq8)
            blend=float(np.clip((self.steps*.01-1.)/.5,0,1))
            for indices,value,limit in (([2,5],u[0],1.),([1,4],u[1],.3),([0,3],u[2],.3)):
                action[indices]=(1-blend)*action[indices]+blend*np.clip(value,-limit,limit)
        return action
