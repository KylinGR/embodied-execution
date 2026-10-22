"""Six measured states, wheel and symmetric knee reference inputs.

Local LQR blends in by a clock after landing; no truth contact switch is used.
Gyro propagation assumes known upright reset and unbiased simulation sensors.
"""
import numpy as np
from identified_balance import IdentifiedLandingEnv


class IdentifiedLanding6Env(IdentifiedLandingEnv):
    def __init__(self, config=None, render_mode=None):
        super().__init__(config,render_mode)
        self.gain6=None
        self.eq6=np.zeros(6)
        self.u6=np.zeros(2)

    def sensor_state6(self):
        base=self.sensor_state()
        return np.r_[base,np.mean(self.data.qpos[self.qi[[1,4]]])+1.,
                     np.mean(self.data.qvel[self.vi[[1,4]]])]

    def feedback_action(self):
        action=super().feedback_action()
        if self.gain6 is not None:
            u=self.u6-self.gain6@(self.sensor_state6()-self.eq6)
            blend=float(np.clip((self.steps*.01-.7)/.3,0,1))
            action[[2,5]]=(1-blend)*action[[2,5]]+blend*np.clip(u[0],-1,1)
            action[[1,4]]=(1-blend)*action[[1,4]]+blend*np.clip(u[1],-.3,.3)
        return action
