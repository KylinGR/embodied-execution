"""Sensor-state controller used for local identification and LQR development.

Gyro propagation assumes the simulator's unbiased gyro and known upright reset;
no simulator pose or velocity is supplied to the controller.
"""
import numpy as np
from landing_gyro_hold import GyroHoldLandingEnv


class IdentifiedLandingEnv(GyroHoldLandingEnv):
    def __init__(self, config=None, render_mode=None):
        super().__init__(config, render_mode)
        self.lqr_gain = None
        self.equilibrium = np.zeros(4)
        self.equilibrium_action = 0.

    def sensor_state(self):
        return np.array([np.arctan2(self.gravity_estimate[0],-self.gravity_estimate[2]),
                         self.data.sensor("gyro").data[1],
                         -np.mean(self.data.qvel[self.vi[[2,5]]])*.1,
                         -np.mean(self.data.qpos[self.qi[[2,5]]])*.1])

    def feedback_action(self):
        action=super().feedback_action()
        if self.lqr_gain is not None:
            action[[2,5]]=np.clip(self.equilibrium_action-self.lqr_gain@(self.sensor_state()-self.equilibrium),-1,1)
        return action
