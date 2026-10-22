"""Simulation-only gyro propagation ablation with known upright reset.

No simulator orientation is supplied to the controller. Unlike a production
observer this omits accelerometer correction and assumes unbiased gyro/reset.
"""
import numpy as np
from diablo_training.model import PHYSICS_DT
from landing_hold import HoldLandingEnv


class GyroHoldLandingEnv(HoldLandingEnv):
    def _update_gravity(self):
        self.gravity_estimate -= np.cross(self.data.sensor("gyro").data,self.gravity_estimate)*PHYSICS_DT
        self.gravity_estimate /= max(np.linalg.norm(self.gravity_estimate),1e-8)
