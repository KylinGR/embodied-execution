"""Planar balance curriculum: fixed leg PD targets and symmetric wheel actions.

The PPO interface remains six-dimensional, but only its paired wheel outputs
are applied. This is a restricted controller, not general six-motor locomotion.
"""

import numpy as np
from diablo_training.env import DiabloEnv


class WheelBalanceEnv(DiabloEnv):
    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError("Action must contain six finite values")
        applied = np.zeros(6)
        applied[[2, 5]] = np.mean(np.clip(action[[2, 5]], -1, 1))
        return super().step(applied)
