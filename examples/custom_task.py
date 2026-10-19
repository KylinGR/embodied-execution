"""Example extension: change task/reward while keeping checked physics and I/O."""

import numpy as np

from diablo_training.env import DiabloEnv


class MyEnv(DiabloEnv):
    def get_command(self, sim_seconds):
        # Stand for two seconds, then request a constant low forward speed.
        return np.array([0.0 if sim_seconds < 2 else 0.1, 0.0, self.reference_height])

    def reward_for_transition(self, transition, action, previous_action):
        # Training-only state is available here, never append it to policy inputs.
        speed_error = transition["velocity"][0] - transition["command"][0]
        return float(
            1
            + np.exp(-(speed_error**2) / 0.04)
            - 0.01 * np.square(action - previous_action).sum()
            - 5 * transition["terminated"]
        )
