"""Single-factor reward ablation; physics, commands and observations stay fixed."""

from diablo_training.env import DiabloEnv


class UprightRewardEnv(DiabloEnv):
    def reward_for_transition(self, transition, action, previous_action):
        reward = super().reward_for_transition(transition, action, previous_action)
        # Hypothesis: stronger tilt cost improves early balance. Measure it;
        # do not compare raw returns across different reward definitions.
        return float(reward - 2.0 * (1.0 - transition["upright"]))
