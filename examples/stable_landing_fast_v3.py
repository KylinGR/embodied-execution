"""Earlier clock-only LQR handoff for the wider free-drop curriculum."""
import numpy as np
from stable_landing_v3 import StableLandingEnv


class FastStableLandingEnv(StableLandingEnv):
    def phase(self):return float(np.clip((self.steps*.01-.12)/.18,0,1))


class WideFastStableLandingEnv(FastStableLandingEnv):
    training_drop_range=(.02,.05)
