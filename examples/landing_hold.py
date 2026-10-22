"""Long-horizon sensor controller candidate; separate from four-second controller."""
import numpy as np
from landing_cushion import CushionLandingEnv


class HoldLandingEnv(CushionLandingEnv):
    def __init__(self, config=None, render_mode=None):
        super().__init__(config, render_mode)
        self.gains = np.array([7.2891376284685325,30.203843227318135,
                               .010403357460483487,1.4234305379392216,-.31098988251019777])
        self.extra_damping = np.array([.25052162446158793,.2028829899438376])
