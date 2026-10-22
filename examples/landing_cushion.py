"""Residual task with extra velocity damping in the leg reference controller."""
import numpy as np
from landing_residual import ResidualLandingEnv, GAINS

FEEDBACK_GAINS = np.array([-7.465466493092423, 27.71311484371056,
                           -2.5205194153626307, 2.039548946630819, -.19687235780313161])
EXTRA_DAMPING = np.array([.10294888340638357, 0.])


class CushionLandingEnv(ResidualLandingEnv):
    def __init__(self, config=None, render_mode=None):
        super().__init__(config, render_mode)
        self.gains = FEEDBACK_GAINS.copy()
        self.extra_damping = EXTRA_DAMPING.copy()

    def feedback_action(self):
        pitch = np.arctan2(self.gravity_estimate[0], -self.gravity_estimate[2])
        gyro = self.data.sensor("gyro").data[1]
        velocity = -np.mean(self.data.qvel[self.vi[[2,5]]])*.1
        position = -np.mean(self.data.qpos[self.qi[[2,5]]])*.1
        action = np.zeros(6)
        action[[2,5]] = np.clip(self.gains @ [pitch,gyro,velocity,position,1.],-1,1)
        # For Kp*(q0+0.5*a-q), this adds -extra_damping*dq torque.
        # Bound the equivalent reference offset; original motor limits still apply.
        for index, stiffness in ((0,15.),(1,18.),(3,15.),(4,18.)):
            damping = self.extra_damping[index % 3]
            action[index] = np.clip(-2*damping*self.data.qvel[self.vi[index]]/stiffness,-.8,.8)
        return action
