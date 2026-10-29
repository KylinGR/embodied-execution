"""Sensor PD joint trajectory for one ground-initiated jump, torque limited."""
import numpy as np
from stable_landing_v3 import StableLandingEnv
from single_jump_torque import TorqueSingleJumpEnv
from diablo_training.control import decode_action,radial_feedforward

class TrajectorySingleJumpEnv(TorqueSingleJumpEnv):
    pulse=(.6,-1.5,.3,.15) # crouch delta knee, extended knee target, crouch/thrust times
    stiffness=60.
    damping=1.
    wheel_scale=0.
    def feedback_action(self):
        position=StableLandingEnv.feedback_action(self)
        q=self.data.qpos[self.qi];dq=self.data.qvel[self.vi]
        requested,_=decode_action(position,q,dq,'position_velocity')
        t=self.steps*.01; crouch,extension,duration,thrust=self.pulse
        if 1.<=t<1.+duration+thrust+.25:
            if t<1.+duration: k=-1.+crouch*np.clip((t-1.)/.1,0,1)
            elif t<1.+duration+thrust: k=extension
            else:k=-1.
            target=np.array([-k/2,k,-k/2,k]);idx=[0,1,3,4]
            requested[idx]=self.stiffness*(target-q[idx])-self.damping*dq[idx]+radial_feedforward(q)[idx]
            requested[[2,5]]*=self.wheel_scale
        return np.clip(requested/17./np.array([1,1,-1,1,1,-1]),-1,1)
