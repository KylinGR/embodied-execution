"""Stage one: free-drop landing surrogate, not active takeoff or platform traversal.

Keeps the measured 125-D observation and all six motor actions. Ground truth
contact/velocity are used only by reward and evaluation, never observations.
Contact forces are sampled at control rate (100 Hz), not physics-rate peaks.
"""
import mujoco
import numpy as np
from diablo_training.env import DiabloEnv
from diablo_training.model import CONTROL_DT


class LandingEnv(DiabloEnv):
    def get_command(self, sim_seconds):
        return np.array([0.0, 0.0, self.reference_height])

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.drop_height = float((options or {}).get("drop_height", self.np_random.uniform(.02, .05)))
        if not np.isfinite(self.drop_height) or not .01 <= self.drop_height <= .08:
            raise ValueError("drop_height must be in [0.01, 0.08] metres")
        self.data.qpos[2] += self.drop_height
        mujoco.mj_forward(self.model, self.data)
        self.landed = False
        self.landing_time = None
        self.stable_steps = 0
        self.success = False
        self.peak_sampled_force = 0.0
        self.history[:] = self._frame()
        return self.history.flatten().copy(), {**info, "drop_height_m": self.drop_height}

    def step(self, action):
        downward_before = max(0., -float(self.data.qvel[2]))
        obs, reward, terminated, truncated, info = super().step(action)
        force = 0.
        wheel_contact = False
        for i, contact in enumerate(self.data.contact):
            if self.floor_id not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == self.floor_id else contact.geom1
            body = self.model.body(self.model.geom_bodyid[other]).name
            wrench = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, wrench)
            force += float(np.linalg.norm(wrench[:3]))
            wheel_contact |= body in ("left_wheel", "right_wheel") and wrench[0] > 1.
        self.peak_sampled_force = max(self.peak_sampled_force, force)
        touchdown = wheel_contact and not self.landed
        if touchdown:
            self.landed = True
            self.landing_time = info["sim_seconds"]
            reward -= 2. * downward_before ** 2
        stable = (wheel_contact and info["upright"] >= .95 and
                  np.linalg.norm(self._body_velocity()[:2]) <= .15 and
                  np.linalg.norm(self.data.sensor("gyro").data) <= 1. and not terminated)
        self.stable_steps = self.stable_steps + 1 if stable else 0
        first_success = self.stable_steps >= round(1. / CONTROL_DT) and not self.success
        self.success |= first_success
        reward += 1. if stable else 0.
        reward -= .00001 * force ** 2
        if first_success:
            reward += 20.
        info.update(drop_height_m=self.drop_height, wheel_contact=bool(wheel_contact),
                    landed=bool(self.landed), landing_time_s=self.landing_time,
                    stable_seconds=self.stable_steps * CONTROL_DT,
                    landing_success=bool(self.success),
                    sampled_contact_force_n=force,
                    peak_sampled_contact_force_n=self.peak_sampled_force)
        return obs, float(reward), terminated, truncated, info
