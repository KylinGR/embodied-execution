"""Sensor feedback plus bounded PPO residual, fixed low-drop curriculum.

Feedback uses IMU and wheel encoders only. Truth velocity and contact forces
are confined to reward/metrics. Leg PD and motor limits stay unchanged.
"""
import numpy as np
import mujoco
import hashlib
import inspect
from pathlib import Path
from diablo_training.env import DiabloEnv
from diablo_training.model import CONTROL_DT
from landing_task import LandingEnv

GAINS = np.array([-7.694922000300018, 28.618788198336446,
                  -3.278883650357984, 1.8083709479119008, -.19215249844148685])
RESIDUAL_SCALE = np.array([.08, .08, .05, .08, .08, .05])


class ResidualLandingEnv(LandingEnv):
    def __init__(self, config=None, render_mode=None):
        # The core contract fingerprints this module; pin its task dependency too.
        digest = hashlib.sha256(Path(inspect.getsourcefile(LandingEnv)).read_bytes()).hexdigest()
        if digest != "33b945d605aca24d1675d8eca75a33fc304bebc45a7346993899bb0463949074":
            raise ValueError("Landing task dependency changed; version the residual task before loading checkpoints")
        super().__init__(config, render_mode)
        if self.config.action_mode != "position_velocity":
            self.close()
            raise ValueError("Feedback gains require the original position_velocity controller")

    def reset(self, *, seed=None, options=None):
        options = {"drop_height": .02, **(options or {})}
        obs, info = super().reset(seed=seed, options=options)
        self.residual = np.zeros(6)
        self.previous_residual = np.zeros(6)
        return obs, info

    def feedback_action(self):
        pitch = np.arctan2(self.gravity_estimate[0], -self.gravity_estimate[2])
        gyro = self.data.sensor("gyro").data[1]
        velocity = -np.mean(self.data.qvel[self.vi[[2, 5]]]) * .1
        position = -np.mean(self.data.qpos[self.qi[[2, 5]]]) * .1
        action = np.zeros(6)
        action[[2, 5]] = np.clip(GAINS @ [pitch, gyro, velocity, position, 1.], -1., 1.)
        return action

    def reward_for_transition(self, transition, action, previous_action):
        velocity = transition["velocity"]
        gyro = transition["gyro"]
        # Continuous gradients toward stopping and upright posture, not merely survival.
        return float(2*np.exp(-20*np.square(velocity[:2]).sum())
                     + 1.5*np.exp(-80*(1-transition["upright"]))
                     + np.exp(-2*np.square(gyro).sum())
                     - .05*np.square(self.residual).sum()
                     - .1*np.square(self.residual-self.previous_residual).sum()
                     - (20. if transition["terminated"] else 0.))

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError("Expected six finite residual actions")
        self.residual = np.clip(action, -1., 1.)
        applied = np.clip(self.feedback_action() + RESIDUAL_SCALE*self.residual, -1., 1.)
        downward_before = max(0., -float(self.data.qvel[2]))
        # Call core directly: deliberately replaces the old landing reward additions.
        obs, reward, done, truncated, info = DiabloEnv.step(self, applied)
        force, wheel_contact = 0., False
        for i, contact in enumerate(self.data.contact):
            if self.floor_id not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == self.floor_id else contact.geom1
            wrench = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, wrench)
            force += float(np.linalg.norm(wrench[:3]))
            wheel_contact |= (self.model.body(self.model.geom_bodyid[other]).name
                              in ("left_wheel", "right_wheel") and wrench[0] > 1.)
        if wheel_contact and not self.landed:
            self.landed, self.landing_time = True, info["sim_seconds"]
            reward -= 2*downward_before**2
        stable = (wheel_contact and not done and info["upright"] >= .95 and
                  np.linalg.norm(self._body_velocity()[:2]) <= .15 and
                  np.linalg.norm(self.data.sensor("gyro").data) <= 1.)
        self.stable_steps = self.stable_steps + 1 if stable else 0
        first_success = self.stable_steps >= round(1/CONTROL_DT) and not self.success
        self.success |= first_success
        # Penalize excess above 3 x body weight, not the normal supporting load.
        body_weight = float(self.model.body_mass.sum()*9.81)
        excess = max(0., force/body_weight - 3.)
        reward -= .1*excess**2
        if first_success:
            reward += 5.
        self.peak_sampled_force = max(self.peak_sampled_force, force)
        self.previous_residual = self.residual.copy()
        info.update(drop_height_m=self.drop_height, wheel_contact=bool(wheel_contact),
                    landed=bool(self.landed), landing_time_s=self.landing_time,
                    stable_seconds=self.stable_steps*CONTROL_DT,
                    landing_success=bool(self.success), sampled_contact_force_n=force,
                    peak_sampled_contact_force_n=self.peak_sampled_force,
                    contact_force_bodyweights=force/body_weight,
                    residual_action=self.residual.tolist(), applied_action=applied.tolist())
        return obs, float(reward), done, truncated, info
