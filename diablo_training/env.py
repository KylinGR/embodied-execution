"""DIABLO flat-ground Gymnasium environment with measured proprioception."""

from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from .model import CONTROL_DT, DEFAULT_Q, JOINTS, LEGS, PHYSICS_DT, WHEELS, make_model


@dataclass
class Config:
    episode_seconds: float = 20.0
    max_speed: float = 0.5
    observation_noise: bool = False

    def __post_init__(self):
        if self.episode_seconds < 18 or not 0.05 <= self.max_speed <= 0.5:
            raise ValueError(
                "Foundation episodes must be >=18 s and speed must be 0.05..0.5 m/s"
            )


def command_profile(t, task, speed):
    """Return commanded forward m/s; all four tasks include a final hold."""
    profiles = (
        ([0, 20], [0, 0]),
        ([0, 2, 4, 8, 10, 20], [0, 0, 1, 1, 0, 0]),
        ([0, 2, 4, 7, 9, 12, 14, 16, 20], [0, 0, 0.4, 0.4, 1, 1, 0.3, 0, 0]),
        ([0, 2, 4, 6, 8, 10, 12, 14, 16, 20], [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]),
    )
    times, values = profiles[task]
    return float(np.interp(t, times, values) * speed)


class DiabloEnv(gym.Env):
    """Single owner of MuJoCo physics; no ROS or hardware connections."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(self, config=None, render_mode=None):
        self.config = config or Config()
        self.render_mode = render_mode
        self.model = make_model()
        self.data = mujoco.MjData(self.model)
        joints = [self.model.joint(n).id for n in JOINTS]
        self.qi = self.model.jnt_qposadr[joints]
        self.vi = self.model.jnt_dofadr[joints]
        self.base_id = self.model.body("diablo_base_link").id
        self.floor_id = self.model.geom("floor").id
        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -100.0, 100.0, shape=(125,), dtype=np.float32
        )
        self.renderer = None
        self.camera = mujoco.MjvCamera()
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 1.6, 135, -20
        self.history = np.zeros((5, 25), dtype=np.float32)
        self.previous_action = np.zeros(6)
        self.gravity_estimate = np.array([0.0, 0.0, -1.0])
        self.steps = 0
        self.task = 0
        self.speed = 0.2
        self.reference_height = 0.2
        self.stop_origin = None

    def _body_velocity(self):
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.base_id, velocity, 1
        )
        return velocity[3:]

    def _command(self):
        return np.array(
            [
                command_profile(self.steps * CONTROL_DT, self.task, self.speed),
                0.0,
                self.reference_height,
            ]
        )

    def _frame(self):
        gyro = self.data.sensor("gyro").data.copy()
        q = self.data.qpos[self.qi[LEGS]] - DEFAULT_Q[LEGS]
        dq = self.data.qvel[self.vi].copy()
        if self.config.observation_noise:
            gyro += self.np_random.normal(0, 0.01, 3)
            dq += self.np_random.normal(0, 0.02, 6)
        return np.concatenate(
            (
                gyro * 0.25,
                self.gravity_estimate,
                self._command() * [10.0, 0.25, 5.0],
                q,
                dq * 0.05,
                self.previous_action,
            )
        ).astype(np.float32)

    def _update_gravity(self):
        gyro = self.data.sensor("gyro").data
        accel = self.data.sensor("accelerometer").data
        self.gravity_estimate -= np.cross(gyro, self.gravity_estimate) * PHYSICS_DT
        norm = np.linalg.norm(accel)
        if abs(norm - 9.81) < 2.0:
            self.gravity_estimate = 0.98 * self.gravity_estimate - 0.02 * accel / norm
        self.gravity_estimate /= max(np.linalg.norm(self.gravity_estimate), 1e-8)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.steps = 0
        self.task = int((options or {}).get("task", self.np_random.integers(0, 4)))
        if self.task not in range(4):
            raise ValueError("task must be 0..3")
        self.speed = float(self.np_random.uniform(0.05, self.config.max_speed))
        self.data.qpos[2] = 0.4
        self.data.qpos[3:7] = [1, 0, 0, 0]
        self.data.qpos[self.qi] = DEFAULT_Q
        mujoco.mj_forward(self.model, self.data)
        wheel_geoms = [
            i
            for i in range(self.model.ngeom)
            if self.model.geom_contype[i]
            and self.model.body(self.model.geom_bodyid[i]).name
            in ("left_wheel", "right_wheel")
        ]
        bottoms = []
        for i in wheel_geoms:
            mesh = self.model.geom_dataid[i]
            first, count = self.model.mesh_vertadr[mesh], self.model.mesh_vertnum[mesh]
            vertices = self.model.mesh_vert[first : first + count]
            world = (
                vertices @ self.data.geom_xmat[i].reshape(3, 3).T
                + self.data.geom_xpos[i]
            )
            bottoms.append(float(world[:, 2].min()))
        bottom = min(bottoms)
        self.data.qpos[2] += 0.005 - bottom
        mujoco.mj_forward(self.model, self.data)
        self.reference_height = float(self.data.xpos[self.base_id, 2])
        self.gravity_estimate[:] = [0, 0, -1]
        self.previous_action[:] = 0
        self.stop_origin = None
        self.history[:] = self._frame()
        return self.history.flatten().copy(), {"task": self.task}

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError("Action must contain six finite values")
        action = np.clip(action, -1, 1)
        saturated = 0
        max_torque = 0.0
        for _ in range(round(CONTROL_DT / PHYSICS_DT)):
            q, dq = self.data.qpos[self.qi], self.data.qvel[self.vi]
            torque = np.array([15, 18, 0, 15, 18, 0]) * (DEFAULT_Q + action * 0.5 - q)
            torque -= np.array([0.3, 0.2, 0.3, 0.3, 0.2, 0.3]) * dq
            torque[WHEELS] += 0.3 * action[WHEELS] * 10
            # Equal-length two-link Jacobian: tau_knee = F * d(length)/dq.
            # Same 60 N radial feedforward as the upstream training example.
            torque[[1, 4]] -= 60 * 0.14 * np.sin((q[[1, 4]] - np.pi + 0.26866) / 2)
            # Conservative motor envelope from URDF limits; it is an assumption.
            lower = -27 * np.clip(1 + dq / 26, 0, 1)
            upper = 27 * np.clip(1 - dq / 26, 0, 1)
            saturated += int(np.count_nonzero((torque < lower) | (torque > upper)))
            self.data.ctrl[:] = np.clip(torque, lower, upper)
            max_torque = max(max_torque, float(np.max(np.abs(self.data.ctrl))))
            mujoco.mj_step(self.model, self.data)
            for warning in (
                mujoco.mjtWarning.mjWARN_BADQPOS,
                mujoco.mjtWarning.mjWARN_BADQVEL,
                mujoco.mjtWarning.mjWARN_BADQACC,
            ):
                if self.data.warning[warning].number:
                    raise FloatingPointError(
                        "MuJoCo numerical warning; automatic state recovery is not accepted"
                    )
            self._update_gravity()
        self.steps += 1
        finite = np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
        if not finite:
            raise FloatingPointError(
                "Nonfinite MuJoCo state; preserve this run and diagnose"
            )
        velocity = self._body_velocity()
        command = self._command()
        rotation = self.data.xmat[self.base_id].reshape(3, 3)
        upright = float(rotation[2, 2])
        body_contact = False
        for contact in self.data.contact:
            if self.floor_id in (contact.geom1, contact.geom2):
                other = (
                    contact.geom2 if contact.geom1 == self.floor_id else contact.geom1
                )
                body_contact |= self.model.geom_bodyid[other] == self.base_id
        terminated = bool(upright < 0.5 or body_contact)
        truncated = self.steps * CONTROL_DT >= self.config.episode_seconds
        error = float(velocity[0] - command[0])
        height_error = float(self.data.xpos[self.base_id, 2] - self.reference_height)
        reward = (
            1.0
            + 2 * np.exp(-(error**2) / 0.04)
            + upright
            - 0.5 * velocity[1] ** 2
            - 5 * height_error**2
            - 0.25 * float(self.data.sensor("gyro").data[2] ** 2)
            - 0.02 * np.square(action - self.previous_action).sum()
            - 0.0002 * np.square(self.data.ctrl).sum()
            - (5.0 if terminated else 0.0)
        )
        if abs(command[0]) < 0.01:
            reward -= 2 * float(np.square(velocity[:2]).sum())
        self.previous_action[:] = action
        self.history[:-1] = self.history[1:]
        self.history[-1] = self._frame()
        info = {
            "task": self.task,
            "sim_seconds": self.steps * CONTROL_DT,
            "forward_mps": float(velocity[0]),
            "command_mps": float(command[0]),
            "speed_error_mps": abs(error),
            "upright": upright,
            "body_ground_contact": bool(body_contact),
            "max_torque_nm": max_torque,
            "saturation_fraction": saturated / 30.0,
            "position_xy_m": self.data.xpos[self.base_id, :2].tolist(),
        }
        if self.steps * CONTROL_DT >= 16:
            if self.stop_origin is None:
                self.stop_origin = self.data.xpos[self.base_id, :2].copy()
            info["hold_drift_m"] = float(
                np.linalg.norm(self.data.xpos[self.base_id, :2] - self.stop_origin)
            )
        return self.history.flatten().copy(), float(reward), terminated, truncated, info

    def render(self):
        if self.renderer is None:
            self.model.vis.headlight.ambient[:] = [0.4, 0.4, 0.4]
            self.model.vis.headlight.diffuse[:] = [0.8, 0.8, 0.8]
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.camera.lookat[:] = self.data.xpos[self.base_id]
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
