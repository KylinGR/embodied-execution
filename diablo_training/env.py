"""DIABLO flat-ground Gymnasium environment with measured proprioception."""

from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from .control import decode_action
from .model import (
    CONTROL_DT,
    DEFAULT_Q,
    JOINTS,
    LEGS,
    OBSERVATION_SIGNS,
    PHYSICS_DT,
    closure_errors,
    make_model,
    set_pose,
)


@dataclass
class Config:
    episode_seconds: float = 20.0
    max_speed: float = 0.5
    observation_noise: bool = False
    action_mode: str = "position_velocity"

    def __post_init__(self):
        if (
            not np.isfinite([self.episode_seconds, self.max_speed]).all()
            or min(self.episode_seconds, self.max_speed) <= 0
        ):
            raise ValueError(
                "Episode duration and command speed must be finite and positive"
            )
        if self.action_mode not in ("position_velocity", "torque"):
            raise ValueError("Unknown action_mode")

### set task table
TASK_NAMES = (
    "stand_balance",
    "single_start_stop",
    "cruise_balance",
    "drive_accel_decel",
    "repeat_start_stop",
)

TASK_PROFILES = (
    ([0, 20], [0, 0]),
    ([0, 2, 4, 8, 10, 20], [0, 0, 1, 1, 0, 0]),
    ([0, 2, 4, 20], [0, 0, 1, 1]),
    ([0, 2, 4, 6, 9, 12, 20], [0, 0, 0.4, 1, 1, 0.4, 0.4]),
    ([0, 2, 4, 6, 8, 10, 12, 14, 16, 20],
     [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]),
)


def command_profile(t, task, speed):
    times, values = TASK_PROFILES[task]
    return float(np.interp(t, times, values) * speed)


class DiabloEnv(gym.Env):
    """Single owner of MuJoCo physics; no ROS or hardware connections."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(self, config=None, render_mode=None):
        self.num_tasks = len(TASK_PROFILES)
        self.config = config or Config()
        self.render_mode = render_mode
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or rgb_array")
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
        self.command = np.zeros(3)
        self._needs_reset, self._closed, self._faulted = True, False, False

    def _body_velocity(self):
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY, self.base_id, velocity, 0
        )
        return self.data.xmat[self.base_id].reshape(3, 3).T @ velocity[3:]

    def get_command(self, sim_seconds):
        """Override for custom tasks; return body vx m/s, yaw rad/s, height m."""
        return np.array(
            [
                command_profile(sim_seconds, self.task, self.speed),
                0.0,
                self.reference_height,
            ]
        )

    def _frame(self):
        gyro = self.data.sensor("gyro").data.copy()
        q = self.data.qpos[self.qi[LEGS]] - DEFAULT_Q[LEGS]
        dq = self.data.qvel[self.vi] * OBSERVATION_SIGNS
        if self.config.observation_noise:
            gyro += self.np_random.normal(0, 0.01, 3)
            dq += self.np_random.normal(0, 0.02, 6)
        frame = np.concatenate(
            (
                gyro * 0.25,
                self.gravity_estimate,
                self.command * [10.0, 0.25, 5.0],
                q,
                dq * 0.05,
                self.previous_action,
            )
        ).astype(np.float32)
        if not np.isfinite(frame).all():
            self._fault("Nonfinite sensor observation")
        return np.clip(frame, -100, 100)

    def _fault(self, message):
        self._faulted = True
        raise FloatingPointError(message)

    def _update_gravity(self):
        gyro = self.data.sensor("gyro").data
        accel = self.data.sensor("accelerometer").data
        self.gravity_estimate -= np.cross(gyro, self.gravity_estimate) * PHYSICS_DT
        norm = np.linalg.norm(accel)
        if abs(norm - 9.81) < 2.0:
            self.gravity_estimate = 0.98 * self.gravity_estimate - 0.02 * accel / norm
        self.gravity_estimate /= max(np.linalg.norm(self.gravity_estimate), 1e-8)

    def reset(self, *, seed=None, options=None):
        if self._closed or self._faulted:
            raise RuntimeError("Closed or faulted environment must be recreated")
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.steps = 0
        self.task = int((options or {}).get("task", self.np_random.integers(0, self.num_tasks)))
        if self.task not in range(self.num_tasks):
            raise ValueError(f"task must be 0..{self.num_tasks - 1}")
        self.speed = float(
            (options or {}).get(
                "speed",
                self.np_random.uniform(
                    0.1 * self.config.max_speed, self.config.max_speed
                ),
            )
        )
        if not np.isfinite(self.speed) or abs(self.speed) > self.config.max_speed:
            raise ValueError("Requested speed is outside the configured command range")
        self.data.qpos[2] = 0.4
        self.data.qpos[3:7] = [1, 0, 0, 0]
        set_pose(self.model, self.data, DEFAULT_Q)
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
            if self.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                axis = self.data.geom_xmat[i].reshape(3, 3)[:, 2]
                radius, halfwidth = self.model.geom_size[i, :2]
                support = radius * np.sqrt(
                    max(0.0, 1 - axis[2] ** 2)
                ) + halfwidth * abs(axis[2])
                bottoms.append(float(self.data.geom_xpos[i, 2] - support))
            else:
                mesh = self.model.geom_dataid[i]
                first, count = (
                    self.model.mesh_vertadr[mesh],
                    self.model.mesh_vertnum[mesh],
                )
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
        self.command = np.asarray(self.get_command(0.0), dtype=float)
        if self.command.shape != (3,) or not np.isfinite(self.command).all():
            self._fault("Custom task command must be finite [vx, yaw_rate, height]")
        self._needs_reset = False
        self.history[:] = self._frame()
        return self.history.flatten().copy(), {"task": self.task}

    def step(self, action):
        if self._needs_reset or self._closed or self._faulted:
            raise RuntimeError(
                "Call reset before stepping; recreate closed/faulted environments"
            )
        action = np.asarray(action, dtype=float)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError("Action must contain six finite values")
        action = np.clip(action, -1, 1)
        command = self.command.copy()
        saturated = 0
        max_torque = 0.0
        for _ in range(round(CONTROL_DT / PHYSICS_DT)):
            q, dq = self.data.qpos[self.qi], self.data.qvel[self.vi]
            requested, applied = decode_action(action, q, dq, self.config.action_mode)
            saturated += int(np.count_nonzero(np.abs(requested - applied) > 1e-9))
            self.data.ctrl[:] = applied
            max_torque = max(max_torque, float(np.max(np.abs(self.data.ctrl))))
            mujoco.mj_step(self.model, self.data)
            for warning in (
                mujoco.mjtWarning.mjWARN_BADQPOS,
                mujoco.mjtWarning.mjWARN_BADQVEL,
                mujoco.mjtWarning.mjWARN_BADQACC,
            ):
                if self.data.warning[warning].number:
                    self._fault(
                        "MuJoCo numerical warning; automatic state recovery is not accepted"
                    )
            self._update_gravity()
        self.steps += 1
        finite = np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
        if not finite:
            self._fault("Nonfinite MuJoCo state; preserve this run and diagnose")
        mujoco.mj_forward(self.model, self.data)
        loop_error = max(closure_errors(self.model, self.data))
        if loop_error > 0.005:
            self._fault(f"Parallelogram closure error {loop_error:.6f} m exceeds 5 mm")
        velocity = self._body_velocity()
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
        height_error = float(self.data.xpos[self.base_id, 2] - command[2])
        transition = {
            "velocity": velocity.copy(),
            "command": command,
            "upright": upright,
            "height_error": height_error,
            "torque": self.data.ctrl.copy(),
            "gyro": self.data.sensor("gyro").data.copy(),
            "terminated": terminated,
        }
        reward = float(
            self.reward_for_transition(transition, action, self.previous_action)
        )
        if not np.isfinite(reward):
            self._fault("Custom reward returned a nonfinite value")
        self.previous_action[:] = action
        self.command = np.asarray(
            self.get_command(self.steps * CONTROL_DT), dtype=float
        )
        if self.command.shape != (3,) or not np.isfinite(self.command).all():
            self._fault("Custom task command must be finite [vx, yaw_rate, height]")
        self.history[:-1] = self.history[1:]
        self.history[-1] = self._frame()
        info = {
            "task": self.task,
            "sim_seconds": self.steps * CONTROL_DT,
            "forward_mps": float(velocity[0]),
            "command_mps": float(command[0]),
            "command": command.tolist(),
            "next_command_mps": float(self.command[0]),
            "speed_error_mps": abs(error),
            "upright": upright,
            "body_ground_contact": bool(body_contact),
            "max_torque_nm": max_torque,
            "saturation_fraction": saturated / 30.0,
            "closure_error_m": loop_error,
            "position_xy_m": self.data.xpos[self.base_id, :2].tolist(),
        }
        if self.steps * CONTROL_DT >= 16 and np.max(np.abs(command[:2])) < 0.01:
            if self.stop_origin is None:
                self.stop_origin = self.data.xpos[self.base_id, :2].copy()
            info["hold_drift_m"] = float(
                np.linalg.norm(self.data.xpos[self.base_id, :2] - self.stop_origin)
            )
        else:
            self.stop_origin = None
        self._needs_reset = terminated or truncated
        return self.history.flatten().copy(), reward, terminated, truncated, info

    def reward_for_transition(self, transition, action, previous_action):
        """Example reward only. Override without changing the physical environment."""
        velocity, command = transition["velocity"], transition["command"]
        error = float(velocity[0] - command[0])
        reward = (
            1.0
            + 2 * np.exp(-(error**2) / 0.04)
            + transition["upright"]
            - 0.5 * velocity[1] ** 2
            - 5 * transition["height_error"] ** 2
            - 0.25 * float(transition["gyro"][2] ** 2)
            - 0.02 * np.square(action - previous_action).sum()
            - 0.0002 * np.square(transition["torque"]).sum()
            - (5.0 if transition["terminated"] else 0.0)
        )
        if abs(command[0]) < 0.01:
            reward -= 2 * float(np.square(velocity[:2]).sum())
        return float(reward)

    def render(self):
        if self.renderer is None:
            self.model.vis.headlight.ambient[:] = [0.4, 0.4, 0.4]
            self.model.vis.headlight.diffuse[:] = [0.8, 0.8, 0.8]
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.camera.lookat[:] = self.data.xpos[self.base_id]
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render()

    def close(self):
        self._closed = True
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
