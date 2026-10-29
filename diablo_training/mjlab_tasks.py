"""MJLab task definitions for the staged Xingtian locomotion curriculum."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import shutil
import uuid
from typing import Any

import mujoco
import numpy as np
import torch

from mjlab.actuator import (
    Actuator,
    ActuatorCfg,
    ActuatorCmd,
    BuiltinPositionActuatorCfg,
    BuiltinPositionActuator,
)
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg, RecomputeLevel, requires_model_fields
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import (
    MjlabOnPolicyRunner,
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlVecEnvWrapper,
)
from mjlab.scene import SceneCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    GridPatternCfg,
    ObjRef,
    RayCastSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.terrains import SubTerrainCfg, TerrainEntityCfg, TerrainGeneratorCfg
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.utils.spec import create_motor_actuator
from mjlab.utils.spec_config import CollisionCfg
from mjlab.viewer import ViewerConfig

from .contract import (
    ACTION_DTYPE,
    ACTION_SHAPE,
    BLIND_SEEDS,
    HEADING_ERROR_OBS_SCALE,
    LIMIT_SEEDS,
    OBSERVATION_DTYPE,
    OBSERVATION_SHAPE,
    PROPRIO_FRAME_DIM,
    PROPRIO_HISTORY_LENGTH,
    SELECTION_PROBE_SEED,
    SELECTION_VERIFY_SEEDS,
    TARGET_BASE_HEIGHT_M,
    budget_iterations,
    checkpoint_interval,
    fully_passes,
    gate_failures,
    gate_score,
    interface_contract,
    load_run_contract,
    load_course,
    rank_policy_candidates,
    sha256,
)
from .model import (
    CONTROL_DT,
    DEFAULT_Q,
    HIP_DAMPING_GAIN,
    HIP_POSITION_GAIN,
    JOINTS,
    KNEE_DAMPING_GAIN,
    KNEE_POSITION_GAIN,
    LEGS,
    NO_LOAD_SPEED,
    OBSERVATION_SIGNS,
    PHYSICS_DT,
    RATED_TORQUE,
    STALL_TORQUE,
    make_robot_spec,
)

COURSE = load_course()
COURSE_HEIGHT_SCALE = 3.0
WHEEL_RADIUS = 0.1
GAIT_JOINTS = ("left_hip_motor", "left_knee_motor", "right_hip_motor", "right_knee_motor")
WHEEL_JOINTS = ("left_wheel_motor", "right_wheel_motor")
PASSIVE_JOINTS = (
    "left_return_passive",
    "left_branch_passive",
    "right_return_passive",
    "right_branch_passive",
)
KNEE_JOINTS = ("left_knee_motor", "right_knee_motor")


def validate_policy_io(
    observations: torch.Tensor | dict[str, torch.Tensor],
    action: torch.Tensor | None = None,
) -> None:
    actor_obs = (
        observations
        if isinstance(observations, torch.Tensor)
        else observations["actor"]
    )
    if (
        actor_obs.ndim != 2
        or tuple(actor_obs.shape[1:]) != OBSERVATION_SHAPE
        or actor_obs.dtype != torch.float32
    ):
        raise RuntimeError(
            f"Policy observation must be {OBSERVATION_DTYPE}[N,{OBSERVATION_SHAPE[0]}], "
            f"got {actor_obs.dtype}{list(actor_obs.shape)}"
        )
    if action is None:
        return
    if (
        action.ndim != 2
        or action.shape[:-1] != actor_obs.shape[:-1]
        or tuple(action.shape[1:]) != ACTION_SHAPE
        or action.dtype != torch.float32
    ):
        raise RuntimeError(
            f"Policy action must be {ACTION_DTYPE}[N,{ACTION_SHAPE[0]}], "
            f"got {action.dtype}{list(action.shape)}"
        )


def _initial_joint_positions() -> dict[str, float]:
    positions = {
        name: float(DEFAULT_Q[i]) for i, name in enumerate(JOINTS)
    }
    for side_index, side in enumerate(("left", "right")):
        knee = float(DEFAULT_Q[3 * side_index + 1])
        positions[f"{side}_return_passive"] = knee
        positions[f"{side}_branch_passive"] = -knee
    return positions


@dataclass(kw_only=True)
class WheelPiActuatorCfg(ActuatorCfg):
    kp: float = 0.3
    ki: float = 4.0
    integral_limit_nm: float = 2.0
    feedforward_nm_per_rad_s: float = 0.05
    target_slew_rad_s2: float = 20.0
    effort_limit: float = STALL_TORQUE

    def build(
        self, entity: Entity, target_ids: list[int], target_names: list[str]
    ) -> WheelPiActuator:
        return WheelPiActuator(self, entity, target_ids, target_names)


class WheelPiActuator(Actuator):
    """GPU PI wheel-speed controller with slew, anti-windup, and motor envelope."""

    cfg: WheelPiActuatorCfg

    def __init__(
        self,
        cfg: WheelPiActuatorCfg,
        entity: Entity,
        target_ids: list[int],
        target_names: list[str],
    ):
        super().__init__(cfg, entity, target_ids, target_names)
        self.integral: torch.Tensor | None = None
        self.target: torch.Tensor | None = None
        self.gain_scale: torch.Tensor | None = None

    def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
        for target_name in target_names:
            actuator = create_motor_actuator(
                spec,
                target_name,
                effort_limit=self.cfg.effort_limit,
                armature=self.cfg.armature,
                frictionloss=self.cfg.frictionloss,
                viscous_damping=self.cfg.viscous_damping,
            )
            actuator.name = f"{target_name}_wheel_pi"
            self._mjs_actuators.append(actuator)

    def initialize(self, mj_model, model, data, device: str) -> None:
        super().initialize(mj_model, model, data, device)
        shape = (data.nworld, len(self._target_ids_list))
        self.integral = torch.zeros(shape, device=device)
        self.target = torch.zeros(shape, device=device)
        self.gain_scale = torch.ones(shape, device=device)

    def set_gain_scale(
        self,
        env_ids: torch.Tensor | slice,
        scale: torch.Tensor,
    ) -> None:
        assert self.gain_scale is not None
        self.gain_scale[env_ids] = scale

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        assert self.integral is not None
        assert self.target is not None
        assert self.gain_scale is not None
        max_delta = self.cfg.target_slew_rad_s2 * PHYSICS_DT
        self.target += torch.clamp(
            cmd.velocity_target - self.target, -max_delta, max_delta
        )
        error = self.target - cmd.vel
        self.integral += error * PHYSICS_DT
        torque_cap = self.cfg.integral_limit_nm / max(self.cfg.ki, 1e-9)
        self.integral.clamp_(-torque_cap, torque_cap)
        kp = self.cfg.kp * self.gain_scale
        ki = self.cfg.ki * self.gain_scale
        feedforward = self.cfg.feedforward_nm_per_rad_s * self.target
        torque = feedforward + kp * error + ki * self.integral
        lower = -self.cfg.effort_limit * (1 + cmd.vel / NO_LOAD_SPEED).clamp(0, 1)
        upper = self.cfg.effort_limit * (1 - cmd.vel / NO_LOAD_SPEED).clamp(0, 1)
        return torch.clamp(torque, lower, upper)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        assert self.integral is not None
        assert self.target is not None
        self.integral[env_ids] = 0.0
        self.target[env_ids] = 0.0


@dataclass(kw_only=True)
class XingtianActionCfg(ActionTermCfg):
    leg_scale: float = 0.5
    wheel_speed_scale: float = 22.0

    def build(self, env: ManagerBasedRlEnv) -> XingtianAction:
        return XingtianAction(self, env)


class XingtianAction(ActionTerm):
    cfg: XingtianActionCfg

    def __init__(self, cfg: XingtianActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        joint_ids, joint_names = self._entity.find_joints(
            JOINTS, preserve_order=True
        )
        if tuple(joint_names) != JOINTS:
            raise ValueError(f"Unexpected DIABLO joint order: {joint_names}")
        self._joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
        joint_id = dict(zip(joint_names, joint_ids, strict=True))
        action_index = {name: i for i, name in enumerate(JOINTS)}
        self._leg_ids = torch.tensor(
            [joint_id[name] for name in GAIT_JOINTS],
            device=self.device,
            dtype=torch.long,
        )
        self._wheel_ids = torch.tensor(
            [joint_id[name] for name in WHEEL_JOINTS],
            device=self.device,
            dtype=torch.long,
        )
        self._leg_channels = torch.tensor(
            [action_index[name] for name in GAIT_JOINTS],
            device=self.device,
            dtype=torch.long,
        )
        self._wheel_channels = torch.tensor(
            [action_index[name] for name in WHEEL_JOINTS],
            device=self.device,
            dtype=torch.long,
        )
        self._default = self._entity.data.default_joint_pos[
            :, self._leg_ids
        ].clone()
        self._raw = torch.zeros((self.num_envs, 6), device=self.device)
        self.position_target = torch.zeros(
            (self.num_envs, len(GAIT_JOINTS)), device=self.device
        )
        self.wheel_velocity_target = torch.zeros(
            (self.num_envs, len(WHEEL_JOINTS)), device=self.device
        )

    @property
    def action_dim(self) -> int:
        return ACTION_SHAPE[0]

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw

    def process_actions(self, actions: torch.Tensor) -> None:
        expected = (self.num_envs, ACTION_SHAPE[0])
        if actions.shape != expected or actions.dtype != torch.float32:
            raise ValueError(
                f"Expected float32 action shape {expected}, got "
                f"{actions.dtype} {actions.shape}"
            )
        self._raw[:] = torch.clamp(actions, -1.0, 1.0)

    def apply_actions(self) -> None:
        leg_target = (
            self._default
            + self.cfg.leg_scale * self._raw[:, self._leg_channels]
        )
        knee_pos = self._entity.data.joint_pos[:, self._leg_ids[1::2]]
        # Native position actuators apply target-kp error directly. Add the
        # same radial-load feedforward as the CPU controller as tau / kp.
        knee_pos_ff = (
            60.0
            * 0.14
            * torch.sin((-2.8729 - knee_pos) / 2.0)
            / KNEE_POSITION_GAIN
        )
        leg_target[:, 1::2] += knee_pos_ff
        wheel_target = (
            -self.cfg.wheel_speed_scale * self._raw[:, self._wheel_channels]
        )
        self.position_target[:] = leg_target
        self.wheel_velocity_target[:] = wheel_target
        self._entity.set_joint_position_target(
            leg_target, joint_ids=self._leg_ids
        )
        self._entity.set_joint_velocity_target(
            wheel_target, joint_ids=self._wheel_ids
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw[env_ids] = 0.0


@dataclass(kw_only=True)
class CourseCommandCfg(CommandTermCfg):
    stage_name: str
    profile_index: int | None = None
    base_height_m: float = TARGET_BASE_HEIGHT_M

    def build(self, env: ManagerBasedRlEnv) -> CourseCommand:
        return CourseCommand(self, env)


class CourseCommand(CommandTerm):
    cfg: CourseCommandCfg

    def __init__(self, cfg: CourseCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.stage = COURSE.stage(cfg.stage_name)
        self.profiles = self.stage["profiles"]
        if cfg.profile_index is not None and not 0 <= cfg.profile_index < len(
            self.profiles
        ):
            raise ValueError("profile_index is outside the stage profile list")
        self.robot: Entity = env.scene["robot"]
        self.command_b = torch.zeros((self.num_envs, 3), device=self.device)
        self.command_b[:, 2] = cfg.base_height_m
        self.profile_ids = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.start_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.start_heading = torch.zeros(self.num_envs, device=self.device)
        self.target_heading = torch.zeros(self.num_envs, device=self.device)
        self._profile_tables = [
            (
                torch.tensor(
                    [point[0] for point in profile["points"]],
                    device=self.device,
                    dtype=torch.float,
                ),
                torch.tensor(
                    [point[1] for point in profile["points"]],
                    device=self.device,
                    dtype=torch.float,
                ),
                torch.tensor(
                    [point[2] for point in profile["points"]],
                    device=self.device,
                    dtype=torch.float,
                ),
            )
            for profile in self.profiles
        ]
        self._heading_tables = []
        for times, _, yaw in self._profile_tables:
            heading_offsets = torch.zeros_like(times)
            heading_offsets[1:] = torch.cumsum(
                0.5
                * (yaw[:-1] + yaw[1:])
                * (times[1:] - times[:-1]),
                dim=0,
            )
            self._heading_tables.append(heading_offsets)
        self.metrics["error_vel_x"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_yaw"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.command_b

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if self.cfg.profile_index is None:
            self.profile_ids[env_ids] = torch.remainder(
                env_ids + self._env.common_step_counter, len(self.profiles)
            )
        else:
            self.profile_ids[env_ids] = self.cfg.profile_index

        # "end" profiles start at the far end and drive backward toward origin.
        terrain = self.stage.get("terrain", {})
        length = float(terrain.get("length_m", 0.0))
        origins = self._env.scene.env_origins[env_ids]
        poses = self.robot.data.root_link_pose_w[env_ids].clone()
        start_offsets = torch.tensor(
            [
                0.75
                if profile.get("start") != "end"
                else min(
                    max(1.0, _planned_forward_distance(profile) + 1.0),
                    max(1.0, length - 1.0),
                )
                for profile in self.profiles
            ],
            device=self.device,
        )[self.profile_ids[env_ids]]
        poses[:, 0] = origins[:, 0] + start_offsets
        poses[:, 1] = origins[:, 1]
        poses[:, 2] = origins[:, 2] + self.cfg.base_height_m
        initial_heading = torch.zeros(len(env_ids), device=self.device)
        heading_range = self.stage.get("reset_heading_range_deg")
        if heading_range is not None:
            if len(heading_range) != 2:
                raise ValueError("reset_heading_range_deg must contain two values")
            heading_min, heading_max = map(math.radians, map(float, heading_range))
            if (
                not math.isfinite(heading_min + heading_max)
                or heading_min > heading_max
            ):
                raise ValueError("reset_heading_range_deg must be finite and ordered")
            initial_heading.uniform_(heading_min, heading_max)
        poses[:, 3] = torch.cos(0.5 * initial_heading)
        poses[:, 4:6] = 0.0
        poses[:, 6] = torch.sin(0.5 * initial_heading)
        self.start_pos[env_ids] = poses[:, :3]
        self.start_heading[env_ids] = 0.0
        self.robot.write_root_link_pose_to_sim(poses, env_ids=env_ids)
        self.robot.write_root_link_velocity_to_sim(
            torch.zeros((len(env_ids), 6), device=self.device), env_ids=env_ids
        )
        self._set_commands(env_ids)

    def _set_commands(self, env_ids: torch.Tensor | None = None) -> None:
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        seconds = self._env.episode_length_buf[ids].float() * self._env.step_dt
        values = torch.zeros((len(ids), 2), device=self.device)
        heading_offsets = torch.zeros(len(ids), device=self.device)
        for profile_index, ((times, linear, yaw), heading_table) in enumerate(
            zip(self._profile_tables, self._heading_tables, strict=True)
        ):
            mask = self.profile_ids[ids] == profile_index
            local_time = seconds[mask]
            right = torch.searchsorted(times, local_time, right=True).clamp(
                1, len(times) - 1
            )
            left = right - 1
            span = (times[right] - times[left]).clamp(min=1e-9)
            blend = ((local_time - times[left]) / span).clamp(0.0, 1.0)
            values[mask, 0] = linear[left] + blend * (linear[right] - linear[left])
            values[mask, 1] = yaw[left] + blend * (yaw[right] - yaw[left])
            heading_offsets[mask] = heading_table[left] + blend * (
                heading_table[right] - heading_table[left]
            )
        self.command_b[ids, :2] = values
        self.command_b[ids, 2] = self.cfg.base_height_m
        self.target_heading[ids] = self.start_heading[ids] + heading_offsets

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        self._set_commands(env_ids)

    def _update_metrics(self) -> None:
        max_steps = max(self._env.max_episode_length, 1)
        self.metrics["error_vel_x"] += (
            self.command_b[:, 0]
            - self.robot.data.root_link_lin_vel_b[:, 0]
        ).abs() / max_steps
        self.metrics["error_yaw"] += (
            self.command_b[:, 1]
            - self.robot.data.root_link_ang_vel_b[:, 2]
        ).abs() / max_steps


def _planned_forward_distance(profile: dict[str, Any]) -> float:
    distance = 0.0
    for start, end in zip(profile["points"], profile["points"][1:]):
        distance += 0.5 * (float(start[1]) + float(end[1])) * (
            float(end[0]) - float(start[0])
        )
    return abs(distance)


def _obstacle_offsets(terrain: dict[str, Any]) -> list[float]:
    kind = terrain.get("kind")
    if kind == "obstacles":
        count = int(terrain["obstacle_count"])
        start = 3.0
        return [start + i * float(terrain["min_spacing_m"]) for i in range(count)]
    if kind == "composite":
        count = int(terrain["obstacle_count"])
        return np.linspace(
            float(terrain["obstacle_start_m"]) + 0.5,
            float(terrain["obstacle_end_m"]) - 0.5,
            count,
        ).tolist()
    return []


def _rough_height(
    length: np.ndarray,
    width: np.ndarray,
    rng,
    amplitude: float,
    max_grade_deg: float,
):
    # ponytail: three seeded waves approximate continuous terrain without a
    # procedural noise dependency. Add Perlin terrain if measured profiles need it.
    x = np.linspace(0.0, 1.0, len(length))
    y = np.linspace(0.0, 1.0, len(width))
    xx, yy = np.meshgrid(x, y, indexing="ij")
    height = np.zeros_like(xx)
    for wavelength in (0.35, 0.65, 1.0):
        phase_x, phase_y = rng.uniform(0.0, 2 * np.pi, 2)
        height += np.sin(2 * np.pi * xx * (60.0 / wavelength) + phase_x)
        height += 0.5 * np.cos(2 * np.pi * yy * (5.0 / wavelength) + phase_y)
    height -= height.min()
    if height.max() > 0:
        height *= amplitude / height.max()
    dx = np.gradient(height, axis=0) / max(length[1] - length[0], 1e-6)
    dy = np.gradient(height, axis=1) / max(width[1] - width[0], 1e-6)
    max_grad = float(np.sqrt(dx * dx + dy * dy).max())
    limit = math.tan(math.radians(max_grade_deg))
    if max_grad > limit:
        height *= limit / max_grad
    return height


@dataclass(kw_only=True)
class XingtianCourseTerrainCfg(SubTerrainCfg):
    terrain: dict[str, Any]
    route_length_m: float | None = None

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        kind = self.terrain["kind"]
        length = float(self.route_length_m or self.terrain["length_m"])
        width = float(self.terrain["width_m"])
        n_x = max(4, round(length / 0.1) + 1)
        n_y = max(4, round(width / 0.1) + 1)
        x = np.linspace(0.0, length, n_x)
        y = np.linspace(0.0, width, n_y)
        xx = np.broadcast_to(x[:, None], (n_x, n_y)).copy()
        height = np.zeros((n_x, n_y), dtype=float)

        if kind in {"rough", "composite"}:
            amp = float(self.terrain.get("amplitude_m", 0.04))
            amp *= max(0.25, difficulty)
            rough = _rough_height(
                x,
                y,
                rng,
                amp,
                float(self.terrain.get("max_grade_deg", 35.0)),
            )
            if kind == "rough":
                height += rough
            else:
                approach = (xx >= 8.0) & (xx < 16.0)
                after = (xx >= 38.0) & (xx < 40.0)
                height += rough * (approach | after)

        if kind in {"slope", "composite"}:
            grade = (
                float(self.terrain.get("grade_deg", 35.0))
                if kind == "slope"
                else 15.0 + 20.0 * difficulty
            )
            if kind == "slope":
                ramp_start = 0.75
                ramp_length = min(3.5, length - 2.0)
                crest_length = min(float(self.terrain.get("crest_length_m", 3.0)), 1.0)
            else:
                ramp_start = float(self.terrain.get("ramp_start_m", 16.0)) + 2.0
                ramp_length = float(self.terrain.get("ramp_length_m", 8.0))
                crest_length = float(self.terrain.get("crest_length_m", 4.0))
            rise = math.tan(math.radians(grade))
            ramp = np.clip((xx - ramp_start) / max(ramp_length, 1e-6), 0.0, 1.0)
            height += rise * ramp_length * ramp
            down_start = ramp_start + ramp_length + crest_length
            descend = np.clip((xx - down_start) / max(ramp_length, 1e-6), 0.0, 1.0)
            height -= rise * ramp_length * descend
            height = np.maximum(height, 0.0)

        h_min = float(height.min())
        h_max = float(height.max())
        h_range = max(h_max - h_min, 1e-3)
        normalized = (height - h_min) / h_range
        field = spec.add_hfield(
            name=f"xingtian_hfield_{uuid.uuid4().hex}",
            size=[length / 2, width / 2, h_range, max(h_range, 1e-3)],
            nrow=n_x,
            ncol=n_y,
            userdata=normalized.astype(np.float32).ravel().tolist(),
        )
        body = spec.body("terrain")
        hfield_geom = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_HFIELD,
            hfieldname=field.name,
            pos=[length / 2, width / 2, h_min],
            rgba=[0.35, 0.42, 0.32, 1.0],
            group=0,
        )
        geometries = [TerrainGeometry(geom=hfield_geom, hfield=field)]
        geometries.extend(
            self._obstacle_geometries(body, difficulty, rng)
        )

        spawn_x = 0.75
        spawn_ix = int(np.argmin(np.abs(x - spawn_x)))
        spawn_y = int(n_y // 2)
        z = float(h_min + normalized[spawn_ix, spawn_y] * h_range)
        return TerrainOutput(
            origin=np.array([spawn_x, width / 2, z]),
            geometries=geometries,
        )

    def _obstacle_geometries(
        self,
        body: mujoco.MjsBody,
        difficulty: float,
        rng: np.random.Generator,
    ) -> list[TerrainGeometry]:
        kind = self.terrain["kind"]
        if kind != "obstacles" and kind != "composite":
            return []
        offsets = _obstacle_offsets(self.terrain)
        types = self.terrain.get("types", ["box", "cylinder", "capsule"])
        low = float(self.terrain.get("min_height_m", 0.02))
        high = float(self.terrain.get("max_height_m", 0.05))
        geometries: list[TerrainGeometry] = []
        patch_id = uuid.uuid4().hex[:8]
        for index, x_offset in enumerate(offsets):
            height = low + (high - low) * float(np.clip(difficulty, 0.0, 1.0))
            obstacle_type = types[index % len(types)]
            if obstacle_type == "box":
                geom = body.add_geom(
                    name=f"roller_obstacle_{patch_id}_{index}",
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=[0.12, 0.75, height / 2],
                    pos=[x_offset, float(self.terrain["width_m"]) / 2, height / 2],
                    rgba=[0.7, 0.45, 0.18, 1.0],
                    group=0,
                )
            elif obstacle_type == "cylinder":
                geom = body.add_geom(
                    name=f"roller_obstacle_{patch_id}_{index}",
                    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    size=[height / 2, 0.75, 0],
                    pos=[
                        x_offset,
                        float(self.terrain["width_m"]) / 2,
                        height / 2,
                    ],
                    quat=[math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0],
                    rgba=[0.55, 0.55, 0.58, 1.0],
                    group=0,
                )
            elif obstacle_type == "capsule":
                geom = body.add_geom(
                    name=f"roller_obstacle_{patch_id}_{index}",
                    type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                    size=[height / 2, 0.75, 0],
                    pos=[
                        x_offset,
                        float(self.terrain["width_m"]) / 2,
                        height / 2,
                    ],
                    quat=[math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0],
                    rgba=[0.30, 0.58, 0.40, 1.0],
                    group=0,
                )
            else:
                raise ValueError(f"Unsupported obstacle type '{obstacle_type}'")
            geom.contype = 2
            geom.conaffinity = 1
            geometries.append(TerrainGeometry(geom=geom))
        return geometries


class XingtianReward:
    """One GPU reward and evaluation-metric term backed by the course JSON."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self._env = env
        self.stage_name = cfg.params["stage_name"]
        self.stage = COURSE.stage(self.stage_name)
        self.weights = self.stage["reward_weights"]
        defaults = dict(COURSE.reward_defaults)
        defaults.update(COURSE.stages.get(self.stage_name, {}).get("reward_defaults", {}))
        self.defaults = defaults
        self.penalties = dict(COURSE.reward_penalties)
        self.penalties.update(self.stage.get("overrides", {}))
        self.robot: Entity = env.scene["robot"]
        self.command_term: CourseCommand = env.command_manager.get_term("course")
        self._profile_peak = torch.tensor(
            [
                max(abs(float(point[1])) for point in profile["points"])
                for profile in self.stage["profiles"]
            ],
            device=self.device,
        )
        joint_ids, joint_names = self.robot.find_joints(
            JOINTS, preserve_order=True
        )
        index = {name: i for i, name in enumerate(joint_names)}
        self._wheel_ids = torch.tensor(
            [joint_ids[index[name]] for name in WHEEL_JOINTS],
            device=self.device,
            dtype=torch.long,
        )
        self._wheel_action_ids = torch.tensor(
            [JOINTS.index(name) for name in WHEEL_JOINTS],
            device=self.device,
            dtype=torch.long,
        )
        self.obstacle_offsets = _obstacle_offsets(self.stage["terrain"])
        self._initialize_buffers()

    def _initialize_buffers(self) -> None:
        n = self.num_envs
        self._initialized = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._last_x = torch.zeros(n, device=self.device)
        self._last_vx = torch.zeros(n, device=self.device)
        self._peak_speed = torch.zeros(n, device=self.device)
        self._last_cmd = torch.zeros(n, device=self.device)
        self._steps = torch.zeros(n, device=self.device)
        self._speed_err = torch.zeros(n, device=self.device)
        self._yaw_err = torch.zeros(n, device=self.device)
        self._upright = torch.zeros(n, device=self.device)
        self._height_err = torch.zeros(n, device=self.device)
        self._height_max = torch.zeros(n, device=self.device)
        self._roll_sum = torch.zeros(n, device=self.device)
        self._roll_max = torch.zeros(n, device=self.device)
        self._pitch_sum = torch.zeros(n, device=self.device)
        self._steady_pitch_sum = torch.zeros(n, device=self.device)
        self._steady_steps = torch.zeros(n, device=self.device)
        self._pitch_max = torch.zeros(n, device=self.device)
        self._heading_error_max = torch.zeros(n, device=self.device)
        self._wheel_slip_sum = torch.zeros(n, device=self.device)
        self._wheel_slip_max = torch.zeros(n, device=self.device)
        self._wheel_slip_excess_ema = torch.zeros(n, device=self.device)
        self._backslide_max = torch.zeros(n, device=self.device)
        self._cross_track_max = torch.zeros(n, device=self.device)
        self._wheel_center_max = torch.zeros(n, device=self.device)
        self._wheel_difference_max = torch.zeros(n, device=self.device)
        self._wheel_target_difference_max = torch.zeros(n, device=self.device)
        self._wheel_sync_excess_ema = torch.zeros(n, device=self.device)
        self._hold_drift_max = torch.zeros(n, device=self.device)
        self._hold_drift_settled_max = torch.zeros(n, device=self.device)
        self._hold_origin = torch.zeros((n, 2), device=self.device)
        self._hold_settled_origin = torch.zeros((n, 2), device=self.device)
        self._hold_steps = torch.zeros(n, device=self.device)
        self._holding = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._hold_settled_active = torch.zeros(
            n, dtype=torch.bool, device=self.device
        )
        self._in_place_max = torch.zeros(n, device=self.device)
        self._in_place_origin = torch.zeros((n, 2), device=self.device)
        self._in_place_active_previous = torch.zeros(
            n, dtype=torch.bool, device=self.device
        )
        self._stop_speed_max = torch.zeros(n, device=self.device)
        self._stop_speed_excess_ema = torch.zeros(n, device=self.device)
        self._power_sum = torch.zeros(n, device=self.device)
        self._stop_cycles = torch.zeros(n, device=self.device)
        self._obstacle_mask = torch.zeros(n, dtype=torch.long, device=self.device)
        self._obstacles = torch.zeros(n, device=self.device)
        self._finished = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._crested = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._fall = torch.zeros(n, dtype=torch.bool, device=self.device)

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    @property
    def device(self) -> str:
        return self._env.device

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        for tensor in (
            self._initialized,
            self._finished,
            self._crested,
            self._fall,
            self._holding,
            self._hold_settled_active,
            self._in_place_active_previous,
        ):
            tensor[env_ids] = False
        for tensor in (
            self._stop_cycles,
            self._obstacle_mask,
        ):
            tensor[env_ids] = 0
        for tensor in (
            self._steps,
            self._speed_err,
            self._yaw_err,
            self._upright,
            self._height_err,
            self._height_max,
            self._roll_sum,
            self._roll_max,
            self._pitch_sum,
            self._steady_pitch_sum,
            self._steady_steps,
            self._pitch_max,
            self._heading_error_max,
            self._wheel_slip_sum,
            self._wheel_slip_max,
            self._wheel_slip_excess_ema,
            self._backslide_max,
            self._cross_track_max,
            self._wheel_center_max,
            self._wheel_difference_max,
            self._wheel_target_difference_max,
            self._wheel_sync_excess_ema,
            self._hold_drift_max,
            self._hold_drift_settled_max,
            self._hold_steps,
            self._in_place_max,
            self._stop_speed_max,
            self._stop_speed_excess_ema,
            self._power_sum,
            self._obstacles,
            self._last_vx,
            self._peak_speed,
            self._last_cmd,
        ):
            tensor[env_ids] = 0.0
        self._hold_origin[env_ids] = 0.0
        self._hold_settled_origin[env_ids] = 0.0
        self._in_place_origin[env_ids] = 0.0

    def _quality(
        self, observed: torch.Tensor, tolerance: float
    ) -> torch.Tensor:
        return torch.exp(-torch.square(observed / max(tolerance, 1e-6)))

    @staticmethod
    def _huber(error: torch.Tensor, tolerance: float) -> torch.Tensor:
        ratio = torch.abs(error) / max(tolerance, 1e-6)
        return torch.where(
            ratio <= 1.0,
            0.5 * torch.square(ratio),
            ratio - 0.5,
        )

    @staticmethod
    def _gate_excess(
        value: torch.Tensor,
        limit: float,
    ) -> torch.Tensor:
        return torch.clamp(value / max(limit, 1e-6) - 1.0, min=0.0)

    @classmethod
    def _gate_cost(
        cls,
        value: torch.Tensor,
        limit: float,
    ) -> torch.Tensor:
        ratio = value / max(limit, 1e-6)
        return 0.05 * torch.square(ratio) + cls._gate_excess(value, limit)

    def _command_gradient(self, command: torch.Tensor) -> torch.Tensor:
        return (command - self._last_cmd) / self._env.step_dt

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        stage_name: str,
    ) -> torch.Tensor:
        del stage_name
        robot = self.robot
        B = self.num_envs
        cmd = self.command_term.command
        vx = robot.data.root_link_lin_vel_b[:, 0]
        vy = robot.data.root_link_lin_vel_b[:, 1]
        vz = robot.data.root_link_lin_vel_b[:, 2]
        wz = robot.data.root_link_ang_vel_b[:, 2]
        wheel_ang = robot.data.root_link_ang_vel_b[:, :2]
        gravity_xy = robot.data.projected_gravity_b[:, :2]
        pos_xy = robot.data.root_link_pos_w[:, :2]
        heading = robot.data.heading_w
        scan = env.scene["terrain_scan"]
        center_ray = scan.num_rays_per_frame // 2
        clearance = scan.data.distances[:, center_ray]

        fresh = self._steps == 0
        if fresh.any():
            self._last_x[fresh] = robot.data.root_link_pos_w[fresh, 0]
            self._last_vx[fresh] = vx[fresh]
            self._initialized[fresh] = True

        speed_tol = float(self.defaults.get("speed_tolerance_base_mps", 0.08))
        yaw_tol = float(self.defaults.get("yaw_tolerance_base_rad_s", 0.12))
        if self.stage["terrain"]["kind"] in {"slope", "rough", "composite"}:
            speed_tol = float(
                self.defaults.get("speed_tolerance_slope", speed_tol)
            )
            yaw_tol = float(
                self.defaults.get("yaw_tolerance_slope", yaw_tol)
            )
        height_tol = float(self.defaults.get("height_tolerance_m", 0.025))
        pitch_tol = float(self.defaults.get("pitch_tolerance_rad", 0.0873))
        slip_tol = float(self.defaults.get("wheel_slip_tolerance_mps", 0.5))
        accel_tol = float(self.defaults.get("acceleration_tolerance_mps2", 0.8))
        zero_tol = float(self.defaults.get("zero_speed_tolerance_mps", 0.03))
        drift_tol = float(self.defaults.get("drift_tolerance_m", 0.2))
        heading_tol = float(self.defaults.get("heading_tolerance_rad", 0.1745))
        straight_heading_tol = float(
            self.defaults.get("straight_heading_tolerance_rad", heading_tol)
        )
        alignment_tol = float(
            self.defaults.get("wheel_alignment_tolerance_m", 0.01)
        )
        straight_yaw_tol = float(
            self.defaults.get("straight_yaw_threshold_rad_s", 0.05)
        )

        speed = self._quality(vx - cmd[:, 0], speed_tol)
        yaw = self._quality(wz - cmd[:, 1], yaw_tol)
        heading_error = torch.atan2(
            torch.sin(heading - self.command_term.target_heading),
            torch.cos(heading - self.command_term.target_heading),
        )
        heading_cost = self._huber(heading_error, straight_heading_tol)
        straight_heading_cost = torch.where(
            torch.abs(cmd[:, 1]) <= straight_yaw_tol,
            heading_cost,
            torch.zeros(B, device=self.device),
        )
        cross_track_error = (
            pos_xy[:, 1] - self.command_term.start_pos[:, 1]
        )
        cross_track_cost = torch.where(
            torch.abs(cmd[:, 1]) <= straight_yaw_tol,
            self._huber(
                cross_track_error,
                float(self.stage["gate"].get("max_cross_track_error_m", 0.05)),
            ),
            torch.zeros(B, device=self.device),
        )
        height = self._quality(clearance - cmd[:, 2], height_tol)
        upright = torch.exp(-torch.sum(torch.square(gravity_xy), dim=1) / 0.04)
        pitch = torch.exp(-torch.square(gravity_xy[:, 0]) / pitch_tol**2)
        wheel_q = robot.data.joint_vel[:, self._wheel_ids]
        wheel_forward = -wheel_q * WHEEL_RADIUS
        wheel_avg = wheel_forward.mean(dim=1)
        wheel_slip = torch.abs(vx - wheel_avg)
        wheel_difference = torch.abs(wheel_forward[:, 0] - wheel_forward[:, 1])
        wheel_target = env.action_manager.get_term("joints").wheel_velocity_target
        wheel_target_difference = torch.abs(
            wheel_target[:, 0] - wheel_target[:, 1]
        )
        raw_wheel_action = env.action_manager.action[:, self._wheel_action_ids]
        previous_wheel_action = env.action_manager.prev_action[
            :, self._wheel_action_ids
        ]
        wheel_direction_flip = (
            raw_wheel_action * previous_wheel_action < 0.0
        ).float().mean(dim=1)
        wheel_slip_reward = torch.exp(-torch.square(wheel_slip / slip_tol))

        moving = torch.abs(cmd[:, 0]) > zero_tol
        stopping = torch.abs(cmd[:, 0]) <= zero_tol
        stand = (torch.abs(cmd[:, 0]) <= zero_tol) & (
            torch.abs(cmd[:, 1]) <= yaw_tol
        )
        straight = torch.abs(cmd[:, 1]) <= straight_yaw_tol
        settle_time = float(self.defaults.get("hold_settle_time_s", 0.0))
        if settle_time < 0.0:
            raise ValueError("hold_settle_time_s must be nonnegative")
        settle_steps = round(settle_time / self._env.step_dt)
        settle_required = settle_steps > 0
        self._heading_error_max = torch.maximum(
            self._heading_error_max, torch.abs(heading_error)
        )
        self._cross_track_max = torch.maximum(
            self._cross_track_max, torch.abs(cross_track_error)
        )
        self._wheel_difference_max = torch.maximum(
            self._wheel_difference_max,
            torch.where(
                straight & self._hold_settled_active
                if settle_required
                else straight,
                wheel_difference,
                torch.zeros_like(wheel_difference),
            ),
        )
        self._wheel_target_difference_max = torch.maximum(
            self._wheel_target_difference_max,
            wheel_target_difference,
        )
        new_hold = stand & ~self._holding
        self._hold_origin[new_hold] = pos_xy[new_hold]
        self._holding[:] = stand
        self._hold_steps = torch.where(
            self._holding,
            self._hold_steps + 1.0,
            torch.zeros(B, device=self.device),
        )
        new_settled_hold = (
            self._holding
            & (self._hold_steps >= settle_steps)
            & ~self._hold_settled_active
        )
        self._hold_settled_origin[new_settled_hold] = pos_xy[new_settled_hold]
        self._hold_settled_active[:] = (
            self._holding & (self._hold_steps >= settle_steps)
        )
        hold_displacement = torch.norm(pos_xy - self._hold_origin, dim=1)
        settled_hold_drift = torch.where(
            self._hold_settled_active,
            torch.norm(pos_xy - self._hold_settled_origin, dim=1),
            torch.zeros(B, device=self.device),
        )
        metric_active = (
            self._hold_settled_active
            if settle_required
            else torch.ones_like(self._holding)
        )
        self._wheel_slip_max = torch.maximum(
            self._wheel_slip_max,
            torch.where(metric_active, wheel_slip, torch.zeros_like(wheel_slip)),
        )
        self._stop_speed_max = torch.maximum(
            self._stop_speed_max,
            torch.where(
                stopping & metric_active,
                torch.abs(vx),
                torch.zeros_like(vx),
            ),
        )
        stationary = torch.where(
            stand,
            torch.exp(-torch.sum(torch.square(robot.data.root_link_lin_vel_b[:, :2]), dim=1) / zero_tol**2),
            torch.zeros(B, device=self.device),
        )
        drift_cost = torch.where(
            self._holding,
            self._huber(hold_displacement, drift_tol),
            torch.zeros(B, device=self.device),
        )
        zero_motion_error = torch.sqrt(
            torch.square(
                torch.norm(robot.data.root_link_lin_vel_b[:, :2], dim=1)
                / max(zero_tol, 1e-6)
            )
            + torch.square(wz / max(yaw_tol, 1e-6))
            + torch.square(heading_error / max(heading_tol, 1e-6))
        )
        zero_motion_cost = torch.where(
            stand,
            torch.log1p(torch.square(zero_motion_error)),
            torch.zeros(B, device=self.device),
        )
        accel = (vx - self._last_vx) / self._env.step_dt
        commanded_accel = self._command_gradient(cmd[:, 0])
        acceleration = torch.where(
            torch.abs(commanded_accel) > 0.01,
            torch.exp(-torch.square((accel - commanded_accel) / accel_tol)),
            torch.zeros(B, device=self.device),
        )
        braking = torch.where(
            commanded_accel < -0.01,
            torch.exp(-torch.square(torch.clamp(accel - commanded_accel, min=0) / accel_tol)),
            torch.zeros(B, device=self.device),
        )
        in_place_active = (torch.abs(cmd[:, 0]) < zero_tol) & (
            torch.abs(cmd[:, 1]) > yaw_tol
        )
        in_place = torch.where(
            in_place_active,
            torch.exp(-torch.square(torch.norm(robot.data.root_link_lin_vel_b[:, :2], dim=1) / 0.2)),
            torch.zeros(B, device=self.device),
        )
        zero_speed_cost = torch.where(
            moving & (torch.abs(vx) < zero_tol),
            torch.ones(B, device=self.device),
            torch.zeros(B, device=self.device),
        )
        stop_quality = torch.where(
            stopping,
            torch.exp(-torch.square(torch.abs(vx) / max(zero_tol, 1e-6))),
            torch.zeros(B, device=self.device),
        )
        slip_fraction_limit = float(
            self.stage["gate"].get("max_wheel_slip_fraction_of_peak", 1.0)
        )
        slip_absolute_limit = float(
            self.stage["gate"].get("max_wheel_slip_mps", 0.5)
        )
        raw_peak_command = self._profile_peak[self.command_term.profile_ids]
        peak_command = torch.where(
            raw_peak_command > 1e-6,
            raw_peak_command,
            torch.ones_like(raw_peak_command),
        )
        slip_limit = torch.minimum(
            peak_command * slip_fraction_limit,
            torch.full_like(peak_command, slip_absolute_limit),
        )
        wheel_slip_ratio = wheel_slip / slip_limit.clamp(min=1e-6)
        wheel_slip_current_cost = (
            0.05 * torch.square(wheel_slip_ratio)
            + torch.clamp(wheel_slip_ratio - 1.0, min=0.0)
        )
        stop_speed_limit = float(
            self.stage["gate"].get("max_stop_speed_mps", 0.1)
        )
        stop_speed_current_cost = torch.where(
            stopping,
            self._gate_cost(
                torch.abs(vx),
                stop_speed_limit,
            ),
            torch.zeros(B, device=self.device),
        )
        wheel_sync_limit = float(
            self.stage["gate"].get(
                "max_straight_wheel_speed_difference_mps",
                0.15,
            )
        )
        wheel_sync_current_cost = torch.where(
            straight,
            self._gate_cost(
                wheel_difference,
                wheel_sync_limit,
            ),
            torch.zeros(B, device=self.device),
        )
        gate_decay = float(
            self.defaults.get("gate_penalty_ema_decay", 0.98)
        )
        self._wheel_slip_excess_ema = (
            gate_decay * self._wheel_slip_excess_ema
            + (1.0 - gate_decay) * wheel_slip_current_cost
        )
        self._stop_speed_excess_ema = (
            gate_decay * self._stop_speed_excess_ema
            + (1.0 - gate_decay) * stop_speed_current_cost
        )
        self._wheel_sync_excess_ema = (
            gate_decay * self._wheel_sync_excess_ema
            + (1.0 - gate_decay) * wheel_sync_current_cost
        )
        wheel_slip_gate_cost = (
            wheel_slip_current_cost + self._wheel_slip_excess_ema
        )
        stop_speed_gate_cost = (
            stop_speed_current_cost + self._stop_speed_excess_ema
        )
        wheel_sync_gate_cost = (
            wheel_sync_current_cost + self._wheel_sync_excess_ema
        )
        wheel_target_scale = max(
            float(
                self.stage.get("control", {}).get(
                    "wheel_speed_target_max_rad_s", 22.0
                )
            ),
            1.0,
        )
        wheel_chatter_cost = (
            (wheel_target_difference / wheel_target_scale).clamp(max=2.0)
            + wheel_direction_flip
        )
        direction = torch.sign(cmd[:, 0])
        backslide = torch.clamp(-direction * vx, min=0.0)
        under = torch.exp(-torch.square(torch.clamp(torch.abs(cmd[:, 0]) - torch.abs(vx), min=0) / speed_tol))
        vertical_cost = torch.square(vz)
        angular_cost = torch.sum(torch.square(wheel_ang), dim=1)

        wheel_poses = robot.data.body_link_pos_w[
            :, robot.find_bodies(("left_wheel", "right_wheel"))[0]
        ]
        wheel_delta = wheel_poses[:, 0, :2] - wheel_poses[:, 1, :2]
        wheel_center_offset = torch.abs(
            wheel_delta[:, 0] * torch.cos(heading)
            + wheel_delta[:, 1] * torch.sin(heading)
        )
        wheel_alignment_cost = torch.where(
            torch.abs(cmd[:, 1]) <= straight_yaw_tol,
            torch.clamp(
                (wheel_center_offset - alignment_tol)
                / max(alignment_tol, 1e-6),
                min=0.0,
                max=1.0,
            ),
            torch.zeros(B, device=self.device),
        )
        soft_limits = robot.data.soft_joint_pos_limits
        q = robot.data.joint_pos
        joint_limit_cost = torch.sum(
            torch.clamp(soft_limits[..., 0] - q, min=0)
            + torch.clamp(q - soft_limits[..., 1], min=0),
            dim=1,
        )
        torque = robot.data.actuator_force
        torque_cost = torch.sum(torch.square(torque), dim=1)
        over_rated_cost = torch.square(
            torch.clamp(torch.abs(torque) - RATED_TORQUE, min=0)
        ).sum(dim=1)
        power = torch.clamp(
            robot.data.qfrc_actuator * robot.data.joint_vel, min=0
        ).sum(dim=1)
        action_rate = torch.sum(
            torch.square(env.action_manager.action - env.action_manager.prev_action),
            dim=1,
        )

        w = self.weights
        p = self.penalties
        reward = (
            float(w.get("speed", 0.0)) * speed
            + float(w.get("underspeed", 0.0)) * under
            + float(w.get("yaw", 0.0)) * yaw
            - float(w.get("straight_heading", 0.0)) * straight_heading_cost
            - float(w.get("cross_track", 0.0)) * cross_track_cost
            + float(w.get("height", 0.0)) * height
            + float(w.get("upright", 0.0)) * upright
            + float(w.get("pitch", 0.0)) * pitch
            + float(w.get("stationary", 0.0)) * stationary
            - float(w.get("drift", 0.0)) * drift_cost
            - float(w.get("zero_motion", 0.0)) * zero_motion_cost
            + float(w.get("acceleration", 0.0)) * acceleration
            + float(w.get("in_place", 0.0)) * in_place
            + float(w.get("wheel_slip", 0.0)) * wheel_slip_reward
            + float(w.get("braking", 0.0)) * braking
            + float(w.get("zero_speed", 0.0)) * stop_quality
            - float(p.get("lateral", 0.0)) * torch.square(vy)
            - float(p.get("action_rate", 0.0)) * action_rate
            - float(p.get("torque", 0.0)) * torque_cost
            - float(w.get("over_rated", p.get("over_rated", 0.0)))
            * over_rated_cost
            - float(p.get("joint_limit", 0.0)) * joint_limit_cost
            - float(p.get("zero_speed", 0.0)) * zero_speed_cost
            - float(p.get("wheel_alignment", 0.0)) * wheel_alignment_cost
            - float(p.get("wheel_sync", 0.0)) * torch.square(wheel_difference)
            - float(w.get("wheel_slip_gate", 0.0)) * wheel_slip_gate_cost
            - float(w.get("stop_speed_gate", 0.0)) * stop_speed_gate_cost
            - float(w.get("wheel_sync_gate", 0.0)) * wheel_sync_gate_cost
            - float(w.get("wheel_chatter", 0.0)) * wheel_chatter_cost
            - float(w.get("vertical_velocity", 0.0)) * vertical_cost
            - float(w.get("angular_xy", 0.0)) * angular_cost
            - float(w.get("backslide", 0.0)) * backslide
            - float(w.get("power", 0.0)) * power
        )

        x = robot.data.root_link_pos_w[:, 0]
        signed_forward = x - self.command_term.start_pos[:, 0]
        progress = torch.abs(signed_forward)
        for obstacle_index, offset in enumerate(self.obstacle_offsets):
            marker = 1 << obstacle_index
            crossed = (
                (self._last_x - (self.command_term.start_pos[:, 0] + offset))
                * (x - (self.command_term.start_pos[:, 0] + offset))
                <= 0
            ) & (self._steps > 0)
            new_cross = crossed & ((self._obstacle_mask & marker) == 0)
            self._obstacle_mask |= new_cross.long() * marker
            self._obstacles += new_cross.float()
            reward += new_cross.float() * float(self.stage.get("events", {}).get("obstacle_pass", 0.0))

        if self.stage["terrain"]["kind"] == "composite":
            terrain = self.stage["terrain"]
            finish_x = float(terrain["length_m"])
            finished = progress >= finish_x
            new_finish = finished & ~self._finished
            self._finished |= finished
            reward += new_finish.float() * float(self.stage.get("events", {}).get("finish", 0.0))
            crest_distance = (
                float(terrain["ramp_start_m"])
                + float(terrain["ramp_length_m"])
                + float(terrain["crest_length_m"])
            )
            crest = progress >= crest_distance
            new_crest = crest & ~self._crested & ~self._fall
            self._crested |= crest
            reward += new_crest.float() * float(
                self.stage.get("events", {}).get("slope_crest", 0.0)
            )

        fell = env.termination_manager.terminated
        # RewardManager also scales by dt. Keep the configured failure cost
        # independent of control frequency so early termination is never cheap.
        reward_scale = (
            self._env.step_dt
            if self._env.cfg.scale_rewards_by_dt
            else 1.0
        )
        reward -= fell.float() * float(p.get("fall", 0.0)) / reward_scale
        self._fall |= fell

        new_stop = (
            (torch.abs(self._last_cmd) > zero_tol)
            & stopping
            & (torch.abs(vx) <= float(self.stage["gate"].get("max_stop_speed_mps", 0.1)))
        )
        self._stop_cycles += new_stop.float()

        self._speed_err += torch.abs(vx - cmd[:, 0])
        self._yaw_err += torch.abs(wz - cmd[:, 1])
        self._upright += upright
        self._height_err += torch.abs(clearance - cmd[:, 2])
        self._height_max = torch.maximum(
            self._height_max, torch.abs(clearance - cmd[:, 2])
        )
        self._pitch_sum += torch.abs(gravity_xy[:, 0])
        self._steady_pitch_sum += torch.where(
            metric_active,
            torch.abs(gravity_xy[:, 0]),
            torch.zeros(B, device=self.device),
        )
        self._steady_steps += metric_active.float()
        self._pitch_max = torch.maximum(self._pitch_max, torch.abs(gravity_xy[:, 0]))
        self._roll_sum += torch.abs(gravity_xy[:, 1])
        self._roll_max = torch.maximum(
            self._roll_max, torch.abs(gravity_xy[:, 1])
        )
        self._wheel_slip_sum += wheel_slip
        self._backslide_max = torch.maximum(self._backslide_max, backslide)
        self._wheel_center_max = torch.maximum(
            self._wheel_center_max, wheel_center_offset
        )
        self._hold_drift_max = torch.maximum(
            self._hold_drift_max,
            torch.where(
                self._holding,
                hold_displacement,
                torch.zeros_like(hold_displacement),
            ),
        )
        self._hold_drift_settled_max = torch.maximum(
            self._hold_drift_settled_max,
            settled_hold_drift,
        )
        new_in_place = in_place_active & ~self._in_place_active_previous
        self._in_place_origin[new_in_place] = pos_xy[new_in_place]
        in_place_displacement = torch.norm(
            pos_xy - self._in_place_origin, dim=1
        )
        self._in_place_max = torch.maximum(
            self._in_place_max,
            torch.where(
                in_place_active,
                in_place_displacement,
                torch.zeros(B, device=self.device),
            ),
        )
        self._in_place_active_previous[:] = in_place_active
        self._power_sum += power
        self._peak_speed = torch.maximum(self._peak_speed, torch.abs(vx))
        self._steps += 1
        self._last_x[:] = x
        self._last_vx[:] = vx
        self._last_cmd[:] = cmd[:, 0]
        if self._env.extras.get("log") is not None:
            wheel_actual = robot.data.joint_vel[:, self._wheel_ids]
            log_values = {
                "Metrics/speed_error_mps": torch.abs(vx - cmd[:, 0]).mean(),
                "Metrics/yaw_error_rad_s": torch.abs(wz - cmd[:, 1]).mean(),
                "Metrics/straight_heading_error_rad": torch.abs(
                    heading_error
                ).mean(),
                "Metrics/heading_error_max_deg": torch.rad2deg(
                    self._heading_error_max
                ).mean(),
                "Metrics/straight_heading_cost": straight_heading_cost.mean(),
                "Metrics/cross_track_cost": cross_track_cost.mean(),
                "Metrics/wheel_slip_gate_cost": wheel_slip_gate_cost.mean(),
                "Metrics/stop_speed_gate_cost": stop_speed_gate_cost.mean(),
                "Metrics/wheel_sync_gate_cost": wheel_sync_gate_cost.mean(),
                "Metrics/wheel_chatter_cost": wheel_chatter_cost.mean(),
                "Metrics/world_heading_deg": torch.rad2deg(heading).mean(),
                "Metrics/upright": upright.mean(),
                "Metrics/height_error_m": torch.abs(
                    clearance - cmd[:, 2]
                ).mean(),
                "Metrics/wheel_slip_mps": wheel_slip.mean(),
                "Metrics/left_wheel_target_rad_s": wheel_target[:, 0].mean(),
                "Metrics/right_wheel_target_rad_s": wheel_target[:, 1].mean(),
                "Metrics/left_wheel_actual_rad_s": wheel_actual[:, 0].mean(),
                "Metrics/right_wheel_actual_rad_s": wheel_actual[:, 1].mean(),
                "Metrics/wheel_target_difference_rad_s": torch.abs(
                    wheel_target[:, 0] - wheel_target[:, 1]
                ).mean(),
                "Metrics/wheel_target_difference_max_rad_s": (
                    self._wheel_target_difference_max.mean()
                ),
                "Metrics/wheel_actual_difference_rad_s": torch.abs(
                    wheel_actual[:, 0] - wheel_actual[:, 1]
                ).mean(),
                "Metrics/forward_progress_m": progress.mean(),
                "Metrics/hold_drift_m": hold_displacement.mean(),
                "Metrics/hold_drift_after_0.5s_m": settled_hold_drift.mean(),
                "Metrics/stop_quality": stop_quality.mean(),
                "Metrics/zero_motion_cost": zero_motion_cost.mean(),
                "Metrics/obstacle_passes": self._obstacles.mean(),
                "Metrics/finish_rate": self._finished.float().mean(),
                "Metrics/fall_rate": self._fall.float().mean(),
            }
            profile_ids = self.command_term.profile_ids
            profile_metrics = {
                "heading_error_deg": torch.rad2deg(self._heading_error_max),
                "cross_track_m": self._cross_track_max,
                "wheel_slip_gate_fraction": self._wheel_slip_max
                / (
                    peak_command * slip_fraction_limit
                ).clamp(min=1e-6),
                "stop_speed_mps": self._stop_speed_max,
                "wheel_sync_mps": self._wheel_difference_max,
                "wheel_target_difference_rad_s": (
                    self._wheel_target_difference_max
                ),
                "wheel_chatter_cost": wheel_chatter_cost,
            }
            for profile_index, profile in enumerate(self.command_term.profiles):
                mask = profile_ids == profile_index
                count = mask.sum().clamp(min=1)
                profile_name = profile["name"]
                log_values[
                    f"Metrics/straight_heading_error_{profile_name}_deg"
                ] = (
                    torch.rad2deg(torch.abs(heading_error)[mask]).sum() / count
                )
                log_values[f"Metrics/hold_drift_{profile_name}_m"] = (
                    hold_displacement[mask].sum() / count
                )
                log_values[
                    f"Metrics/hold_drift_after_0.5s_{profile_name}_m"
                ] = settled_hold_drift[mask].sum() / count
                for metric_name, metric_values in profile_metrics.items():
                    masked = torch.where(
                        mask,
                        metric_values,
                        torch.zeros_like(metric_values),
                    )
                    log_values[
                        f"Metrics/{profile_name}/{metric_name}_mean"
                    ] = masked.sum() / count
                    log_values[
                        f"Metrics/{profile_name}/{metric_name}_worst"
                    ] = masked.max()
            self._env.extras["log"].update(log_values)
        return reward

    def metrics(self) -> dict[str, torch.Tensor]:
        steps = self._steps.clamp(min=1)
        ratio = self._steps / max(self._env.max_episode_length, 1)
        raw_peak_command = torch.tensor(
            [
                max(abs(float(point[1])) for point in profile["points"])
                for profile in self.stage["profiles"]
            ],
            device=self.device,
        )[self.command_term.profile_ids]
        has_speed_command = raw_peak_command > 1e-6
        peak_command = torch.where(
            has_speed_command,
            raw_peak_command,
            torch.ones_like(raw_peak_command),
        )
        fraction_scale = has_speed_command.float()
        progress = torch.abs(
            self._last_x - self.command_term.start_pos[:, 0]
        )
        completed = (self._steps >= self._env.max_episode_length) & ~self._fall
        if self.stage["terrain"]["kind"] == "composite":
            completed = self._finished & ~self._fall
        speed_mae = self._speed_err / steps
        wheel_slip_fraction = (
            self._wheel_slip_max / peak_command * fraction_scale
        )
        stop_speed_fraction = (
            self._stop_speed_max / peak_command * fraction_scale
        )
        planned_peak_fraction = (
            self._peak_speed / peak_command * fraction_scale
        ).clamp(max=1.0)
        return {
            "pass_rate": completed.float(),
            "duration_fraction": ratio,
            "speed_mae_mps": speed_mae,
            "speed_mae_fraction_of_peak": speed_mae / peak_command * fraction_scale,
            "yaw_mae_rad_s": self._yaw_err / steps,
            "upright": self._upright / steps,
            "mean_abs_pitch_deg": torch.rad2deg(self._pitch_sum / steps),
            "pitch_deg": torch.rad2deg(self._pitch_max),
            "mean_abs_steady_pitch_deg": torch.rad2deg(
                self._steady_pitch_sum / self._steady_steps.clamp(min=1)
            ),
            "mean_abs_roll_deg": torch.rad2deg(self._roll_sum / steps),
            "roll_deg": torch.rad2deg(self._roll_max),
            "final_heading_error_deg": torch.rad2deg(
                torch.abs(
                    torch.atan2(
                        torch.sin(
                            self.robot.data.heading_w
                            - self.command_term.target_heading
                        ),
                        torch.cos(
                            self.robot.data.heading_w
                            - self.command_term.target_heading
                        ),
                    )
                )
            ),
            "mean_height_error_m": self._height_err / steps,
            "height_deviation_m": self._height_max,
            "wheel_slip_mps": self._wheel_slip_max,
            "wheel_slip_max_mps": self._wheel_slip_max,
            "wheel_slip_fraction_of_peak": wheel_slip_fraction,
            "backslide_mps": self._backslide_max,
            "cross_track_error_m": self._cross_track_max,
            "wheel_center_offset_m": self._wheel_center_max,
            "straight_wheel_speed_difference_mps": self._wheel_difference_max,
            "hold_drift_m": self._hold_drift_max,
            "settled_hold_drift_m": self._hold_drift_settled_max,
            "hold_drift_after_0.5s_m": self._hold_drift_settled_max,
            "in_place_displacement_m": self._in_place_max,
            "stop_speed_mps": self._stop_speed_max,
            "stop_speed_fraction_of_peak": stop_speed_fraction,
            "stop_cycles": self._stop_cycles,
            "obstacle_passes": self._obstacles,
            "forward_progress_m": progress,
            "planned_peak_progress_fraction": planned_peak_fraction,
            "finished": self._finished.float(),
            "fell": self._fall.float(),
            "mean_positive_power_w": self._power_sum / steps,
        }


def _proprio(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot: Entity = env.scene["robot"]
    command = env.command_manager.get_command("course")
    command_term = env.command_manager.get_term("course")
    joint_ids, _ = robot.find_joints(JOINTS, preserve_order=True)
    leg_ids = torch.tensor(
        [joint_ids[i] for i in LEGS], device=env.device, dtype=torch.long
    )
    joint_ids_t = torch.tensor(joint_ids, device=env.device, dtype=torch.long)
    signs = torch.tensor(OBSERVATION_SIGNS, device=env.device, dtype=torch.float)
    heading_error = torch.atan2(
        torch.sin(robot.data.heading_w - command_term.target_heading),
        torch.cos(robot.data.heading_w - command_term.target_heading),
    )
    command_observation = torch.stack(
        (
            command[:, 0] * 10.0,
            command[:, 1] * 0.25,
            heading_error * HEADING_ERROR_OBS_SCALE,
        ),
        dim=-1,
    )
    frame = torch.cat(
        (
            robot.data.root_link_ang_vel_b * 0.25,
            robot.data.projected_gravity_b,
            command_observation,
            robot.data.joint_pos[:, leg_ids] - robot.data.default_joint_pos[:, leg_ids],
            robot.data.joint_vel[:, joint_ids_t] * signs * 0.05,
            env.action_manager.prev_action,
        ),
        dim=-1,
    )
    if frame.shape[-1] != PROPRIO_FRAME_DIM or frame.dtype != torch.float32:
        raise RuntimeError(
            f"Proprio observation must be float32[{PROPRIO_FRAME_DIM}], got "
            f"{frame.dtype}{list(frame.shape)}"
        )
    return frame


def _height_scan(env: ManagerBasedRlEnv) -> torch.Tensor:
    values = envs_mdp.height_scan(
        env,
        sensor_name="terrain_scan",
        offset=TARGET_BASE_HEIGHT_M,
        miss_value=COURSE_HEIGHT_SCALE,
    )
    return values / COURSE_HEIGHT_SCALE


def _privileged(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot: Entity = env.scene["robot"]
    scan = env.scene["terrain_scan"]
    normals = scan.data.normals_w[:, scan.num_rays_per_frame // 2]
    normals /= normals.norm(dim=1, keepdim=True).clamp(min=1e-6)
    contact = env.scene["wheel_ground_contact"]
    found = contact.data.found.float().mean(dim=1, keepdim=True)
    return torch.cat(
        (
            robot.data.root_link_lin_vel_b,
            robot.data.root_link_pos_w[:, 2:3],
            normals,
            found,
        ),
        dim=-1,
    )


def _terminate_body_contact(
    env: ManagerBasedRlEnv, sensor_name: str, seconds: float = 0.0
) -> torch.Tensor:
    del seconds
    sensor = env.scene[sensor_name]
    return sensor.data.found.any(dim=-1)


def _reset_passive_joints(
    env: ManagerBasedRlEnv, env_ids: torch.Tensor | None
) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    robot: Entity = env.scene["robot"]
    passive_ids, passive_names = robot.find_joints(
        PASSIVE_JOINTS, preserve_order=True
    )
    if tuple(passive_names) != PASSIVE_JOINTS:
        raise RuntimeError(f"Unexpected passive joint order: {passive_names}")
    knee_ids, knee_names = robot.find_joints(KNEE_JOINTS, preserve_order=True)
    if tuple(knee_names) != KNEE_JOINTS:
        raise RuntimeError(f"Unexpected knee joint order: {knee_names}")
    passive_ids_t = torch.tensor(
        passive_ids, device=env.device, dtype=torch.long
    )
    knee_ids_t = torch.tensor(knee_ids, device=env.device, dtype=torch.long)
    knees = robot.data.joint_pos[env_ids][:, knee_ids_t]
    knee_velocities = robot.data.joint_vel[env_ids][:, knee_ids_t]
    signs = torch.tensor(
        (1.0, -1.0, 1.0, -1.0), device=env.device, dtype=knees.dtype
    )
    passive_positions = knees[:, [0, 0, 1, 1]] * signs
    passive_velocities = knee_velocities[:, [0, 0, 1, 1]] * signs
    robot.write_joint_state_to_sim(
        passive_positions,
        passive_velocities,
        joint_ids=passive_ids_t,
        env_ids=env_ids,
    )


def verify_reset_state(
    env: ManagerBasedRlEnv, tolerance: float = 1e-5
) -> dict[str, float | int]:
    """Fail fast when GPU reset starts outside the closed-loop configuration."""
    robot: Entity = env.scene["robot"]
    passive_ids, passive_names = robot.find_joints(
        PASSIVE_JOINTS, preserve_order=True
    )
    if tuple(passive_names) != PASSIVE_JOINTS:
        raise RuntimeError(f"Unexpected passive joint order: {passive_names}")
    knee_ids, knee_names = robot.find_joints(KNEE_JOINTS, preserve_order=True)
    if tuple(knee_names) != KNEE_JOINTS:
        raise RuntimeError(f"Unexpected knee joint order: {knee_names}")
    passive_ids_t = torch.tensor(
        passive_ids, device=env.device, dtype=torch.long
    )
    knee_ids_t = torch.tensor(knee_ids, device=env.device, dtype=torch.long)
    knees = robot.data.joint_pos[:, knee_ids_t]
    expected = torch.stack(
        (knees[:, 0], -knees[:, 0], knees[:, 1], -knees[:, 1]), dim=1
    )
    max_error = float(
        torch.abs(robot.data.joint_pos[:, passive_ids_t] - expected).max().item()
    )
    if max_error > tolerance:
        raise RuntimeError(
            "GPU reset violates the parallelogram closure: "
            f"passive joint error {max_error:.6g} rad > {tolerance:.6g}"
        )
    contact_counts = {}
    for sensor_name in ("body_ground_contact", "leg_ground_contact"):
        contact = env.scene[sensor_name].data.found
        contact_counts[sensor_name] = int(
            contact.reshape(contact.shape[0], -1).any(dim=1).sum().item()
        )
    grounded = sum(contact_counts.values())
    if grounded:
        details = ", ".join(
            f"{name}={count}" for name, count in contact_counts.items()
        )
        raise RuntimeError(
            f"GPU reset starts with non-wheel ground contact ({details})"
        )
    return {
        "passive_joint_max_error_rad": max_error,
        "body_ground_contact_count": contact_counts["body_ground_contact"],
        "leg_ground_contact_count": contact_counts["leg_ground_contact"],
    }


def verify_zero_action_stance(
    env: ManagerBasedRlEnv,
    steps: int = 20,
    minimum_base_height: float = 0.23,
) -> dict[str, float]:
    """Require the default PD stance to survive the initial control horizon."""
    action = env.action_manager.action.new_zeros((env.num_envs, 6))
    robot: Entity = env.scene["robot"]
    minimum_height = float("inf")
    previous_auto_reset = env.cfg.auto_reset
    env.cfg.auto_reset = False
    try:
        for step in range(steps):
            _, reward, terminated, truncated, _ = env.step(action)
            if not bool(reward.isfinite().all().item()):
                raise RuntimeError("Zero-action stance produced a non-finite reward")
            if bool((terminated | truncated).any().item()):
                action_term: XingtianAction = env.action_manager.get_term("joints")
                actuator_gain = env.sim.model.actuator_gainprm[:, :, 0]
                actuator_bias = env.sim.model.actuator_biasprm[:, :, 1:3]
                terms = {
                    name: bool(env.termination_manager.get_term(name).any().item())
                    for name in (
                        "body_ground",
                        "leg_ground",
                        "bad_orientation",
                        "time_out",
                    )
                }
                leg_ids, _ = robot.find_joints(GAIT_JOINTS, preserve_order=True)
                leg_ids_t = torch.tensor(
                    leg_ids, device=env.device, dtype=torch.long
                )
                wheel_contact = env.scene["wheel_ground_contact"].data.found
                raise RuntimeError(
                    "Default standing pose terminated during stance check: "
                    f"step={step + 1}, base_height="
                    f"{float(robot.data.root_link_pos_w[:, 2].min().item()):.4f} m, "
                    f"leg_pos={robot.data.joint_pos[0, leg_ids_t].tolist()}, "
                    f"leg_vel={robot.data.joint_vel[0, leg_ids_t].tolist()}, "
                    f"actuator_force={robot.data.actuator_force[0].tolist()}, "
                    f"position_target={action_term.position_target[0].tolist()}, "
                    f"wheel_velocity_target="
                    f"{action_term.wheel_velocity_target[0].tolist()}, "
                    f"gainprm={actuator_gain[0].tolist()}, "
                    f"biasprm={actuator_bias[0].tolist()}, "
                    f"wheel_contact={int(wheel_contact.any().item())}, "
                    f"terms={terms}"
                )
            minimum_height = min(
                minimum_height,
                float(robot.data.root_link_pos_w[:, 2].min().item()),
            )
    finally:
        env.cfg.auto_reset = previous_auto_reset
    if minimum_height < minimum_base_height:
        raise RuntimeError(
            "Default standing pose collapsed: "
            f"minimum base height {minimum_height:.4f} m < "
            f"{minimum_base_height:.4f} m"
        )
    return {
        "steps": steps,
        "minimum_base_height_m": minimum_height,
    }


def _course_terrain_levels(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    terrain = env.scene.terrain
    if terrain is None:
        return {}
    reward_term: XingtianReward = env.reward_manager.cfg["course"].func
    metrics = reward_term.metrics()
    minimum_progress = float(reward_term.stage["gate"].get("min_forward_progress_m", 0.0))
    progress = metrics["forward_progress_m"][env_ids]
    failed = metrics["fell"][env_ids].bool()
    move_up = progress >= max(minimum_progress, 1.0)
    move_down = failed | (progress < 0.5 * max(minimum_progress, 1.0))
    if env.common_step_counter == 0:
        move_up[:] = False
        move_down[:] = False
    move_down &= ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return {
        "mean": terrain.terrain_levels.float().mean(),
        "max": terrain.terrain_levels.max(),
    }


def _randomize_wheel_gains(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    scale_range: tuple[float, float],
) -> None:
    ids = (
        torch.arange(env.num_envs, device=env.device)
        if env_ids is None
        else env_ids
    )
    robot: Entity = env.scene["robot"]
    for actuator in robot.actuators:
        if isinstance(actuator, WheelPiActuator):
            scale = torch.empty(
                (len(ids), len(actuator.target_names)), device=env.device
            ).uniform_(*scale_range)
            actuator.set_gain_scale(ids, scale)


@requires_model_fields(
    "actuator_gainprm", "actuator_biasprm"
)
def _randomize_leg_gains(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    scale_range: tuple[float, float],
) -> None:
    ids = (
        torch.arange(env.num_envs, device=env.device)
        if env_ids is None
        else env_ids
    )
    robot: Entity = env.scene["robot"]
    gain_default = env.sim.get_default_field("actuator_gainprm")
    bias_default = env.sim.get_default_field("actuator_biasprm")
    for actuator in robot.actuators:
        if not isinstance(actuator, BuiltinPositionActuator):
            continue
        ctrl_ids = actuator.global_ctrl_ids
        shape = (len(ids), len(ctrl_ids))
        kp_scale = torch.empty(shape, device=env.device).uniform_(*scale_range)
        kd_scale = torch.empty(shape, device=env.device).uniform_(*scale_range)
        env.sim.model.actuator_gainprm[ids[:, None], ctrl_ids, 0] = (
            gain_default[ctrl_ids, 0] * kp_scale
        )
        env.sim.model.actuator_biasprm[ids[:, None], ctrl_ids, 1] = (
            bias_default[ctrl_ids, 1] * kp_scale
        )
        env.sim.model.actuator_biasprm[ids[:, None], ctrl_ids, 2] = (
            bias_default[ctrl_ids, 2] * kd_scale
        )


@requires_model_fields(
    "body_mass", "body_inertia", recompute=RecomputeLevel.set_const
)
def _randomize_mass_and_inertia(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    scale_range: tuple[float, float],
) -> None:
    ids = (
        torch.arange(env.num_envs, device=env.device)
        if env_ids is None
        else env_ids
    )
    robot: Entity = env.scene["robot"]
    body_ids = robot.indexing.body_ids
    scale = torch.empty((len(ids), 1), device=env.device).uniform_(*scale_range)
    mass_default = env.sim.get_default_field("body_mass")[body_ids]
    inertia_default = env.sim.get_default_field("body_inertia")[body_ids]
    env.sim.model.body_mass[ids[:, None], body_ids] = mass_default * scale
    env.sim.model.body_inertia[ids[:, None], body_ids] = (
        inertia_default * scale.unsqueeze(-1)
    )


def _make_robot_cfg(name: str) -> EntityCfg:
    control = COURSE.stage(name).get("control", {})
    actuators = (
        BuiltinPositionActuatorCfg(
            target_names_expr=(".*_hip_motor",),
            stiffness=HIP_POSITION_GAIN,
            damping=HIP_DAMPING_GAIN,
            effort_limit=STALL_TORQUE,
            armature=0.01,
            frictionloss=0.05,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=(".*_knee_motor",),
            stiffness=KNEE_POSITION_GAIN,
            damping=KNEE_DAMPING_GAIN,
            effort_limit=STALL_TORQUE,
            armature=0.01,
            frictionloss=0.05,
        ),
        WheelPiActuatorCfg(
            target_names_expr=(".*_wheel_motor",),
            kp=float(control.get("wheel_speed_kp", 0.3)),
            ki=float(control.get("wheel_speed_ki", 4.0)),
            integral_limit_nm=float(
                control.get("wheel_integral_limit_nm", 2.0)
            ),
            feedforward_nm_per_rad_s=float(
                control.get("wheel_feedforward_nm_per_rad_s", 0.05)
            ),
            target_slew_rad_s2=float(
                control.get("wheel_target_slew_rad_s2", 20.0)
            ),
            armature=0.01,
            effort_limit=STALL_TORQUE,
        ),
    )
    collision = CollisionCfg(
        geom_names_expr=(r"^.*_collision_\d+$",),
        contype=1,
        conaffinity=2,
        condim=3,
        priority=0,
        friction=(0.8, 0.005, 0.0001),
        solimp=(0.95, 0.99, 0.001),
    )
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, TARGET_BASE_HEIGHT_M),
            joint_pos=_initial_joint_positions(),
            joint_vel={".*": 0.0},
        ),
        spec_fn=lambda: make_robot_spec(
            include_floor=False,
            include_actuators=False,
            configure_options=False,
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=actuators, soft_joint_pos_limit_factor=0.95
        ),
        collisions=(collision,),
        sort_actuators=True,
    )


def _terrain_cfg(name: str, play: bool = False) -> TerrainEntityCfg:
    terrain = COURSE.stage(name)["terrain"]
    if terrain["kind"] == "flat":
        return TerrainEntityCfg(terrain_type="plane", env_spacing=3.0)
    route_length = 60.0 if name == "composite" else None
    generator = TerrainGeneratorCfg(
        seed=7,
        curriculum=not play,
        size=(route_length or float(terrain["length_m"]), float(terrain["width_m"])),
        border_width=1.0,
        border_height=0.4,
        num_rows=1 if play else 6,
        num_cols=1,
        color_scheme="height",
        difficulty_range=(1.0, 1.0) if play else (0.0, 1.0),
        sub_terrains={
            name: XingtianCourseTerrainCfg(
                proportion=1.0, terrain=terrain, route_length_m=route_length
            )
        },
        add_lights=True,
    )
    terrain_collision = CollisionCfg(
        geom_names_expr=(".*",),
        contype=2,
        conaffinity=1,
        condim=3,
        priority=1,
        friction=(0.8, 0.005, 0.0001),
    )
    return TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=generator,
        max_init_terrain_level=0 if play else 2,
        collisions=(terrain_collision,),
    )


def _make_env_cfg(name: str, play: bool = False) -> ManagerBasedRlEnvCfg:
    stage = COURSE.stage(name)
    terrain = stage["terrain"]
    observation_noise = float(stage.get("observation_noise", 0.02))
    if observation_noise < 0.0 or not math.isfinite(observation_noise):
        raise ValueError("observation_noise must be finite and nonnegative")
    scan = RayCastSensorCfg(
        name="terrain_scan",
        frame=ObjRef(type="body", name="diablo_base_link", entity="robot"),
        ray_alignment="yaw",
        pattern=GridPatternCfg(size=(1.5, 1.0), resolution=0.25),
        max_distance=5.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
    )
    wheel_contact = ContactSensorCfg(
        name="wheel_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_wheel|right_wheel)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=2,
        track_air_time=False,
    )
    body_contact = ContactSensorCfg(
        name="body_ground_contact",
        primary=ContactMatch(
            mode="body",
            pattern=r"^diablo_base_link$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    leg_contact = ContactSensorCfg(
        name="leg_ground_contact",
        primary=ContactMatch(
            mode="body",
            pattern=(
                r"^(left_fake_thigh_link|left_thigh_link|left_shank_link|"
                r"left_parallel_link|right_fake_thigh_link|right_thigh_link|"
                r"right_shank_link|right_parallel_link)$"
            ),
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    observations = {
        "actor": ObservationGroupCfg(
            terms={
                "proprio": ObservationTermCfg(
                    func=_proprio,
                    noise=Unoise(
                        n_min=-observation_noise,
                        n_max=observation_noise,
                    ),
                    history_length=PROPRIO_HISTORY_LENGTH,
                    flatten_history_dim=True,
                ),
            },
            concatenate_terms=True,
            enable_corruption=True,
        ),
        "critic": ObservationGroupCfg(
            terms={
                "proprio": ObservationTermCfg(
                    func=_proprio,
                    history_length=PROPRIO_HISTORY_LENGTH,
                    flatten_history_dim=True,
                ),
                "height_scan": ObservationTermCfg(func=_height_scan),
                "privileged": ObservationTermCfg(func=_privileged),
            },
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }
    actions = {
        "joints": XingtianActionCfg(
            entity_name="robot",
            wheel_speed_scale=float(
                stage.get("control", {}).get(
                    "wheel_speed_target_max_rad_s", 22.0
                )
            ),
        ),
    }
    command_time = max(float(p["points"][-1][0]) for p in stage["profiles"])
    commands = {
        "course": CourseCommandCfg(
            stage_name=name,
            resampling_time_range=(1e9, 1e9),
            profile_index=0 if play else None,
        )
    }
    events = {
        "reset_scene": EventTermCfg(func=envs_mdp.reset_scene_to_default, mode="reset"),
        "reset_joints": EventTermCfg(
            func=envs_mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.02, 0.02),
                "velocity_range": (-0.02, 0.02),
                "asset_cfg": SceneEntityCfg("robot", joint_names=JOINTS),
            },
        ),
        "reset_passive_joints": EventTermCfg(
            func=_reset_passive_joints,
            mode="reset",
        ),
    }
    if not play:
        events.update(
            {
                "friction": EventTermCfg(
                    func=dr.geom_friction,
                    mode="startup",
                    params={
                        "asset_cfg": SceneEntityCfg("robot"),
                        "operation": "abs",
                        "ranges": (
                            COURSE.randomization["friction_min"],
                            COURSE.randomization["friction_max"],
                        ),
                        "shared_random": True,
                    },
                ),
                "mass": EventTermCfg(
                    func=_randomize_mass_and_inertia,
                    mode="startup",
                    params={
                        "scale_range": (
                            COURSE.randomization["mass_scale_min"],
                            COURSE.randomization["mass_scale_max"],
                        ),
                    },
                ),
                "leg_gains": EventTermCfg(
                    func=_randomize_leg_gains,
                    mode="startup",
                    params={
                        "scale_range": (
                            COURSE.randomization["actuator_scale_min"],
                            COURSE.randomization["actuator_scale_max"],
                        ),
                    },
                ),
                "wheel_gains": EventTermCfg(
                    func=_randomize_wheel_gains,
                    mode="startup",
                    params={
                        "scale_range": (
                            COURSE.randomization["actuator_scale_min"],
                            COURSE.randomization["actuator_scale_max"],
                        )
                    },
                ),
            }
        )
    if not play and name in {"rough", "roller_obstacle", "composite"}:
        events["push"] = EventTermCfg(
            func=envs_mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(3.0, 7.0),
            params={
                "velocity_range": {
                    "x": (-0.25, 0.25),
                    "y": (-0.2, 0.2),
                    "yaw": (-0.3, 0.3),
                },
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
    rewards = {
        "course": RewardTermCfg(
            func=XingtianReward,
            weight=1.0,
            params={"stage_name": name},
        )
    }
    termination_tilt_deg = float(stage.get("termination_tilt_deg", 60.0))
    if not 0.0 < termination_tilt_deg <= 180.0:
        raise ValueError("termination_tilt_deg must be in (0, 180]")
    terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "body_ground": TerminationTermCfg(
            func=_terminate_body_contact,
            params={"sensor_name": body_contact.name},
        ),
        "leg_ground": TerminationTermCfg(
            func=_terminate_body_contact,
            params={"sensor_name": leg_contact.name},
        ),
        "bad_orientation": TerminationTermCfg(
            func=velocity_mdp.bad_orientation,
            params={"limit_angle": math.radians(termination_tilt_deg)},
        ),
    }
    curriculum = {}
    if terrain["kind"] != "flat" and not play:
        curriculum["terrain_levels"] = CurriculumTermCfg(
            func=_course_terrain_levels,
            params={},
        )
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=_terrain_cfg(name, play=play),
            entities={"robot": _make_robot_cfg(name)},
            sensors=(scan, wheel_contact, body_contact, leg_contact),
            num_envs=1 if play else 4096,
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum=curriculum,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="diablo_base_link",
            distance=2.0,
            elevation=-15.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            nconmax=100,
            njmax=1000,
            contact_sensor_maxmatch=512,
            mujoco=MujocoCfg(
                timestep=PHYSICS_DT,
                integrator="implicitfast",
                iterations=40,
                ls_iterations=30,
                ccd_iterations=50,
            ),
        ),
        decimation=round(CONTROL_DT / PHYSICS_DT),
        episode_length_s=command_time,
        is_finite_horizon=True,
        auto_reset=True,
    )


def _ppo_cfg(name: str, num_envs: int = 4096) -> RslRlOnPolicyRunnerCfg:
    stage = COURSE.stage(name)
    ppo = dict(COURSE.stage("flat_balance").get("ppo", {}))
    ppo.update(stage.get("ppo", {}))
    iterations = budget_iterations(int(stage["budget"]), num_envs, 24)
    return RslRlOnPolicyRunnerCfg(
        seed=42,
        num_steps_per_env=24,
        max_iterations=iterations,
        save_interval=checkpoint_interval(iterations),
        experiment_name="diablo_xingtian_course",
        run_name=name,
        logger="tensorboard",
        upload_model=False,
        clip_actions=1.0,
        actor=RslRlModelCfg(
            hidden_dims=(256, 256),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(256, 256),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            num_learning_epochs=int(ppo.get("n_epochs", 5)),
            num_mini_batches=4,
            learning_rate=float(ppo.get("learning_rate", 2e-4)),
            schedule="adaptive",
            gamma=float(ppo.get("gamma", 0.99)),
            lam=float(ppo.get("gae_lambda", 0.95)),
            entropy_coef=float(ppo.get("ent_coef", 0.005)),
            desired_kl=float(ppo.get("target_kl", 0.02)),
            max_grad_norm=float(ppo.get("max_grad_norm", 0.5)),
            clip_param=float(ppo.get("clip_range", 0.2)),
            value_loss_coef=float(ppo.get("vf_coef", 0.5)),
        ),
    )


def get_task_cfgs(name: str, play: bool = False) -> tuple[ManagerBasedRlEnvCfg, RslRlOnPolicyRunnerCfg]:
    return _make_env_cfg(name, play=play), _ppo_cfg(name)


def register() -> None:
    for name in COURSE.order:
        register_mjlab_task(
            task_id=(
                {
                    "flat_balance": "Mjlab-Xingtian-Flat-Balance",
                    "flat_motion": "Mjlab-Xingtian-Flat-Motion",
                    "flat_turning": "Mjlab-Xingtian-Flat-Turning",
                    "flat_start_stop": "Mjlab-Xingtian-Flat-StartStop",
                    "slope": "Mjlab-Xingtian-Slope",
                    "rough": "Mjlab-Xingtian-Rough",
                    "roller_obstacle": "Mjlab-Xingtian-Obstacles",
                    "composite": "Mjlab-Xingtian-Composite",
                }.get(
                    name,
                    f"Mjlab-Xingtian-Speed-{name.removeprefix('flat_motion_')}",
                )
            ),
            env_cfg=_make_env_cfg(name),
            play_env_cfg=_make_env_cfg(name, play=True),
            rl_cfg=_ppo_cfg(name),
            runner_cls=MjlabOnPolicyRunner,
        )


register()


def require_cuda(device: str = "cuda:0") -> str:
    if not device.startswith("cuda"):
        raise RuntimeError("MJLab training requires an NVIDIA CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; MJLab will not fall back to CPU")
    return device


def _eval_env_cfg(
    stage_name: str, seed: int, profile_index: int
) -> tuple[ManagerBasedRlEnvCfg, RslRlOnPolicyRunnerCfg]:
    env_cfg, agent_cfg = get_task_cfgs(stage_name, play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = seed
    env_cfg.auto_reset = False
    env_cfg.commands["course"].profile_index = profile_index
    env_cfg.observations["actor"].enable_corruption = False
    if env_cfg.scene.terrain is not None and env_cfg.scene.terrain.terrain_generator:
        generator = env_cfg.scene.terrain.terrain_generator
        generator.seed = seed
        generator.curriculum = False
        generator.num_rows = 1
        generator.num_cols = 1
        generator.difficulty_range = (1.0, 1.0)
        env_cfg.scene.terrain.max_init_terrain_level = 0
    return env_cfg, agent_cfg


def _load_runner(
    stage_name: str,
    seed: int,
    profile_index: int,
    checkpoint: Path,
    device: str,
):
    contract_path = checkpoint.parent / "contract.json"
    load_run_contract(contract_path, stage_name)
    env_cfg, agent_cfg = _eval_env_cfg(stage_name, seed, profile_index)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), device=device)
    runner.load(
        str(checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )
    return env, wrapped, runner


def _run_episode(
    stage_name: str,
    profile_index: int,
    seed: int,
    checkpoint: Path,
    device: str,
) -> dict[str, Any]:
    env, wrapped, runner = _load_runner(
        stage_name, seed, profile_index, checkpoint, device
    )
    try:
        wrapped.seed(seed)
        obs, _ = wrapped.reset()
        validate_policy_io(obs)
        verify_reset_state(env)
        policy = runner.get_inference_policy(device=device)
        episode_return = 0.0
        for _ in range(env.max_episode_length):
            with torch.inference_mode():
                action = policy(obs)
                validate_policy_io(obs, action)
                obs, reward, done, extras = wrapped.step(action)
            if not torch.isfinite(reward).all():
                raise FloatingPointError("Evaluation produced a non-finite reward")
            episode_return += float(reward[0].item())
            if bool(done[0].item()):
                break
        reward_term: XingtianReward = env.reward_manager.cfg["course"].func
        metrics = {
            name: value[0].item()
            for name, value in reward_term.metrics().items()
        }
        nonfinite = [
            name
            for name, value in metrics.items()
            if isinstance(value, (int, float)) and not math.isfinite(float(value))
        ]
        if nonfinite:
            raise FloatingPointError(
                f"Evaluation metrics are non-finite: {nonfinite}"
            )
        metrics["episode_return"] = episode_return
        metrics["terminated"] = bool(env.reset_terminated[0].item())
        metrics["truncated"] = bool(env.reset_time_outs[0].item())
        profile = COURSE.stage(stage_name)["profiles"][profile_index]
        return {
            "seed": seed,
            "profile": profile["name"],
            "metrics": metrics,
            "failures": gate_failures(metrics, COURSE.stage(stage_name)["gate"]),
        }
    finally:
        env.close()


def evaluate_checkpoint(
    stage_name: str,
    checkpoint: Path,
    seeds: tuple[int, ...] | list[int],
    device: str = "cuda:0",
) -> dict[str, Any]:
    device = require_cuda(device)
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    stage = COURSE.stage(stage_name)
    runs = []
    for profile_index in range(len(stage["profiles"])):
        for seed in seeds:
            runs.append(
                _run_episode(
                    stage_name, profile_index, int(seed), checkpoint, device
                )
            )
    passed = fully_passes((run["metrics"] for run in runs), stage["gate"])
    result: dict[str, Any] = {
        "stage": stage_name,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "iteration": _checkpoint_iteration(checkpoint),
        "seeds": list(map(int, seeds)),
        "runs": runs,
        "passed": passed,
    }
    if stage_name == "composite":
        scored = [
            gate_score(run["metrics"], stage["gate"]) for run in runs
        ]
        by_seed: dict[int, list[float]] = {}
        for run, score in zip(runs, scored):
            by_seed.setdefault(int(run["seed"]), []).append(score["score"])
        result.update(
            {
                "score": float(np.mean([score["score"] for score in scored])),
                "worst_seed_score": min(
                    float(np.mean(values)) for values in by_seed.values()
                ),
                "minimum_margin": min(
                    score["minimum_margin"] for score in scored
                ),
                "components": {
                    name: float(np.mean([score["components"][name] for score in scored]))
                    for name in scored[0]["components"]
                },
                "mean_speed_mae_mps": float(
                    np.mean([run["metrics"]["speed_mae_mps"] for run in runs])
                ),
                "mean_yaw_mae_rad_s": float(
                    np.mean([run["metrics"]["yaw_mae_rad_s"] for run in runs])
                ),
            }
        )
    return result


def _checkpoint_iteration(path: Path) -> int:
    try:
        return int(path.stem.removeprefix("model_"))
    except ValueError:
        return 0


def candidate_checkpoints(run_dir: Path) -> list[Path]:
    unique: dict[str, Path] = {}
    for path in sorted(run_dir.glob("model_*.pt"), key=_checkpoint_iteration):
        unique.setdefault(sha256(path), path)
    if not unique:
        raise FileNotFoundError(f"No model_*.pt checkpoints found in {run_dir}")
    return list(unique.values())


def _attach_onnx_metadata(path: Path, metadata: dict[str, Any]) -> None:
    import onnx

    model = onnx.load(str(path))
    for tensor in (*model.graph.input, *model.graph.output):
        batch = tensor.type.tensor_type.shape.dim[0]
        batch.ClearField("dim_value")
        batch.dim_param = "batch"
    for key, value in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = str(key)
        entry.value = (
            ",".join(map(str, value)) if isinstance(value, (list, tuple)) else str(value)
        )
        model.metadata_props.append(entry)
    onnx.save(model, str(path))


def export_checkpoint(
    stage_name: str,
    checkpoint: Path,
    output_dir: Path,
    filename: str = "policy.onnx",
    device: str = "cuda:0",
) -> Path:
    device = require_cuda(device)
    env, wrapped, runner = _load_runner(
        stage_name, 42, 0, checkpoint.resolve(), device
    )
    del wrapped
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        sample_tensor = env.observation_manager.compute()["actor"][:1]
        validate_policy_io(sample_tensor)
        sample = sample_tensor.cpu().numpy()
        runner.export_policy_to_onnx(str(output_dir), filename=filename)
        onnx_path = output_dir / filename
        import onnxruntime as ort

        session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        model_input = session.get_inputs()[0]
        model_output = session.get_outputs()[0]
        input_name = model_input.name
        output = session.run(None, {input_name: sample})[0]
        if (
            model_input.type != "tensor(float)"
            or model_input.shape[-1] != OBSERVATION_SHAPE[0]
            or sample.dtype != np.float32
        ):
            raise RuntimeError(
                "Exported ONNX input must be float32 with "
                f"{OBSERVATION_SHAPE[0]} features, got {sample.dtype} "
                f"{model_input.type}{model_input.shape}"
            )
        if (
            model_output.type != "tensor(float)"
            or output.shape != (sample.shape[0], ACTION_SHAPE[0])
            or output.dtype != np.float32
            or not np.isfinite(output).all()
        ):
            raise RuntimeError(
                "Exported ONNX output must be finite float32"
                f"[N,{ACTION_SHAPE[0]}], got {output.dtype}{output.shape}"
            )
        contract = interface_contract(COURSE)
        _attach_onnx_metadata(
            onnx_path,
            {
                "stage": stage_name,
                "checkpoint": checkpoint.name,
                "checkpoint_sha256": sha256(checkpoint),
                "interface_sha256": contract["sha256"],
                "observation_shape": f"{sample.shape[0]},{OBSERVATION_SHAPE[0]}",
                "observation_dtype": OBSERVATION_DTYPE,
                "action_shape": f"{output.shape[0]},{ACTION_SHAPE[0]}",
                "action_dtype": ACTION_DTYPE,
                "control_dt_s": CONTROL_DT,
                "simulation_only": True,
                "hardware_authorized": False,
            },
        )
        return onnx_path
    finally:
        env.close()


def _selection_metrics(result: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    scored = [gate_score(run["metrics"], gate) for run in result["runs"]]
    by_seed: dict[int, list[float]] = {}
    for run, score in zip(result["runs"], scored, strict=True):
        by_seed.setdefault(int(run["seed"]), []).append(float(score["score"]))
    result = dict(result)
    result.update(
        {
            "score": float(np.mean([score["score"] for score in scored])),
            "pass_rate": float(np.mean([score["passed"] for score in scored])),
            "worst_seed_score": min(float(np.mean(values)) for values in by_seed.values()),
            "minimum_margin": min(score["minimum_margin"] for score in scored),
            "mean_speed_mae_mps": float(
                np.mean([run["metrics"]["speed_mae_mps"] for run in result["runs"]])
            ),
            "mean_yaw_mae_rad_s": float(
                np.mean([run["metrics"]["yaw_mae_rad_s"] for run in result["runs"]])
            ),
        }
    )
    return result


def select_policy_best(
    run_dir: Path,
    device: str = "cuda:0",
    stage_name: str = "composite",
) -> dict[str, Any]:
    device = require_cuda(device)
    run_dir = run_dir.resolve()
    contract_path = run_dir / "contract.json"
    load_run_contract(contract_path, stage_name)
    stage = COURSE.stage(stage_name)
    gate = stage["gate"]
    output_dir = run_dir / "bestmodel"
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in (
        output_dir / "bestmodel.pt",
        output_dir / "bestmodel.onnx",
        output_dir / "contract.json",
        output_dir / "selection.json",
        run_dir / "selection.json",
        run_dir / "policy-best.pt",
        run_dir / "policy-best.onnx",
    ):
        path.unlink(missing_ok=True)

    candidates = candidate_checkpoints(run_dir)
    probes = []
    verifications = []
    for checkpoint in candidates:
        probe = _selection_metrics(
            evaluate_checkpoint(
                stage_name,
                checkpoint,
                [SELECTION_PROBE_SEED],
                device,
            ),
            gate,
        )
        probes.append(probe)
        print(
            f"[probe] {checkpoint.name} seed={SELECTION_PROBE_SEED}: "
            f"{'PASS' if probe['passed'] else 'FAIL'}"
        )
        if not probe["passed"]:
            continue
        candidate = _selection_metrics(
            evaluate_checkpoint(
                stage_name,
                checkpoint,
                SELECTION_VERIFY_SEEDS,
                device,
            ),
            gate,
        )
        candidate["probe_runs"] = probe["runs"]
        verifications.append(candidate)
        print(
            f"[verify] {checkpoint.name}: "
            f"{'PASS' if candidate['passed'] else 'FAIL'} "
            f"score={candidate.get('score', 0.0):.3f}"
        )

    qualified = [candidate for candidate in verifications if candidate["passed"]]
    try:
        ranking = rank_policy_candidates(qualified)
    except ValueError as error:
        failure = {
            "stage": stage_name,
            "run_dir": str(run_dir),
            "probe_seed": SELECTION_PROBE_SEED,
            "verification_seeds": list(SELECTION_VERIFY_SEEDS),
            "probe_count": len(probes),
            "verification_count": len(verifications),
            "qualified_count": len(qualified),
            "probes": probes,
            "verifications": verifications,
            "qualified": qualified,
            "error": str(error),
        }
        (output_dir / "selection.json").write_text(
            json.dumps(failure, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        raise

    best = ranking[0]
    selected_checkpoint = Path(best["checkpoint"])
    shutil.copy2(contract_path, output_dir / "contract.json")
    best_copy = output_dir / "bestmodel.pt"
    shutil.copy2(selected_checkpoint, best_copy)
    if sha256(best_copy) != best["checkpoint_sha256"]:
        raise RuntimeError("bestmodel.pt does not match the selected checkpoint")

    onnx_path = export_checkpoint(
        stage_name,
        best_copy,
        output_dir,
        filename="bestmodel.onnx",
        device=device,
    )
    blind = evaluate_checkpoint(
        stage_name, best_copy, BLIND_SEEDS, device
    )
    limit = evaluate_checkpoint(
        stage_name, best_copy, LIMIT_SEEDS, device
    )
    selection = {
        "stage": stage_name,
        "run_dir": str(run_dir),
        "probe_seed": SELECTION_PROBE_SEED,
        "verification_seeds": list(SELECTION_VERIFY_SEEDS),
        "probe_count": len(probes),
        "verification_count": len(verifications),
        "qualified_count": len(qualified),
        "blind_seeds": list(BLIND_SEEDS),
        "limit_seeds": list(LIMIT_SEEDS),
        "score_weights": {
            "completion": 0.25,
            "tracking": 0.25,
            "stability": 0.25,
            "terrain": 0.15,
            "efficiency": 0.10,
        },
        "probes": probes,
        "verifications": verifications,
        "qualified": qualified,
        "ranking": ranking,
        "selected": best,
        "source_checkpoint": str(selected_checkpoint),
        "source_checkpoint_sha256": best["checkpoint_sha256"],
        "bestmodel_pt": str(best_copy),
        "bestmodel_pt_sha256": sha256(best_copy),
        "bestmodel_onnx": str(onnx_path),
        "bestmodel_onnx_sha256": sha256(onnx_path),
        "blind": blind,
        "limit": limit,
    }
    (output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return selection


def latest_run(log_root: Path, stage_name: str) -> Path:
    experiment = log_root / "diablo_xingtian_course"
    runs = [path for path in experiment.glob(f"*_{stage_name}") if path.is_dir()]
    if not runs:
        raise FileNotFoundError(
            f"No run for stage '{stage_name}' under {experiment}"
        )
    return max(runs, key=lambda path: path.stat().st_mtime)


def write_run_contract(
    run_dir: Path,
    stage_name: str,
    num_envs: int,
    iterations: int,
) -> None:
    stage = COURSE.stage(stage_name)
    contract = interface_contract(COURSE)
    payload = {
        "contract": contract,
        "stage": stage_name,
        "budget": int(stage["budget"]),
        "num_envs": num_envs,
        "iterations": iterations,
        "budget_iterations": budget_iterations(
            int(stage["budget"]), num_envs, 24
        ),
    }
    (run_dir / "contract.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
