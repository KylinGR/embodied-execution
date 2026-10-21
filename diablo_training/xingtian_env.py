"""Reference flat-ground XINGTIAN environment built on the validated model."""

from dataclasses import dataclass

import numpy as np

from .env import Config, DiabloEnv
from .model import LEGS, OBSERVATION_SIGNS
from .xingtian import (
    CommandConfig,
    CommandScheduler,
    LocomotionReward,
    RewardWeights,
)
from .xingtian_terrain import (
    STAGES,
    build_terrain_profile,
    make_terrain_model,
    profile_height,
)


@dataclass
class XingtianConfig(Config):
    max_backward_speed: float = 0.8
    max_yaw_rate: float = 1.5
    min_height: float = 0.16
    max_height: float = 0.34
    command_duration_min: float = 2.0
    command_duration_max: float = 6.0
    command_ramp_seconds: float = 0.35

    def __post_init__(self):
        super().__post_init__()
        values = (
            self.max_backward_speed,
            self.max_yaw_rate,
            self.command_duration_min,
            self.command_duration_max,
            self.command_ramp_seconds,
        )
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("XINGTIAN command limits must be finite and positive")
        if self.min_height >= self.max_height:
            raise ValueError("min_height must be lower than max_height")
        if self.command_duration_min > self.command_duration_max:
            raise ValueError(
                "command_duration_min must not exceed command_duration_max"
            )


@dataclass
class XingtianStageConfig(XingtianConfig):
    stage: str = "B"
    terrain_level: int = 1
    terrain_seed: int = 1
    track_length_m: float = 40.0
    track_width_m: float = 6.0
    obstacle_height_m: float = 0.0
    max_lateral_y_m: float = 2.0

    def __post_init__(self):
        super().__post_init__()
        if self.stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}")
        if self.terrain_level < 0:
            raise ValueError("terrain_level must be non-negative")
        if min(self.track_length_m, self.track_width_m, self.max_lateral_y_m) <= 0:
            raise ValueError("track dimensions and lateral limit must be positive")
        if self.obstacle_height_m < 0:
            raise ValueError("obstacle_height_m must be non-negative")


class XingtianFlatEnv(DiabloEnv):
    """Flat reference task with forward/backward, yaw, height and stop commands."""

    def __init__(
        self,
        config: XingtianConfig | None = None,
        reward_weights: RewardWeights | None = None,
        render_mode=None,
    ):
        self.xingtian_config = config or XingtianConfig()
        self.reward_function = LocomotionReward(reward_weights)
        self.command_scheduler = None
        self._previous_previous_action = np.zeros(6)
        super().__init__(self.xingtian_config, render_mode=render_mode)

    def _new_scheduler(self) -> CommandScheduler:
        command_config = CommandConfig(
            max_forward_speed=self.xingtian_config.max_speed,
            max_backward_speed=self.xingtian_config.max_backward_speed,
            max_yaw_rate=self.xingtian_config.max_yaw_rate,
            min_height=self.xingtian_config.min_height,
            max_height=self.xingtian_config.max_height,
            duration_min=self.xingtian_config.command_duration_min,
            duration_max=self.xingtian_config.command_duration_max,
            ramp_seconds=self.xingtian_config.command_ramp_seconds,
        )
        return CommandScheduler(command_config, self.np_random)

    def reset(self, *, seed=None, options=None):
        self.command_scheduler = None
        self._previous_previous_action[:] = 0
        return super().reset(seed=seed, options=options)

    def get_command(self, sim_seconds):
        if self.command_scheduler is None:
            self.command_scheduler = self._new_scheduler()
        return self.command_scheduler.command_at(sim_seconds)

    def reward_for_transition(self, transition, action, previous_action):
        qvel = self.data.qvel[self.vi]
        torque = transition["torque"]
        transition = dict(transition)
        transition["joint_power"] = torque * qvel
        transition["joint_acceleration"] = np.zeros(4)
        joint_ids = [self.model.joint(name).id for name in (
            "left_hip_motor",
            "left_knee_motor",
            "right_hip_motor",
            "right_knee_motor",
        )]
        joint_range = self.model.jnt_range[joint_ids]
        margin = 0.05 * (joint_range[:, 1] - joint_range[:, 0])
        q = self.data.qpos[self.qi[LEGS]]
        transition["soft_limit_excess"] = np.maximum(
            np.maximum(joint_range[:, 0] + margin - q, q - joint_range[:, 1] + margin),
            0.0,
        )
        wheel_speed = qvel[[2, 5]] * OBSERVATION_SIGNS[[2, 5]]
        transition["wheel_velocity"] = wheel_speed * 0.1
        half_track = 0.18755
        transition["wheel_linear_velocity"] = np.array(
            [
                transition["velocity"][0] - half_track * transition["gyro"][2],
                transition["velocity"][0] + half_track * transition["gyro"][2],
            ]
        )
        breakdown = self.reward_function.evaluate(
            transition,
            action,
            previous_action,
            previous_difference=self._previous_previous_action,
        )
        self._previous_previous_action[:] = previous_action
        return breakdown.total


class XingtianStageEnv(XingtianFlatEnv):
    """B/C/D curriculum environment sharing the flat A observation/action contract."""

    def __init__(
        self,
        config: XingtianStageConfig | None = None,
        reward_weights: RewardWeights | None = None,
        render_mode=None,
    ):
        self.stage_config = config or XingtianStageConfig()
        self.terrain_profile = build_terrain_profile(
            stage=self.stage_config.stage,
            level_index=self.stage_config.terrain_level,
            seed=self.stage_config.terrain_seed,
            track_length_m=self.stage_config.track_length_m,
            track_width_m=self.stage_config.track_width_m,
            obstacle_height_m=self.stage_config.obstacle_height_m or None,
        )
        super().__init__(
            self.stage_config,
            reward_weights=reward_weights,
            render_mode=render_mode,
        )

    def _make_model(self):
        return make_terrain_model(self.terrain_profile)

    def ground_height(self, x, y):
        return profile_height(self.terrain_profile, x, y)

    def _sample_start_xy(self):
        low, high = self.terrain_profile.start_x_range
        x = float(self.np_random.uniform(low, high))
        if self.terrain_profile.obstacles:
            for _ in range(100):
                if all(abs(x - obs["x"]) > 0.8 for obs in self.terrain_profile.obstacles):
                    break
                x = float(self.np_random.uniform(low, high))
        y = float(self.np_random.uniform(-0.25, 0.25))
        return x, y

    def reward_for_transition(self, transition, action, previous_action):
        transition = dict(transition)
        x, y = self.data.xpos[self.base_id, :2]
        transition["height_error"] = float(
            self.data.xpos[self.base_id, 2]
            - self.ground_height(float(x), float(y))
            - transition["command"][2]
        )
        return super().reward_for_transition(transition, action, previous_action)

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        x, y = self.data.xpos[self.base_id, :2]
        ground = self.ground_height(float(x), float(y))
        lateral_failure = abs(float(y)) > self.stage_config.max_lateral_y_m
        if lateral_failure:
            terminated = True
            self._needs_reset = True
            reward -= 20.0
        info.update(
            {
                "stage": self.stage_config.stage,
                "terrain_level": self.stage_config.terrain_level,
                "terrain_seed": self.stage_config.terrain_seed,
                "ground_height_m": ground,
                "relative_base_height_m": float(
                    self.data.xpos[self.base_id, 2] - ground
                ),
                "lateral_failure": lateral_failure,
            }
        )
        return observation, float(reward), terminated, truncated, info
