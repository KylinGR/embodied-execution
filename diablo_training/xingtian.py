"""Reference task components for the XINGTIAN locomotion curriculum.

These utilities are intentionally independent of a particular physics backend.
They can feed the existing MuJoCo environment or a GPU-native Isaac/MJX port.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CommandSample:
    """Body-frame command: forward m/s, yaw rad/s, base height m."""

    forward: float
    yaw_rate: float
    height: float
    duration: float


@dataclass(frozen=True)
class CommandConfig:
    max_forward_speed: float = 1.0
    max_backward_speed: float = 0.8
    max_yaw_rate: float = 1.5
    min_height: float = 0.16
    max_height: float = 0.34
    duration_min: float = 2.0
    duration_max: float = 6.0
    ramp_seconds: float = 0.35
    zero_probability: float = 0.20
    forward_probability: float = 0.25
    backward_probability: float = 0.20
    turn_probability: float = 0.25

    def validate(self) -> None:
        values = (
            self.max_forward_speed,
            self.max_backward_speed,
            self.max_yaw_rate,
            self.min_height,
            self.max_height,
            self.duration_min,
            self.duration_max,
            self.ramp_seconds,
        )
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("Command limits must be finite and positive")
        if self.min_height >= self.max_height:
            raise ValueError("min_height must be lower than max_height")
        if self.duration_min > self.duration_max:
            raise ValueError("duration_min must not exceed duration_max")
        probabilities = (
            self.zero_probability,
            self.forward_probability,
            self.backward_probability,
            self.turn_probability,
        )
        if min(probabilities) < 0 or sum(probabilities) > 1:
            raise ValueError("Command probabilities must be non-negative and sum to <= 1")


class CommandScheduler:
    """Sample flat-ground commands with smooth transitions and explicit stops."""

    def __init__(self, config: CommandConfig, rng: np.random.Generator):
        config.validate()
        self.config = config
        self.rng = rng
        self.started_at = 0.0
        self.target = self._sample()
        self.previous = np.array([0.0, 0.0, self._nominal_height()])
        self._next_at = self.target.duration

    def _nominal_height(self) -> float:
        return 0.5 * (self.config.min_height + self.config.max_height)

    def _height(self) -> float:
        return float(
            self.rng.uniform(self.config.min_height, self.config.max_height)
        )

    def _sample(self) -> CommandSample:
        draw = float(self.rng.random())
        if draw < self.config.zero_probability:
            forward, yaw_rate = 0.0, 0.0
        elif draw < self.config.zero_probability + self.config.forward_probability:
            forward = float(
                self.rng.uniform(0.05, self.config.max_forward_speed)
            )
            yaw_rate = float(
                self.rng.uniform(-0.2, 0.2) * self.config.max_yaw_rate
            )
        elif (
            draw
            < self.config.zero_probability
            + self.config.forward_probability
            + self.config.backward_probability
        ):
            forward = float(
                self.rng.uniform(-self.config.max_backward_speed, -0.05)
            )
            yaw_rate = float(
                self.rng.uniform(-0.2, 0.2) * self.config.max_yaw_rate
            )
        elif (
            draw
            < self.config.zero_probability
            + self.config.forward_probability
            + self.config.backward_probability
            + self.config.turn_probability
        ):
            forward = float(
                self.rng.uniform(-0.3, 0.3) * self.config.max_forward_speed
            )
            sign = -1.0 if self.rng.random() < 0.5 else 1.0
            yaw_rate = float(
                sign * self.rng.uniform(0.25, 1.0) * self.config.max_yaw_rate
            )
        else:
            # Mixed command: switch direction and yaw together.
            forward = float(
                self.rng.uniform(
                    -self.config.max_backward_speed,
                    self.config.max_forward_speed,
                )
            )
            yaw_rate = float(
                self.rng.uniform(-self.config.max_yaw_rate, self.config.max_yaw_rate)
            )
        duration = float(
            self.rng.uniform(self.config.duration_min, self.config.duration_max)
        )
        return CommandSample(forward, yaw_rate, self._height(), duration)

    def command_at(self, sim_seconds: float) -> np.ndarray:
        if sim_seconds < 0 or not np.isfinite(sim_seconds):
            raise ValueError("sim_seconds must be finite and non-negative")
        while sim_seconds >= self._next_at:
            self.started_at = self._next_at
            self.previous = np.array(
                [self.target.forward, self.target.yaw_rate, self.target.height]
            )
            self.target = self._sample()
            self._next_at = self.started_at + self.target.duration

        target = np.array(
            [self.target.forward, self.target.yaw_rate, self.target.height]
        )
        elapsed = sim_seconds - self.started_at
        alpha = min(1.0, elapsed / self.config.ramp_seconds)
        return (1.0 - alpha) * self.previous + alpha * target


@dataclass(frozen=True)
class TerrainLevel:
    name: str
    rms_height: float
    peak_height: float
    feature_scale: float
    max_slope_deg: float
    obstacle_height: float
    friction_min: float
    friction_max: float


TERRAIN_LEVELS = (
    TerrainLevel("flat", 0.0, 0.0, 1.00, 0.0, 0.0, 0.65, 1.0),
    TerrainLevel("rough-l1", 0.010, 0.025, 0.80, 10.0, 0.015, 0.55, 1.0),
    TerrainLevel("rough-l2", 0.020, 0.050, 0.50, 18.0, 0.025, 0.50, 1.0),
    TerrainLevel("rough-l3", 0.035, 0.075, 0.30, 28.0, 0.040, 0.45, 1.0),
    TerrainLevel("rough-l4", 0.055, 0.110, 0.18, 35.0, 0.055, 0.40, 1.0),
)


def _smooth_upsample(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Bilinear interpolation using only NumPy."""
    target_x, target_y = shape
    source_x = np.linspace(0.0, 1.0, values.shape[0])
    source_y = np.linspace(0.0, 1.0, values.shape[1])
    x = np.linspace(0.0, 1.0, target_x)
    y = np.linspace(0.0, 1.0, target_y)
    horizontal = np.column_stack(
        [np.interp(x, source_x, values[:, column]) for column in range(values.shape[1])]
    )
    return np.vstack(
        [np.interp(y, source_y, horizontal[row, :]) for row in range(horizontal.shape[0])]
    )


def fractal_heightfield(
    level: TerrainLevel,
    *,
    seed: int,
    length_m: float = 20.0,
    width_m: float = 8.0,
    horizontal_resolution_m: float = 0.05,
) -> np.ndarray:
    """Generate a deterministic, bounded rough heightfield in metres."""
    nx = int(round(length_m / horizontal_resolution_m)) + 1
    ny = int(round(width_m / horizontal_resolution_m)) + 1
    rng = np.random.default_rng(seed)
    if level.rms_height == 0:
        return np.zeros((nx, ny), dtype=np.float32)

    field = np.zeros((nx, ny), dtype=np.float64)
    octaves = ((0.80, 0.45), (0.35, 0.30), (0.15, 0.18), (0.06, 0.07))
    for cells, amplitude in octaves:
        coarse_nx = max(3, int(length_m / max(cells * level.feature_scale, 0.05)))
        coarse_ny = max(3, int(width_m / max(cells * level.feature_scale, 0.05)))
        coarse = rng.normal(0.0, 1.0, (coarse_nx, coarse_ny))
        field += amplitude * _smooth_upsample(coarse, (nx, ny))

    field -= field.mean()
    measured_rms = float(np.sqrt(np.mean(field**2)))
    if measured_rms < 1e-9:
        return np.zeros((nx, ny), dtype=np.float32)
    field *= level.rms_height / measured_rms
    peak = max(float(np.max(np.abs(field))), 1e-9)
    field *= min(1.0, level.peak_height / peak)
    return field.astype(np.float32)


@dataclass(frozen=True)
class ObstacleSpec:
    kind: str
    height_m: float
    width_m: float
    length_m: float
    x_m: float
    y_m: float
    radius_m: float = 0.0


def sample_obstacles(
    level: TerrainLevel,
    *,
    seed: int,
    count: int = 8,
    course_length_m: float = 20.0,
    course_width_m: float = 4.0,
) -> tuple[ObstacleSpec, ...]:
    """Create explicit box/cylinder obstacles; do not encode sharp edges as hfield."""
    if level.obstacle_height <= 0 or count <= 0:
        return ()
    rng = np.random.default_rng(seed + 10_000)
    obstacles = []
    for index in range(count):
        x = (index + 0.5) * course_length_m / count
        height = level.obstacle_height * float(rng.uniform(0.55, 1.0))
        kind = "cylinder" if rng.random() < 0.35 else "box"
        obstacles.append(
            ObstacleSpec(
                kind=kind,
                height_m=height,
                width_m=float(rng.uniform(0.20, 0.50)),
                length_m=float(rng.uniform(0.10, 0.40)),
                x_m=x,
                y_m=float(rng.uniform(-course_width_m / 3, course_width_m / 3)),
                radius_m=float(rng.uniform(0.005, 0.020))
                if kind == "box"
                else height / 2,
            )
        )
    return tuple(obstacles)


@dataclass(frozen=True)
class RewardWeights:
    forward_tracking: float = 1.2
    yaw_tracking: float = 0.8
    upright: float = 0.8
    height: float = 0.5
    stand_still: float = 0.8
    vertical_velocity: float = 1.0
    angular_velocity_xy: float = 0.05
    action_rate: float = 0.02
    action_second_difference: float = 0.01
    joint_acceleration: float = 2e-7
    torque_squared: float = 1e-5
    power: float = 1e-4
    wheel_slip: float = 0.05
    soft_limit: float = 1.0
    termination: float = 20.0


@dataclass
class RewardBreakdown:
    terms: dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return float(sum(self.terms.values()))


class LocomotionReward:
    """Backend-independent reward used by flat and terrain stages."""

    def __init__(self, weights: RewardWeights | None = None):
        self.weights = weights or RewardWeights()

    def evaluate(
        self,
        transition: dict,
        action: np.ndarray,
        previous_action: np.ndarray,
        *,
        dt: float = 0.01,
        previous_difference: np.ndarray | None = None,
    ) -> RewardBreakdown:
        action = np.asarray(action, dtype=float)
        previous_action = np.asarray(previous_action, dtype=float)
        command = np.asarray(transition["command"], dtype=float)
        velocity = np.asarray(transition["velocity"], dtype=float)
        gyro = np.asarray(transition["gyro"], dtype=float)
        torque = np.asarray(transition["torque"], dtype=float)
        upright = float(np.clip(transition["upright"], -1.0, 1.0))
        height_error = float(transition["height_error"])
        vx_error = float(velocity[0] - command[0])
        yaw_error = float(gyro[2] - command[1])

        w = self.weights
        terms = {
            "forward_tracking": w.forward_tracking * np.exp(-(vx_error**2) / 0.04),
            "yaw_tracking": w.yaw_tracking * (np.exp(-(yaw_error**2) / 0.25) if abs(command[1]) > 0.1 else np.exp(-(gyro[2]**2) / 0.01)),
            "upright": w.upright * np.exp(
                -max(0.0, (1.0 - upright) ** 2) / 0.02
            ),
            "height": w.height * np.exp(-(height_error**2) / 0.005),
            "vertical_velocity": -w.vertical_velocity * float(velocity[2] ** 2),
            "angular_velocity_xy": -w.angular_velocity_xy
            * float(np.sum(gyro[:2] ** 2)),
            "action_rate": -w.action_rate * float(np.sum((action - previous_action) ** 2)),
            "torque_squared": -w.torque_squared * float(np.sum(torque**2)),
        }
        acceleration = np.asarray(transition.get("joint_acceleration", np.zeros(4)))
        terms["joint_acceleration"] = -w.joint_acceleration * float(
            np.sum(acceleration**2)
        )
        wheel_velocity = np.asarray(transition.get("wheel_velocity", np.zeros(2)))
        wheel_linear_velocity = np.asarray(
            transition.get("wheel_linear_velocity", np.zeros(2))
        )
        terms["wheel_slip"] = -w.wheel_slip * float(
            np.sum((wheel_velocity - wheel_linear_velocity) ** 2)
        )
        power = np.asarray(transition.get("joint_power", np.zeros_like(torque)))
        terms["power"] = -w.power * float(np.sum(np.abs(power)))
        if previous_difference is not None:
            second = action - 2.0 * previous_action + np.asarray(previous_difference)
            terms["action_second_difference"] = -w.action_second_difference * float(
                np.sum(second**2)
            )
        if np.max(np.abs(command[:2])) < 0.01:
            terms["stand_still"] = w.stand_still * float(
                np.exp(-(np.sum(velocity[:2] ** 2) + gyro[2] ** 2) / 0.01)
            )
        soft_limit = np.asarray(transition.get("soft_limit_excess", np.zeros(4)))
        terms["soft_limit"] = -w.soft_limit * float(np.sum(np.maximum(soft_limit, 0)))
        scaled = {name: dt * value for name, value in terms.items()}
        if transition["terminated"]:
            scaled["termination"] = -w.termination
        if not np.isfinite(list(scaled.values())).all():
            raise FloatingPointError("Reward contains non-finite terms")
        return RewardBreakdown(scaled)


def body_height_ladder(
    leg_min_m: float,
    leg_max_m: float,
    *,
    fractions: Iterable[float] = (0.1, 0.3, 0.5, 0.7, 0.9),
    safety_margin_m: float = 0.005,
) -> tuple[float, ...]:
    if leg_max_m <= leg_min_m:
        raise ValueError("leg_max_m must be greater than leg_min_m")
    low = leg_min_m + safety_margin_m
    high = leg_max_m - safety_margin_m
    values = []
    for fraction in fractions:
        if not 0 <= fraction <= 1:
            raise ValueError("fractions must be in [0, 1]")
        values.append(float(low + fraction * (high - low)))
    return tuple(values)


def wheel_lift_matrix(
    body_heights: Iterable[float],
    lift_heights_mm: Iterable[float] = (10, 20, 30, 40, 50, 60, 70, 80),
    hold_seconds: Iterable[float] = (0.25, 0.5, 1.0, 2.0),
) -> tuple[dict, ...]:
    return tuple(
        {
            "side": side,
            "body_height_m": float(body_height),
            "target_clearance_m": float(lift_mm) / 1000.0,
            "hold_seconds": float(hold),
        }
        for side in ("left", "right")
        for body_height in body_heights
        for lift_mm in lift_heights_mm
        for hold in hold_seconds
    )
