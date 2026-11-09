"""Explicit motor-space action mapping, separated from policy and reward."""

import numpy as np
import torch

from .model import DEFAULT_Q, NO_LOAD_SPEED, OBSERVATION_SIGNS, STALL_TORQUE, WHEELS

YAW_COMMAND_HARD_LIMIT_RAD_S = 2.0


def closed_loop_yaw_target(
    command_yaw: torch.Tensor,
    heading_error: torch.Tensor,
    *,
    gain: float = 0.75,
    correction_limit: float = 0.2,
) -> torch.Tensor:
    """Add bounded heading feedback to the open-loop yaw command."""
    correction = torch.clamp(
        -gain * heading_error,
        -correction_limit,
        correction_limit,
    )
    return torch.clamp(
        command_yaw + correction,
        -YAW_COMMAND_HARD_LIMIT_RAD_S,
        YAW_COMMAND_HARD_LIMIT_RAD_S,
    )


def decompose_wheel_residual(
    residual_action: torch.Tensor,
    *,
    common_limit: float | None = None,
    active: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split residual into common/differential parts with optional common cap."""
    if residual_action.ndim != 2 or residual_action.shape[-1] != 2:
        raise ValueError(
            "residual_action must have shape [batch,2], got "
            f"{tuple(residual_action.shape)}"
        )
    if common_limit is not None and not 0.0 <= common_limit <= 1.0:
        raise ValueError("common_limit must be in [0,1] or None")
    if active is not None and active.shape != (residual_action.shape[0], 1):
        raise ValueError(
            "active must have shape [batch,1], got "
            f"{tuple(active.shape)}"
        )
    common = residual_action.mean(dim=-1, keepdim=True)
    differential = residual_action - common
    if common_limit is not None:
        bounded_common = torch.clamp(
            common,
            -common_limit,
            common_limit,
        )
        common = (
            bounded_common
            if active is None
            else torch.where(active, bounded_common, common)
        )
    return common + differential, common, differential


def wheel_command_targets(
    command: torch.Tensor,
    residual_action: torch.Tensor,
    *,
    radius: float,
    half_track: float,
    feedforward_gain: float,
    yaw_feedforward_gain: float,
    residual_scale_rad_s: float,
    speed_limit: float = NO_LOAD_SPEED,
    common_residual_limit: float | None = None,
    common_residual_turn_threshold: float = 0.05,
) -> torch.Tensor:
    """Map forward/yaw commands and two residual actions to wheel velocities."""
    if command.ndim != 2 or command.shape[-1] != 3:
        raise ValueError(
            f"command must have shape [batch,3], got {tuple(command.shape)}"
        )
    if residual_action.shape != (command.shape[0], 2):
        raise ValueError(
            "residual_action must have shape "
            f"[{command.shape[0]},2], got {tuple(residual_action.shape)}"
        )
    if radius <= 0.0 or half_track < 0.0 or residual_scale_rad_s < 0.0:
        raise ValueError("Wheel geometry and residual scale must be valid")
    if speed_limit <= 0.0:
        raise ValueError("speed_limit must be positive")
    if common_residual_turn_threshold < 0.0:
        raise ValueError("common residual turn threshold must be nonnegative")

    command = command.to(
        device=residual_action.device,
        dtype=residual_action.dtype,
    )
    forward = command[:, 0]
    yaw = command[:, 1].clamp(
        -YAW_COMMAND_HARD_LIMIT_RAD_S,
        YAW_COMMAND_HARD_LIMIT_RAD_S,
    )
    left_speed = forward - yaw_feedforward_gain * yaw * half_track
    right_speed = forward + yaw_feedforward_gain * yaw * half_track
    nominal = -feedforward_gain * torch.stack(
        (left_speed, right_speed), dim=-1
    ) / radius
    effective_residual, _, _ = decompose_wheel_residual(
        residual_action,
        common_limit=common_residual_limit,
        active=(
            command[:, 1].abs() > common_residual_turn_threshold
        ).unsqueeze(-1),
    )
    target = nominal - residual_scale_rad_s * effective_residual
    return torch.clamp(target, -speed_limit, speed_limit)


def radial_feedforward(motor_q, force=60.0):
    result = np.zeros(6)
    result[[1, 4]] = force * 0.14 * np.sin((-2.8729 - motor_q[[1, 4]]) / 2)
    return result


def decode_action(action, motor_q, motor_dq, mode="position_velocity"):
    if mode == "torque":
        requested = action * STALL_TORQUE * OBSERVATION_SIGNS
    elif mode == "position_velocity":
        requested = np.array([15, 18, 0, 15, 18, 0]) * (
            DEFAULT_Q + action * 0.5 - motor_q
        )
        requested -= np.array([0.3, 0.2, 0.3, 0.3, 0.2, 0.3]) * motor_dq
        requested[WHEELS] -= 0.3 * action[WHEELS] * 10
        requested += radial_feedforward(motor_q)
    else:
        raise ValueError("action_mode must be position_velocity or torque")
    lower = -STALL_TORQUE * np.clip(1 + motor_dq / NO_LOAD_SPEED, 0, 1)
    upper = STALL_TORQUE * np.clip(1 - motor_dq / NO_LOAD_SPEED, 0, 1)
    applied = np.clip(requested, lower, upper)
    return requested, applied
