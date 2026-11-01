"""Explicit motor-space action mapping, separated from policy and reward."""

import numpy as np
import torch

from .model import DEFAULT_Q, NO_LOAD_SPEED, OBSERVATION_SIGNS, STALL_TORQUE, WHEELS


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

    command = command.to(
        device=residual_action.device,
        dtype=residual_action.dtype,
    )
    forward = command[:, 0]
    yaw = command[:, 1]
    left_speed = forward - yaw_feedforward_gain * yaw * half_track
    right_speed = forward + yaw_feedforward_gain * yaw * half_track
    nominal = -feedforward_gain * torch.stack(
        (left_speed, right_speed), dim=-1
    ) / radius
    target = nominal - residual_scale_rad_s * residual_action
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
