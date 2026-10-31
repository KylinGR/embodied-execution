"""Explicit motor-space action mapping, separated from policy and reward."""

import numpy as np

from .model import DEFAULT_Q, NO_LOAD_SPEED, OBSERVATION_SIGNS, STALL_TORQUE, WHEELS


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
