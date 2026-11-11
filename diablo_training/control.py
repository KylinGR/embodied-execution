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
    elif mode == "wheel_torque":
        # 腿：和解析控制器同一套 PD（kp=60, kd=2）固定到 DEFAULT_Q。
        # 注意不能用 pv 模式的 15/18——那套软 4 倍，托不住机身（实测 132 步就摔）。
        # 这里用显式下标，避免再改 control.py 的 import。
        requested = np.zeros(6)
        for _j in (0, 1, 3, 4):
            requested[_j] = 60.0 * (DEFAULT_Q[_j] - motor_q[_j]) - 2.0 * motor_dq[_j]
        # 轮：动作直接映射成力矩，全权限（±STALL_TORQUE）
        requested[WHEELS] = action[WHEELS] * STALL_TORQUE * OBSERVATION_SIGNS[WHEELS]
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
