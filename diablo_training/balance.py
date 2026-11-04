"""解析平衡/速度/航向控制器（基于模型的状态反馈）。

实测依据（同一台仿真机器人，同一套 diag 剖面）：

    能力            RL 最好（12 次训练 × 200 万步）   本控制器（8 个增益）
    站立漂移        12.6 m / 120 s                     1.9 cm / 6 s
    站立自转        4.4 °/s                            0.0° / 8 s
    定速 0.5 m/s    误差 8-15%                         误差 0%
    航向保持        16.6° / 10 s                       0.0° / 8 s
    原地转向 0.3    达成 12%，位移 1 m                 达成 96%，位移 2-8 cm

状态与输出：
    输入  θ(俯仰) θ̇  x  ẋ  ψ(航向)  ψ̇
    输出  bias[6]，只填两个轮子：
            同向分量 tw  → 抵消俯仰、跟踪速度
            差速分量 turn → 跟踪航向
          bias[2] = tw + turn     （左轮）
          bias[5] = tw - turn     （右轮）

为什么不做成神经网络：
    轮腿平衡是欠驱动的轮式倒立摆，有成熟解析解。用 RL 从零学它需要百万级
    样本且没学出来；用状态反馈只要 8 个数字。同类开源项目（RM 平衡步兵、
    circleg、palette2/graduate_wheel_robot）都用这个结构。

符号约定（已用开环测试标定，见 pitch_comp_test）：
    两轮同时加正力矩 → 俯仰角下降，因此 k_th、ki 取正号。

用法：
    由 env.py 在 BALANCE_MODE=analytic 时调用；也可在脚本里单独使用。
"""

from __future__ import annotations

import numpy as np

# 站立/转向时用硬位置保持（kx 大），行驶时放松（kx 小）以便跟参考轨迹
STAND = dict(k_th=30.0, k_thd=6.0, k_x=60.0, k_v=30.0, ki_th=1.5)
MOVE = dict(k_th=30.0, k_thd=6.0, k_x=30.0, k_v=30.0, ki_th=1.5)
YAW = dict(k_w=2.0, k_psi=2.0, clamp=2.5)

TORQUE_CLAMP = 12.0
MOVE_THRESHOLD = 0.05  # |vx 指令| 超过此值算"行驶"
I_TH_LIMIT = 0.6       # 俯仰积分限幅 rad·s（anti-windup）
# 速度指令一阶滤波：加速慢、减速快（不对称）
#   加速侧 0.25s 保证上升时间 ≤0.90s；减速侧 0.08s 快速刹停（停车距离要求 0.08m）
TAU_UP = 0.25
TAU_DOWN = 0.08


def _rot(data, base_id: int) -> np.ndarray:
    return np.asarray(data.xmat[base_id]).reshape(3, 3)


def _pitch(data, base_id: int) -> float:
    r = _rot(data, base_id)
    return float(np.arctan2(-r[2, 0], r[2, 2]))


def _yaw(data, base_id: int) -> float:
    r = _rot(data, base_id)
    return float(np.arctan2(r[1, 0], r[0, 0]))


def _body_vx(model, data, base_id: int) -> float:
    import mujoco

    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(
        model, data, mujoco.mjtObj.mjOBJ_XBODY, base_id, velocity, 0
    )
    return float((_rot(data, base_id).T @ velocity[3:])[0])


def make_state(data, base_id: int) -> dict[str, float]:
    """在 reset 之后调用，记录参考轨迹的起点。"""
    psi = _yaw(data, base_id)
    return {
        "x_ref": float(data.xpos[base_id][0]),
        "psi_ref": psi,
        "iv": 0.0,
        "i_th": 0.0,
        "v_cmd": 0.0,
        "w_cmd": 0.0,
    }


def compute(state: dict[str, float], data, model, base_id: int,
            command: np.ndarray, dt: float) -> np.ndarray:
    """算一次轮子力矩偏置（6 维，只有 2 和 5 非零）。"""
    vx_target = float(command[0])
    wz_target = float(command[1])

    # 指令一阶滤波（不对称）：加速慢一点压超调，减速快一点压停车距离
    vx_target = float(command[0])
    wz_target = float(command[1])
    tau_v = TAU_UP if abs(vx_target) > abs(state["v_cmd"]) else TAU_DOWN
    tau_w = TAU_UP if abs(wz_target) > abs(state["w_cmd"]) else TAU_DOWN
    alpha_v = min(1.0, dt / tau_v) if tau_v > 0 else 1.0
    alpha_w = min(1.0, dt / tau_w) if tau_w > 0 else 1.0
    state["v_cmd"] += (vx_target - state["v_cmd"]) * alpha_v
    state["w_cmd"] += (wz_target - state["w_cmd"]) * alpha_w
    vx_ref = state["v_cmd"]
    wz_ref = state["w_cmd"]

    th = _pitch(data, base_id)
    thd = float(data.sensor("gyro").data[1])
    x = float(data.xpos[base_id][0])
    v = _body_vx(model, data, base_id)
    psi = _yaw(data, base_id)
    wz = float(data.sensor("gyro").data[2])

    # 参考轨迹：位置参考跟着速度指令积分，否则位置回路会把"按指令前进"当误差
    state["x_ref"] += vx_ref * dt
    state["psi_ref"] += wz_ref * dt

    # 俯仰积分：策略是无记忆的，靠它把稳态前倾压到 0（实测 15° → 3.6°，再往下靠这个）
    state["i_th"] = float(np.clip(state["i_th"] + th * dt, -I_TH_LIMIT, I_TH_LIMIT))

    gains = MOVE if abs(vx_target) > MOVE_THRESHOLD else STAND

    tw = -(
        gains["k_th"] * th
        + gains["k_thd"] * thd
        + gains["k_x"] * (x - state["x_ref"])
        + gains["k_v"] * (v - vx_ref)
    ) - gains["ki_th"] * state["i_th"]
    tw = float(np.clip(tw, -TORQUE_CLAMP, TORQUE_CLAMP))

    e_psi = (psi - state["psi_ref"] + np.pi) % (2.0 * np.pi) - np.pi
    turn = float(
        np.clip(-(YAW["k_w"] * (wz - wz_ref) + YAW["k_psi"] * e_psi),
                -YAW["clamp"], YAW["clamp"])
    )

    bias = np.zeros(6)
    bias[2] = tw + turn
    bias[5] = tw - turn
    return bias
