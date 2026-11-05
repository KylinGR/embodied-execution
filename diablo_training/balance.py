"""解析平衡/速度/航向控制器（基于模型的状态反馈）。

实测依据（同一台仿真机器人，同一套 diag 剖面）：

    能力            RL 最好（12 次训练 × 200 万步）   本控制器（8 个增益）
    站立漂移        12.6 m / 120 s                     1.9 cm / 6 s
    站立自转        4.4 °/s                            0.0° / 8 s
    定速 0.5 m/s    误差 8-15%                         误差 0%
    航向保持        16.6° / 10 s                       0.0° / 8 s
    原地转向 0.3    达成 12%，位移 1 m                 达成 96%，位移 2-8 cm

状态与输出：
    输入  θ(俯仰) θ̇  s(行程)  ẋ  ψ(航向)  ψ̇
    输出  bias[6]，只填两个轮子：
            同向分量 tw  → 抵消俯仰、跟踪速度
            差速分量 turn → 跟踪航向
          bias[2] = tw + turn     （左轮）
          bias[5] = tw - turn     （右轮）

为什么用行程 s 而不是世界 x：
    x 是世界系坐标，航向转到 ψ 之后，机体以 v 前进时世界 x 的变化率是
    v·cosψ。若位置参考仍按 x_ref += v_cmd·dt 积分，控制器为了"追上"这个
    x_ref 必须让机体速度变成 v_cmd/cosψ —— 实测航向 35° 时指令 0.4 跑成
    0.51（+47%），⑥ 行驶中转弯就是这么挂的。改积分机体前向速度就与
    航向无关。

为什么停车要单独一段逻辑：
    停车瞬间速度误差一步到 -0.4 m/s，制动扭矩把机身压成前倾（-13°），
    俯仰阻尼项 k_thd·θ̇ 立刻回一个 +24 N·m 的正扭矩把机体往前推，制动
    只作用了一个控制周期。实测平均减速度只有 0.8 m/s²（物理上限 5+），
    停车距离 0.0967 m 卡在阈值 0.08 上。
    → 制动期间单独一套增益：俯仰刚度和阻尼都缩小、关掉位置/积分项、
      力矩上限放到 16，让制动敢持续用力。

符号约定（已用开环测试标定，见 pitch_comp_test）：
    两轮同时加正力矩 → 俯仰角下降，因此 k_th、ki 取正号。

用法：
    由 env.py 在 BALANCE_MODE=analytic 时调用；也可在脚本里单独使用。
"""

from __future__ import annotations

import numpy as np

# 站立/转向时用硬位置保持（kx 大），行驶时放松（kx 小）以便跟参考轨迹
# MOVE 的 k_x 从 30 降到 10：位置项在校准追赶时会把"参考轨迹还没走到的部分"
# 也算成误差，起步阶段额外加速 → 前向超调 31%。k_v 30→45 补回定速精度。
STAND = dict(k_th=30.0, k_thd=6.0, k_x=60.0, k_v=30.0, ki_th=1.5)
MOVE = dict(k_th=30.0, k_thd=6.0, k_x=10.0, k_v=45.0, ki_th=1.5)
YAW = dict(k_w=2.0, k_psi=2.0, clamp=2.5)

TORQUE_CLAMP = 12.0
MOVE_THRESHOLD = 0.05  # |vx 指令| 超过此值算"行驶"
I_TH_LIMIT = 0.6       # 俯仰积分限幅 rad·s（anti-windup）
# 速度指令一阶滤波：加速慢、减速快（不对称）
#   加速侧 0.35s：0.25 时起步太猛（前向超调 31%），0.35 后剩 1%，
#                 上升时间 0.66s，仍低于 0.90s 判据
#   减速侧 0.03s 让参考速度先行归零
TAU_UP = 0.35
TAU_DOWN = 0.03
STOP_THRESHOLD = 0.02  # |vx 指令| 低于此值算"停车"
STOP_ANCHOR_EACH = 0.0  # 1=每个停车周期都重钉行程参考；0=只在停车那一刻钉一次
                        # （0 更稳：站立时位置回路还在，实测漂移 0.063 → 0.007 m/s）

# 延迟补偿：传感器/执行延迟 τ 时，轮子真正出力那一刻的俯仰角约等于 θ + θ̇·τ。
# 用预测值做俯仰反馈可补掉大部分相位滞后；0 = 不补偿。
# 实测（100 Hz 控制周期）：10 ms 延迟 start_stop 就开始掉、20 ms 连站立都崩。
TH_PREDICT_S = 0.0

# 制动段专用增益。k_*_scale 是相对 STAND 的缩放。
# 实测（start_stop, 3 次启停）：停车距离 0.099 → 0.067 m，min_upright 0.934。
#   k_v 加大是主因：原 30 时速度误差 0.4 m/s 只出 -12 N·m，且只维持一个周期；
#   75 才能在俯仰回路回顶之后仍保持制动。
#   k_th 放大到 2.5 倍是配套的：制动时俯仰刚度不够会被 -13° 前倾拖散（跌倒）。
STOP = dict(
    k_th_scale=2.5,
    k_thd_scale=1.0,
    k_x_scale=0.0,     # 制动时不追位置参考，只追速度
    k_v=75.0,
    ki_scale=0.0,      # 制动时冻结俯仰积分，避免松刹车时被积分项推着走
    clamp=17.0,
    pos_limit=99.0,    # 制动时正扭矩（"接住身体"那一下）的上限，单独限
    decel=0.0,         # >0 时参考速度按此减速度线性下降（m/s²），0=用滤波
    exit_v=0.10,       # |v| 低于此值退出制动段（0.05 太小：站立时 0.06 m/s 的
                       # 自然蠕行会一直被判成"制动"，位置回路被关掉，越漂越远）
)


# 最近一拍的分项力矩，供 posture_probe 做姿态体检（不参与控制）
LAST: dict[str, float] = {}


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
        "s": 0.0,
        "s_ref": 0.0,
        "psi_ref": psi,
        "iv": 0.0,
        "i_th": 0.0,
        "v_cmd": 0.0,
        "w_cmd": 0.0,
        "stopped": 0.0,
    }


def _gains(vx_target: float, braking: bool) -> tuple[float, float, float, float, float, float]:
    """返回 (k_th, k_thd, k_x, k_v, ki_th, clamp)。"""
    if braking:
        return (
            STAND["k_th"] * STOP["k_th_scale"],
            STAND["k_thd"] * STOP["k_thd_scale"],
            STAND["k_x"] * STOP["k_x_scale"],
            STOP["k_v"],
            STAND["ki_th"] * STOP["ki_scale"],
            STOP["clamp"],
        )
    g = MOVE if abs(vx_target) > MOVE_THRESHOLD else STAND
    return (g["k_th"], g["k_thd"], g["k_x"], g["k_v"], g["ki_th"], TORQUE_CLAMP)


def compute(state: dict[str, float], data, model, base_id: int,
            command: np.ndarray, dt: float) -> np.ndarray:
    """算一次轮子力矩偏置（6 维，只有 2 和 5 非零）。"""
    vx_target = float(command[0])
    wz_target = float(command[1])

    # 指令一阶滤波（不对称）：加速慢一点压超调，减速快一点压停车距离。
    # 停车段可以改成"按固定减速度斜坡归零"：一步把参考速度打到 0 会让速度
    # 误差瞬间到 -0.4 m/s，制动扭矩只作用一个周期就被俯仰回路顶回去。
    stopping = abs(vx_target) < STOP_THRESHOLD
    if stopping:
        if STOP["decel"] > 0.0:
            step = STOP["decel"] * dt
            state["v_cmd"] = (0.0 if abs(state["v_cmd"]) <= step
                              else state["v_cmd"] - np.sign(state["v_cmd"]) * step)
        else:
            state["v_cmd"] += (0.0 - state["v_cmd"]) * min(1.0, dt / TAU_DOWN)
    else:
        tau_v = TAU_UP if abs(vx_target) > abs(state["v_cmd"]) else TAU_DOWN
        state["v_cmd"] += (vx_target - state["v_cmd"]) * min(1.0, dt / tau_v)

    tau_w = TAU_UP if abs(wz_target) > abs(state["w_cmd"]) else TAU_DOWN
    state["w_cmd"] += (wz_target - state["w_cmd"]) * min(1.0, dt / tau_w)
    vx_ref = state["v_cmd"]
    wz_ref = state["w_cmd"]

    th = _pitch(data, base_id)
    thd = float(data.sensor("gyro").data[1])
    v = _body_vx(model, data, base_id)
    psi = _yaw(data, base_id)
    wz = float(data.sensor("gyro").data[2])

    # 行程：积分机体前向速度，和世界系航向无关（见文件头）
    state["s"] += v * dt

    if stopping:
        if STOP_ANCHOR_EACH or not state["stopped"]:
            state["s_ref"] = state["s"]
        state["stopped"] = 1.0
    else:
        state["stopped"] = 0.0

    state["s_ref"] += vx_ref * dt
    state["psi_ref"] += wz_ref * dt

    braking = stopping and abs(v) > STOP["exit_v"]
    # 俯仰积分：只在非制动段累积，制动时冻结（否则松刹车会被积分项推着走）
    if not braking:
        state["i_th"] = float(np.clip(state["i_th"] + th * dt, -I_TH_LIMIT, I_TH_LIMIT))

    k_th, k_thd, k_x, k_v, ki_th, clamp = _gains(vx_target, braking)

    # 延迟补偿：用 θ + θ̇·τ 近似"力矩真正作用到轮子时"的俯仰角
    th_ctl = th + thd * TH_PREDICT_S if TH_PREDICT_S else th

    tw = -(
        k_th * th_ctl
        + k_thd * thd
        + k_x * (state["s"] - state["s_ref"])
        + k_v * (v - vx_ref)
    ) - ki_th * state["i_th"]
    if braking:
        tw = float(np.clip(tw, -clamp, min(clamp, STOP["pos_limit"])))
    else:
        tw = float(np.clip(tw, -clamp, clamp))

    # 姿态体检用：把这一拍各项力矩分量留在模块级，posture_probe 直接读
    LAST.clear()
    LAST.update(
        th=float(th), thd=float(thd), v=float(v), v_ref=float(vx_ref),
        ds=float(state["s"] - state["s_ref"]), dv=float(v - vx_ref),
        term_th=float(k_th * th), term_thd=float(k_thd * thd),
        term_s=float(k_x * (state["s"] - state["s_ref"])),
        term_v=float(k_v * (v - vx_ref)), term_i=float(ki_th * state["i_th"]),
        i_th=float(state["i_th"]), tw=float(tw), braking=float(braking),
        k_th=k_th, k_thd=k_thd, k_x=k_x, k_v=k_v, clamp=clamp,
    )

    e_psi = (psi - state["psi_ref"] + np.pi) % (2.0 * np.pi) - np.pi
    turn = float(
        np.clip(-(YAW["k_w"] * (wz - wz_ref) + YAW["k_psi"] * e_psi),
                -YAW["clamp"], YAW["clamp"])
    )

    bias = np.zeros(6)
    bias[2] = tw + turn
    bias[5] = tw - turn
    return bias
