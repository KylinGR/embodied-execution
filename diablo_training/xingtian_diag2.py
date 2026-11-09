"""A 阶段应用级诊断 diag-2：自己下指令、自己量位移、失败必给实数。

diag-1 的三个致命问题（已定位并修掉）：
  1. `reset(options={"profile": ...})` 根本不认 `profile` —— 环境只读 `task` / `speed`。
     结果 `stand` 和 `forward` 跑出逐字节相同的轨迹，阶段名是假的。
     → diag-2 用官方钩子 `DiabloEnv.get_command(sim_seconds)` 覆写指令，
       签名见 env.py:110，返回 [vx m/s, yaw rad/s, height m]，env.py:301 每步调用。
  2. 所有指标都从 `info` 里取，`info` 缺字段时静默变 0 / 默认值，
     于是"缺数据"被印成"速度误差 0%"。
     → diag-2 的速度、位移、航向全部由 `data.xpos` / `data.xmat` 自己算；
       `info` 只用来做交叉校验，并在 JSON 里留下误差值。
  3. 机身高度基准硬编码 0.25，而 `base_z` 是机身连杆原点、不等于整机高度。
     → diag-2 从指令里读真实高度，且**不拿它判通过**（基准尚未标定，见附录 C），
       只判高度波动和标准差（这两个和基准无关，天然可信）。

用法：
  python -m diablo_training.xingtian_diag2 \
      --model /home/yuewenzhe/outputs/A-stage/selection/policy-best.zip \
      --seeds 2002 2003 2004 --out runs/diag/A2

  # 只跑某几个阶段
  python -m diablo_training.xingtian_diag2 --model ... --only stand forward

  # 用训练时的 0.35 s 斜坡重跑（更公平，但阶跃响应会被斜坡糊掉）
  python -m diablo_training.xingtian_diag2 --model ... --ramp 0.35

  # 录像
  python -m diablo_training.xingtian_diag2 --model ... --video

注意：本文件不在 policy_contract 指纹里，改动不影响权重加载。
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

VERSION = "diag-5"

# 记录这几个关节的实际角度：用来把 base_z 对到可达高度网格上，
# 也用来判断策略有没有落到平行四边形的镜像支链。
JOINTS_DIAG = (
    "left_hip_motor", "left_knee_motor", "left_wheel_motor",
    "right_hip_motor", "right_knee_motor", "right_wheel_motor",
)


# ---------------------------------------------------------------- 指令剖面

def _pt(time_s: float, vx: float = 0.0, yaw: float = 0.0, h: float = 0.25) -> dict[str, Any]:
    """一个指令转折点。[vx m/s, yaw rad/s, height m]。"""
    return {"time": float(time_s), "command": [float(vx), float(yaw), float(h)]}


STAGES: dict[str, dict[str, Any]] = {
    # 原地平衡：15 秒纯零指令
    "stand": {"duration": 15.0, "profile": [_pt(0.0)]},
    # 持续站立：120 秒纯零指令。手册写实机"最大连续站立 ≥90 min"，
    # 仿真里至少验到分钟级，否则"能站住"和"能站 0.7 秒"分不开。
    "stand_long": {"duration": 120.0, "profile": [_pt(0.0)]},
    # 直线前进：1 秒静止后阶跃到 0.5
    "forward": {"duration": 15.0, "profile": [_pt(0.0), _pt(1.0, 0.5)]},
    # 直线后退
    "backward": {"duration": 15.0, "profile": [_pt(0.0), _pt(1.0, -0.3)]},
    # 原地转向（0.3 已证伪 0.8，先证能不能转）
    "turn": {"duration": 12.0, "profile": [_pt(0.0), _pt(1.0, yaw=0.3)]},
    "turn_left": {"duration": 12.0, "profile": [_pt(0.0), _pt(1.0, yaw=0.3)]},
    "turn_right": {"duration": 12.0, "profile": [_pt(0.0), _pt(1.0, yaw=-0.3)]},
    # 行进中左拐→直行→右拐→直行（原评测完全没测过）
    "turn_in_motion": {
        "duration": 18.0,
        "profile": [
            _pt(0.0), _pt(1.0, 0.4), _pt(3.0, 0.4, 0.3),
            _pt(8.0, 0.4), _pt(11.0, 0.4, -0.3), _pt(16.0, 0.4),
        ],
    },
    # 一流程三次启停（每次 3 s 走、3 s 停）
    "start_stop": {
        "duration": 22.0,
        "profile": [
            _pt(0.0), _pt(1.0, 0.4), _pt(5.0), _pt(8.0, 0.4),
            _pt(12.0), _pt(15.0, 0.4), _pt(19.0),
        ],
    },
    # 直线高速：配置允许 1.0 m/s，但 A 阶段只测过 0.5，单独加两档验证上限
    "forward_fast_08": {"duration": 15.0, "profile": [_pt(0.0), _pt(1.0, 0.8)]},
    "forward_fast_10": {"duration": 15.0, "profile": [_pt(0.0), _pt(1.0, 1.0)]},
    # 低速→高速→中速，看定速和过渡
    "accel": {
        "duration": 14.0,
        "profile": [_pt(0.0), _pt(1.0, 0.1), _pt(5.0, 0.5), _pt(10.0, 0.2)],
    },
}

DEFAULT_STAGES = (
    "stand", "stand_long", "forward", "backward", "turn",
    "turn_in_motion", "start_stop", "accel",
)


# ---------------------------------------------------------------- 阈值

THRESHOLDS: dict[str, dict[str, float]] = {
    "common": {"min_upright": 0.90, "duration_ratio": 0.98},
    "stand": {
        "still_turn_dps": 0.20,     # °/s，和时长解耦，15 s 和 120 s 用同一条线
        "still_drift_mps": 0.02,
        "still_coverage": 0.85,
        "tilt_rms_deg": 2.0,
        "tilt_max_deg": 6.0,
    },
    "stand_long": {
        "still_turn_dps": 0.20,
        "still_drift_mps": 0.02,
        "still_coverage": 0.85,
        "tilt_rms_deg": 2.0,
        "tilt_max_deg": 6.0,
    },
    "forward": {
        "speed_rel_err": 0.08, "distance_ratio": 0.90, "lateral_drift_m": 0.08,
        "heading_deg": 3.0, "rise_time_s": 0.8, "overshoot": 0.10,
    },
    "backward": {
        "speed_rel_err": 0.08, "distance_ratio": 0.90, "lateral_drift_m": 0.08,
        "heading_deg": 3.0,
    },
    "turn": {"yaw_mae": 0.12, "min_sign_ratio": 0.90},
    "turn_left": {"yaw_mae": 0.12, "min_sign_ratio": 0.90},
    "turn_right": {"yaw_mae": 0.12, "min_sign_ratio": 0.90},
    "turn_in_motion": {"yaw_mae_turn": 0.15, "speed_rel_err_turn": 0.15},
    "start_stop": {"n_cycles": 3.0, "rise_time_s": 0.8, "stop_distance_m": 0.08},
    "accel": {"speed_rel_err": 0.10},
    "forward_fast_08": {
        "speed_rel_err": 0.08, "distance_ratio": 0.90, "lateral_drift_m": 0.08,
        "heading_deg": 3.0, "rise_time_s": 0.8, "overshoot": 0.10,
    },
    "forward_fast_10": {
        "speed_rel_err": 0.08, "distance_ratio": 0.90, "lateral_drift_m": 0.08,
        "heading_deg": 3.0, "rise_time_s": 0.8, "overshoot": 0.10,
    },
    # 高度只判波动/标准差；与指令的偏差依赖尚未标定的 base_z 基准，不判通过
    "height": {"ripple_m": 0.04, "std_m": 0.015},
}

DT_FALLBACK = 0.01


# ---------------------------------------------------------------- 环境

def _resolve_class(name: str):
    module, _, attr = name.partition(":")
    if not attr:
        raise ValueError(f"需要 'module:Class' 形式，收到 {name!r}")
    return getattr(importlib.import_module(module), attr)


def build_env(resolved: dict[str, Any]):
    """按 resolved.json 构建环境，等价于 cli.load_policy 的做法。"""
    from diablo_training.contract import environment_class

    env_cls = environment_class(resolved["environment_class"])
    params = dict(resolved["environment"])

    cfg = None
    cfg_name = resolved.get("config_class")
    if cfg_name:
        try:
            cfg = _resolve_class(cfg_name)(**params)
        except Exception as exc:
            print(f"[warn] config_class 构建失败（{exc}），退回 diablo_training.env.Config")
    if cfg is None:
        from diablo_training.env import Config

        cfg = Config(**params)

    return env_cls(cfg)


def load_policy(policy_path: Path):
    from stable_baselines3 import PPO

    return PPO.load(str(policy_path), device="cpu")


# ---------------------------------------------------------------- 自己下指令

def command_at(profile: list[dict[str, Any]], t: float, ramp: float) -> np.ndarray:
    """按剖面取 t 时刻的指令。ramp>0 时在每个转折点前 ramp 秒内线性过渡。"""
    if t <= profile[0]["time"]:
        return np.asarray(profile[0]["command"], dtype=float)

    for i in range(1, len(profile)):
        a, b = profile[i - 1], profile[i]
        if t < b["time"]:
            c0 = np.asarray(a["command"], dtype=float)
            c1 = np.asarray(b["command"], dtype=float)
            if ramp <= 0.0:
                return c0
            start = b["time"] - ramp
            frac = 0.0 if t <= start else min(1.0, (t - start) / ramp)
            return c0 + (c1 - c0) * frac
    return np.asarray(profile[-1]["command"], dtype=float)


def install_profile(env, profile: list[dict[str, Any]], ramp: float) -> None:
    """覆写官方钩子 get_command(sim_seconds)。

    env.py:301 每步用 self.steps * CONTROL_DT 调用它，
    所以这里返回的就是策略在观测 [6:9] 里看到的指令。
    """
    env.get_command = lambda sim_seconds: command_at(profile, float(sim_seconds), ramp)
    # env.py:245 在本步开头 copy 上一步末尾写入的 self.command，
    # 所以 reset 之后必须立刻补一次 t=0 的指令，否则第一步用的是调度器残留值。
    env.command = command_at(profile, 0.0, ramp)


# ---------------------------------------------------------------- 从 data 自量

def _rot(env) -> np.ndarray:
    return np.asarray(env.data.xmat[env.base_id]).reshape(3, 3)


def _body_yaw(env) -> float:
    r = _rot(env)
    return math.atan2(float(r[1, 0]), float(r[0, 0]))


def _upright(env) -> float:
    return float(_rot(env)[2, 2])


def _base_xyz(env) -> tuple[float, float, float]:
    p = env.data.xpos[env.base_id]
    return float(p[0]), float(p[1]), float(p[2])


_SIGNS_FALLBACK = (1.0, 1.0, -1.0, 1.0, 1.0, -1.0)


def _wheel_omega(env) -> np.ndarray | None:
    """两个轮子的角速度（已乘观测符号，与策略看到的一致）。"""
    try:
        from diablo_training.model import OBSERVATION_SIGNS

        signs = np.asarray(OBSERVATION_SIGNS, dtype=float)
    except Exception:
        signs = np.asarray(_SIGNS_FALLBACK, dtype=float)
    try:
        wheels = [2, 5]
        return np.asarray(env.data.qvel[env.vi[wheels]], dtype=float) * signs[wheels]
    except Exception:
        return None


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# ---------------------------------------------------------------- 单阶段运行

def run_stage(
    env,
    model,
    name: str,
    spec: dict[str, Any],
    seed: int,
    *,
    ramp: float,
    wheel_radius: float,
    video_dir: Path | None = None,
) -> dict[str, Any]:
    install_profile(env, spec["profile"], ramp)
    # 训练配置里 episode_seconds 通常是 30 s，环境到点就 trunc，
    # 那 120 s 的 stand_long 永远跑不完（duation_ratio 只有 0.25）。
    # 诊断的时长由剖面自己决定，所以要先把环境的回合上限抬上去。
    need_s = float(spec["duration"]) + 1.0
    try:
        current = float(getattr(env.config, "episode_seconds", 0.0))
        if current < need_s:
            env.config.episode_seconds = need_s
    except Exception as exc:
        print(f"[warn] 无法抬高 episode_seconds（{exc}），长剖面会被截断")

    obs, _ = env.reset(seed=seed, options={})

    dt = float(getattr(env.config, "control_dt_s", DT_FALLBACK) or DT_FALLBACK)
    limit = int(round(spec["duration"] / dt))

    keys = (
        "t", "cmd_v", "cmd_w", "cmd_h", "v_true", "v_info", "wheel_vx",
        "w_z", "yaw", "x", "y", "z", "up", "roll", "pitch", "wl_z", "wr_z",
    )
    rec: dict[str, list[float]] = {k: [] for k in keys}
    for jname in JOINTS_DIAG:
        rec[f"q_{jname}"] = []
    qadr = {jname: int(env.model.jnt_qposadr[env.model.joint(jname).id])
            for jname in JOINTS_DIAG}
    try:
        wheel_bid = {"left": int(env.model.body("left_wheel").id),
                     "right": int(env.model.body("right_wheel").id)}
    except Exception:
        wheel_bid = None

    renderer = writer_top = writer_side = None
    cam_top = cam_side = None
    if video_dir is not None:
        import imageio.v2 as iio
        import mujoco

        video_dir.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(env.model, height=480, width=640)

        cam_top = mujoco.MjvCamera()
        cam_top.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam_top.distance, cam_top.azimuth, cam_top.elevation = 3.0, 90.0, -89.0

        cam_side = mujoco.MjvCamera()
        cam_side.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam_side.distance, cam_side.azimuth, cam_side.elevation = 2.0, 135.0, -20.0

        writer_top = iio.get_writer(str(video_dir / f"{name}-top.mp4"), fps=25)
        writer_side = iio.get_writer(str(video_dir / f"{name}-side.mp4"), fps=25)

    terminated = False
    truncated = False
    idx = 0
    prev = None
    yaw_prev = None
    try:
        for i in range(limit):
            action, _ = model.predict(obs, deterministic=True)
            obs, _r, term, trunc, info = env.step(action)

            x, y, z = _base_xyz(env)
            yaw = _body_yaw(env)
            rot = _rot(env)
            up = float(rot[2, 2])
            # ZYX 分解：横滚绕 x、俯仰绕 y。两者含义完全不同——
            # 俯仰 15° 是"前倾走路"，横滚 15° 是"一个轮子抬起来了"。
            pitch = math.atan2(-float(rot[2, 0]),
                               math.hypot(float(rot[2, 1]), float(rot[2, 2])))
            roll = math.atan2(float(rot[2, 1]), float(rot[2, 2]))

            if prev is None:
                v_true = 0.0
            else:
                px, py, _pz = prev
                dyaw = _wrap(yaw - yaw_prev)
                ym = yaw_prev + 0.5 * dyaw
                v_true = ((x - px) * math.cos(ym) + (y - py) * math.sin(ym)) / dt

            omega = _wheel_omega(env)
            wheel_vx = float("nan") if omega is None else float(wheel_radius * omega.mean())

            cmd = np.asarray(env.command, dtype=float).reshape(-1)
            rec["t"].append(i * dt)
            rec["cmd_v"].append(float(cmd[0]))
            rec["cmd_w"].append(float(cmd[1]))
            rec["cmd_h"].append(float(cmd[2]) if cmd.size > 2 else float("nan"))
            rec["v_true"].append(float(v_true))
            rec["v_info"].append(float(info.get("forward_mps", float("nan"))))
            rec["wheel_vx"].append(wheel_vx)
            rec["w_z"].append(float(env.data.sensor("gyro").data[2]))
            rec["yaw"].append(float(yaw))
            rec["x"].append(x)
            rec["y"].append(y)
            rec["z"].append(z)
            rec["up"].append(up)
            rec["roll"].append(math.degrees(roll))
            rec["pitch"].append(math.degrees(pitch))
            if wheel_bid is not None:
                rec["wl_z"].append(float(env.data.xpos[wheel_bid["left"]][2]))
                rec["wr_z"].append(float(env.data.xpos[wheel_bid["right"]][2]))
            else:
                rec["wl_z"].append(float("nan"))
                rec["wr_z"].append(float("nan"))
            for jname in JOINTS_DIAG:
                rec[f"q_{jname}"].append(float(env.data.qpos[qadr[jname]]))

            prev = (x, y, z)
            yaw_prev = yaw

            if renderer is not None and i % 4 == 0:
                for cam, writer in ((cam_top, writer_top), (cam_side, writer_side)):
                    cam.lookat[:] = env.data.xpos[env.base_id]
                    renderer.update_scene(env.data, camera=cam)
                    writer.append_data(renderer.render())

            idx = i + 1
            if term or trunc:
                terminated = bool(term)
                truncated = bool(trunc)
                break
    finally:
        for w in (writer_top, writer_side):
            if w is not None:
                w.close()
        if renderer is not None:
            renderer.close()

    metrics = compute_metrics(rec, dt)
    metrics["stage"] = name
    metrics["seed"] = seed
    metrics["steps"] = idx
    metrics["terminated"] = terminated
    metrics["truncated"] = truncated
    metrics["duration_ratio"] = metrics["duration_s"] / spec["duration"]
    metrics["ramp_s"] = ramp
    metrics["wheel_radius"] = wheel_radius
    ok, reasons = judge(name, metrics)
    metrics["pass"] = ok
    metrics["reasons"] = reasons
    return metrics


# ---------------------------------------------------------------- 指标

def _edge_indices(cv: np.ndarray, enter: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """指令从零起步 / 回到零的步号。"""
    moved = np.abs(cv) > enter
    starts = np.where(moved[1:] & ~moved[:-1])[0] + 1
    stops = np.where(~moved[1:] & moved[:-1])[0] + 1
    return starts, stops


def compute_metrics(rec: dict[str, list[float]], dt: float) -> dict[str, Any]:
    t = np.asarray(rec["t"], dtype=float)
    cv = np.asarray(rec["cmd_v"], dtype=float)
    cw = np.asarray(rec["cmd_w"], dtype=float)
    ch = np.asarray(rec["cmd_h"], dtype=float)
    v = np.asarray(rec["v_true"], dtype=float)
    vi = np.asarray(rec["v_info"], dtype=float)
    wv = np.asarray(rec["wheel_vx"], dtype=float)
    w = np.asarray(rec["w_z"], dtype=float)
    yaw = np.asarray(rec["yaw"], dtype=float)
    x = np.asarray(rec["x"], dtype=float)
    y = np.asarray(rec["y"], dtype=float)
    z = np.asarray(rec["z"], dtype=float)
    up = np.asarray(rec["up"], dtype=float)

    out: dict[str, Any] = {}
    out["duration_s"] = float(t[-1] + dt) if t.size else 0.0
    out["min_upright"] = float(up.min()) if up.size else float("nan")
    out["upright_mean"] = float(up.mean()) if up.size else float("nan")
    # 倾角：up 是机身 z 轴与世界 z 轴的点积，夹角就是倾斜角。
    # 单看 min_upright 分不清"一直歪着"和"闪了一下"，所以给 RMS 和均值。
    tilt = np.degrees(np.arccos(np.clip(up, -1.0, 1.0))) if up.size else np.zeros(0)
    if tilt.size:
        out["tilt_mean_deg"] = float(tilt.mean())
        out["tilt_rms_deg"] = float(np.sqrt(np.mean(tilt ** 2)))
        out["tilt_max_deg"] = float(tilt.max())

    # 横滚/俯仰分解 + 左右轮高度差：判断"倾"是前倾还是单轮抬起
    roll = np.asarray(rec.get("roll", []), dtype=float)
    pitch = np.asarray(rec.get("pitch", []), dtype=float)
    if roll.size:
        out["roll_mean_deg"] = float(roll.mean())
        out["roll_rms_deg"] = float(np.sqrt(np.mean(roll ** 2)))
        out["roll_absmax_deg"] = float(np.abs(roll).max())
        out["pitch_mean_deg"] = float(pitch.mean())
        out["pitch_rms_deg"] = float(np.sqrt(np.mean(pitch ** 2)))
        out["pitch_absmax_deg"] = float(np.abs(pitch).max())
    wl = np.asarray(rec.get("wl_z", []), dtype=float)
    wr = np.asarray(rec.get("wr_z", []), dtype=float)
    if wl.size and np.isfinite(wl).any() and np.isfinite(wr).any():
        diff = wl - wr
        out["wheel_z_diff_mean_mm"] = float(np.nanmean(diff) * 1000.0)
        out["wheel_z_diff_absmax_mm"] = float(np.nanmax(np.abs(diff)) * 1000.0)

    # 航向：先解缠，避免跨 ±π 时算出 360° 跳变
    if yaw.size > 1:
        dyaw = np.array([_wrap(yaw[i] - yaw[i - 1]) for i in range(1, yaw.size)])
        out["heading_change_deg"] = float(np.degrees(dyaw.sum()))
    else:
        out["heading_change_deg"] = 0.0

    step = np.hypot(np.diff(x), np.diff(y)) if x.size > 1 else np.zeros(0)
    path = float(step.sum())
    out["path_length_m"] = path
    disp = float(math.hypot(x[-1] - x[0], y[-1] - y[0])) if x.size else 0.0
    out["displacement_m"] = disp
    out["straightness"] = float(disp / path) if path > 1e-6 else 0.0

    # 横向漂移：所有点到首末连线的最大垂直距离
    out["lateral_drift_m"] = 0.0
    if x.size > 10 and path > 0.05:
        d = np.array([x[-1] - x[0], y[-1] - y[0]])
        n = float(np.linalg.norm(d))
        if n > 1e-6:
            rel = np.stack([x - x[0], y - y[0]], axis=1)
            out["lateral_drift_m"] = float(np.max(np.abs(rel @ np.array([-d[1], d[0]]) / n)))

    # 应该走多远（指令积分），和实际位移比
    expected = float(np.abs(cv).sum() * dt)
    out["expected_distance_m"] = expected
    out["distance_ratio"] = float(disp / expected) if expected > 1e-6 else 0.0

    # 定速
    steady = (np.abs(cv) > 0.05) & (np.abs(cw) < 0.05)
    out["steady_coverage"] = float(steady.mean()) if steady.size else 0.0
    if steady.sum() > 50:
        cmd_level = float(np.abs(cv[steady]).mean())
        err = np.abs(v[steady] - cv[steady])
        out["speed_mae_steady"] = float(err.mean())
        out["speed_rel_err"] = float(err.mean() / max(cmd_level, 1e-6))
        out["v_true_mean_steady"] = float(v[steady].mean())
        out["cmd_v_mean_steady"] = cmd_level
        out["v_true_std_steady"] = float(v[steady].std())
        b = np.abs(v[steady]) > 0.05
        out["speed_sign_follow"] = float(np.mean(np.sign(v[steady][b]) == np.sign(cv[steady][b]))) \
            if b.any() else float("nan")

    # 转向
    turn_win = np.abs(cw) > 0.05
    out["turn_coverage"] = float(turn_win.mean()) if turn_win.size else 0.0
    if turn_win.sum() > 50:
        e = w[turn_win] - cw[turn_win]
        out["yaw_mae_turn"] = float(np.abs(e).mean())
        out["yaw_bias_turn"] = float(e.mean())
        out["yaw_sign_ratio"] = float(np.mean(np.sign(w[turn_win]) == np.sign(cw[turn_win])))
        out["yaw_mae"] = out["yaw_mae_turn"]
        out["yaw_rate_mean"] = float(w[turn_win].mean())
        out["yaw_cmd_mean"] = float(cw[turn_win].mean())

    # 静止
    still = (np.abs(cv) < 0.02) & (np.abs(cw) < 0.02)
    out["still_coverage"] = float(still.mean()) if still.size else 0.0
    if still.sum() > 100:
        still_s = float(still.sum()) * dt
        out["still_duration_s"] = still_s
        out["still_yaw_rate"] = float(np.abs(w[still]).mean())
        out["still_turn_deg"] = float(np.degrees(np.abs(w[still]).sum() * dt))
        # 角速度口径：0.2 °/s 就是"15 秒转 3°"，也等于"120 秒转 24°"
        out["still_turn_dps"] = out["still_turn_deg"] / max(still_s, 1e-9)
        out["still_drift_mps"] = float(np.abs(v[still]).mean())

    # 启停：每次从零起步 / 回到零都单独算
    starts, stops = _edge_indices(cv)
    rises: list[float] = []
    overs: list[float] = []
    for s in starts:
        target = abs(cv[s])
        seg = np.abs(v[s:s + int(3.0 / dt)])
        if seg.size == 0 or target <= 0:
            continue
        k10 = np.where(seg > 0.1 * target)[0]
        k90 = np.where(seg > 0.9 * target)[0]
        if k90.size:
            rises.append(float(k90[0] * dt))
            overs.append(float(max(0.0, seg.max() / target - 1.0)))
        else:
            rises.append(float("nan"))   # 没到 90%，如实记 NaN
    stop_d: list[float] = []
    for s in stops:
        seg = np.abs(v[s:s + int(3.0 / dt)])
        below = np.where(seg < 0.05)[0]
        stop_d.append(float(seg[: below[0] + 1].sum() * dt) if below.size else float("nan"))
    out["n_starts"] = int(starts.size)
    out["n_stops"] = int(stops.size)
    out["n_cycles"] = int(min(starts.size, stops.size))
    out["rise_times_s"] = [None if not np.isfinite(r) else round(r, 3) for r in rises]
    out["overshoot"] = max(overs) if overs else None
    out["stop_distances_m"] = [None if not np.isfinite(d) else round(d, 4) for d in stop_d]
    if starts.size:
        s0 = starts[0]
        out["rise_time_s"] = out["rise_times_s"][0]

    # 机身高度
    moving = np.abs(v) > 0.1
    out["base_z_mean_all"] = float(z.mean()) if z.size else float("nan")
    if moving.sum() > 50:
        out["base_z_mean_moving"] = float(z[moving].mean())
        out["height_ripple_m"] = float(z[moving].max() - z[moving].min())
        out["height_std_m"] = float(z[moving].std())
        if np.isfinite(ch[moving]).any():
            cmd_h = float(np.nanmean(ch[moving]))
            out["height_cmd"] = cmd_h
            # 仅作参考：base_z 是机身连杆原点，不是整机高度（附录 C 待标定）
            out["base_z_minus_cmd"] = float(z[moving].mean() - cmd_h)

    # 速度通道校验：轮速反推 vs 真实速度
    good = moving & np.isfinite(wv)
    if good.sum() > 50:
        out["wheel_vx_err"] = float(np.abs(wv[good] - v[good]).mean())
        denom = float(np.dot(v[good], v[good]))
        out["wheel_vx_slope"] = float(np.dot(wv[good], v[good]) / denom) if denom > 1e-9 else float("nan")
        if good.sum() > 10 and wv[good].std() > 1e-9 and v[good].std() > 1e-9:
            out["wheel_vx_corr"] = float(np.corrcoef(wv[good], v[good])[0, 1])
        else:
            out["wheel_vx_corr"] = float("nan")
    good_i = moving & np.isfinite(vi)
    if good_i.sum() > 50:
        out["info_vx_err"] = float(np.abs(vi[good_i] - v[good_i]).mean())

    # 关节角：判断策略站在哪个姿态、有没有翻支链
    pose: dict[str, dict[str, float]] = {}
    for jname in JOINTS_DIAG:
        q = np.asarray(rec.get(f"q_{jname}", []), dtype=float)
        if q.size == 0:
            continue
        pose[jname] = {
            "mean": float(q.mean()), "min": float(q.min()), "max": float(q.max()),
            "std": float(q.std()),
        }
        out[f"q_{jname}_mean"] = float(q.mean())
        out[f"q_{jname}_std"] = float(q.std())
    if pose:
        out["pose"] = pose

    return out


def judge(stage: str, m: dict[str, Any]) -> tuple[bool, list[str]]:
    """每条不通过都必须是实数。指标缺失本身算不通过，不再用默认值掩盖。"""
    # 闸门：阶段名不认识就直接崩，不许静默 PASS。
    # 这条是为了堵住一类真实事故——变量被覆盖导致 stage 变成关节名，
    # THRESHOLDS.get(未知, {}) 返回空字典，于是所有检查都不执行、结果全 PASS。
    if stage not in STAGES:
        raise ValueError(
            f"judge 收到未知阶段 {stage!r}；合法值：{sorted(STAGES)}。"
            " 如果这是关节名，说明 run_stage 里的循环变量把阶段名覆盖了。"
        )
    reasons: list[str] = []
    common = THRESHOLDS["common"]

    if m["terminated"]:
        reasons.append("跌倒")
    if m.get("duration_ratio", 0.0) < common["duration_ratio"]:
        reasons.append(f"未跑满时长 ({m.get('duration_ratio', 0):.2f})")
    up = m.get("min_upright")
    if up is None or not np.isfinite(up) or up < common["min_upright"]:
        reasons.append(f"姿态 {up}")

    def hi(key: str, th: float, label: str, fmt: str = "{:.3f}") -> None:
        val = m.get(key)
        if val is None or not np.isfinite(val):
            reasons.append(f"{label} 无数据")
        elif val > th:
            reasons.append(f"{label} {fmt.format(val)} > {fmt.format(th)}")

    def lo(key: str, th: float, label: str, fmt: str = "{:.3f}") -> None:
        val = m.get(key)
        if val is None or not np.isfinite(val):
            reasons.append(f"{label} 无数据")
        elif val < th:
            reasons.append(f"{label} {fmt.format(val)} < {fmt.format(th)}")

    def hi_abs(key: str, th: float, label: str, fmt: str = "{:.3f}") -> None:
        """对有符号量取绝对值再比。

        这是一个真实事故的产物：航向偏差有正负，用 hi() 直接比 val > th 时，
        -34.7° > 8° 恒为 False，于是**所有负向航向偏差永远通过**。
        这种 bug 只会漏判不会误判，方向是最危险的那种。
        """
        val = m.get(key)
        if val is None or not np.isfinite(val):
            reasons.append(f"{label} 无数据")
        elif abs(val) > th:
            reasons.append(f"{label} {fmt.format(abs(val))} > {fmt.format(th)}")

    th = THRESHOLDS.get(stage, {})

    if stage in ("stand", "stand_long"):
        lo("still_coverage", th["still_coverage"], "零指令覆盖率", "{:.0%}")
        hi("still_turn_dps", th["still_turn_dps"], "静止自转", "{:.2f} °/s")
        hi("still_drift_mps", th["still_drift_mps"], "静止漂移", "{:.3f} m/s")
        hi("tilt_rms_deg", th["tilt_rms_deg"], "倾角 RMS", "{:.2f}°")
        hi("tilt_max_deg", th["tilt_max_deg"], "倾角峰值", "{:.1f}°")

    if stage in ("forward", "backward", "accel",
                 "forward_fast_08", "forward_fast_10"):
        hi("speed_rel_err", th["speed_rel_err"], "速度误差", "{:.0%}")
        if stage != "accel":
            lo("distance_ratio", th["distance_ratio"], "位移比", "{:.0%}")
            hi("lateral_drift_m", th["lateral_drift_m"], "横向漂移", "{:.2f} m")
            hi_abs("heading_change_deg", th["heading_deg"], "航向偏", "{:.1f}°")
        if stage == "forward":
            rt = m.get("rise_time_s")
            if rt is None or not np.isfinite(rt):
                reasons.append("上升时间 未达到 90%")
            else:
                if rt > th["rise_time_s"]:
                    reasons.append(f"上升时间 {rt:.2f}s > {th['rise_time_s']:.2f}s")
                hi("overshoot", th["overshoot"], "超调", "{:.0%}")

    if stage in ("turn", "turn_left", "turn_right"):
        lo("yaw_sign_ratio", th["min_sign_ratio"], "方向正确率", "{:.0%}")
        hi("yaw_mae", th["yaw_mae"], "yaw MAE")

    if stage == "turn_in_motion":
        hi("yaw_mae_turn", th["yaw_mae_turn"], "转弯段 yaw MAE")
        hi("speed_rel_err", th["speed_rel_err_turn"], "直行段速度误差", "{:.0%}")

    if stage == "start_stop":
        lo("n_cycles", th["n_cycles"], "启停次数", "{:.0f}")
        bad = [r for r in (m.get("rise_times_s") or []) if r is None or r > th["rise_time_s"]]
        if bad:
            reasons.append(f"上升时间 {m.get('rise_times_s')} 超 {th['rise_time_s']:.2f}s")
        bad_d = [d for d in (m.get("stop_distances_m") or []) if d is None or d > th["stop_distance_m"]]
        if bad_d:
            reasons.append(f"停车距离 {m.get('stop_distances_m')} 超 {th['stop_distance_m']:.2f}m")

    # 高度：只判与基准无关的波动/标准差
    if stage in ("forward", "backward", "turn_in_motion", "accel"):
        h = THRESHOLDS["height"]
        hi("height_ripple_m", h["ripple_m"], "高度波动", "{:.3f} m")
        hi("height_std_m", h["std_m"], "高度标准差", "{:.3f} m")

    return (len(reasons) == 0), reasons


# ---------------------------------------------------------------- 输出

def to_markdown(rows: list[dict[str, Any]]) -> str:
    head = (
        "| stage | seed | 通过 | 时长 | min_up | 倾角RMS | 指令v | 实测v | 速度误差 "
        "| 位移比 | 轮速斜率 | 横漂 | 航向 | yaw MAE | 高度波动 | 原因 |"
    )
    lines = [head, "|" + "---|" * 16]
    for r in rows:
        def g(key: str, fmt: str = "{:.3f}", dash: str = "—") -> str:
            val = r.get(key)
            if val is None:
                return dash
            try:
                return fmt.format(val)
            except Exception:
                return str(val)

        lines.append(
            "| {s} | {seed} | {ok} | {dur} | {up} | {tilt} | {cv} | {vt} | {se} | {dr} "
            "| {sl} | {lat} | {hd} | {ym} | {rip} | {why} |".format(
                s=r["stage"], seed=r["seed"], ok="✅" if r.get("pass") else "❌",
                dur=f"{r.get('duration_s', 0):.1f}",
                up=g("min_upright"), tilt=g("tilt_rms_deg", "{:.2f}"),
                cv=g("cmd_v_mean_steady", "{:.2f}"),
                vt=g("v_true_mean_steady", "{:.2f}"),
                se=g("speed_rel_err", "{:.0%}"), dr=g("distance_ratio", "{:.0%}"),
                sl=g("wheel_vx_slope", "{:.2f}"), lat=g("lateral_drift_m", "{:.2f}"),
                hd=g("heading_change_deg", "{:.1f}"), ym=g("yaw_mae"),
                rip=g("height_ripple_m", "{:.3f}"),
                why="; ".join(r.get("reasons") or []) or "—",
            )
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="A 阶段应用级诊断 diag-2")
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--resolved", type=Path, default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seeds", nargs="+", type=int, default=[2002])
    ap.add_argument("--only", nargs="+", choices=tuple(STAGES), default=None)
    ap.add_argument("--ramp", type=float, default=0.0,
                    help="指令斜坡秒数；训练用 0.35，默认 0 是真阶跃")
    ap.add_argument("--wheel-radius", type=float, default=0.1)
    ap.add_argument("--video", action="store_true")
    args = ap.parse_args()

    model_path: Path = args.model
    resolved_path = args.resolved
    if resolved_path is None:
        for cand in (model_path.parent / "resolved.json",
                     model_path.parent.parent / "resolved.json",
                     model_path.with_name("resolved.json")):
            if cand.exists():
                resolved_path = cand
                break
    if resolved_path is None or not resolved_path.exists():
        raise SystemExit("找不到 resolved.json，请用 --resolved 指定")

    resolved = json.loads(resolved_path.read_text())
    model = load_policy(model_path)
    names = args.only or list(DEFAULT_STAGES)

    print(f"[{VERSION}] model={model_path}")
    print(f"[{VERSION}] resolved={resolved_path}  ramp={args.ramp}s  "
          f"wheel_radius={args.wheel_radius}m")

    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    t0 = time.monotonic()

    for seed in args.seeds:
        for name in names:
            spec = STAGES[name]
            env = build_env(resolved)
            try:
                video_dir = (args.out / "video" / f"seed{seed}") if args.video else None
                m = run_stage(
                    env, model, name, spec, seed,
                    ramp=args.ramp, wheel_radius=args.wheel_radius,
                    video_dir=video_dir,
                )
            finally:
                try:
                    env.close()
                except Exception:
                    pass
            rows.append(m)
            print(
                f"[seed {seed}] {name:<14} {'PASS' if m['pass'] else 'FAIL'}  "
                f"dur={m['duration_s']:.1f}s  up={m['min_upright']:.3f}  "
                f"位移={m['displacement_m']:.2f}m/{m['expected_distance_m']:.2f}m  "
                f"hip={m.get('q_left_hip_motor_mean', float('nan')):+.2f} "
                f"knee={m.get('q_left_knee_motor_mean', float('nan')):+.2f}  "
                f"roll={m.get('roll_mean_deg', float('nan')):+.1f}° "
                f"pitch={m.get('pitch_mean_deg', float('nan')):+.1f}° "
                f"轮差={m.get('wheel_z_diff_mean_mm', float('nan')):+.0f}mm  "
                f"{'; '.join(m['reasons']) if m['reasons'] else 'ok'}"
            )

    wall = time.monotonic() - t0
    summary = {
        "version": VERSION,
        "model": str(model_path),
        "resolved": str(resolved_path),
        "seeds": args.seeds,
        "stages": names,
        "ramp_s": args.ramp,
        "wheel_radius": args.wheel_radius,
        "wall_seconds": wall,
        "thresholds": THRESHOLDS,
        "rows": rows,
    }
    (args.out / "diag.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    md = to_markdown(rows)
    (args.out / "diag.md").write_text(md + "\n")

    n_pass = sum(1 for r in rows if r["pass"])
    print(f"\n{len(rows)} 次运行，通过 {n_pass}，墙钟 {wall:.1f}s")
    print(f"输出：{args.out}/diag.json 与 diag.md")
    return 0 if n_pass == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
