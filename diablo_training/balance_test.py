"""轮腿平衡控制原型验证（不碰 env.py，不训练）。

目的：
    11 次端到端 RL 尝试都失败在"平衡"上。开源项目（RM 平衡步兵、circleg、
    palette2 等）几乎都用"基于模型的状态反馈"做平衡：腿用 PD 固定，轮子由
    线性状态反馈（LQR 结构）驱动。

    本脚本先回答一个最小问题：
        **腿用 PD 固定在默认姿态 + 轮子用 4 状态线性反馈，这台机器人能不能站住？**

    如果能，那 RL 只需要学"腿高、转向、速度参考"这些上层决策，
    平衡不用它管——问题会被拆开。

状态反馈形式（就是 LQR 的结构，增益用手工网格搜而非解 Riccati）：
    τ = -( kθ·θ + kθ̇·θ̇ + kx·(x - x_ref) + kv·(ẋ - v_ref) )

用法：
    python -m diablo_training.balance_test --resolved runs/A-fix14-tilt15/selection/resolved.json
    python -m diablo_training.balance_test --resolved ... --vref 0.5     # 测试速度跟踪
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def _rot(data, bid: int) -> np.ndarray:
    return np.asarray(data.xmat[bid]).reshape(3, 3)


def _pitch(data, bid: int) -> float:
    r = _rot(data, bid)
    return float(np.arctan2(-r[2, 0], r[2, 2]))


def _pitch_rate(data, bid: int) -> float:
    r = _rot(data, bid)
    return float(data.cvel[bid][1])


def _body_velocity(model, data, bid: int) -> np.ndarray:
    import mujoco

    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(
        model, data, mujoco.mjtObj.mjOBJ_XBODY, bid, velocity, 0
    )
    return _rot(data, bid).T @ velocity[3:]


def simulate(env, gains, seconds, v_ref=0.0, leg_kp=60.0, leg_kd=2.0,
             yaw_gains=None, w_ref=0.0, turn_clamp=6.0, ki_v=0.0):
    """腿 PD 固定 + 轮子线性状态反馈（可选：偏航差速控制）。"""
    import mujoco

    from diablo_training.model import DEFAULT_Q, PHYSICS_DT

    k_th, k_thd, k_x, k_v = gains
    data, model, bid = env.data, env.model, env.base_id
    k_w, k_psi = (0.0, 0.0) if yaw_gains is None else yaw_gains

    qi = np.asarray(env.qi)
    vi = np.asarray(env.vi)
    leg_idx = [0, 1, 3, 4]
    wheel_idx = [2, 5]

    obs, _ = env.reset(seed=2002, options={})
    x_ref = float(data.xpos[bid][0])
    psi_start = float(np.arctan2(_rot(data, bid)[1, 0], _rot(data, bid)[0, 0]))
    psi_ref = psi_start

    steps = int(round(seconds / PHYSICS_DT))
    min_up = 1.0
    trace = []
    iv = 0.0
    for i in range(steps):
        q = data.qpos[qi].copy()
        dq = data.qvel[vi].copy()

        tau = np.zeros(6)
        for j in leg_idx:
            tau[j] = leg_kp * (DEFAULT_Q[j] - q[j]) - leg_kd * dq[j]

        th = _pitch(data, bid)
        thd = _pitch_rate(data, bid)
        x = float(data.xpos[bid][0])
        v = float(_body_velocity(model, data, bid)[0])

        # 参考轨迹：x_ref 跟着速度指令走，否则位置回路会把"按指令前进"当成误差拉回来
        x_ref += v_ref * PHYSICS_DT
        psi_ref += w_ref * PHYSICS_DT
        iv = float(np.clip(iv + (v - v_ref) * PHYSICS_DT, -5.0, 5.0))

        tw = -(k_th * th + k_thd * thd + k_x * (x - x_ref) + k_v * (v - v_ref)
               + ki_v * iv)
        tw = float(np.clip(tw, -12.0, 12.0))

        turn = 0.0
        if yaw_gains is not None:
            rot = _rot(data, bid)
            psi = float(np.arctan2(rot[1, 0], rot[0, 0]))
            wz = float(data.sensor("gyro").data[2])
            # 航向误差要绕回 ±π
            e_psi = (psi - psi_ref + np.pi) % (2 * np.pi) - np.pi
            turn = float(np.clip(-(k_w * (wz - w_ref) + k_psi * e_psi),
                                 -turn_clamp, turn_clamp))

        tau[wheel_idx[0]] = tw + turn
        tau[wheel_idx[1]] = tw - turn

        data.ctrl[:] = tau
        mujoco.mj_step(model, data)

        up = float(_rot(data, bid)[2, 2])
        min_up = min(min_up, up)
        if (i + 1) % int(round(0.5 / PHYSICS_DT)) == 0:
            rot = _rot(data, bid)
            psi = float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0])))
            roll = float(np.degrees(np.arctan2(rot[2, 1], rot[2, 2])))
            trace.append((round((i + 1) * PHYSICS_DT, 1), np.degrees(th), roll, x, v, psi))
        if not np.isfinite(data.qpos).all() or up < 0.5:
            return {"ok": False, "min_up": min_up, "trace": trace,
                    "pitch_deg": float(np.degrees(th)), "x": x, "v": v,
                    "psi_deg": 0.0, "psi_start": float(np.degrees(psi_start))}

    rot = _rot(data, bid)
    psi_end = float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0])))
    return {"ok": True, "min_up": min_up, "trace": trace,
            "pitch_deg": float(np.degrees(th)), "x": x, "v": v,
            "psi_deg": psi_end, "psi_start": float(np.degrees(psi_start))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolved", required=True, type=Path)
    ap.add_argument("--vref", type=float, default=0.0)
    ap.add_argument("--wref", type=float, default=0.0, help="指令偏航角速度 rad/s")
    ap.add_argument("--yaw", action="store_true", help="搜索偏航增益（平衡增益固定）")
    ap.add_argument("--gains", nargs=4, type=float, default=None,
                    help="直接指定 kθ kθ̇ kx kv（跳过搜索）")
    ap.add_argument("--yawset", nargs=3, type=float, default=None,
                    help="直接指定 kψ̇ kψ 符号（跳过搜索）")
    ap.add_argument("--clamp", type=float, default=2.5, help="差速力矩限幅 N·m")
    ap.add_argument("--ki-v", type=float, default=0.0, help="速度积分增益")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    from diablo_training.xingtian_diag2 import build_env

    resolved = json.loads(args.resolved.read_text(encoding="utf-8"))
    env = build_env(resolved)

    if args.gains is not None:
        g = tuple(args.gains)
        yg = None
        if args.yawset is not None:
            kw, kp_, sg = args.yawset
            yg = (sg * kw, sg * kp_)
        r = simulate(env, g, args.seconds, v_ref=args.vref, yaw_gains=yg,
                     w_ref=args.wref, turn_clamp=args.clamp, ki_v=args.ki_v)
        print(f"直接评估：kθ={g[0]} kθ̇={g[1]} kx={g[2]} kv={g[3]}  "
              f"偏航={yg}  限幅={args.clamp}  vref={args.vref}  wref={args.wref}")
        print(f"站住={r['ok']}  末倾角={r['pitch_deg']:+.2f}°  "
              f"末位移={r['x']:+.3f}m  末速度={r['v']:+.3f}m/s")
        print()
        print("轨迹：时间  pitch°  roll°  x  v  yaw°")
        prev_psi = None
        total_psi = 0.0
        for row in r["trace"]:
            t, th, roll, x, v, psi = row
            if prev_psi is not None:
                d = (psi - prev_psi + 180.0) % 360.0 - 180.0
                total_psi += d
            else:
                total_psi = 0.0
            prev_psi = psi
            print(f"  {t:>5.1f}  {th:>+8.2f} {roll:>+7.2f} {x:>+7.3f} {v:>+7.3f}"
                  f" {psi:>+8.1f}   累计转角 {total_psi:>+8.1f}")
        return 0

    if args.yaw:
        base = (45.0, 6.0, 60.0, 30.0)
        print(f"偏航增益搜索：平衡增益固定 {base}    vref={args.vref:+.2f}  wref={args.wref:+.2f}")
        print()
        results = []
        expected = np.degrees(args.wref * args.seconds)
        for k_w, k_psi, sign, clamp in itertools.product(
            [0.3, 0.8, 2.0], [2.0, 5.0, 10.0], [1.0, -1.0], [1.5, 2.5]
        ):
            r = simulate(env, base, args.seconds, v_ref=args.vref,
                         yaw_gains=(sign * k_w, sign * k_psi), w_ref=args.wref,
                         turn_clamp=clamp)
            if not r["ok"]:
                score = 1e6
            else:
                drift = r["psi_deg"] - r["psi_start"]
                if args.wref == 0.0:
                    score = (abs(r["v"]) * 50 + abs(r["x"]) * 20
                             + abs(r["pitch_deg"]) + abs(drift) * 3)
                else:
                    # 目标是"转够角度且不跑偏"
                    score = (abs(drift - expected) + abs(r["x"]) * 20
                             + abs(r["pitch_deg"]) * 2)
            results.append((score, (k_w, k_psi, sign, clamp), r))
        results.sort(key=lambda t: t[0])
        print(f"{'kψ̇':>6}{'kψ':>6}{'符号':>5}{'限幅':>6}  {'站住':>4}{'末倾角°':>9}"
              f"{'末位移m':>9}{'末速度':>8}{'航向°':>9}{'漂移°':>9}{'目标°':>8}")
        print("-" * 88)
        for _s, g, r in results[: args.top]:
            drift = r["psi_deg"] - r["psi_start"]
            print(f"{g[0]:>6.1f}{g[1]:>6.1f}{g[2]:>5.0f}{g[3]:>6.1f}"
                  f"  {'是' if r['ok'] else '否':>4}"
                  f"{r['pitch_deg']:>9.2f}{r['x']:>9.2f}{r['v']:>8.2f}"
                  f"{r['psi_deg']:>9.1f}{drift:>+9.1f}{expected:>+8.1f}")
        best = results[0]
        print("-" * 88)
        print(f"最优：kψ̇={best[1][0]:.1f} kψ={best[1][1]:.1f} 符号={best[1][2]:.0f} 限幅={best[1][3]:.1f}")
        print("轨迹：时间  pitch°  roll°  x  v  yaw°")
        for row in best[2]["trace"]:
            t, th, roll, x, v, psi = row
            print(f"  {t:>5.1f}  {th:>+8.2f} {roll:>+7.2f} {x:>+7.3f} {v:>+7.3f} {psi:>+8.1f}")
        return 0

    print(f"腿 PD 固定（kp=60 kd=2）+ 轮子状态反馈    指令速度 {args.vref:+.2f} m/s")
    print()

    grid = itertools.product(
        [20.0, 30.0],                # kθ  N·m/rad
        [2.0, 6.0],                  # kθ̇  N·m/(rad/s)
        [30.0, 60.0],                # kx  N·m/m
        [30.0, 60.0, 90.0],          # kv  N·m/(m/s)
        [0.0, 5.0, 20.0],            # ki_v 速度积分（消除稳态误差）
    )

    results = []
    for gains in grid:
        r = simulate(env, gains[:4], args.seconds, v_ref=args.vref, ki_v=gains[4])
        if not r["ok"]:
            score = 1e6
        else:
            # 站立的关键是"别动"：速度权重最高，其次位移，倾角权重最低
            score = abs(r["v"] - args.vref) * 50 + abs(r["x"]) * 20 + abs(r["pitch_deg"])
        results.append((score, gains, r))

    results.sort(key=lambda t: t[0])

    print(f"{'kθ':>7}{'kθ̇':>7}{'kx':>7}{'kv':>7}{'ki_v':>7}  {'站住':>4}"
          f"{'末倾角°':>9}{'末位移m':>9}{'末速度':>8}")
    print("-" * 68)
    for _score, gains, r in results[: args.top]:
        print(f"{gains[0]:>7.0f}{gains[1]:>7.1f}{gains[2]:>7.1f}{gains[3]:>7.1f}"
              f"{gains[4]:>7.1f}"
              f"  {'是' if r['ok'] else '否':>4}"
              f"{r['pitch_deg']:>9.2f}{r['x']:>9.2f}{r['v']:>8.2f}")

    best = results[0]
    print("-" * 68)
    print(f"最优增益：kθ={best[1][0]:.0f} kθ̇={best[1][1]:.1f} "
          f"kx={best[1][2]:.1f} kv={best[1][3]:.1f} ki_v={best[1][4]:.1f}")

    results.sort(key=lambda t: t[0])

    print(f"{'kθ':>7}{'kθ̇':>7}{'kx':>7}{'kv':>7}  {'站住':>4}{'末倾角°':>9}{'末位移m':>9}{'末速度':>8}")
    print("-" * 62)
    for _score, gains, r in results[: args.top]:
        print(f"{gains[0]:>7.0f}{gains[1]:>7.1f}{gains[2]:>7.1f}{gains[3]:>7.1f}"
              f"  {'是' if r['ok'] else '否':>4}"
              f"{r['pitch_deg']:>9.2f}{r['x']:>9.2f}{r['v']:>8.2f}")

    best = results[0]
    print("-" * 62)
    print(f"最优增益：kθ={best[1][0]:.0f} kθ̇={best[1][1]:.1f} "
          f"kx={best[1][2]:.1f} kv={best[1][3]:.1f}")
    print()
    print("轨迹（每 0.5 秒）：时间  pitch°  roll°  x  v  yaw°")
    for row in best[2]["trace"]:
        t, th, roll, x, v, psi = row
        print(f"  {t:>5.1f}  {th:>+8.2f} {roll:>+7.2f} {x:>+7.3f} {v:>+7.3f} {psi:>+8.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
