"""姿态体检：跑一个剖面，把俯仰/横滚/高度/各分项力矩打印成可判断的表。

"姿态好不好"要看四件事，这里一次给全：
    1. 稳态俯仰有没有偏置（该不该前倾由加速度决定，不是常数）
    2. 俯仰的动态幅度（起步、制动、转弯各多大）
    3. 横滚耦合（转弯时一侧是不是被压下去了）
    4. 机身高度波动（腿 PD 撑不撑得住）

用法：
  BALANCE_MODE=analytic LEG_MODE=pd python -m diablo_training.posture_probe \
      --resolved runs/A-fix14-tilt15/selection/resolved.json --stage forward --trace
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("BALANCE_MODE", "analytic")
os.environ.setdefault("LEG_MODE", "pd")

from diablo_training import balance  # noqa: E402
from diablo_training import xingtian_diag2 as D  # noqa: E402


class _ZeroPolicy:
    def predict(self, obs, deterministic=True):  # noqa: ARG002
        return np.zeros(6, dtype=np.float32), None


def _attitude(env) -> tuple[float, float, float]:
    """返回 (pitch, roll, base_z)，单位 度 / 米。"""
    rot = np.asarray(env.data.xmat[env.base_id]).reshape(3, 3)
    pitch = np.degrees(np.arctan2(-rot[2, 0], rot[2, 2]))
    roll = np.degrees(np.arctan2(rot[2, 1], rot[2, 2]))
    return float(pitch), float(roll), float(env.data.xpos[env.base_id][2])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolved", required=True, type=Path)
    ap.add_argument("--stage", default="forward")
    ap.add_argument("--seed", type=int, default=2002)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--every", type=float, default=0.25)
    ap.add_argument("--set", action="append", default=[],
                    help="覆盖 balance 常数，如 STOP.k_v=60")
    args = ap.parse_args()

    for item in args.set:
        key, _, val = item.partition("=")
        if "." in key:
            grp, name = key.split(".", 1)
            getattr(balance, grp)[name] = float(val)
        else:
            setattr(balance, key, float(val))

    resolved = json.loads(args.resolved.read_text(encoding="utf-8"))
    spec = D.STAGES[args.stage]
    env = D.build_env(resolved)
    D.install_profile(env, spec["profile"], 0.0)
    env.config.episode_seconds = float(spec["duration"]) + 1.0
    env.reset(seed=args.seed, options={})

    dt = float(getattr(env.config, "control_dt_s", 0.01) or 0.01)
    n = int(round(spec["duration"] / dt))
    rows = []
    for i in range(n):
        _obs, _r, term, trunc, _info = env.step(np.zeros(6))
        pitch, roll, z = _attitude(env)
        v = balance.LAST.get("v", float("nan"))
        cmd = float(np.asarray(env.command).reshape(-1)[0])
        rows.append({
            "t": i * dt, "cmd": cmd, "v": v, "pitch": pitch, "roll": roll, "z": z,
            **{k: balance.LAST.get(k, float("nan")) for k in
               ("thd", "term_th", "term_thd", "term_s", "term_v", "term_i",
                "i_th", "tw", "braking", "k_th", "k_v")},
        })
        if term or trunc:
            break

    t = np.array([r["t"] for r in rows])
    pitch = np.array([r["pitch"] for r in rows])
    roll = np.array([r["roll"] for r in rows])
    z = np.array([r["z"] for r in rows])
    v = np.array([r["v"] for r in rows])
    cmd = np.array([r["cmd"] for r in rows])

    print(f"\n=== {args.stage} ===  {len(rows)} 步 ({t[-1]:.1f}s)")
    print(f"  俯仰  mean {pitch.mean():+6.2f}°  std {pitch.std():5.2f}  "
          f"min {pitch.min():+6.2f}  max {pitch.max():+6.2f}")
    print(f"  横滚  mean {roll.mean():+6.2f}°  std {roll.std():5.2f}  "
          f"|max| {np.abs(roll).max():5.2f}")
    print(f"  机身  z mean {z.mean():.3f} m  min {z.min():.3f}  max {z.max():.3f}  "
          f"峰峰 {(z.max() - z.min()) * 1000:.0f} mm")
    print(f"  速度  指令 {np.abs(cmd).mean():.2f}  实测 {v.mean():+.2f}  "
          f"std {v.std():.3f}")

    # 停车距离（与 xingtian_diag2 同口径）：指令归零后，|v| 累计到掉下 0.05 为止
    moved = np.abs(cmd) > 0.05
    stops = np.where(~moved[1:] & moved[:-1])[0] + 1
    if stops.size:
        dists = []
        for s in stops:
            seg = np.abs(v[s:s + int(round(3.0 / dt))])
            below = np.where(seg < 0.05)[0]
            dists.append(float(seg[:below[0] + 1].sum() * dt) if below.size else float("nan"))
        print(f"  停车  {' '.join(f'{d:.4f}' for d in dists)} m"
              f"   （俯仰最低 {pitch.min():+.1f}° / 最高 {pitch.max():+.1f}°）")

    # 稳态（最后 3 秒且无制动）的力矩分解：看俯仰偏置被谁撑着
    steady = (t > t[-1] - 3.0) & (np.array([r["braking"] for r in rows]) < 0.5)
    if steady.sum() > 20:
        print("  稳态力矩分解（N·m，同向单轮）：")
        for key, label in (("term_th", "kθ·θ"), ("term_thd", "kθ̇·θ̇"),
                           ("term_s", "kx·Δs"), ("term_v", "kv·Δv"),
                           ("term_i", "ki·∫θ")):
            vals = np.array([r[key] for r in rows])[steady]
            print(f"    {label:<10} {vals.mean():+7.3f}   (std {vals.std():.3f})")
        print(f"    合计 tw    {np.array([r['tw'] for r in rows])[steady].mean():+7.3f}")
        print(f"    ∫θ {np.array([r['i_th'] for r in rows])[steady].mean():+.3f} rad·s"
              f"  (限幅 ±{balance.I_TH_LIMIT})")

    if args.trace:
        stride = max(1, int(round(args.every / dt)))
        print(f"\n  {'t':>5} {'cmd':>5} {'v':>6} {'pitch':>7} {'roll':>6} {'z':>6} "
              f"{'tw':>6} {'kθ·θ':>6} {'kv·Δv':>6} {'ki·∫θ':>6}")
        for i in range(0, len(rows), stride):
            r = rows[i]
            print(f"  {r['t']:>5.2f} {r['cmd']:>5.2f} {r['v']:>+6.2f} "
                  f"{r['pitch']:>+7.2f} {r['roll']:>+6.2f} {r['z']:>6.3f} "
                  f"{r['tw']:>+6.2f} {r['term_th']:>+6.2f} {r['term_v']:>+6.2f} "
                  f"{r['term_i']:>+6.2f}")
    try:
        env.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
