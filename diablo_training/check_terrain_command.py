"""检查地形阶段的指令调度器是否真的覆盖了验收剖面。

判据
----
A 阶段的验收剖面是"恒定速度连跑 14 秒"，而仓库自带调度器每段只有 2–6 s。
如果新调度器里长段（≥10 s）的占比太低，说明覆盖仍然不足，训练出来的策略
在验收剖面上仍会分布外失败。

跑法：
    python -m diablo_training.check_terrain_command
"""

from __future__ import annotations

import argparse

import numpy as np

from .xingtian_terrain_reward import TerrainCommandScheduler


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-long-fraction", type=float, default=0.35,
                    help="长段（≥10 s）应占时间比例的下限")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    scheduler = TerrainCommandScheduler(rng)
    steps = int(args.seconds / 0.01)
    segments: list[tuple[float, float, float]] = []
    previous = None
    for i in range(steps):
        scheduler.command_at(i * 0.01)
        marker = id(scheduler.target)
        if marker != previous:
            segments.append(
                (float(scheduler.target[0]), float(scheduler.target[1]),
                 float(scheduler.duration))
            )
            previous = marker

    durations = np.array([s[2] for s in segments])
    forward = np.array([s[0] for s in segments])
    yaw = np.array([s[1] for s in segments])

    long_mask = durations >= 10.0
    hold_mask = long_mask & (np.abs(yaw) < 1e-9) & (np.abs(forward) >= 0.3)
    long_fraction = float(durations[long_mask].sum() / durations.sum())
    hold_fraction = float(durations[hold_mask].sum() / durations.sum())

    print(f"总时长        : {args.seconds:.0f} s，段数 {len(segments)}")
    print(f"段长 平均/最短/最长: {durations.mean():.1f} / {durations.min():.1f} / {durations.max():.1f} s")
    print(f"长段(>=10 s)  : {long_mask.sum()} 段，占时间 {long_fraction:.0%}")
    print(f"其中定速直行段: {hold_mask.sum()} 段，占时间 {hold_fraction:.0%}")
    print(f"速度范围      : {forward.min():+.2f} ~ {forward.max():+.2f} m/s")
    print(f"偏航范围      : {yaw.min():+.2f} ~ {yaw.max():+.2f} rad/s")

    ok = True
    if long_fraction < args.min_long_fraction:
        print(f"❌ 长段只占 {long_fraction:.0%}，低于 {args.min_long_fraction:.0%}，覆盖不足")
        ok = False
    if hold_mask.sum() < 3:
        print("❌ 定速直行段少于 3 个，验收剖面仍有分布外风险")
        ok = False
    print("✅ 调度器覆盖验收剖面" if ok else "❌ 覆盖不足")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
