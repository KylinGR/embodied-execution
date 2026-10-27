"""奖励体检：用合成 transition 量出每一项的实际贡献，训练前先跑。

为什么需要
----------
前五轮里有三轮是我把某一项调过头或写错语义，每次代价是一次 16 分钟训练：
  · A-fix3  upright 收紧 100 倍  → forward 位移比 110% → 11%
  · A-fix4  upright 按指令分支   → 跌倒 10/24
  · A-fix5  position_hold 相对出发点 → forward 位移比 2%（策略不敢动）

问题出在：这些都可以在**训练前用几行算术发现**。这个脚本不跑物理，
直接构造 transition 喂给奖励函数，看每一项占多少。

它用"扰动法"而不是读内部字段：给定基线，单独改一个量，看总分变化多少。
这样不依赖 LocomotionReward 的内部结构，代码也不会和实现漂移。

用法：
  python -m diablo_training.reward_audit
"""

from __future__ import annotations

import numpy as np

try:
    from diablo_training.control import CONTROL_DT
except Exception:
    try:
        from diablo_training.env import CONTROL_DT
    except Exception:
        CONTROL_DT = 0.01

from diablo_training.xingtian import LocomotionReward, RewardWeights


def base_transition(**over) -> dict:
    """一个"完美静止"的 transition，按需覆盖字段。"""
    t = {
        "velocity": np.zeros(6),
        "command": np.zeros(3),
        "gyro": np.zeros(3),
        "upright": 1.0,
        "torque": np.zeros(6),
        "height_error": 0.0,
        "heading_error": 0.0,
        "pos_error": 0.0,
        "joint_power": np.zeros(4),
        "joint_acceleration": np.zeros(4),
        "soft_limit_excess": np.zeros(4),
        "wheel_velocity": np.zeros(2),
        "wheel_linear_velocity": np.zeros(2),
        "terminated": False,
    }
    t.update(over)
    return t


def total(reward, transition) -> float:
    zero = np.zeros(6)
    out = reward.evaluate(transition, zero, zero, previous_difference=zero)
    return float(getattr(out, "total", out))


def main() -> None:
    reward = LocomotionReward(RewardWeights())
    zero = np.zeros(6)
    baseline = reward.evaluate(base_transition(), zero, zero, previous_difference=zero)
    r0 = float(getattr(baseline, "total", baseline))

    print("=" * 74)
    print("奖励体检：单项扰动的每步影响（缩放前 / 缩放后 = ×CONTROL_DT）")
    print("=" * 74)

    terms = getattr(baseline, "terms", None)
    if isinstance(terms, dict) and terms:
        print("\n  基线（完美静止）各分项：")
        for k in sorted(terms, key=lambda n: -abs(float(terms[n]))):
            print(f"    {k:<24}{float(terms[k]):+.5f}")
    print(f"\n  完美静止基线合计：{r0:+.5f}/步\n")

    cmd_fwd = np.array([0.5, 0.0, 0.25])
    cmd_stand = np.array([0.0, 0.0, 0.25])

    cases = [
        ("站着，航向偏 0.20 rad (11°)", dict(command=cmd_stand, heading_error=0.20)),
        ("站着，航向偏 1.00 rad (57°)", dict(command=cmd_stand, heading_error=1.00)),
        ("站着，位置偏 0.20 m", dict(command=cmd_stand, pos_error=0.20)),
        ("站着，位置偏 0.50 m", dict(command=cmd_stand, pos_error=0.50)),
        ("停下时已离出发点 2.00 m", dict(command=cmd_stand, pos_error=2.00)),
        ("站着，自转 0.30 rad/s", dict(command=cmd_stand, gyro=np.array([0, 0, 0.30]))),
        ("前进 0.5 指令，实际 0.34", dict(command=cmd_fwd, velocity=np.array([0.34, 0, 0, 0, 0, 0]))),
        ("前进 0.5 指令，完全不动", dict(command=cmd_fwd)),
        ("巡航，航向偏 1.00 rad", dict(command=cmd_fwd,
                                    velocity=np.array([0.5, 0, 0, 0, 0, 0]),
                                    heading_error=1.00)),
        ("巡航，前倾 8° (upright 0.990)", dict(command=cmd_fwd,
                                            velocity=np.array([0.5, 0, 0, 0, 0, 0]),
                                            upright=0.9903)),
        ("站立，前倾 15° (upright 0.966)", dict(command=cmd_stand, upright=0.9659)),
        ("高度误差 0.05 m", dict(command=cmd_stand, height_error=0.05)),
        ("高度误差 0.15 m", dict(command=cmd_stand, height_error=0.15)),
    ]

    print(f"  {'情形':<34}{'Δ每步(已含dt)':>16}{'未缩放':>14}{'倍数':>9}")
    print("  " + "-" * 34 + "-" * 16 + "-" * 14 + "-" * 9)
    ref = abs(RewardWeights().forward_tracking * CONTROL_DT)  # forward 项理论上限

    rows = []
    for label, over in cases:
        # .total 与 terms 已经是乘过 CONTROL_DT 的每步值，不要再乘
        d = total(reward, base_transition(**over)) - r0
        rows.append((label, d))

    for label, d in rows:
        ratio = abs(d) / ref if ref > 0 else float("nan")
        flag = "  ← 过强" if ratio > 1.5 else ("  ← 偏强" if ratio > 0.8 else "")
        print(f"  {label:<34}{d:>+16.6f}{d / CONTROL_DT:>+14.4f}{ratio:>8.2f}×{flag}")

    print(f"\n  参照：forward_tracking 理论上限 = {ref:+.5f}/步（权重 "
          f"{RewardWeights().forward_tracking} × dt {CONTROL_DT}）")
    print("\n  判读规则：任何一项在**正常工况下**超过 1.5× forward 上限，")
    print("  就说明它会压死运动能力——这正是 A-fix3/A-fix5 失败的方式。")


if __name__ == "__main__":
    main()
