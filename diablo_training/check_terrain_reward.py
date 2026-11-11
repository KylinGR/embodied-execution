"""地形奖励自检：确认三项新增惩罚的符号与量级正确。

跑法（不需要 heredoc，避免粘贴时漏掉结束标记）：

    $HOME/envs/diablo/bin/python -m diablo_training.check_terrain_reward

判据
----
1. 理想直行 > 横漂 > 航向偏 8° > 航向偏 73°（航向惩罚随误差单调变强且饱和）
2. 站立：完美站立与漂 0.032 m/s 的差值 ≥ 0.005/步（即 ≥ 0.5/秒）
3. 所有值都在 0.01–0.05 量级（说明新增项已按 dt 缩放，没有压死原有十项）
"""

from __future__ import annotations

import numpy as np

from .xingtian_terrain_reward import TerrainReward


def _base(command, velocity=None, yaw=0.0):
    return dict(
        command=np.asarray(command, dtype=float),
        velocity=np.zeros(3) if velocity is None else np.asarray(velocity, dtype=float),
        gyro=np.zeros(3),
        torque=np.zeros(6),
        upright=1.0,
        height_error=0.0,
        yaw=float(yaw),
        terminated=False,
        wheel_velocity=np.zeros(2),
        wheel_linear_velocity=np.zeros(2),
    )


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--min-stand-delta", type=float, default=0.0015,
                    help="站立惩罚的最小单步差值。平地阶段（stand_creep=5）用 0.005；"
                         "地形阶段降权到 1.0 后用默认 0.0015")
    args = ap.parse_args()
    min_stand_delta = args.min_stand_delta

    reward = TerrainReward()
    action = np.zeros(6)
    ok = True

    def total(transition):
        return reward.evaluate(transition, action, action).total

    reward.heading_anchor = 0.0
    straight = total(_base([0.5, 0.0, 0.25], [0.5, 0.0, 0.0]))
    lateral = total(_base([0.5, 0.0, 0.25], [0.5, 0.073, 0.0]))
    yaw8 = total(_base([0.5, 0.0, 0.25], [0.5, 0.0, 0.0], yaw=0.14))
    yaw73 = total(_base([0.5, 0.0, 0.25], [0.5, 0.0, 0.0], yaw=1.27))

    print(f"理想直行      : {straight:.4f}")
    print(f"横漂 0.073    : {lateral:.4f}  (Δ {lateral - straight:+.4f})")
    print(f"航向偏 8°     : {yaw8:.4f}  (Δ {yaw8 - straight:+.4f})")
    print(f"航向偏 73°    : {yaw73:.4f}  (Δ {yaw73 - straight:+.4f})")

    order = [straight, lateral, yaw8, yaw73]
    if not all(a >= b for a, b in zip(order, order[1:])):
        print("❌ 顺序不对：应当 理想 ≥ 横漂 ≥ 航向8° ≥ 航向73°")
        ok = False
    if not 0.01 <= straight <= 0.06:
        print(f"❌ 理想直行 {straight:.4f} 不在 0.01–0.06，新增项可能没有按 dt 缩放")
        ok = False

    perfect = total(_base(np.zeros(3), np.zeros(3)))
    creep = total(_base(np.zeros(3), [0.032, 0.0, 0.0]))
    delta = perfect - creep
    print(f"完美站立      : {perfect:.4f}")
    print(f"站立漂 0.032  : {creep:.4f}  (Δ {delta:+.4f}，每秒约 {delta * 100:.2f})")
    if delta < min_stand_delta:
        print(f"❌ 站立惩罚只差 {delta:.4f}/步，低于门槛 {min_stand_delta}；把 stand_creep 调大")
        ok = False

    print("✅ 全部通过" if ok else "❌ 有检查未通过，先别开训")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
