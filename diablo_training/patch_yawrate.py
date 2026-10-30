"""加一条「直行/站立时的偏航角速度线性惩罚」yaw_rate_hold。

为什么需要（师兄的判断 + 我们的数据）：

    师兄说"直行不行是平衡没搞好"。数据支持：
        站立自转 3.6 / 4.4 / 5.5 / 10.3 °/s
        forward 航向 ≤10° / 4° / 29° / 206°      ← 单调恶化

    代码缺口：
        "angular_velocity_xy": -w * sum(gyro[:2]**2)     只有 roll/pitch
        "yaw_tracking" 的静止分支: clip(1 - |gyro_z|/0.30)  ← 饱和核
    偏航角速度超过 0.3 rad/s（17 °/s）后梯度为零，而实测就在 4-10 °/s 附近游走。

    这和 tilt_hold 是同一个病：饱和核在小/中误差处梯度过弱。
    tilt_hold 把 (1-cos) 换成 sin 线性后，倾角 19.3° → 11.5°。

改动：
    - RewardWeights 加字段 yaw_rate（默认 0.3）
    - terms 加 "yaw_rate_hold"，仅在 |cmd_yaw| <= 0.1 时生效
      （即直行和站立时惩罚偏航角速度；主动转向时不受影响）

量级（相对 forward 上限 0.012/步）：
    5 °/s → 0.15×    10 °/s → 0.29×    20 °/s → 0.58×

用法：
    python -m diablo_training.patch_yawrate
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

DEFAULT = Path("diablo_training/xingtian.py")

WEIGHT_OLD = "    tilt: float = 0.15\n"
WEIGHT_NEW = "    tilt: float = 0.15\n    yaw_rate: float = 0.3\n"

ANCHOR_OLD = (
    '            "vertical_velocity": -w.vertical_velocity * float(velocity[2] ** 2),\n'
)
ANCHOR_NEW = (
    "            # 直行/站立时惩罚偏航角速度：线性、永不饱和。\n"
    "            # yaw_tracking 的静止分支是 clip(1-|wz|/0.30)，超过 0.3 rad/s\n"
    "            # 梯度为零；而实测自转就在 4-10 °/s（0.07-0.17 rad/s）附近。\n"
    "            # 主动转向时（|cmd_yaw|>0.1）不生效，不影响原地转向。\n"
    '            "yaw_rate_hold": (\n'
    "                -w.yaw_rate * abs(gyro[2]) / 0.15\n"
    "                if abs(command[1]) <= 0.1 else 0.0\n"
    "            ),\n"
    '            "vertical_velocity": -w.vertical_velocity * float(velocity[2] ** 2),\n'
)


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    if not path.exists():
        print(f"[失败] 找不到 {path}")
        return 1

    text = path.read_text(encoding="utf-8")

    if "yaw_rate_hold" in text:
        print("[跳过] 已经打过 yaw_rate_hold 补丁")
        return 0

    for name, old in (("RewardWeights.tilt", WEIGHT_OLD),
                      ("vertical_velocity 锚点", ANCHOR_OLD)):
        n = text.count(old)
        if n != 1:
            print(f"[失败] {name} 期望匹配 1 处，实际 {n} 处 —— 先别打补丁")
            return 1

    new = text.replace(WEIGHT_OLD, WEIGHT_NEW, 1)
    new = new.replace(ANCHOR_OLD, ANCHOR_NEW, 1)

    try:
        ast.parse(new)
    except SyntaxError as exc:
        print(f"[失败] 语法不通：{exc}")
        return 1

    path.write_text(new, encoding="utf-8")
    print(f"[完成] {path} 已加入 yaw_rate_hold（权重 0.3 / 分母 0.15 / 仅直行与站立时生效）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
