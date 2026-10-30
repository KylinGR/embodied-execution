"""回退 yaw_rate_hold 补丁，把 xingtian.py 恢复到 A-fix16 的状态。

为什么回退：
    A-fix25 实测：自转从 4.43 降到 2.37/3.17（降 28-56%，机制有效），
    但 forward 航向不一致（A27 恶化到 51.2°，A28 达标），
    并且丢掉了 ② 前进 和 ④ 加减速。
    净损失，所以回退，把"能压自转"这个结论单独记录下来。

用法：
    python -m diablo_training.unpatch_yawrate
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

DEFAULT = Path("diablo_training/xingtian.py")

WEIGHT = "    yaw_rate: float = 0.3\n"

BLOCK = (
    "            # 直行/站立时惩罚偏航角速度：线性、永不饱和。\n"
    "            # yaw_tracking 的静止分支是 clip(1-|wz|/0.30)，超过 0.3 rad/s\n"
    "            # 梯度为零；而实测自转就在 4-10 °/s（0.07-0.17 rad/s）附近。\n"
    "            # 主动转向时（|cmd_yaw|>0.1）不生效，不影响原地转向。\n"
    '            "yaw_rate_hold": (\n'
    "                -w.yaw_rate * abs(gyro[2]) / 0.15\n"
    "                if abs(command[1]) <= 0.1 else 0.0\n"
    "            ),\n"
)


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    if not path.exists():
        print(f"[失败] 找不到 {path}")
        return 1

    text = path.read_text(encoding="utf-8")

    if "yaw_rate_hold" not in text:
        print("[跳过] 没有 yaw_rate_hold，已经是干净的")
        return 0

    new = text
    for name, chunk in (("RewardWeights.yaw_rate", WEIGHT), ("yaw_rate_hold 块", BLOCK)):
        n = new.count(chunk)
        if n != 1:
            print(f"[失败] {name} 期望匹配 1 处，实际 {n} 处 —— 请手工处理，不要盲删")
            return 1
        new = new.replace(chunk, "", 1)

    if "yaw_rate_hold" in new:
        print("[失败] 替换后仍有残留，中止")
        return 1

    try:
        ast.parse(new)
    except SyntaxError as exc:
        print(f"[失败] 语法不通：{exc}")
        return 1

    path.write_text(new, encoding="utf-8")
    print(f"[完成] {path} 已回退到 A-fix16 状态（无 yaw_rate_hold）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
