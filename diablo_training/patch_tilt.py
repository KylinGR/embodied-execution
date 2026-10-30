"""加一条「站立时倾斜角线性惩罚」tilt_hold。

为什么要这一条：

    upright 的核是 exp(-max(0, 1-cos)²/0.02)。(1-cos) 在 19° 处只有 0.055，
    平方后 0.003，除以 0.02 —— 惩罚几乎为零。reward_audit 实测：
        站立，前倾 15°  →  0.04×forward（基本没有信号）
    之前三次调 upright 都失败，就是因为在调一个天生在小角度处梯度为零的核。

    sin(倾角) 在 19° 是 0.326，比 0.055 大 6 倍，且小角度处梯度恒定。
    权重 0.3、分母 0.15 时的信号强度：
        8°  → 0.23×    15° → 0.43×    19° → 0.54×

    门控：只在「指令站着不动」时生效（|cmd_vx|<0.05 且 |cmd_yaw|≤0.1），
    这样前进/转弯时机器人的正常前倾不会被误罚，不会拖坏 ④⑥。

    upright = cos(倾角)，所以 sin(倾角) = sqrt(1 - upright²)，
    不需要新的观测量，也不改 obs 维度（125 维红线安全）。

用法：
    python -m diablo_training.patch_tilt                 # 改默认文件
    python -m diablo_training.patch_tilt 某个文件.py     # 改指定文件
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

DEFAULT = Path("diablo_training/xingtian.py")

WEIGHT_OLD = "    upright: float = 0.8\n"
WEIGHT_NEW = "    upright: float = 0.8\n    tilt: float = 0.3\n"

UPRIGHT_OLD = '        upright = float(np.clip(transition["upright"], -1.0, 1.0))\n'
UPRIGHT_NEW = (
    '        upright = float(np.clip(transition["upright"], -1.0, 1.0))\n'
    "        # upright 是 cos(倾角)，所以下面这个就是 sin(倾角)。\n"
    "        tilt_sin = float(np.sqrt(max(0.0, 1.0 - upright * upright)))\n"
)

ANCHOR_OLD = '            "height": -w.height * abs(height_error) / 0.05,\n'
ANCHOR_NEW = (
    "            # 站立时直接惩罚倾斜角：sin 在小角度有恒定梯度，(1-cos) 没有。\n"
    "            # 只作用于「指令站立」，避免干扰前进/转弯时的正常前倾。\n"
    '            "tilt_hold": (\n'
    "                -w.tilt * tilt_sin / 0.15\n"
    "                if abs(command[0]) < 0.05 and abs(command[1]) <= 0.1 else 0.0\n"
    "            ),\n"
    '            "height": -w.height * abs(height_error) / 0.05,\n'
)


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    if not path.exists():
        print(f"[失败] 找不到 {path}")
        return 1

    text = path.read_text(encoding="utf-8")

    if "tilt_hold" in text:
        print("[跳过] 已经打过 tilt_hold 补丁")
        return 0

    for name, old in (("RewardWeights.upright", WEIGHT_OLD),
                      ("upright 赋值", UPRIGHT_OLD),
                      ("height 项锚点", ANCHOR_OLD)):
        n = text.count(old)
        if n != 1:
            print(f"[失败] {name} 期望匹配 1 处，实际 {n} 处 —— 文件已变，先别打补丁")
            return 1

    new = text.replace(WEIGHT_OLD, WEIGHT_NEW, 1)
    new = new.replace(UPRIGHT_OLD, UPRIGHT_NEW, 1)
    new = new.replace(ANCHOR_OLD, ANCHOR_NEW, 1)

    try:
        ast.parse(new)
    except SyntaxError as exc:
        print(f"[失败] 打完补丁语法不通：{exc}")
        return 1

    path.write_text(new, encoding="utf-8")
    print(f"[完成] {path} 已加入 tilt_hold（权重 0.3 / 分母 0.15 / 仅站立时生效）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
