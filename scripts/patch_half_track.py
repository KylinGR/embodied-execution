"""修正 xingtian_env.py 里的 half_track（轮距）常数。

问题
----
`xingtian_env.py` 的奖励函数里：

    half_track = 0.18755

这个值其实是 URDF 里 `fake_hip_joint` 的 y 偏移（**髋关节间距**），
不是左右轮的**轮心距**。

实测与正运动学
--------------
- 实物：外缘跨度 528 mm − 单轮宽 50 mm ≈ 478 mm
- MuJoCo 正运动学：左轮中心 y=+0.2412、右轮中心 y=−0.2413 → 轮心距 482.5 mm
- 两者差 0.9%，说明模型可信

因此正确值是 **0.24125**（482.5 mm 的一半），原来的 0.18755 偏小 22.3%。

影响
----
只作用于奖励的 `wheel_slip` 项（左右轮线速度换算），不影响观测、动作、
控制器或任何推理路径。BC 交付件是监督学习产物，奖励未参与训练，故行为不变。

用法
----
    python scripts/patch_half_track.py          # 应用
    python scripts/patch_half_track.py --check  # 只检查
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "diablo_training" / "xingtian_env.py"

OLD = "        half_track = 0.18755\n"
NEW = (
    "        # 轮心距的一半（482.5 mm / 2）。实测外缘跨度 528 mm − 轮宽 50 mm ≈ 478 mm，\n"
    "        # MuJoCo 正运动学 482.5 mm，两者差 0.9%。原值 0.18755 是髋关节间距，偏小 22.3%。\n"
    "        half_track = 0.24125\n"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只检查，不修改")
    args = ap.parse_args()

    text = TARGET.read_text(encoding="utf-8")

    if "half_track = 0.24125" in text:
        print("[跳过] 已经是 0.24125")
        return 0
    if OLD not in text:
        found = re.findall(r"half_track\s*=\s*[0-9.]+", text)
        print(f"[失败] 没找到预期的那一行，当前为：{found}")
        return 1

    patched = text.replace(OLD, NEW, 1)
    try:
        ast.parse(patched)
    except SyntaxError as exc:
        print(f"[失败] 语法不通：{exc}")
        return 1

    if args.check:
        print("[待修改] 当前 0.18755 → 目标 0.24125（去掉 --check 即应用）")
        return 0

    TARGET.write_text(patched, encoding="utf-8")
    print("[完成] half_track: 0.18755 → 0.24125")
    print("       下一步：重签交付件（adopt_policy.py），再重跑诊断确认行为不变")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
