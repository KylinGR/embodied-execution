"""给 xingtian_diag2 加两个直线高速剖面：0.8 m/s 和 1.0 m/s。

背景：
    配置允许 max_speed=1.0，但 A 阶段所有诊断只测到 0.5 m/s。
    制动距离 ∝ v²、起步俯仰 ∝ 加速度，所以"能跑多快"必须单独测。

改动（三处，纯增量）：
    STAGES     增加 forward_fast_08 / forward_fast_10
    THRESHOLDS 增加对应判据
    judge      把这两个阶段并进 forward 那一支

用法：
    python scripts/patch_diag_fast.py
    python -m diablo_training.xingtian_diag2 --only forward_fast_08 forward_fast_10 ...
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "diablo_training" / "xingtian_diag2.py"

ANCHOR_STAGES = "    # 低速→高速→中速，看定速和过渡\n"
NEW_STAGES = '''    # 直线高速：配置允许 1.0 m/s，但 A 阶段只测过 0.5，单独加两档验证上限
    "forward_fast_08": {"duration": 15.0, "profile": [_pt(0.0), _pt(1.0, 0.8)]},
    "forward_fast_10": {"duration": 15.0, "profile": [_pt(0.0), _pt(1.0, 1.0)]},
'''

ANCHOR_THRESH = '    "accel": {"speed_rel_err": 0.10},\n'
NEW_THRESH = '''    "accel": {"speed_rel_err": 0.10},
    "forward_fast_08": {
        "speed_rel_err": 0.08, "distance_ratio": 0.90, "lateral_drift_m": 0.08,
        "heading_deg": 3.0, "rise_time_s": 0.8, "overshoot": 0.10,
    },
    "forward_fast_10": {
        "speed_rel_err": 0.08, "distance_ratio": 0.90, "lateral_drift_m": 0.08,
        "heading_deg": 3.0, "rise_time_s": 0.8, "overshoot": 0.10,
    },
'''

ANCHOR_JUDGE = '    if stage in ("forward", "backward", "accel"):\n'
NEW_JUDGE = ('    if stage in ("forward", "backward", "accel",\n'
             '                 "forward_fast_08", "forward_fast_10"):\n')


def main() -> int:
    text = DIAG.read_text(encoding="utf-8")
    if "forward_fast_08" in text:
        print("[跳过] 已经加过高速剖面")
        return 0

    for name, anchor in (("STAGES", ANCHOR_STAGES), ("THRESHOLDS", ANCHOR_THRESH),
                         ("judge", ANCHOR_JUDGE)):
        if text.count(anchor) != 1:
            print(f"[失败] {name} 的锚点匹配到 {text.count(anchor)} 处，期望 1 处")
            return 1

    text = text.replace(ANCHOR_STAGES, NEW_STAGES + ANCHOR_STAGES, 1)
    text = text.replace(ANCHOR_THRESH, NEW_THRESH, 1)
    text = text.replace(ANCHOR_JUDGE, NEW_JUDGE, 1)

    try:
        ast.parse(text)
    except SyntaxError as exc:
        print(f"[失败] 语法不通：{exc}")
        return 1

    DIAG.write_text(text, encoding="utf-8")
    print("[完成] xingtian_diag2.py 增加 forward_fast_08 / forward_fast_10")
    print("       注意：这两个不复进 DEFAULT_STAGES，只有显式 --only 才会跑")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
