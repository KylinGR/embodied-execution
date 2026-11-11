"""把多个 diag.json 并排成一张对照表，省得一个个翻。

用法：
  python -m diablo_training.diag_compare runs/diag/A12/diag.json runs/diag/A13-s2003/diag.json

输出：
  · 每个阶段的"跨 seed 平均值"并排成表
  · 每个 run 的通过项数
  · 每行标出哪个 run 最好
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# 阶段 → [(字段, 显示名, 越好还是越差, 格式)]
STAGE_FIELDS: dict[str, list[tuple[str, str, str, str]]] = {
    "stand": [
        ("still_turn_dps", "自转°/s", "low", "{:.2f}"),
        ("still_drift_mps", "漂移m/s", "low", "{:.3f}"),
        ("tilt_rms_deg", "倾角RMS°", "low", "{:.2f}"),
    ],
    "stand_long": [
        ("still_turn_dps", "自转°/s", "low", "{:.2f}"),
        ("displacement_m", "120s位移m", "low", "{:.2f}"),
        ("tilt_rms_deg", "倾角RMS°", "low", "{:.2f}"),
    ],
    "forward": [
        ("distance_ratio", "位移比", "high", "{:.2f}"),
        ("speed_rel_err", "速度误差", "low", "{:.2f}"),
        ("heading_change_deg", "航向°", "low", "{:.1f}"),
        ("lateral_drift_m", "横漂m", "low", "{:.2f}"),
        ("rise_time_s", "上升s", "low", "{:.2f}"),
    ],
    "backward": [
        ("distance_ratio", "位移比", "high", "{:.2f}"),
        ("speed_rel_err", "速度误差", "low", "{:.2f}"),
        ("lateral_drift_m", "横漂m", "low", "{:.2f}"),
    ],
    "turn": [
        ("yaw_sign_ratio", "方向率", "high", "{:.2f}"),
        ("yaw_mae", "yawMAE", "low", "{:.2f}"),
    ],
    "turn_in_motion": [
        ("distance_ratio", "位移比", "high", "{:.2f}"),
        ("yaw_mae_turn", "转弯MAE", "low", "{:.2f}"),
        ("speed_rel_err", "直行误差", "low", "{:.2f}"),
    ],
    "start_stop": [
        ("n_cycles", "启停数", "high", "{:.0f}"),
        ("max_stop_distance_m", "最大停车m", "low", "{:.3f}"),
        ("max_rise_time_s", "最大上升s", "low", "{:.2f}"),
    ],
    "accel": [
        ("distance_ratio", "位移比", "high", "{:.2f}"),
        ("speed_rel_err", "速度误差", "low", "{:.2f}"),
        ("height_ripple_m", "高度波动", "low", "{:.3f}"),
    ],
}


def _stage_value(row: dict, key: str) -> float:
    """start_stop 的停车距离/上升时间是列表，取最差（最大）那个。"""
    if key == "max_stop_distance_m":
        vals = [v for v in (row.get("stop_distances_m") or []) if v is not None]
        return max(vals) if vals else float("nan")
    if key == "max_rise_time_s":
        vals = [v for v in (row.get("rise_times_s") or []) if v is not None]
        return max(vals) if vals else float("nan")
    val = row.get(key)
    return float(val) if isinstance(val, (int, float)) else float("nan")


def load(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    buckets: dict[str, dict[str, list[float]]] = {}
    for row in data["rows"]:
        stage = row["stage"]
        buckets.setdefault(stage, {})
        for key, _label, _dir, _fmt in STAGE_FIELDS.get(stage, []):
            buckets[stage].setdefault(key, []).append(_stage_value(row, key))
    stages = {s: {k: float(np.nanmean(v)) for k, v in d.items()}
              for s, d in buckets.items()}
    n_pass = sum(1 for r in data["rows"] if r.get("pass"))
    return stages, n_pass, len(data["rows"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args()

    runs = [(p.parent.name, *load(p)) for p in args.files]
    names = [n for n, _s, _p, _t in runs]
    stage_maps = {n: s for n, s, _p, _t in runs}

    print("=" * 78)
    print("diag 对照表（每个阶段取跨 seed 平均）")
    print("=" * 78)

    for stage, fields in STAGE_FIELDS.items():
        present = [n for n in names if stage in stage_maps[n]]
        if not present:
            continue
        print(f"\n【{stage}】")
        print("  " + "指标".ljust(12) + "".join(n.ljust(20) for n in names))
        for key, label, better, fmt in fields:
            cells = []
            values = []
            for name in names:
                val = stage_maps[name].get(stage, {}).get(key, float("nan"))
                values.append(val)
                cells.append("—".ljust(20) if not np.isfinite(val)
                             else fmt.format(val).ljust(20))
            finite = [v for v in values if np.isfinite(v)]
            mark = ""
            if len(finite) == len(values) and len(finite) > 1:
                best = min(finite) if better == "low" else max(finite)
                mark = f"   ← {names[values.index(best)]}"
            print("  " + label.ljust(12) + "".join(cells) + mark)

    print("\n" + "=" * 78)
    print("通过情况")
    print("=" * 78)
    for name, _stages, n_pass, n_total in runs:
        print(f"  {name:<28} 通过 {n_pass}/{n_total}")


if __name__ == "__main__":
    main()
