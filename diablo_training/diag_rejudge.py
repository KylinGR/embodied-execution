"""用新阈值重判历史 diag.json，不需要重新训练。

为什么需要：
  阈值是我拍的，现在要放宽。但"放宽后谁能通过"不能靠猜——
  所有 diag.json 里都有原始指标，直接套新阈值重判就行。

做法：
  复用 xingtian_diag2.judge()，只是替换它的 THRESHOLDS 字典。
  这样判据逻辑只有一份，不会和主程序漂移。

用法：
  python -m diablo_training.diag_rejudge runs/diag/A12/diag.json runs/diag/A11/diag.json ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import xingtian_diag2 as D

# ---------------------------------------------------------------- 放宽后的阈值
# 每条都写了依据；标注（不变）的是经过物理推导后决定保留的。
RELAXED: dict[str, dict[str, float]] = {
    "common": {"min_upright": 0.90, "duration_ratio": 0.98},

    # 15 秒累计 15°；实机 IMU 零偏量级 0.1-0.5 °/s，仿真无零偏但 0.2 过苛
    "stand": {
        "still_turn_dps": 1.0,
        "still_drift_mps": 0.05,
        "still_coverage": 0.85,
        "tilt_rms_deg": 8.0,
        "tilt_max_deg": 15.0,
    },
    "stand_long": {
        "still_turn_dps": 1.0,
        "still_drift_mps": 0.05,
        "still_coverage": 0.85,
        "tilt_rms_deg": 8.0,
        "tilt_max_deg": 15.0,
    },

    # 差速机器人 7 m 行程的方向容差 8°、侧偏 4%（0.30 m）
    "forward": {
        "speed_rel_err": 0.15,
        "distance_ratio": 0.90,
        "lateral_drift_m": 0.30,
        "heading_deg": 8.0,
        "rise_time_s": 0.9,
        "overshoot": 0.20,
    },
    "backward": {
        "speed_rel_err": 0.20,
        "distance_ratio": 0.90,
        "lateral_drift_m": 0.30,
        "heading_deg": 8.0,
    },

    # 两轮差速原地转向本身就难；MAE 0.20 仍高于历史最好 0.275
    "turn": {"yaw_mae": 0.20, "min_sign_ratio": 0.75},
    "turn_left": {"yaw_mae": 0.20, "min_sign_ratio": 0.75},
    "turn_right": {"yaw_mae": 0.20, "min_sign_ratio": 0.75},
    "turn_in_motion": {"yaw_mae_turn": 0.25, "speed_rel_err_turn": 0.25},

    # 停车距离【不变】：物理推导 0.06 m（340 N / 22.9 kg / 0.1 s 延迟），
    # A-fix3 实测做到过 0.028 m，说明 0.08 不苛刻
    "start_stop": {"n_cycles": 3.0, "rise_time_s": 0.9, "stop_distance_m": 0.08},

    "accel": {"speed_rel_err": 0.25},
    "height": {"ripple_m": 0.06, "std_m": 0.025},
}

# 6 项要求 → 对应剖面
REQUIREMENTS = {
    "① 原地平衡": ("stand", "stand_long"),
    "② 前进": ("forward",),
    "② 后退": ("backward",),
    "③ 多次启停": ("start_stop",),
    "④ 加减速": ("accel",),
    "⑤ 原地转向": ("turn",),
    "⑥ 行驶中转弯": ("turn_in_motion",),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args()

    D.THRESHOLDS.clear()
    D.THRESHOLDS.update(RELAXED)

    print("=" * 76)
    print("按【放宽后】的阈值重判历史数据（不需要重新训练）")
    print("=" * 76)

    summary: dict[str, dict[str, int]] = {}
    totals: dict[str, list[str]] = {}

    for path in args.files:
        name = path.parent.name
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data["rows"]
        by_stage: dict[str, list[dict]] = {}
        for row in rows:
            by_stage.setdefault(row["stage"], []).append(row)

        n_pass = 0
        passed_stages: list[str] = []
        for stage, stage_rows in by_stage.items():
            oks = []
            for row in stage_rows:
                ok, why = D.judge(stage, row)
                oks.append((ok, why))
            n_ok = sum(1 for ok, _ in oks if ok)
            n_pass += n_ok
            if n_ok == len(stage_rows):
                passed_stages.append(stage)
            summary.setdefault(name, {})[stage] = n_ok
            mark = "✅" if n_ok == len(stage_rows) else ("◐" if n_ok else "✗")
            why = "" if n_ok == len(stage_rows) else \
                " | ".join((oks[0][1] or [])[:2])
            print(f"  {name:<22} {stage:<15} {mark} {n_ok}/{len(stage_rows)}"
                  + (f"   {why}" if why else ""))

        totals[name] = passed_stages
        print(f"  {'':<22} → 通过 {n_pass}/{len(rows)} 次运行\n")

    print("=" * 76)
    print("按 6 项要求汇总（3 个 seed 全过才算通过）")
    print("=" * 76)
    print(f"  {'要求':<16}" + "".join(f"{n[:14]:<15}" for n in summary))
    for label, stages in REQUIREMENTS.items():
        cells = []
        for name in summary:
            got = all(st in totals[name] for st in stages)
            cells.append(("✅" if got else "✗").ljust(15))
        print(f"  {label:<16}" + "".join(cells))

    print("\n  已完全通过的要求：")
    any_pass = False
    for name, stages in totals.items():
        for label, need in REQUIREMENTS.items():
            if all(st in stages for st in need):
                print(f"    {name} → {label}")
                any_pass = True
    if not any_pass:
        print("    （无）")


if __name__ == "__main__":
    main()
