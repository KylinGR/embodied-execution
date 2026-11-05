"""把一批 diag.json 汇总成一张「扰动档 × 剖面」的表，判据用 diag_rejudge.RELAXED。

用法：
    python scripts/dr_summary.py runs/dr/*/diag.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diablo_training import diag_rejudge as R  # noqa: E402
from diablo_training import xingtian_diag2 as D  # noqa: E402

STAGES = ["stand", "stand_long", "forward", "backward", "turn",
          "turn_in_motion", "start_stop", "accel"]
SHORT = {"stand": "stand", "stand_long": "long", "forward": "fwd",
         "backward": "bwd", "turn": "turn", "turn_in_motion": "tim",
         "start_stop": "s/s", "accel": "accel"}


def judge_relaxed(stage: str, row: dict) -> bool:
    old = {k: dict(v) for k, v in D.THRESHOLDS.items()}
    D.THRESHOLDS.clear()
    D.THRESHOLDS.update(R.RELAXED)
    try:
        return D.judge(stage, row)[0]
    finally:
        D.THRESHOLDS.clear()
        D.THRESHOLDS.update(old)


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print(__doc__)
        return 2

    head = f"{'档位':<14}" + "".join(f"{SHORT[s]:>7}" for s in STAGES)
    head += f"{'停车m':>16}{'俯仰峰°':>9}{'倾角RMS':>8}{'min_up':>7}"
    print(head)
    print("-" * len(head))

    for path in sorted(paths):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data["rows"]
        by_stage: dict[str, list[dict]] = {}
        for row in rows:
            by_stage.setdefault(row["stage"], []).append(row)

        cells = ""
        for stage in STAGES:
            group = by_stage.get(stage, [])
            ok = sum(1 for row in group if judge_relaxed(stage, row))
            cells += f"{ok}/{len(group)}".rjust(7)

        stops = []
        for row in by_stage.get("start_stop", []):
            stops += [d for d in (row.get("stop_distances_m") or []) if d is not None]
        stop_txt = " ".join(f"{s:.4f}" for s in stops) if stops else "—"
        if len(stop_txt) > 15:
            stop_txt = stop_txt[:15]

        pitch = max((row.get("pitch_absmax_deg", 0.0) for row in rows), default=0.0)
        tilt = max((row.get("tilt_rms_deg", 0.0) for row in rows), default=0.0)
        up = min((row.get("min_upright", 1.0) for row in rows), default=1.0)
        print(f"{path.parent.name:<14}{cells}{stop_txt:>16}{pitch:>9.1f}{tilt:>8.2f}{up:>7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
