"""在一个 run 的多个 checkpoint 里按实际 diag 指标选优。

为什么需要：
    select-best 的判据只有"短评测里没摔倒"，六次里从没选中过 2M 的
    checkpoint（选的都是 1.0M–1.9M），说明最后一个往往不是最好的。
    而 diag 有真实指标（位移比、左右转方向率、yaw MAE），应该用它选。

做法：
    对每个 checkpoint 调 diablo_training.xingtian_diag2 跑一组剖面，
    读回 diag.json，打印一行摘要并给出综合分。

用法：
    python -m diablo_training.ck_sweep --run runs/A-fix14-s2003 \
        --resolved runs/A-fix14-tilt15/selection/resolved.json \
        --stages forward turn turn_left turn_right --every 400000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def checkpoints(run: Path, every: int) -> list[Path]:
    cks = sorted(run.glob("checkpoints/*.zip"), key=lambda p: int(
        p.stem.split("_")[2]))
    if every <= 0:
        return cks
    kept = []
    for ck in cks:
        steps = int(ck.stem.split("_")[2])
        if steps % every == 0:
            kept.append(ck)
    return kept or cks


def one(ck: Path, resolved: Path, stages: list[str]) -> dict | None:
    with tempfile.TemporaryDirectory() as td:
        cmd = [
            sys.executable, "-m", "diablo_training.xingtian_diag2",
            "--model", str(ck), "--resolved", str(resolved),
            "--seeds", "2002", "--only", *stages, "--out", td,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        dj = Path(td) / "diag.json"
        if not dj.exists():
            print(f"  [失败] {ck.name}: {proc.stderr.strip()[-200:]}")
            return None
        return json.loads(dj.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--resolved", required=True, type=Path)
    ap.add_argument("--stages", nargs="+",
                    default=["forward", "turn", "turn_left", "turn_right"])
    ap.add_argument("--every", type=int, default=400000)
    args = ap.parse_args()

    cks = checkpoints(args.run, args.every)
    print(f"{args.run}  共 {len(cks)} 个 checkpoint 待评")
    print(f"{'步数':>9} {'fwd位移比':>10} {'跌倒':>5} {'左转':>7} {'右转':>7} {'转弯MAE':>9}")
    print("-" * 56)

    rows = []
    for ck in cks:
        steps = int(ck.stem.split("_")[2])
        data = one(ck, args.resolved, args.stages)
        if data is None:
            continue
        by = {r["stage"]: r for r in data["rows"]}
        fwd = by.get("forward", {})
        left = by.get("turn_left", {})
        right = by.get("turn_right", {})
        fell = sum(1 for r in data["rows"] if r.get("terminated"))

        dist = fwd.get("distance_ratio")
        l_r = left.get("yaw_sign_ratio")
        r_r = right.get("yaw_sign_ratio")
        mae = left.get("yaw_mae")

        print(f"{steps:>9} {dist if dist is not None else float('nan'):>10.2f}"
              f" {fell:>5}"
              f" {l_r * 100 if l_r is not None else float('nan'):>6.0f}%"
              f" {r_r * 100 if r_r is not None else float('nan'):>6.0f}%"
              f" {mae if mae is not None else float('nan'):>9.3f}")
        rows.append((steps, dist, fell, l_r, r_r, mae))

    if not rows:
        print("没有可评的 checkpoint")
        return 1

    # 综合分：
    #   1) distance_ratio 必须裁剪 —— 摔倒滑行会给出 18.09 这种假数字
    #   2) 跌倒要硬性排除，否则"摔出去很远"会被评成最优
    #   3) 左右转要都高、且两边平衡
    eligible = [r for r in rows
                if r[2] == 0 and r[1] is not None and r[1] >= 0.80]
    if eligible:
        best = max(eligible, key=lambda r: (
            0.4 * ((r[3] or 0.0) + (r[4] or 0.0))
            - 0.3 * abs((r[3] or 0.0) - (r[4] or 0.0))
            + 0.2 * min(r[1] or 0.0, 1.2)
        ))
        why = "合格（无跌倒 且 位移比≥0.80）中左右最平衡"
    else:
        best = max(rows, key=lambda r: min(r[1] or 0.0, 1.2) - 0.5 * r[2])
        why = "【警告】没有合格 checkpoint，只能选相对最好的"

    print("-" * 56)
    steps, d, fell, lv, rv = best[0], best[1], best[2], best[3], best[4]
    print(f"综合最优：{steps} 步   forward={d:.2f}  左={(lv or 0) * 100:.0f}%"
          f"  右={(rv or 0) * 100:.0f}%  跌倒={fell}")
    print(f"  依据：{why}")
    print(f"  {args.run}/checkpoints/rl_model_{steps}_steps.zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
