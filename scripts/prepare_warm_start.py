"""把行为克隆策略准备成 `diablo train --warm-start` 的合法来源。

为什么需要两步处理
------------------
1. **resolved.json**：`cli.py` 的 warm-start 会读 checkpoint 旁边的 `resolved.json`，
   比对四个字段（`action_mode` / `observation.shape` / `model.control_dt_s` /
   `joints|actuators|closure_constraints`）。BC 策略没有这个文件，需要补一份
   `action_mode` 匹配的。

2. **log_std**：SB3 的 `PPO.save()` 把 `log_std` 当参数一起存；warm-start 用
   `load_state_dict(strict=True)` 会把它一并拷过去。AB 阶段实测：
   `log_std=0`（std=1.0，动作噪声 ±17 N·m）时微调，策略被噪声淹掉，
   300k 步后 24 项诊断全崩（0/24）。所以必须先压到 std≈0.08。

用法
----
    python scripts/prepare_warm_start.py \\
        --policy deliverables/BAL-1/policy.zip \\
        --resolved deliverables/BAL-1/resolved.json \\
        --out runs/warmstart/BAL-B  \\
        --log-std -2.5

产出：
    runs/warmstart/BAL-B/policy.zip     已压低 log_std
    runs/warmstart/BAL-B/resolved.json  与目标契约的 action_mode 匹配
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from stable_baselines3 import PPO


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--resolved", type=Path, required=True,
                    help="一份 action_mode / obs / dt 与目标阶段匹配的 resolved.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--log-std", type=float, default=-2.5,
                    help="std = exp(log_std)；−2.5 ≈ 0.082（对应 ±1.4 N·m 噪声）")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    model = PPO.load(str(args.policy), device="cpu")
    with torch.no_grad():
        model.policy.log_std.data.fill_(args.log_std)
    target = args.out / "policy.zip"
    model.save(str(target))

    shutil.copyfile(args.resolved, args.out / "resolved.json")

    sigma = float(model.policy.log_std.exp().mean())
    print(f"log_std -> {args.log_std}（std ≈ {sigma:.3f}，约 ±{sigma * 17:.1f} N·m）")
    print(f"写入 {target}")
    print(f"写入 {args.out / 'resolved.json'}")
    print("下一步：diablo train --warm-start " + str(target) + " ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
