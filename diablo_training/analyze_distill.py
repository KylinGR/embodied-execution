"""诊断蒸馏数据集和学生模型：到底是"没数据"还是"没学会"。

用法：
    python -m diablo_training.analyze_distill --data runs/distill_pv2/data.npz
    python -m diablo_training.analyze_distill --data runs/distill_pv2/data.npz \
        --model runs/distill_pv2/policy.zip --config examples/xingtian_balance_student_pv.json

观测布局（每帧 25 维，5 帧共 125，最新帧在最后）：
    gyro(3) gravity(3) command(3) q-DEFAULT_Q(4) dq(6) prev_action(6)
    缩放：gyro×0.25、command×[10, 0.25, 5]、dq×0.05
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CMD_SCALE = np.array([10.0, 0.25, 5.0])


def cmd_of(obs: np.ndarray) -> np.ndarray:
    """从 125 维观测里取出当前指令 [vx, wz, height]。"""
    return obs[:, -19:-16] / CMD_SCALE


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--config", type=Path,
                    default=Path("examples/xingtian_balance_student_pv.json"))
    args = ap.parse_args()

    data = np.load(args.data)
    obs, act = data["obs"], data["act"]
    cmd = cmd_of(obs)
    vx, wz = cmd[:, 0], cmd[:, 1]

    print(f"样本 {len(obs)}")
    moving = np.abs(vx) > 0.05
    turning = np.abs(wz) > 0.05
    print(f"  指令 |vx|>0.05: {moving.mean():5.1%}   |wz|>0.05: {turning.mean():5.1%}")
    print(f"  vx 分位 10/50/90%: {np.percentile(vx, [10, 50, 90]).round(2)}")
    print(f"  动作逐关节 std: {act.std(axis=0).round(3)}")
    print(f"  动作逐关节 absmean: {np.abs(act).mean(axis=0).round(3)}")

    buckets = {
        "stand": ~moving & ~turning,
        "forward": vx > 0.05,
        "backward": vx < -0.05,
        "turn": turning,
    }
    for name, mask in buckets.items():
        if mask.sum() == 0:
            print(f"  [{name}] 无样本")
            continue
        print(f"  [{name}] {mask.sum():6d} 样本  "
              f"动作 absmean {np.abs(act[mask]).mean():.3f}  "
              f"轮子 absmean {np.abs(act[mask][:, [2, 5]]).mean():.3f}")

    if args.model is None:
        print("\n（没有 --model：只看了数据覆盖；加 --model 看学生在各类样本上的误差）")
        return 0

    from stable_baselines3 import PPO
    import torch

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    model = PPO.load(str(args.model), device="cpu")
    arch = cfg.get("ppo", {}).get("policy_kwargs", {}).get("net_arch")
    print(f"\n学生模型 {args.model}  net_arch={arch}")

    X = torch.as_tensor(obs, dtype=torch.float32)
    with torch.no_grad():
        pred, _v, _lp = model.policy(X, deterministic=True)
    pred = pred.cpu().numpy()
    err = pred - act

    print(f"  总体 mse {np.mean(err ** 2):.6f}   rmse {np.sqrt(np.mean(err ** 2)):.4f}")
    for name, mask in buckets.items():
        if mask.sum() == 0:
            continue
        mse = float(np.mean(err[mask] ** 2))
        print(f"  [{name}] mse {mse:.6f}  rmse {np.sqrt(mse):.4f}  "
              f"轮子 rmse {np.sqrt(np.mean(err[mask][:, [2, 5]] ** 2)):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
