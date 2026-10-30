"""量出观测里每个分块的统计量。

为什么需要：
    _frame() 里各分块的缩放大不相同：
        gyro*0.25            ~0.05-0.5
        gravity              ~1
        command*[10,0.25,5]  vx 可达 ±10，yaw 只有 ±0.075  ← 差 130 倍
        q (腿角)             ~1.5
        dq*0.05              ~0.5
        previous_action      [-1,1]
    "偏航指令看不见"是我的推断，必须先量出来再改。

用法：
    python -m diablo_training.obs_stats \
        --model runs/A-fix16-kl/checkpoints/rl_model_1300000_steps.zip \
        --resolved runs/A-fix14-tilt15/selection/resolved.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BLOCKS = [
    ("gyro*0.25", 0, 3),
    ("gravity", 3, 6),
    ("cmd_vx*10", 6, 7),
    ("cmd_yaw*0.25", 7, 8),
    ("cmd_h*5", 8, 9),
    ("q 腿角", 9, 13),
    ("dq*0.05", 13, 19),
    ("prev_action", 19, 25),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path)
    ap.add_argument("--resolved", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--command", nargs=3, type=float, default=[0.5, 0.3, 0.25])
    ap.add_argument("--seed", type=int, default=2002)
    args = ap.parse_args()

    from diablo_training.xingtian_diag2 import build_env

    resolved = json.loads(args.resolved.read_text(encoding="utf-8"))
    env = build_env(resolved)

    cmd = np.asarray(args.command, dtype=float)
    env.get_command = lambda sim_seconds: cmd
    env.command = cmd

    obs, _ = env.reset(seed=args.seed, options={})

    model = None
    if args.model is not None:
        from stable_baselines3 import PPO

        model = PPO.load(str(args.model), device="cpu")

    frames = []
    for _ in range(args.steps):
        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, _r, term, trunc, _info = env.step(action)
        frames.append(np.asarray(obs, dtype=float)[:25])
        if term or trunc:
            obs, _ = env.reset(seed=args.seed, options={})

    data = np.stack(frames)
    print(f"观测维度 {data.shape[1]}（应为 25）   采样 {len(frames)} 步   指令 {cmd.tolist()}")
    print()
    print(f"{'分块':<14}{'均值':>10}{'标准差':>10}{'最小值':>10}{'最大值':>10}{'|最大|':>10}")
    print("-" * 64)
    for name, lo, hi in BLOCKS:
        block = data[:, lo:hi]
        print(
            f"{name:<14}{block.mean():>10.3f}{block.std():>10.3f}"
            f"{block.min():>10.3f}{block.max():>10.3f}"
            f"{np.abs(block).max():>10.3f}"
        )
    print("-" * 64)
    print(f"{'整体':<14}{data.mean():>10.3f}{data.std():>10.3f}"
          f"{data.min():>10.3f}{data.max():>10.3f}{np.abs(data).max():>10.3f}")
    print()
    print("判读：|最大| 最大和最小的分块差多少倍。超过 50 倍就说明条件数很差。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
