"""带自适应学习率的训练入口。

为什么需要：
    RSL-RL（腿足 RL 主流实现）的 PPO 每次 mini-batch 后按实测 KL 调学习率：
        kl > desired_kl * 2   → lr /= 1.5
        kl < desired_kl / 2   → lr *= 1.5
    而 SB3 的 PPO 只有 target_kl（KL 超了就提前结束本轮 epoch），学习率不变。

    实测证据：我们加 target_kl=0.01 后，日志出现 "Early stopping at step 1"
    ——它在"撞墙停下"，而不是"走小步"。虽然解决了震荡（checkpoint 从孤峰
    变成平台），但每轮只剩 1 个 epoch 的学习量。自适应 lr 能让它真的迈小步，
    同时保留 5 个 epoch 的更新次数。

实现方式：
    不改仓库代码，在进程启动时给 stable_baselines3 的 PPO.train 套一层，
    读 KL → 调 lr → 再交回原来的 train。然后直接调用仓库自己的 cli.main()。

用法（参数与 diablo train 完全一致）：
    python -m diablo_training.train_adaptive train \
        --config examples/xingtian_stage_a.json \
        --steps 2000000 --seed 2002 --n-envs 16 --device cuda \
        --output runs/A-fix18-adaptlr

    DESIRED_KL=0.02 python -m diablo_training.train_adaptive train ...   # 放宽目标
"""

from __future__ import annotations

import os
import sys

DESIRED_KL = float(os.environ.get("DESIRED_KL", "0.01"))
LR_MIN = 1e-5
LR_MAX = 1e-2
FACTOR = 1.5


def install() -> None:
    from stable_baselines3.ppo import PPO

    original = PPO.train
    state = {"calls": 0}

    def train(self):  # type: ignore[no-untyped-def]
        out = original(self)
        state["calls"] += 1

        kl = None
        logger = getattr(self, "logger", None)
        if logger is not None and getattr(logger, "name_to_value", None):
            kl = logger.name_to_value.get("train/approx_kl")
        if kl is None and isinstance(out, tuple) and len(out) >= 3:
            kl = out[2]

        if kl is not None:
            try:
                kl = float(kl)
            except (TypeError, ValueError):
                kl = None

        if os.environ.get("ADAPTIVE_DEBUG") == "1" and state["calls"] <= 3:
            keys = sorted(getattr(logger, "name_to_value", {}) or {})
            print(f"[adaptive-lr:debug] 第 {state['calls']} 次 train()", flush=True)
            print(f"  返回类型 {type(out).__name__}  len={len(out) if isinstance(out, tuple) else 'n/a'}", flush=True)
            print(f"  logger 里的 key: {keys}", flush=True)
            print(f"  取到的 kl = {kl}", flush=True)
            print(f"  当前 lr = {getattr(self, 'learning_rate', None)}", flush=True)

        if kl is not None and kl > 0.0:
            lr = float(getattr(self, "learning_rate", LR_MAX))
            new_lr = lr
            if kl > DESIRED_KL * 2.0:
                new_lr = max(LR_MIN, lr / FACTOR)
            elif kl < DESIRED_KL / 2.0:
                new_lr = min(LR_MAX, lr * FACTOR)

            if abs(new_lr - lr) > 1e-12:
                self.learning_rate = new_lr
                for group in self.policy.optimizer.param_groups:
                    group["lr"] = new_lr
                print(f"[adaptive-lr] kl={kl:.4f}  lr {lr:.2e} -> {new_lr:.2e}", flush=True)

        return out

    PPO.train = train  # type: ignore[method-assign]


def main() -> int:
    install()
    from diablo_training.cli import main as cli_main

    return cli_main() or 0


if __name__ == "__main__":
    print(f"[adaptive-lr] desired_kl={DESIRED_KL}  范围 [{LR_MIN:.0e}, {LR_MAX:.0e}]", flush=True)
    raise SystemExit(main())
