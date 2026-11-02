"""用门控双专家（简化 MoE）训练的策略注入入口。

做法：在进程启动时给 stable_baselines3 的 PPO 注入 features_extractor_class，
然后调用仓库自己的 cli.main()。**不改仓库任何代码。**

为什么能这样做：
    MoE 的门控只依赖观测（指令就在 obs 里），所以整个专家结构可以塞进
    features_extractor。而 SB3 的 policy_kwargs 正好接受 features_extractor_class。

用法（参数与 diablo train 完全一致）：
    python -m diablo_training.train_moe train \
        --config examples/xingtian_stage_a.json \
        --steps 300000 --seed 2002 --n-envs 16 --device cuda \
        --warm-start runs/A-fix21-yawscale/checkpoints/rl_model_1900000_steps.zip \
        --output runs/A-fix32-moe

环境变量：
    MOE_GATE=1.0        门控阈值（cmd_yaw 观测值，观测系数是 10 时 1.0 对应 0.1 rad/s）
    MOE_FEATURES=64     专家输出维度
    MOE_NET_ARCH=64     覆盖 config 里的 pi/vf 层宽（默认沿用 config 的 [128,64]）
"""

from __future__ import annotations

import os

from diablo_training.moe_policy import MoEFeatures


def install() -> None:
    from stable_baselines3 import PPO

    original_init = PPO.__init__

    def patched_init(self, policy, env, *args, **kwargs):  # type: ignore[no-untyped-def]
        policy_kwargs = dict(kwargs.get("policy_kwargs") or {})
        policy_kwargs["features_extractor_class"] = MoEFeatures
        policy_kwargs.setdefault("features_extractor_kwargs", {})

        net_arch = os.environ.get("MOE_NET_ARCH")
        if net_arch is not None:
            policy_kwargs["net_arch"] = [int(x) for x in net_arch.split(",") if x]

        kwargs["policy_kwargs"] = policy_kwargs
        original_init(self, policy, env, *args, **kwargs)

    PPO.__init__ = patched_init  # type: ignore[method-assign]

    # 训练结束前打印一次门控使用情况，确认专家真的在分流
    original_learn = PPO.learn

    def patched_learn(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        try:
            return original_learn(self, *args, **kwargs)
        finally:
            extractor = getattr(self.policy, "features_extractor", None)
            if extractor is not None and hasattr(extractor, "gate_usage"):
                fwd, turn = extractor.gate_usage()
                total = fwd + turn
                if total:
                    print(
                        f"[moe] 门控统计：前进专家 {fwd} ({fwd / total:.1%})  "
                        f"转向专家 {turn} ({turn / total:.1%})",
                        flush=True,
                    )

    PPO.learn = patched_learn  # type: ignore[method-assign]


def main() -> int:
    install()
    from diablo_training.cli import main as cli_main

    print(
        f"[moe] 双专家已注入：门控阈值 {os.environ.get('MOE_GATE', '1.0')}，"
        f"偏航索引 {os.environ.get('MOE_YAW_INDEX', '107')}",
        flush=True,
    )
    return cli_main() or 0


if __name__ == "__main__":
    raise SystemExit(main())
