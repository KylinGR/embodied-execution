"""把一个"非 diablo train 产出"的策略（行为克隆）登记进仓库的契约体系。

为什么需要：
    `diablo evaluate` / `diablo export` 会做两步校验——
        require_compatible(resolved, env)      # resolved["contract"] 必须等于当前代码算出的指纹
        require_checkpoint(policy, resolved)   # policy.diablo_contract_sha256 必须存在且相等
    `PPO.save()` 直接存的策略没有 diablo_contract_sha256，也没有配套 resolved.json，
    所以过不了这两步。本脚本补上，等价于 diablo train 结尾写 resolved 的动作。

用法：
    python scripts/adopt_policy.py \
        --config examples/xingtian_balance_student_wt.json \
        --policy runs/distill_wt3/policy.zip \
        --run-dir runs/BAL-1
产出：
    runs/BAL-1/checkpoints/policy.zip        已带 diablo_contract_sha256
    runs/BAL-1/selection/policy-best.zip     同上（仓库习惯的交付位置）
    runs/BAL-1/resolved.json                 与 checkpoints/ 同级的契约
    runs/BAL-1/selection/resolved.json       与 selection/ 同级的契约
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from stable_baselines3 import PPO

from diablo_training.contract import environment_class, import_class, policy_contract
from diablo_training.model import metadata


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg_json = json.loads(args.config.read_text(encoding="utf-8"))
    config_cls = import_class(cfg_json["config_class"])
    config = config_cls(**cfg_json["environment"])
    env_cls = environment_class(cfg_json["environment_class"])

    env = env_cls(config)
    try:
        contract = policy_contract(env)
        model_meta = metadata(env.model)
    finally:
        env.close()

    resolved = {
        "environment": asdict(config),
        "environment_class": cfg_json["environment_class"],
        "config_class": cfg_json["config_class"],
        "contract": contract,
        "ppo": cfg_json.get("ppo", {}),
        "model": model_meta,
        "seed": args.seed,
        "device": "cpu",
        "source_policy": str(args.policy),
        "note": "policy adopted from behavior-cloning distillation, not diablo train",
    }

    run = args.run_dir
    for sub in ("checkpoints", "selection"):
        (run / sub).mkdir(parents=True, exist_ok=True)

    policy = PPO.load(str(args.policy), device="cpu")
    policy.diablo_contract_sha256 = contract["sha256"]

    ckpt = run / "checkpoints" / "policy.zip"
    best = run / "selection" / "policy-best.zip"
    policy.save(str(ckpt))
    policy.save(str(best))

    (run / "resolved.json").write_text(
        json.dumps(resolved, indent=2, ensure_ascii=False), encoding="utf-8")
    (run / "selection" / "resolved.json").write_text(
        json.dumps(resolved, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"contract sha256: {contract['sha256'][:32]}…")
    print(f"action_mode    : {contract['action_mode']}")
    print(f"obs shape      : {contract['observation']['shape']}")
    print(f"写入 {ckpt}")
    print(f"写入 {best}")
    print(f"写入 {run / 'resolved.json'} 与 selection/resolved.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
