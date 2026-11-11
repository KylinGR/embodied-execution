"""Select a stage checkpoint using fixed command profiles and development seeds."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import time

import numpy as np

from .cli import load_policy, write_json
from .contract import environment_class
from .xingtian_env import XingtianFlatEnv, XingtianStageConfig
from .xingtian_eval import ProfileXingtianEnvMixin, STAGES


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=2001)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=tuple(STAGES),
        default=("stand", "forward", "backward", "turn", "start_stop"),
    )
    parser.add_argument("--min-success", type=float, default=1.0)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return status 2 when no checkpoint reaches --min-success",
    )
    return parser.parse_args()


def checkpoint_paths(run_dir: Path) -> list[Path]:
    paths = sorted((run_dir / "checkpoints").glob("*.zip"))
    final = run_dir / "policy.zip"
    if final.is_file():
        paths.append(final)
    return sorted(set(path.resolve() for path in paths))


def evaluate_checkpoint(path: Path, stages: tuple[str, ...], seed: int):
    model, config, resolved = load_policy(path)
    env_type = environment_class(resolved["environment_class"])
    if not issubclass(env_type, XingtianFlatEnv):
        raise TypeError("select-best requires an XingtianFlatEnv-derived task")
    if isinstance(config, XingtianStageConfig):
        config = replace(config, terrain_seed=seed + 50_000)

    class EvalEnv(ProfileXingtianEnvMixin, env_type):
        pass

    env = EvalEnv(config)
    reports = []
    try:
        for stage_name in stages:
            stage = STAGES[stage_name]
            obs, _ = env.reset(seed=seed, options={"profile": stage["profile"]})
            speed_errors, yaw_errors, uprights = [], [], []
            terminated = False
            info = {}
            for _ in range(round(stage["duration"] / 0.01)):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                command = info["command"]
                speed_errors.append(abs(info["forward_mps"] - command[0]))
                yaw_errors.append(
                    abs(float(env.data.sensor("gyro").data[2]) - command[1])
                )
                uprights.append(abs(info["upright"]))
                if terminated or truncated:
                    break
            success = not terminated and (
                info.get("sim_seconds", 0.0) >= stage["duration"] - 0.02
                or info.get("sim_seconds", 0.0) >= config.episode_seconds - 0.02
            )
            reports.append(
                {
                    "stage": stage_name,
                    "success": bool(success),
                    "sim_seconds": float(info.get("sim_seconds", 0.0)),
                    "speed_mae_mps": float(np.mean(speed_errors)),
                    "yaw_mae_rad_s": float(np.mean(yaw_errors)),
                    "min_upright": float(np.min(uprights)),
                }
            )
    finally:
        env.close()

    success_rate = float(np.mean([item["success"] for item in reports]))
    speed_error = float(np.mean([item["speed_mae_mps"] for item in reports]))
    yaw_error = float(np.mean([item["yaw_mae_rad_s"] for item in reports]))
    min_upright = float(np.min([item["min_upright"] for item in reports]))
    score = (
        100.0 * (1.0 - success_rate)
        + 100.0 * speed_error
        + 20.0 * yaw_error
        + 20.0 * max(0.0, 0.8 - min_upright)
    )
    return {
        "checkpoint": str(path),
        "timesteps": int(model.num_timesteps),
        "score": score,
        "success_rate": success_rate,
        "speed_mae_mps": speed_error,
        "yaw_mae_rad_s": yaw_error,
        "min_upright": min_upright,
        "stages": reports,
    }


def run_selection(args):
    run_dir = args.run.resolve()
    if not (run_dir / "resolved.json").is_file():
        raise FileNotFoundError(f"Missing {run_dir / 'resolved.json'}")
    paths = checkpoint_paths(run_dir)
    if not paths:
        raise FileNotFoundError("No checkpoint .zip files found")
    output = (args.output or run_dir / "selection").resolve()
    output.mkdir(parents=True, exist_ok=False)

    began = time.monotonic()
    candidates = []
    for path in paths:
        print(f"Evaluating {path}", flush=True)
        candidates.append(evaluate_checkpoint(path, tuple(args.stages), args.seed))
    qualified = [
        item for item in candidates if item["success_rate"] >= args.min_success
    ]
    pool = qualified or candidates
    best = min(pool, key=lambda item: (item["score"], item["timesteps"]))
    shutil.copy2(best["checkpoint"], output / "policy-best.zip")
    shutil.copy2(run_dir / "resolved.json", output / "resolved.json")
    result = {
        "run": str(run_dir),
        "seed": args.seed,
        "stages": list(args.stages),
        "min_success": args.min_success,
        "wall_seconds": time.monotonic() - began,
        "best": best,
        "qualified": len(qualified),
        "candidates": candidates,
    }
    write_json(output / "selection.json", result)
    print(json.dumps(result["best"], ensure_ascii=False, indent=2))
    return 2 if args.strict and not qualified else 0


def main():
    return run_selection(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
