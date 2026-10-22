"""Run paired PPO baselines and a tilt-cost ablation, retaining all failures.

Use the project's Python environment. Assets must already pass `diablo doctor`.
Outputs are local experiment artifacts, not pre-trained walking policies.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys


def run(arguments, log, environment, allowed=(0,)):
    with log.open("x") as stream:
        result = subprocess.run(
            [sys.executable, "-m", "diablo_training.cli", *arguments],
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=environment,
        )
    if result.returncode not in allowed:
        raise RuntimeError(f"Command failed ({result.returncode}); see {log}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=16384)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps <= 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("steps must be positive and seeds must be unique")
    root = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(root / "examples"), str(root), environment.get("PYTHONPATH", "")]
    )
    manifest = {
        "steps_requested_per_run": args.steps,
        "training_seeds": args.seeds,
        "evaluation_seed": 314,
        "n_envs": 4,
        "hypothesis": "Extra 2*(1-upright) cost may improve balance",
        "limitation": "Short paired pilot, not convergence or locomotion qualification",
    }
    (out / "experiment.json").write_text(json.dumps(manifest, indent=2) + "\n")
    columns = ["variant", "seed", "task", "steps", "sim_seconds", "terminated", "speed_mae_mps"]
    with (out / "comparison.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for seed in args.seeds:
            for variant in ("baseline", "upright"):
                name = f"{variant}-{seed}"
                training, evaluation = out / name, out / f"{name}-eval"
                command = ["train", "--steps", str(args.steps), "--seed", str(seed),
                           "--n-envs", "4", "--output", str(training)]
                if variant == "upright":
                    command += ["--config", str(root / "examples/upright_reward.json")]
                run(command, out / f"{name}.log", environment)
                result = json.loads((training / "result.json").read_text())
                if not result["parameters_changed"]:
                    raise RuntimeError(f"Policy did not update: {training}")
                run(["evaluate", "--model", str(training / "policy.zip"),
                     "--seed", "314", "--output", str(evaluation)],
                    out / f"{name}-eval.log", environment, allowed=(0, 2))
                report = json.loads((evaluation / "result.json").read_text())
                for episode in report["episodes"]:
                    writer.writerow({"variant": variant, "seed": seed,
                                     "task": episode["task"],
                                     "steps": result["actual_total_timesteps"],
                                     **{key: episode[key] for key in
                                        ("sim_seconds", "terminated", "speed_mae_mps")}})
                stream.flush()
                print(f"Completed {name}: {training}", flush=True)
    print(f"Evidence: {out / 'comparison.csv'}", flush=True)


if __name__ == "__main__":
    main()
