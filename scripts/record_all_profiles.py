"""Record one headless rollout video for every profile in a training stage."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diablo_training.contract import load_course, load_run_contract, stage_profiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage", default="flat_adaptive_speed")
    parser.add_argument(
        "--profile-set",
        choices=("all", "baseline", "target", "diagnostic"),
        default="all",
    )
    parser.add_argument("--seed", type=int, default=2001)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the recording commands without running them",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    contract = checkpoint.parent / "contract.json"
    if not contract.is_file():
        raise FileNotFoundError(
            f"Missing {contract}; keep the checkpoint in its training run directory"
        )
    course = load_course()
    try:
        stage = course.stage(args.stage)
        all_profiles = stage["profiles"]
        selected = stage_profiles(
            stage,
            profile_set=args.profile_set,
        )
        profiles = [
            (all_profiles.index(profile), profile)
            for profile in selected
        ]
    except ValueError as error:
        raise SystemExit(f"Invalid stage '{args.stage}': {error}") from error
    load_run_contract(contract, args.stage)

    out_root = (
        args.out
        if args.out is not None
        else ROOT
        / "runs"
        / "videos"
        / checkpoint.parent.name
        / (
            f"{checkpoint.stem}-seed{args.seed}-"
            f"{datetime.now():%Y%m%dT%H%M%S}"
        )
    ).resolve()
    if out_root.is_file() or (
        out_root.is_dir() and any(out_root.iterdir())
    ):
        raise FileExistsError(
            f"Output directory is not empty: {out_root}; choose another --out"
        )

    commands = []
    for profile_index, profile in profiles:
        output = out_root / f"{profile_index:02d}_{profile['name']}"
        commands.append(
            [
                sys.executable,
                str(ROOT / "record_headless.py"),
                "--checkpoint",
                str(checkpoint),
                "--stage",
                args.stage,
                "--seed",
                str(args.seed),
                "--profile-index",
                str(profile_index),
                "--device",
                args.device,
                "--out",
                str(output),
            ]
        )

    print(f"Output: {out_root}")
    failed = []
    for index, command in enumerate(commands, start=1):
        profile_name = profiles[index - 1][1]["name"]
        print(f"[{index}/{len(commands)}] {profile_name}", flush=True)
        if args.dry_run:
            print(" ", shlex.join(command))
            continue
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode:
            failed.append(profile_name)

    if args.dry_run:
        return 0
    if failed:
        print(f"Failed profiles: {', '.join(failed)}", file=sys.stderr)
        return 1
    print(f"Recorded {len(commands)} profiles under {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
