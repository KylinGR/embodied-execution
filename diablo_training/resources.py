"""Fetch an immutable upstream model without modifying its files."""

import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def source_path():
    lock = json.loads((ROOT / "upstream.lock.json").read_text())
    return ROOT / lock["directory"], lock


def fetch():
    destination, lock = source_path()
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--no-checkout", lock["repository"], str(destination)],
            check=True,
            timeout=120,
        )
        subprocess.run(
            ["git", "-C", str(destination), "checkout", "--detach", lock["commit"]],
            check=True,
            timeout=30,
        )
    verify_source()
    print(f"Model source ready: {destination} @ {lock['commit']}")


def verify_source():
    destination, lock = source_path()
    if not (destination / ".git").exists():
        raise FileNotFoundError("Model source is missing. Run: diablo fetch")
    revision = subprocess.check_output(
        ["git", "-C", str(destination), "rev-parse", "HEAD"],
        text=True,
        timeout=10,
    ).strip()
    changes = subprocess.check_output(
        [
            "git",
            "-C",
            str(destination),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        text=True,
        timeout=10,
    ).strip()
    if revision != lock["commit"] or changes:
        raise RuntimeError(
            "Upstream model checkout differs from its lock; preserve your changes and use a fresh checkout."
        )
    robot = destination / lock["robot"]
    if not robot.is_file():
        raise FileNotFoundError(robot)
    return robot
