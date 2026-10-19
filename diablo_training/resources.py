"""Fetch an immutable upstream model without modifying its files."""

import json
import os
from pathlib import Path
import subprocess

PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path.cwd()
ASSETS = (
    Path(
        os.environ.get(
            "DIABLO_ASSET_DIR",
            str(
                PROJECT
                if (PROJECT / "pyproject.toml").is_file()
                else Path.home() / ".cache" / "embodied-execution"
            ),
        )
    )
    .expanduser()
    .resolve()
)


def source_path():
    lock = json.loads(Path(__file__).with_name("upstream.lock.json").read_text())
    return ASSETS / lock["directory"], lock


def fetch():
    _, lock = source_path()
    for entry in (lock, lock["mechanism"]):
        destination = ASSETS / entry["directory"]
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--no-checkout",
                    entry["repository"],
                    str(destination),
                ],
                check=True,
                timeout=120,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(destination),
                    "checkout",
                    "--detach",
                    entry["commit"],
                ],
                check=True,
                timeout=30,
            )
    verify_source()
    print(f"Pinned model and mechanism sources ready in: {ASSETS / 'third_party'}")


def verify_source():
    destination, lock = source_path()
    for entry in (lock, lock["mechanism"]):
        _verify_checkout(ASSETS / entry["directory"], entry)
    robot = destination / lock["robot"]
    if not robot.is_file():
        raise FileNotFoundError(robot)
    return robot


def mechanism_path():
    _, lock = source_path()
    return ASSETS / lock["mechanism"]["directory"] / lock["mechanism"]["file"]


def _verify_checkout(destination, lock):
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
