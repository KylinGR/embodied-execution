"""Course loading, gate checks, scoring, and checkpoint consistency helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

PACKAGE_DIR = Path(__file__).resolve().parent
CURRICULUM_PATH = PACKAGE_DIR / "curriculum.json"
SPEED_STAGES_PATH = PACKAGE_DIR / "flat_motion_speed_stages.json"

TARGET_BASE_HEIGHT_M = 0.270943
TARGET_WHOLE_HEIGHT_M = 0.37958
PROPRIO_HISTORY_LENGTH = 5
OBSERVATION_SHAPE = (125,)
ACTION_SHAPE = (6,)
OBSERVATION_DTYPE = "float32"
ACTION_DTYPE = "float32"
HEADING_ERROR_OBS_SCALE = 5.0
PROPRIO_FRAME_DIM = OBSERVATION_SHAPE[0] // PROPRIO_HISTORY_LENGTH
SCREEN_SEEDS = (2001, 2002, 2003)
VERIFY_SEEDS = (2004, 2005, 2006, 2007, 2008)
SELECTION_PROBE_SEED = 2001
SELECTION_VERIFY_SEEDS = (2002, 2003, 2004, 2005, 2006)
BLIND_SEEDS = tuple(range(3101, 3111))
LIMIT_SEEDS = (4101, 4102, 4103)

TASK_IDS = {
    "flat_balance": "Mjlab-Xingtian-Flat-Balance",
    "flat_motion": "Mjlab-Xingtian-Flat-Motion",
    "flat_turning": "Mjlab-Xingtian-Flat-Turning",
    "flat_start_stop": "Mjlab-Xingtian-Flat-StartStop",
    "slope": "Mjlab-Xingtian-Slope",
    "rough": "Mjlab-Xingtian-Rough",
    "roller_obstacle": "Mjlab-Xingtian-Obstacles",
    "composite": "Mjlab-Xingtian-Composite",
}

_PROPORTIONAL_MINIMUMS = {
    "min_forward_progress_m",
    "min_obstacle_passes",
    "min_stop_cycles",
}
_MAX_EQUIVALENT = {"mean_height_tolerance_m"}
_UNIT_MINIMUMS = {
    "min_duration_fraction",
    "min_pass_rate",
    "min_planned_peak_progress_fraction",
    "min_upright",
}


@dataclass(frozen=True)
class Course:
    stages: dict[str, dict[str, Any]]
    order: tuple[str, ...]
    reward_defaults: dict[str, Any]
    reward_penalties: dict[str, float]
    randomization: dict[str, float]
    validation_seeds: tuple[int, ...]
    blind_seeds: tuple[int, ...]
    limit_seeds: tuple[int, ...]

    def stage(self, name: str) -> dict[str, Any]:
        try:
            return self.stages[name]
        except KeyError as error:
            raise ValueError(f"Unknown stage '{name}'") from error


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_course() -> Course:
    """Load both supplied courses and apply the hardware height correction."""
    base = _read_json(CURRICULUM_PATH)
    speed = _read_json(SPEED_STAGES_PATH)
    for source in (base, speed):
        if float(source.get("target_base_height_m", -1)) != TARGET_BASE_HEIGHT_M:
            raise ValueError(
                "Course target_base_height_m must match the calibrated "
                f"{TARGET_BASE_HEIGHT_M} m model height"
            )
        if float(source.get("target_whole_height_m", -1)) != TARGET_WHOLE_HEIGHT_M:
            raise ValueError("Course target_whole_height_m is inconsistent")
    stages = dict(base["stages"])

    overlap = set(stages) & set(speed["stages"])
    if overlap:
        raise ValueError(f"Duplicate curriculum stages: {sorted(overlap)}")
    for name in speed["order"]:
        stages[name] = speed["stages"][name]

    order = (
        "flat_balance",
        *speed["order"],
        "flat_motion",
        "flat_turning",
        "flat_start_stop",
        "slope",
        "rough",
        "roller_obstacle",
        "composite",
    )
    if set(order) != set(stages) or len(order) != len(stages):
        raise ValueError("Curriculum stage order does not cover every stage exactly once")
    for name in order:
        _validate_stage(name, stages[name])
    if set(SCREEN_SEEDS) & set(VERIFY_SEEDS):
        raise ValueError("Screen and verification seeds must be disjoint")
    exposed = set(SCREEN_SEEDS) | set(VERIFY_SEEDS)
    if exposed & (set(BLIND_SEEDS) | set(LIMIT_SEEDS)):
        raise ValueError("Blind and limit seeds must not participate in selection")
    selection_exposed = {SELECTION_PROBE_SEED, *SELECTION_VERIFY_SEEDS}
    if selection_exposed & (set(BLIND_SEEDS) | set(LIMIT_SEEDS)):
        raise ValueError("Blind and limit seeds must not participate in selection")

    return Course(
        stages=stages,
        order=order,
        reward_defaults=base["reward_defaults"],
        reward_penalties=base["reward_defaults"]["common_penalties"],
        randomization=base["randomization"],
        validation_seeds=tuple(base["validation_seeds"]),
        blind_seeds=tuple(base["blind_seeds"]),
        limit_seeds=tuple(base["limit_seeds"]),
    )


def _validate_stage(name: str, stage: dict[str, Any]) -> None:
    if not stage.get("profiles"):
        raise ValueError(f"{name}: at least one command profile is required")
    if int(stage.get("budget", 0)) <= 0:
        raise ValueError(f"{name}: budget must be positive")
    previous = -math.inf
    for profile in stage["profiles"]:
        points = profile.get("points") or []
        if len(points) < 2:
            raise ValueError(f"{name}/{profile.get('name')}: profile is too short")
        times = [float(point[0]) for point in points]
        if times[0] != 0 or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError(
                f"{name}/{profile.get('name')}: profile times must start at zero "
                "and strictly increase"
            )
        previous = max(previous, times[-1])
    if previous <= 0:
        raise ValueError(f"{name}: episode must have positive duration")


def profile_command(
    profile: dict[str, Any], seconds: float
) -> tuple[float, float]:
    """Linearly interpolate a course profile and hold its terminal value."""
    points = profile["points"]
    if seconds <= points[0][0]:
        return float(points[0][1]), float(points[0][2])
    for start, end in zip(points, points[1:]):
        if seconds <= end[0]:
            span = float(end[0] - start[0])
            blend = 0.0 if span == 0 else (seconds - float(start[0])) / span
            return (
                float(start[1] + blend * (end[1] - start[1])),
                float(start[2] + blend * (end[2] - start[2])),
            )
    return float(points[-1][1]), float(points[-1][2])


def episode_seconds(stage: dict[str, Any]) -> float:
    return max(float(profile["points"][-1][0]) for profile in stage["profiles"])


def budget_iterations(budget: int, num_envs: int, steps_per_env: int = 24) -> int:
    if min(budget, num_envs, steps_per_env) <= 0:
        raise ValueError("Budget, environment count, and rollout length must be positive")
    return math.ceil(budget / (num_envs * steps_per_env))


def checkpoint_interval(iterations: int, percent: int = 10) -> int:
    """Return the iteration interval for roughly the requested percentage."""
    if iterations <= 0 or not 0 < percent <= 100:
        raise ValueError("Iterations must be positive and percent must be in (0, 100]")
    return max(1, iterations * percent // 100)


def task_id(stage_name: str) -> str:
    if stage_name in TASK_IDS:
        return TASK_IDS[stage_name]
    if stage_name.startswith("flat_motion_"):
        speed = stage_name.removeprefix("flat_motion_")
        if speed.isdigit():
            return f"Mjlab-Xingtian-Speed-{speed}"
    raise ValueError(f"No MJLab task ID exists for stage '{stage_name}'")


def stage_task_ids(course: Course) -> dict[str, str]:
    return {name: task_id(name) for name in course.order}


def metric_name(gate_name: str) -> str:
    if gate_name.startswith("max_"):
        return gate_name[4:]
    if gate_name.startswith("min_"):
        return gate_name[4:]
    if gate_name == "mean_height_tolerance_m":
        return "mean_height_error_m"
    if gate_name == "require_finish":
        return "finished"
    raise ValueError(f"Unsupported gate key '{gate_name}'")


def gate_failures(metrics: dict[str, float | bool], gate: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for key, limit in gate.items():
        name = metric_name(key)
        if name not in metrics:
            failures.append(f"{key}: metric '{name}' is missing")
            continue
        value = metrics[name]
        if key == "require_finish":
            if bool(value) != bool(limit):
                failures.append(f"{key}: expected {limit}, got {value}")
        elif key.startswith("max_") or key in _MAX_EQUIVALENT:
            if float(value) > float(limit):
                failures.append(f"{key}: {value} > {limit}")
        elif float(value) < float(limit):
            failures.append(f"{key}: {value} < {limit}")
    return failures


def _quality(key: str, value: float, limit: float) -> float:
    if key.startswith("max_") or key in _MAX_EQUIVALENT:
        if limit <= 0:
            return 1.0 if value <= 0 else 0.0
        return float(max(0.0, min(1.0, 1.0 - value / limit)))
    if key in _UNIT_MINIMUMS:
        if limit >= 1.0:
            return 1.0 if value >= limit else 0.0
        return float(max(0.0, min(1.0, (value - limit) / (1.0 - limit))))
    if key in _PROPORTIONAL_MINIMUMS:
        if limit <= 0:
            return 1.0
        return float(max(0.0, min(1.0, value / limit - 1.0)))
    raise ValueError(f"Unsupported score gate '{key}'")


def gate_score(metrics: dict[str, float | bool], gate: dict[str, Any]) -> dict[str, Any]:
    """Compute the agreed robust 25/25/25/15/10 policy score."""
    values: dict[str, float] = {}
    for key, limit in gate.items():
        if key == "min_pass_rate":
            continue
        name = metric_name(key)
        if key == "require_finish":
            values[key] = 1.0 if bool(metrics.get(name, False)) else 0.0
        else:
            values[key] = _quality(key, float(metrics[name]), float(limit))

    def mean(*keys: str) -> float:
        present = [values[key] for key in keys if key in values]
        return sum(present) / len(present) if present else 0.0

    components = {
        "completion": mean(
            "min_duration_fraction",
            "min_forward_progress_m",
            "require_finish",
        ),
        "tracking": mean("max_speed_mae_mps", "max_yaw_mae_rad_s"),
        "stability": mean(
            "min_upright",
            "mean_height_tolerance_m",
            "max_height_deviation_m",
            "max_wheel_center_offset_m",
            "max_straight_wheel_speed_difference_mps",
        ),
        "terrain": mean("min_obstacle_passes", "max_backslide_mps"),
        "efficiency": mean("max_wheel_slip_mps"),
    }
    power = max(0.0, float(metrics.get("mean_positive_power_w", 0.0)))
    components["efficiency"] = 0.9 * components["efficiency"] + 0.1 * max(
        0.0, 1.0 - power / 235.2
    )

    score = 100.0 * (
        0.25 * components["completion"]
        + 0.25 * components["tracking"]
        + 0.25 * components["stability"]
        + 0.15 * components["terrain"]
        + 0.10 * components["efficiency"]
    )
    passed = not gate_failures(metrics, gate)
    return {
        "score": round(score, 6),
        "components": components,
        "minimum_margin": round(min(values.values(), default=0.0), 6),
        "passed": passed,
        "failures": gate_failures(metrics, gate),
    }


def fully_passes(
    runs: Iterable[dict[str, Any]], gate: dict[str, Any]
) -> bool:
    return all(not gate_failures(run, gate) for run in runs)


def rank_policy_candidates(candidates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return passed candidates in the agreed deterministic score order."""
    eligible = [candidate for candidate in candidates if candidate.get("passed")]
    if not eligible:
        raise ValueError("No candidate checkpoint passed every gate on all five seeds")
    return sorted(
        eligible,
        key=lambda candidate: (
            -float(candidate["score"]),
            -float(candidate["worst_seed_score"]),
            -float(candidate["minimum_margin"]),
            float(candidate["mean_speed_mae_mps"]),
            float(candidate["mean_yaw_mae_rad_s"]),
            int(candidate["iteration"]),
            str(candidate["checkpoint"]),
        ),
    )


def select_policy_best(candidates: Iterable[dict[str, Any]]) -> dict[str, Any]:
    return rank_policy_candidates(candidates)[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def policy_interface_contract(course: Course) -> dict[str, Any]:
    return {
        "version": 7,
        "backend": "mjlab-rsl-rl",
        "mjlab": "1.6.0",
        "mujoco": "3.11.x",
        "observation_shape": OBSERVATION_SHAPE[0],
        "observation_dtype": OBSERVATION_DTYPE,
        "observation_breakdown": {
            "proprio_history": OBSERVATION_SHAPE[0],
        },
        "action_shape": ACTION_SHAPE[0],
        "action_dtype": ACTION_DTYPE,
        "action_order": [
            "left_hip_motor",
            "left_knee_motor",
            "left_wheel_motor",
            "right_hip_motor",
            "right_knee_motor",
            "right_wheel_motor",
        ],
        "control_dt_s": 0.01,
        "target_base_height_m": TARGET_BASE_HEIGHT_M,
        "target_whole_height_m": TARGET_WHOLE_HEIGHT_M,
    }


def interface_contract(course: Course) -> dict[str, Any]:
    contract = policy_interface_contract(course)
    contract.update(
        {
            "stage_order": list(course.order),
            "task_ids": stage_task_ids(course),
            "source_sha256": {
                path.name: sha256(path)
                for path in (
                    PACKAGE_DIR / "model.py",
                    PACKAGE_DIR / "control.py",
                    PACKAGE_DIR / "mjlab_tasks.py",
                    PACKAGE_DIR / "upstream.lock.json",
                    CURRICULUM_PATH,
                    SPEED_STAGES_PATH,
                )
            },
            "screen_seeds": list(SCREEN_SEEDS),
            "verify_seeds": list(VERIFY_SEEDS),
            "blind_seeds": list(BLIND_SEEDS),
            "limit_seeds": list(LIMIT_SEEDS),
        }
    )
    contract["sha256"] = hashlib.sha256(
        json.dumps(contract, sort_keys=True).encode()
    ).hexdigest()
    return contract


def verify_interface_contract(content: dict[str, Any]) -> None:
    expected = content.get("sha256")
    unsigned = {key: value for key, value in content.items() if key != "sha256"}
    actual = hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest()
    if expected != actual:
        raise ValueError("MJLab run contract is missing or inconsistent")
    current = policy_interface_contract(load_course())
    mismatched = [
        field
        for field, expected_value in current.items()
        if content.get(field) != expected_value
    ]
    if mismatched:
        raise ValueError(
            "Checkpoint policy interface is incompatible; mismatched fields: "
            + ", ".join(mismatched)
        )


def verify_run_contract(
    payload: dict[str, Any],
    stage_name: str | None = None,
) -> dict[str, Any]:
    contract = payload.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("Run contract is missing the nested 'contract' object")
    verify_interface_contract(contract)
    actual_stage = payload.get("stage")
    if stage_name is not None and actual_stage != stage_name:
        raise ValueError(
            f"Run stage is {actual_stage}, requested {stage_name}"
        )
    return contract


def load_run_contract(
    path: Path,
    stage_name: str | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing run contract: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return verify_run_contract(payload, stage_name)


if __name__ == "__main__":
    loaded = load_course()
    print(json.dumps(interface_contract(loaded), indent=2, ensure_ascii=False))
