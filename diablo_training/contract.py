"""Course loading, gate checks, scoring, and checkpoint consistency helpers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

PACKAGE_DIR = Path(__file__).resolve().parent
CURRICULUM_PATH = PACKAGE_DIR / "curriculum.json"
SPEED_STAGES_PATH = PACKAGE_DIR / "flat_motion_speed_stages.json"
TURN_STAGES_PATH = PACKAGE_DIR / "flat_turning_speed_stages.json"
ADAPTIVE_STAGES_PATH = PACKAGE_DIR / "flat_adaptive_speed_stages.json"
OBJECTIVES_PATH = PACKAGE_DIR / "training_objectives.json"

TARGET_BASE_HEIGHT_M = 0.350
TARGET_STANDING_HEIGHT_M = 0.477859
RESET_CLEARANCE_M = 0.005
RESET_BASE_HEIGHT_M = TARGET_BASE_HEIGHT_M + RESET_CLEARANCE_M
TARGET_WHOLE_HEIGHT_M = 0.491
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
    "flat_turn_in_place_040": "Mjlab-Xingtian-Turn-InPlace-040",
    "flat_turn_moving_040": "Mjlab-Xingtian-Turn-Moving-040",
    "flat_turn_in_place_080": "Mjlab-Xingtian-Turn-InPlace-080",
    "flat_turn_moving_080": "Mjlab-Xingtian-Turn-Moving-080",
    "flat_turn_in_place_120": "Mjlab-Xingtian-Turn-InPlace-120",
    "flat_turn_moving_120": "Mjlab-Xingtian-Turn-Moving-120",
    "flat_turn_in_place_160_200": "Mjlab-Xingtian-Turn-InPlace-160-200",
    "flat_adaptive_speed": "Mjlab-Xingtian-Flat-AdaptiveSpeed",
}

_PROPORTIONAL_MINIMUMS = {
    "min_forward_progress_m",
    "min_obstacle_passes",
    "min_stop_cycles",
    "min_turn_radius_m",
}
_MAX_EQUIVALENT = {"mean_height_tolerance_m"}
_UNIT_MINIMUMS = {
    "min_duration_fraction",
    "min_pass_rate",
    "min_planned_peak_progress_fraction",
    "min_cruise_speed_fraction",
    "min_upright",
}
_TRAINING_GATE_KEYS = {
    "max_cross_track_error_m",
    "max_s_turn_path_error_m",
    "max_segment_turn_angle_error_rad",
    "max_settled_mean_height_error_m",
    "max_wheel_slip_mps",
    "max_settled_wheel_slip_mps",
    "max_mean_steady_wheel_slip_mps",
    "max_transient_wheel_slip_mps",
    "max_wheel_slip_fraction_of_peak",
    "max_settled_wheel_slip_fraction_of_peak",
    "max_stop_speed_mps",
    "max_cruise_speed_mae_mps",
    "max_turn_angle_error_rad",
    "max_yaw_rate_peak_rad_s",
    "max_straight_wheel_speed_difference_mps",
    "max_settled_straight_wheel_speed_difference_mps",
    "min_cruise_speed_fraction",
    "min_forward_progress_m",
}
_TRAINING_INTERFACE_FIELDS = {
    "observation_shape",
    "observation_dtype",
    "observation_breakdown",
    "action_shape",
    "action_dtype",
    "action_order",
    "control_dt_s",
    "target_base_height_m",
    "target_standing_height_m",
    "reset_base_height_m",
}
KNOWN_METRICS = {
    "pass_rate",
    "duration_fraction",
    "speed_mae_mps",
    "speed_mae_fraction_of_peak",
    "cruise_speed_mps",
    "cruise_speed_mae_mps",
    "cruise_speed_fraction",
    "planned_peak_progress_fraction",
    "yaw_mae_rad_s",
    "yaw_underspeed_rad_s",
    "yaw_rate_peak_rad_s",
    "heading_error_max_deg",
    "turn_angle_error_rad",
    "s_turn_path_error_m",
    "segment_turn_angle_error_rad",
    "turn_radius_min_m",
    "turn_radius_max_m",
    "upright",
    "mean_abs_pitch_deg",
    "pitch_deg",
    "mean_abs_steady_pitch_deg",
    "mean_abs_roll_deg",
    "roll_deg",
    "final_heading_error_deg",
    "heading_drift_after_0.5s_deg",
    "final_speed_mps",
    "mean_height_error_m",
    "height_deviation_m",
    "settled_mean_height_error_m",
    "settled_height_deviation_m",
    "wheel_slip_mps",
    "settled_wheel_slip_mps",
    "mean_steady_wheel_slip_mps",
    "transient_wheel_slip_mps",
    "wheel_slip_fraction_of_peak",
    "settled_wheel_slip_fraction_of_peak",
    "backslide_mps",
    "cross_track_error_m",
    "wheel_center_offset_m",
    "straight_wheel_speed_difference_mps",
    "settled_straight_wheel_speed_difference_mps",
    "mean_settled_straight_wheel_speed_difference_mps",
    "hold_drift_m",
    "hold_drift_after_0.5s_m",
    "in_place_displacement_m",
    "stop_speed_mps",
    "stop_speed_fraction_of_peak",
    "stop_cycles",
    "obstacle_passes",
    "forward_progress_m",
    "finished",
    "obstacle_passes",
}
_WARM_START_FIELDS = {
    "version",
    "policy_architecture",
    "observation_shape",
    "observation_dtype",
    "observation_breakdown",
    "action_shape",
    "action_dtype",
    "action_order",
    "control_dt_s",
}


@dataclass(frozen=True)
class Course:
    stages: dict[str, dict[str, Any]]
    order: tuple[str, ...]
    turn_order: tuple[str, ...]
    adaptive_order: tuple[str, ...]
    objectives: dict[str, Any]
    gate_templates: dict[str, dict[str, Any]]
    ppo: dict[str, Any]
    speed_ppo: dict[str, Any]
    turn_ppo: dict[str, Any]
    reward_defaults: dict[str, Any]
    turn_reward_defaults: dict[str, Any]
    rehearsal_gate: dict[str, Any] | None
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

    @property
    def all_order(self) -> tuple[str, ...]:
        return (
            self.order
            + self.turn_order
            + self.adaptive_order
        )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


_PARAMETER_REFERENCE = re.compile(r"^(-?)\$\{([A-Za-z0-9_.]+)\}$")


def _resolve_config_parameters(
    value: Any,
    parameters: dict[str, Any],
    location: str = "config",
) -> Any:
    if isinstance(value, dict):
        return {
            key: _resolve_config_parameters(
                item,
                parameters,
                f"{location}.{key}",
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _resolve_config_parameters(
                item,
                parameters,
                f"{location}[{index}]",
            )
            for index, item in enumerate(value)
        ]
    if not isinstance(value, str) or "${" not in value:
        return value
    match = _PARAMETER_REFERENCE.fullmatch(value)
    if match is None:
        raise ValueError(f"{location}: invalid parameter reference '{value}'")
    sign, path = match.groups()
    resolved: Any = parameters
    for part in path.split("."):
        if not isinstance(resolved, dict) or part not in resolved:
            raise ValueError(f"{location}: unknown parameter '{path}'")
        resolved = resolved[part]
    if sign == "-":
        if isinstance(resolved, bool) or not isinstance(resolved, (int, float)):
            raise ValueError(f"{location}: negated parameter '{path}' must be numeric")
        return -float(resolved)
    if isinstance(resolved, (int, float, bool, str, list, dict)):
        return deepcopy(resolved)
    raise ValueError(f"{location}: parameter '{path}' has unsupported type")


def _merge_gate_layers(*layers: dict[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for layer in layers:
        if not layer:
            continue
        for key, value in layer.items():
            if key == "remove_gate_keys":
                continue
            if value is None:
                merged.pop(key, None)
            else:
                merged[key] = value
    for layer in layers:
        if not layer:
            continue
        for key in layer.get("remove_gate_keys", ()):
            merged.pop(str(key), None)
    return merged


def _validate_objectives(objectives: dict[str, Any]) -> None:
    try:
        review = objectives["review"]
        target = objectives["target"]
        baseline = objectives["baseline"]
        safety = objectives["safety"]
        timing = objectives["timing"]
        grid = objectives["adaptive_grid"]
        gates = objectives["gates"]
    except KeyError as error:
        raise ValueError(f"training_objectives.json is missing {error.args[0]}") from error
    if not math.isclose(float(review["straight_speed_min_mps"]), 0.2):
        raise ValueError("review.straight_speed_min_mps must be 0.2")
    if float(review["straight_speed_max_mps"]) < float(
        review["straight_speed_min_mps"]
    ):
        raise ValueError("review speed maximum is below its minimum")
    if float(target["straight_speed_max_mps"]) < float(
        review["straight_speed_max_mps"]
    ):
        raise ValueError("target speed must not be below review speed")
    if float(review["in_place_yaw_min_rad_s"]) < 0.4:
        raise ValueError("review in-place yaw minimum must be at least 0.4 rad/s")
    if float(target["in_place_yaw_max_rad_s"]) < float(
        review["in_place_yaw_min_rad_s"]
    ):
        raise ValueError("target in-place yaw is below the review minimum")
    protected_cells = baseline["protected_cells"]
    if not isinstance(protected_cells, dict) or set(protected_cells) != {
        "balance",
        "straight",
        "in_place_turn",
        "moving_turn",
    }:
        raise ValueError("baseline protected_cells must define every task mode")
    health_gate = grid.get("health_gate")
    if not isinstance(health_gate, dict) or not health_gate:
        raise ValueError("adaptive_grid.health_gate must be a non-empty object")
    if "min_pass_rate" in health_gate:
        raise ValueError("adaptive health gate must not contain min_pass_rate")
    for key in health_gate:
        if metric_name(str(key)) not in KNOWN_METRICS:
            raise ValueError(f"adaptive health gate '{key}' has no metric")
    if float(target["moving_turn_yaw_max_rad_s"]) > 1.2:
        raise ValueError("moving-turn yaw cap must not exceed 1.2 rad/s")
    moving_turn_speed = float(target["moving_turn_speed_max_mps"])
    if not math.isfinite(moving_turn_speed) or moving_turn_speed <= 0.0:
        raise ValueError("moving-turn speed cap must be positive")
    if float(review["moving_turn_speed_max_mps"]) > moving_turn_speed:
        raise ValueError("moving-turn review speed exceeds its target speed")
    if moving_turn_speed + 1e-9 < (
        float(safety["min_turn_radius_m"])
        * float(target["moving_turn_yaw_max_rad_s"])
    ):
        raise ValueError(
            "moving-turn speed cap cannot produce the minimum safe radius "
            "at maximum yaw"
        )
    if not 0.0 < float(safety["min_turn_radius_m"]) <= float(
        safety["max_turn_radius_m"]
    ):
        raise ValueError("turn-radius limits are invalid")
    if float(safety["turn_radius_window_min_speed_mps"]) <= 0.0:
        raise ValueError("turn-radius speed window must be positive")
    if not 0.0 < float(
        safety["turn_radius_window_min_yaw_rad_s"]
    ) <= float(target["moving_turn_yaw_max_rad_s"]):
        raise ValueError("turn-radius yaw window is outside the moving-turn range")
    if float(timing["balance_hold_s"]) != 20.0:
        raise ValueError("balance hold must be 20 seconds")
    if float(timing["in_place_hold_s"]) != 10.0:
        raise ValueError("in-place hold must be 10 seconds")
    if float(timing["in_place_yaw_ramp_s"]) != 2.0:
        raise ValueError("in-place yaw ramp must be 2 seconds")
    if float(timing["moving_turn_yaw_ramp_s"]) != 2.0:
        raise ValueError("moving-turn yaw ramp must be 2 seconds")
    if float(timing["straight_accel_s"]) != 4.0 or float(
        timing["straight_decel_s"]
    ) != 4.0:
        raise ValueError("straight acceleration and deceleration must be 4 seconds")
    if int(grid["window_size"]) != int(grid["minimum_window_episodes"]):
        raise ValueError("grid window size must equal minimum_window_episodes")
    if float(grid["vx_resolution_mps"]) <= 0.0 or float(
        grid["wz_resolution_rad_s"]
    ) <= 0.0:
        raise ValueError("adaptive grid resolution must be positive")
    mode_fractions = grid["mode_minimum_fractions"]
    mode_names = {
        "balance",
        "straight",
        "in_place_turn",
        "moving_turn",
    }
    if not isinstance(mode_fractions, dict) or set(mode_fractions) != mode_names:
        raise ValueError(
            "adaptive grid mode_minimum_fractions must define all task modes"
        )
    fractions = [float(mode_fractions[mode]) for mode in mode_names]
    if any(value <= 0.0 for value in fractions) or sum(fractions) > 1.0 + 1e-9:
        raise ValueError(
            "adaptive grid mode minimum fractions must be positive and sum at most 1"
        )
    if float(grid["stop_after_motion_arm_speed_mps"]) <= 0.0:
        raise ValueError("stop-after-motion arm speed must be positive")
    vx_step = float(grid["vx_resolution_mps"])
    wz_step = float(grid["wz_resolution_rad_s"])
    for value in (
        review["straight_speed_min_mps"],
        review["straight_speed_max_mps"],
        target["straight_speed_max_mps"],
    ):
        if not math.isclose(
            float(value) / vx_step,
            round(float(value) / vx_step),
            abs_tol=1e-6,
        ):
            raise ValueError("speed limits must align with vx grid resolution")
    for value in (
        review["in_place_yaw_min_rad_s"],
        target["in_place_yaw_max_rad_s"],
        target["moving_turn_yaw_max_rad_s"],
    ):
        if not math.isclose(
            float(value) / wz_step,
            round(float(value) / wz_step),
            abs_tol=1e-6,
        ):
            raise ValueError("yaw limits must align with wz grid resolution")
    allowed = {
        "balance",
        "straight",
        "in_place_turn",
        "moving_turn",
        "terrain",
        "composite",
    }
    if "common" not in gates or not allowed.issubset(gates):
        raise ValueError(
            "training objectives must define common and all task gate templates"
        )


def _profile_target(profile: dict[str, Any], axis: int) -> float:
    values = [float(point[axis]) for point in profile["points"]]
    value = max(values, key=abs)
    return float(value)


def _set_profile_points(
    profile: dict[str, Any],
    points: list[list[float]],
) -> None:
    profile["points"] = points


def _cap_profile_speed(
    stage: dict[str, Any],
    max_speed_mps: float,
) -> None:
    for profile in (
        *stage["profiles"],
        *stage.get("training_profiles", ()),
    ):
        for point in profile["points"]:
            point[1] = max(
                -max_speed_mps,
                min(max_speed_mps, float(point[1])),
            )


def _apply_objective_timing(
    stage_name: str,
    stage: dict[str, Any],
    objectives: dict[str, Any],
) -> None:
    timing = objectives["timing"]
    if "adaptive_grid" in stage:
        settle = float(timing["straight_settle_s"])
        accel = float(timing["straight_accel_s"])
        cruise = float(timing["straight_cruise_s"])
        decel = float(timing["straight_decel_s"])
        yaw_ramp = float(timing["in_place_yaw_ramp_s"])
        in_place_hold = float(timing["in_place_hold_s"])
        turn_hold = float(timing["moving_turn_hold_s"])
        for profile in stage["profiles"]:
            task_type = profile["task_type"]
            speed = _profile_target(profile, 1)
            yaw = _profile_target(profile, 2)
            if task_type == "balance":
                points = [
                    [0.0, 0.0, 0.0],
                    [float(timing["balance_hold_s"]), 0.0, 0.0],
                ]
            elif task_type == "straight":
                points = [
                    [0.0, 0.0, 0.0],
                    [settle, 0.0, 0.0],
                    [settle + accel, speed, 0.0],
                    [settle + accel + cruise, speed, 0.0],
                    [settle + accel + cruise + decel, 0.0, 0.0],
                    [
                        settle + accel + cruise + decel + settle,
                        0.0,
                        0.0,
                    ],
                ]
            elif task_type == "in_place_turn":
                ramp_end = settle + yaw_ramp
                hold_end = ramp_end + in_place_hold
                down_end = hold_end + yaw_ramp
                points = [
                    [0.0, 0.0, 0.0],
                    [settle, 0.0, 0.0],
                    [ramp_end, 0.0, yaw],
                    [hold_end, 0.0, yaw],
                    [down_end, 0.0, 0.0],
                    [
                        down_end + float(timing["in_place_end_hold_s"]),
                        0.0,
                        0.0,
                    ],
                ]
            elif task_type == "moving_turn":
                turn_yaw = abs(yaw)
                switch = settle + accel + yaw_ramp
                reverse = switch + turn_hold
                settle_end = reverse + turn_hold + yaw_ramp
                stop = settle_end + decel
                points = [
                    [0.0, 0.0, 0.0],
                    [settle, 0.0, 0.0],
                    [settle + accel, speed, 0.0],
                    [switch, speed, -turn_yaw],
                    [reverse, speed, -turn_yaw],
                    [reverse + yaw_ramp, speed, turn_yaw],
                    [settle_end, speed, turn_yaw],
                    [settle_end + yaw_ramp, speed, 0.0],
                    [stop, 0.0, 0.0],
                    [stop + settle, 0.0, 0.0],
                ]
            else:
                continue
            _set_profile_points(profile, points)
        return
    if stage_name.startswith("flat_motion_") or stage_name == "flat_motion":
        for profile in stage["profiles"]:
            target = abs(_profile_target(profile, 1))
            sign = -1.0 if _profile_target(profile, 1) < 0.0 else 1.0
            if profile["name"] in {"forward", "backward"}:
                _set_profile_points(
                    profile,
                    [
                        [0.0, 0.0, 0.0],
                        [2.0, 0.0, 0.0],
                        [6.0, sign * target, 0.0],
                        [14.0, sign * target, 0.0],
                        [18.0, 0.0, 0.0],
                        [20.0, 0.0, 0.0],
                    ],
                )
            elif profile["name"] == "accel_decel":
                _set_profile_points(
                    profile,
                    [
                        [0.0, 0.0, 0.0],
                        [2.0, 0.0, 0.0],
                        [6.0, target, 0.0],
                        [8.0, target, 0.0],
                        [12.0, 0.0, 0.0],
                        [14.0, 0.0, 0.0],
                        [18.0, -target, 0.0],
                        [20.0, -target, 0.0],
                        [24.0, 0.0, 0.0],
                        [26.0, 0.0, 0.0],
                    ],
                )
        return
    if stage_name == "flat_start_stop":
        for profile in stage["profiles"]:
            points = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
            now = 2.0
            targets = sorted(
                {
                    abs(float(point[1]))
                    for point in profile["points"]
                    if abs(float(point[1])) > 1e-6
                }
            )
            for target in targets:
                points.append([now + 4.0, target, 0.0])
                points.append([now + 6.0, target, 0.0])
                points.append([now + 10.0, 0.0, 0.0])
                points.append([now + 12.0, 0.0, 0.0])
                now += 12.0
            _set_profile_points(profile, points)
        return
    if "in_place" in stage_name or stage_name == "flat_turning":
        for profile in stage["profiles"]:
            yaw = _profile_target(profile, 2)
            moving = "turn_" in profile["name"] and profile["name"] not in {
                "turn_left",
                "turn_right",
            }
            if moving:
                speed = _profile_target(profile, 1)
                yaw = abs(_profile_target(profile, 2))
                _set_profile_points(
                    profile,
                    [
                        [0.0, 0.0, 0.0],
                        [2.0, 0.0, 0.0],
                        [6.0, speed, 0.0],
                        [8.0, speed, -yaw],
                        [14.0, speed, -yaw],
                        [16.0, speed, yaw],
                        [22.0, speed, yaw],
                        [24.0, speed, 0.0],
                        [28.0, 0.0, 0.0],
                        [30.0, 0.0, 0.0],
                    ],
                )
            else:
                ramp_end = 2.0 + float(timing["in_place_yaw_ramp_s"])
                hold_end = ramp_end + float(timing["in_place_hold_s"])
                down_end = hold_end + float(timing["in_place_yaw_ramp_s"])
                _set_profile_points(
                    profile,
                    [
                        [0.0, 0.0, 0.0],
                        [2.0, 0.0, 0.0],
                        [ramp_end, 0.0, yaw],
                        [hold_end, 0.0, yaw],
                        [down_end, 0.0, 0.0],
                        [
                            down_end + float(timing["in_place_end_hold_s"]),
                            0.0,
                            0.0,
                        ],
                    ],
                )
        for profile in stage.get("training_profiles", ()):
            speed = _profile_target(profile, 1)
            _set_profile_points(
                profile,
                [
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [6.0, speed, 0.0],
                    [10.0, speed, 0.0],
                    [14.0, 0.0, 0.0],
                    [18.0, 0.0, 0.0],
                    [20.0, 0.0, 0.0],
                ],
            )
        return
    if "moving" in stage_name:
        hold = float(timing["moving_turn_hold_s"])
        ramp = float(timing["moving_turn_yaw_ramp_s"])
        switch = 2.0 + 4.0 + ramp
        reverse = switch + hold
        settle = reverse + hold + ramp
        stop = settle + 4.0
        for profile in stage["profiles"]:
            speed = _profile_target(profile, 1)
            yaw = abs(_profile_target(profile, 2))
            _set_profile_points(
                profile,
                [
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [6.0, speed, 0.0],
                    [switch, speed, -yaw],
                    [reverse, speed, -yaw],
                    [reverse + ramp, speed, yaw],
                    [settle, speed, yaw],
                    [settle + ramp, speed, 0.0],
                    [stop, 0.0, 0.0],
                    [stop + 2.0, 0.0, 0.0],
                ],
            )


def load_course() -> Course:
    """Load base, turning, and adaptive courses with centralized objectives."""
    objectives = _read_json(OBJECTIVES_PATH)
    _validate_objectives(objectives)
    base = _resolve_config_parameters(
        _read_json(CURRICULUM_PATH), objectives, CURRICULUM_PATH.name
    )
    speed = _resolve_config_parameters(
        _read_json(SPEED_STAGES_PATH), objectives, SPEED_STAGES_PATH.name
    )
    turn = _resolve_config_parameters(
        _read_json(TURN_STAGES_PATH), objectives, TURN_STAGES_PATH.name
    )
    adaptive = _resolve_config_parameters(
        _read_json(ADAPTIVE_STAGES_PATH), objectives, ADAPTIVE_STAGES_PATH.name
    )
    for source in (base, speed, turn, adaptive):
        if float(source.get("target_base_height_m", -1)) != TARGET_BASE_HEIGHT_M:
            raise ValueError(
                "Course target_base_height_m must match the calibrated "
                f"{TARGET_BASE_HEIGHT_M} m model height"
            )
        if (
            float(source.get("target_standing_height_m", -1))
            != TARGET_STANDING_HEIGHT_M
        ):
            raise ValueError("Course target_standing_height_m is inconsistent")
        if float(source.get("target_whole_height_m", -1)) != TARGET_WHOLE_HEIGHT_M:
            raise ValueError("Course target_whole_height_m is inconsistent")
    overlap = set(base["stages"]) & set(speed["stages"])
    if overlap:
        raise ValueError(f"Duplicate curriculum stages: {sorted(overlap)}")
    stages = dict(base["stages"])
    target_speed = float(objectives["target"]["straight_speed_max_mps"])
    speed_order = tuple(
        name
        for name in speed["order"]
        if name.startswith("flat_motion_")
        and name[-3:].isdigit()
        and int(name[-3:]) / 100.0 <= target_speed + 1e-9
    )
    for name in speed["order"]:
        stages[name] = speed["stages"][name]

    order = (
        "flat_balance",
        *speed_order,
        "flat_motion",
        "flat_turning",
        "flat_start_stop",
        "slope",
        "rough",
        "roller_obstacle",
        "composite",
    )
    main_stages = set(base["stages"]) | set(speed_order)
    if set(order) != main_stages or len(order) != len(main_stages):
        raise ValueError("Curriculum stage order does not cover every stage exactly once")

    turn_order = tuple(turn["order"])
    if (
        len(turn_order) != len(set(turn_order))
        or set(turn_order) != set(turn["stages"])
    ):
        raise ValueError(
            "Turning course order does not cover every stage exactly once"
        )
    overlap = set(stages) & set(turn["stages"])
    if overlap:
        raise ValueError(f"Duplicate curriculum stages: {sorted(overlap)}")
    stages.update(turn["stages"])
    adaptive_order = tuple(adaptive["order"])
    if (
        len(adaptive_order) != len(set(adaptive_order))
        or set(adaptive_order) != set(adaptive["stages"])
    ):
        raise ValueError(
            "Adaptive course order does not cover every stage exactly once"
        )
    overlap = set(stages) & set(adaptive["stages"])
    if overlap:
        raise ValueError(f"Duplicate curriculum stages: {sorted(overlap)}")
    stages.update(adaptive["stages"])
    validation_order = tuple(
        dict.fromkeys((*order, *speed["stages"], *turn_order, *adaptive_order))
    )
    for name in validation_order:
        _apply_objective_timing(name, stages[name], objectives)
    for name in (*order, *turn_order, *adaptive_order):
        _cap_profile_speed(
            stages[name],
            float(objectives["target"]["straight_speed_max_mps"]),
        )
    for name in turn_order:
        if "moving" in name:
            _cap_profile_speed(
                stages[name],
                float(objectives["target"]["moving_turn_speed_max_mps"]),
            )
    for name in validation_order:
        _validate_stage(name, stages[name])
        if name in turn_order:
            _validate_turn_stage(name, stages[name])
    baseline = objectives["baseline"]
    if baseline["source_stage"] not in stages:
        raise ValueError("baseline source_stage does not exist")
    protected_cells = baseline["protected_cells"]
    vx_step = float(objectives["adaptive_grid"]["vx_resolution_mps"])
    wz_step = float(objectives["adaptive_grid"]["wz_resolution_rad_s"])
    moving_limits = (
        max(abs(float(cell[0])) * vx_step for cell in protected_cells["moving_turn"]),
        max(abs(float(cell[1])) * wz_step for cell in protected_cells["moving_turn"]),
    )
    if not math.isclose(
        moving_limits[0],
        float(objectives["review"]["moving_turn_speed_max_mps"]),
    ) or not math.isclose(
        moving_limits[1],
        float(objectives["review"]["moving_turn_yaw_max_rad_s"]),
    ):
        raise ValueError("baseline moving protected cells do not match review")
    source_stage = stages[baseline["source_stage"]]
    source_speed = max(
        abs(float(point[1]))
        for profile in source_stage["profiles"]
        for point in profile["points"]
    )
    source_yaw = max(
        abs(float(point[2]))
        for profile in source_stage["profiles"]
        for point in profile["points"]
    )
    if not math.isclose(source_speed, moving_limits[0]) or not math.isclose(
        source_yaw,
        moving_limits[1],
    ):
        raise ValueError("baseline source stage does not match protected profile")
    rehearsal_gate = turn.get("rehearsal_gate")
    if any(stage.get("training_profiles") for stage in stages.values()):
        if not isinstance(rehearsal_gate, dict):
            raise ValueError(
                "Turning rehearsal profiles require a rehearsal_gate"
            )
    if set(SCREEN_SEEDS) & set(VERIFY_SEEDS):
        raise ValueError("Screen and verification seeds must be disjoint")
    exposed = set(SCREEN_SEEDS) | set(VERIFY_SEEDS)
    if exposed & (set(BLIND_SEEDS) | set(LIMIT_SEEDS)):
        raise ValueError("Blind and limit seeds must not participate in selection")
    selection_exposed = {SELECTION_PROBE_SEED, *SELECTION_VERIFY_SEEDS}
    if selection_exposed & (set(BLIND_SEEDS) | set(LIMIT_SEEDS)):
        raise ValueError("Blind and limit seeds must not participate in selection")

    course = Course(
        stages=stages,
        order=order,
        turn_order=turn_order,
        adaptive_order=adaptive_order,
        objectives=objectives,
        gate_templates=objectives["gates"],
        ppo=base["ppo"],
        speed_ppo=speed["ppo"],
        turn_ppo=turn["ppo"],
        reward_defaults=base["reward_defaults"],
        turn_reward_defaults=turn["reward_defaults"],
        rehearsal_gate=rehearsal_gate,
        reward_penalties=base["reward_defaults"]["common_penalties"],
        randomization=base["randomization"],
        validation_seeds=tuple(base["validation_seeds"]),
        blind_seeds=tuple(base["blind_seeds"]),
        limit_seeds=tuple(base["limit_seeds"]),
    )
    for name in course.stages:
        stage = course.stage(name)
        for profile in stage_profiles(stage):
            gate = resolved_profile_gate(course, name, profile)
            for key in gate:
                metric = metric_name(str(key))
                if metric not in KNOWN_METRICS:
                    raise ValueError(
                        f"{name}/{profile['name']}: "
                        f"gate '{key}' maps to unknown metric '{metric}'"
                    )
    return course


def reward_defaults_for_stage(course: Course, stage_name: str) -> dict[str, Any]:
    """Resolve common and stage reward defaults in runtime precedence order."""
    stage = course.stage(stage_name)
    turn_stages = set(course.turn_order)
    defaults = dict(
        course.turn_reward_defaults
        if stage_name in turn_stages
        else course.reward_defaults
    )
    defaults.update(stage.get("reward_defaults", {}))
    if "hold_settle_time_s" in stage:
        defaults["hold_settle_time_s"] = stage["hold_settle_time_s"]
    settle_time = float(defaults.get("hold_settle_time_s", 0.0))
    if not math.isfinite(settle_time) or settle_time < 0.0:
        raise ValueError(
            f"{stage_name}: hold_settle_time_s must be finite and nonnegative"
        )
    defaults["hold_settle_time_s"] = settle_time
    grace_time = float(defaults.get("evaluation_grace_s", 0.0))
    if not math.isfinite(grace_time) or grace_time < 0.0:
        raise ValueError(
            f"{stage_name}: evaluation_grace_s must be finite and nonnegative"
        )
    defaults["evaluation_grace_s"] = grace_time
    gate = stage["gate"]

    def cap(name: str, limit: float | None, ratio: float = 0.8) -> None:
        if limit is None or name not in defaults:
            return
        defaults[name] = min(float(defaults[name]), float(limit) * ratio)

    cap("speed_tolerance_base_mps", gate.get("max_speed_mae_mps"))
    cap("yaw_tolerance_base_rad_s", gate.get("max_yaw_mae_rad_s"))
    cap("yaw_underspeed_tolerance_rad_s", gate.get("max_yaw_mae_rad_s"))
    pitch_limit = gate.get("max_mean_abs_pitch_deg")
    cap(
        "pitch_tolerance_rad",
        math.radians(float(pitch_limit)) if pitch_limit is not None else None,
    )
    height_limit = gate.get(
        "max_settled_mean_height_error_m",
        gate.get("mean_height_tolerance_m"),
    )
    cap("height_tolerance_m", height_limit)
    slip_limit = gate.get(
        "max_settled_wheel_slip_mps",
        gate.get("max_wheel_slip_mps", gate.get("max_mean_steady_wheel_slip_mps")),
    )
    cap("wheel_slip_tolerance_mps", slip_limit)
    sync_limit = gate.get(
        "max_settled_straight_wheel_speed_difference_mps",
        gate.get("max_straight_wheel_speed_difference_mps"),
    )
    cap("wheel_sync_tolerance_mps", sync_limit)
    alignment_limit = gate.get("max_wheel_center_offset_m")
    cap("wheel_alignment_tolerance_m", alignment_limit)
    cap("turn_wheel_alignment_tolerance_m", alignment_limit, ratio=2.0 / 3.0)
    cap(
        "drift_tolerance_m",
        gate.get("max_hold_drift_after_0.5s_m", gate.get("max_hold_drift_m")),
    )
    cap("zero_speed_tolerance_mps", gate.get("max_stop_speed_mps"))
    final_heading_limit = gate.get("max_final_heading_error_deg")
    if final_heading_limit is not None:
        heading_limit = math.radians(float(final_heading_limit))
        defaults.setdefault(
            "straight_heading_tolerance_rad",
            defaults.get("heading_tolerance_rad", heading_limit),
        )
        cap("straight_heading_tolerance_rad", heading_limit)
        cap("heading_tolerance_rad", heading_limit)
        turn_limits = [heading_limit]
        turn_error_limit = gate.get("max_turn_angle_error_rad")
        if turn_error_limit is not None:
            turn_limits.append(float(turn_error_limit))
        cap("turn_angle_tolerance_rad", min(turn_limits))
    cross_track_limit = gate.get("max_cross_track_error_m")
    if cross_track_limit is not None:
        defaults.setdefault("cross_track_tolerance_m", float(cross_track_limit))
        cap("cross_track_tolerance_m", cross_track_limit)
    heading_yaw_gain = float(defaults.get("heading_yaw_gain", 0.75))
    yaw_correction_limit = float(
        defaults.get("heading_yaw_correction_limit_rad_s", 0.2)
    )
    if not math.isfinite(heading_yaw_gain) or heading_yaw_gain < 0.0:
        raise ValueError(f"{stage_name}: heading_yaw_gain is invalid")
    if not math.isfinite(yaw_correction_limit) or yaw_correction_limit < 0.0:
        raise ValueError(
            f"{stage_name}: heading_yaw_correction_limit_rad_s is invalid"
        )
    defaults["heading_yaw_gain"] = heading_yaw_gain
    defaults["heading_yaw_correction_limit_rad_s"] = yaw_correction_limit
    return defaults


def _validate_stage(name: str, stage: dict[str, Any]) -> None:
    if not stage.get("profiles"):
        raise ValueError(f"{name}: at least one command profile is required")
    if int(stage.get("budget", 0)) <= 0:
        raise ValueError(f"{name}: budget must be positive")
    training_profiles = stage.get("training_profiles", [])
    if not isinstance(training_profiles, list):
        raise ValueError(f"{name}: training_profiles must be a list")
    profiles = [*stage["profiles"], *training_profiles]
    profile_names = []
    for profile in profiles:
        profile_name = profile.get("name")
        if not isinstance(profile_name, str) or not profile_name:
            raise ValueError(f"{name}: every profile requires a string name")
        profile_names.append(profile_name)
    if len(profile_names) != len(set(profile_names)):
        raise ValueError(f"{name}: profile names must be unique")
    previous = -math.inf
    for profile in profiles:
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
    base_duration = max(
        float(profile["points"][-1][0]) for profile in stage["profiles"]
    )
    if any(
        float(profile["points"][-1][0]) > base_duration + 1e-9
        for profile in training_profiles
    ):
        raise ValueError(
            f"{name}: training profiles must not extend the evaluation episode"
        )
    for profile in profiles:
        task_type = profile.get("task_type")
        if task_type is not None and task_type not in {
            "balance",
            "straight",
            "in_place_turn",
            "moving_turn",
            "terrain",
            "composite",
        }:
            raise ValueError(
                f"{name}/{profile['name']}: unknown task_type '{task_type}'"
            )
        overrides = profile.get("gate_overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError(
                f"{name}/{profile['name']}: gate_overrides must be an object"
            )
    if training_profiles:
        training_profile_schedule(stage)
    if stage.get("adaptive_grid") is not None:
        _validate_adaptive_grid(name, stage)


def _validate_adaptive_grid(name: str, stage: dict[str, Any]) -> None:
    grid = stage["adaptive_grid"]
    if not isinstance(grid, dict):
        raise ValueError(f"{name}: adaptive_grid must be an object")
    numeric_keys = (
        "vx_resolution_mps",
        "wz_resolution_rad_s",
        "protected_fraction",
        "protected_recovery_fraction",
        "window_size",
        "minimum_window_episodes",
        "expand_success_rate",
        "deactivate_success_rate",
        "protected_health_floor",
        "protected_health_windows",
        "consecutive_windows",
        "straight_speed_min_mps",
        "review_straight_speed_max_mps",
        "target_straight_speed_max_mps",
        "target_moving_turn_speed_max_mps",
        "in_place_yaw_min_rad_s",
        "in_place_yaw_max_rad_s",
        "moving_turn_yaw_max_rad_s",
        "min_turn_radius_m",
        "max_turn_radius_m",
        "straight_yaw_deadband_rad_s",
        "balance_hold_s",
        "straight_settle_s",
        "straight_accel_s",
        "straight_cruise_s",
        "straight_decel_s",
        "in_place_settle_s",
        "in_place_yaw_ramp_s",
        "in_place_hold_s",
        "in_place_end_hold_s",
        "moving_turn_hold_s",
        "moving_turn_yaw_ramp_s",
    )
    for key in numeric_keys:
        value = float(grid.get(key, math.nan))
        if not math.isfinite(value):
            raise ValueError(f"{name}: adaptive_grid.{key} must be finite")
    if not 0.0 < float(grid["protected_fraction"]) <= 1.0:
        raise ValueError(f"{name}: protected_fraction must be in (0, 1]")
    if not 0.0 < float(grid["protected_recovery_fraction"]) <= 1.0:
        raise ValueError(f"{name}: protected_recovery_fraction must be in (0, 1]")
    if not 0.75 <= float(grid["expand_success_rate"]) <= 0.85:
        raise ValueError(f"{name}: expand_success_rate must be in [0.75, 0.85]")
    health_gate = grid.get("health_gate")
    if not isinstance(health_gate, dict) or not health_gate:
        raise ValueError(f"{name}: health_gate must be a non-empty object")
    for key, value in health_gate.items():
        if key == "min_pass_rate":
            raise ValueError(f"{name}: health_gate must not contain min_pass_rate")
        if metric_name(str(key)) not in KNOWN_METRICS:
            raise ValueError(f"{name}: health_gate '{key}' has no metric")
        if key not in stage["gate"]:
            raise ValueError(f"{name}: health_gate '{key}' is not a stage metric")
        resolved = float(value)
        if not math.isfinite(resolved) or resolved < 0.0:
            raise ValueError(
                f"{name}: health_gate.{key} must be finite and nonnegative"
            )
    if int(grid["protected_health_windows"]) <= 0:
        raise ValueError(f"{name}: protected_health_windows must be positive")
    if not 0.0 <= float(grid["deactivate_success_rate"]) < float(
        grid["expand_success_rate"]
    ):
        raise ValueError(f"{name}: deactivate rate must be below expand rate")
    if float(grid["target_straight_speed_max_mps"]) < float(
        grid["review_straight_speed_max_mps"]
    ):
        raise ValueError(f"{name}: target speed is below review speed")
    if float(grid["target_moving_turn_speed_max_mps"]) + 1e-9 < (
        float(grid["min_turn_radius_m"])
        * float(grid["moving_turn_yaw_max_rad_s"])
    ):
        raise ValueError(
            f"{name}: moving-turn speed cap cannot preserve minimum radius"
        )
    protected_cells = grid["protected_cells"]
    if (
        not isinstance(protected_cells, dict)
        or set(protected_cells)
        != {"balance", "straight", "in_place_turn", "moving_turn"}
    ):
        raise ValueError(f"{name}: protected_cells must define every task mode")
    for mode, cells in protected_cells.items():
        if not isinstance(cells, list) or not cells:
            raise ValueError(f"{name}: protected {mode} cells are missing")
        for cell in cells:
            if (
                not isinstance(cell, list)
                or len(cell) != 2
                or any(isinstance(value, bool) for value in cell)
            ):
                raise ValueError(f"{name}: invalid protected {mode} cell")
            vx_index, yaw_index = map(int, cell)
            if mode == "balance" and (vx_index != 0 or yaw_index != 0):
                raise ValueError(f"{name}: balance protection must use (0, 0)")
            if mode == "straight" and yaw_index != 0:
                raise ValueError(f"{name}: straight protection must use yaw 0")
            if mode == "in_place_turn" and vx_index != 0:
                raise ValueError(f"{name}: in-place protection must use speed 0")
            if mode == "moving_turn" and (vx_index == 0 or yaw_index == 0):
                raise ValueError(
                    f"{name}: moving-turn protection needs both axes"
                )
            if mode == "moving_turn":
                initial_radius = (
                    abs(vx_index) * float(grid["vx_resolution_mps"])
                    / (abs(yaw_index) * float(grid["wz_resolution_rad_s"]))
                )
                if not (
                    float(grid["min_turn_radius_m"])
                    <= initial_radius
                    <= float(grid["max_turn_radius_m"])
                ):
                    raise ValueError(
                        f"{name}: protected moving-turn cell is outside "
                        "radius limits"
                    )
    if float(grid["moving_turn_yaw_max_rad_s"]) > 1.2:
        raise ValueError(f"{name}: moving-turn yaw cap exceeds 1.2 rad/s")
    if float(grid["in_place_yaw_min_rad_s"]) < 0.4:
        raise ValueError(f"{name}: in-place yaw minimum is below 0.4 rad/s")
    fractions = grid["mode_minimum_fractions"]
    if not isinstance(fractions, dict) or set(fractions) != {
        "balance",
        "straight",
        "in_place_turn",
        "moving_turn",
    }:
        raise ValueError(
            f"{name}: mode_minimum_fractions must define all task modes"
        )
    values = [float(value) for value in fractions.values()]
    if any(value <= 0.0 for value in values) or sum(values) > 1.0 + 1e-9:
        raise ValueError(
            f"{name}: mode minimum fractions must be positive and sum at most 1"
        )


def training_profile_schedule(stage: dict[str, Any]) -> tuple[str, ...]:
    """Expand the deterministic weighted schedule used only during training."""
    training_profiles = stage.get("training_profiles", ())
    mixture = stage.get("training_mix")
    if not training_profiles:
        if mixture is not None:
            raise ValueError("training_mix requires training_profiles")
        return ()
    if not isinstance(mixture, list) or not mixture:
        raise ValueError("training_profiles require a nonempty training_mix")

    profile_names = [
        profile["name"]
        for profile in (*stage["profiles"], *training_profiles)
    ]
    expected = set(profile_names)
    seen: set[str] = set()
    schedule: list[str] = []
    for entry in mixture:
        if not isinstance(entry, dict):
            raise ValueError("training_mix entries must be objects")
        profile_name = entry.get("profile")
        weight = entry.get("weight")
        if not isinstance(profile_name, str) or profile_name not in expected:
            raise ValueError(
                f"training_mix references unknown profile '{profile_name}'"
            )
        if profile_name in seen:
            raise ValueError(
                f"training_mix contains duplicate profile '{profile_name}'"
            )
        if isinstance(weight, bool) or not isinstance(weight, int) or weight <= 0:
            raise ValueError("training_mix weights must be positive integers")
        schedule.extend([profile_name] * weight)
        seen.add(profile_name)
    missing = expected - seen
    if missing:
        raise ValueError(
            f"training_mix is missing profiles: {sorted(missing)}"
        )
    return tuple(schedule)


def _validate_turn_stage(name: str, stage: dict[str, Any]) -> None:
    points = [
        point
        for profile in (
            *stage["profiles"],
            *stage.get("training_profiles", ()),
        )
        for point in profile["points"]
    ]
    speed_limit = 0.9 if "moving" in name else 0.8
    if max(abs(float(point[1])) for point in points) > speed_limit + 1e-9:
        raise ValueError(
            f"{name}: turning course must not exceed {speed_limit} m/s"
        )
    if max(abs(float(point[2])) for point in points) > 2.0 + 1e-9:
        raise ValueError(f"{name}: turning course must not exceed 2.0 rad/s")
    if "moving" in name and max(
        abs(float(point[2])) for point in points
    ) > 1.2 + 1e-9:
        raise ValueError(f"{name}: moving-turn course must not exceed 1.2 rad/s")
    if "in_place" in name:
        for profile in stage.get("training_profiles", ()):
            if any(abs(float(point[2])) > 1e-9 for point in profile["points"]):
                raise ValueError(
                    f"{name}/{profile['name']}: in-place rehearsal must be straight"
                )


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
    return {name: task_id(name) for name in course.all_order}


def ppo_defaults_for_stage(course: Course, stage_name: str) -> dict[str, Any]:
    turn_stages = set(course.turn_order)
    if stage_name in turn_stages:
        return dict(course.turn_ppo)
    speed_stage = (
        stage_name.startswith("flat_motion_") and stage_name != "flat_motion"
    )
    return dict(course.speed_ppo if speed_stage else course.ppo)


def metric_name(gate_name: str) -> str:
    if gate_name == "min_turn_radius_m":
        return "turn_radius_min_m"
    if gate_name == "max_turn_radius_m":
        return "turn_radius_max_m"
    if gate_name.startswith("max_"):
        return gate_name[4:]
    if gate_name.startswith("min_"):
        return gate_name[4:]
    if gate_name == "mean_height_tolerance_m":
        return "mean_height_error_m"
    if gate_name == "require_finish":
        return "finished"
    raise ValueError(f"Unsupported gate key '{gate_name}'")


def profile_task_type(
    stage_name: str,
    stage: dict[str, Any],
    profile: dict[str, Any],
) -> str:
    explicit = profile.get("task_type")
    if explicit:
        return str(explicit)
    profile_name = str(profile.get("name", ""))
    terrain_kind = str(stage.get("terrain", {}).get("kind", "flat"))
    if stage_name == "flat_balance" or (
        "hold" in profile_name and "rehearsal" not in profile_name
    ):
        return "balance"
    if "in_place" in stage_name or profile_name in {
        "turn_left",
        "turn_right",
        "clockwise",
        "counterclockwise",
        "clockwise_160",
        "counterclockwise_160",
        "clockwise_200",
        "counterclockwise_200",
    }:
        return "in_place_turn"
    if stage_name.startswith("flat_turn_") or stage_name == "flat_turning":
        return "moving_turn"
    if terrain_kind != "flat":
        return "composite" if stage_name == "composite" else "terrain"
    return "straight"


def resolved_profile_gate(
    course: Course,
    stage_name: str,
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Resolve a task-aware gate while preserving legacy homogeneous stages."""
    stage = course.stage(stage_name)
    if "task_type" not in profile and "gate_overrides" not in profile:
        return dict(stage["gate"])
    task_type = profile_task_type(stage_name, stage, profile)
    gate = _merge_gate_layers(
        course.gate_templates["common"],
        course.gate_templates[task_type],
        stage.get("gate_overrides"),
        profile.get("gate_overrides"),
    )
    if not gate:
        raise ValueError(f"{stage_name}/{profile['name']}: resolved gate is empty")
    return gate


def resolved_task_gate(
    course: Course,
    stage_name: str,
    task_type: str,
) -> dict[str, Any]:
    """Resolve common and task-specific gate layers without merging tasks."""
    stage = course.stage(stage_name)
    gate = _merge_gate_layers(
        course.gate_templates["common"],
        course.gate_templates[task_type],
        stage.get("gate_overrides"),
    )
    if not gate:
        raise ValueError(f"{stage_name}/{task_type}: resolved gate is empty")
    return gate


def stage_profiles(
    stage: dict[str, Any],
    include_training_profiles: bool = False,
    profile_set: str = "required",
) -> list[dict[str, Any]]:
    profiles = list(stage["profiles"])
    if include_training_profiles:
        profiles.extend(stage.get("training_profiles", ()))
    if profile_set == "all":
        return profiles
    if profile_set == "required":
        return [
            profile
            for profile in profiles
            if profile.get("profile_set", "baseline") != "diagnostic"
        ]
    return [
        profile
        for profile in profiles
        if profile.get("profile_set", "all") in {profile_set, "all"}
    ]


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
        "tracking": mean(
            "max_speed_mae_mps",
            "max_cruise_speed_mae_mps",
            "min_cruise_speed_fraction",
            "max_turn_angle_error_rad",
            "max_yaw_mae_rad_s",
            "max_yaw_rate_peak_rad_s",
        ),
        "stability": mean(
            "min_upright",
            "mean_height_tolerance_m",
            "max_settled_mean_height_error_m",
            "max_height_deviation_m",
            "max_settled_height_deviation_m",
            "max_wheel_center_offset_m",
            "max_straight_wheel_speed_difference_mps",
            "max_settled_straight_wheel_speed_difference_mps",
            "max_mean_settled_straight_wheel_speed_difference_mps",
        ),
        "terrain": mean("min_obstacle_passes", "max_backslide_mps"),
        "efficiency": mean(
            "max_wheel_slip_mps",
            "max_settled_wheel_slip_mps",
            "max_mean_steady_wheel_slip_mps",
            "max_transient_wheel_slip_mps",
            "max_wheel_slip_fraction_of_peak",
            "max_settled_wheel_slip_fraction_of_peak",
        ),
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


def stage_training_contract(course: Course, stage_name: str) -> dict[str, Any]:
    """Return a stable fingerprint of only what the current stage consumes."""
    stage = course.stage(stage_name)
    ppo = ppo_defaults_for_stage(course, stage_name)
    ppo.update(stage.get("ppo", {}))
    reward_defaults = reward_defaults_for_stage(course, stage_name)
    reward_defaults.pop("evaluation_grace_s", None)
    penalties = dict(course.reward_penalties)
    penalties.update(stage.get("overrides", {}))
    stage_semantics = {
        key: value
        for key, value in stage.items()
        if key
        not in {
            "budget",
            "gate",
            "ppo",
            "reward_defaults",
            "overrides",
        }
    }
    interface = policy_interface_contract(course)
    content = {
        "version": 1,
        "stage": stage_name,
        "interface": {
            key: interface[key] for key in sorted(_TRAINING_INTERFACE_FIELDS)
        },
        "stage_semantics": stage_semantics,
        "profile_gates": {
            profile["name"]: {
                key: value
                for key, value in resolved_profile_gate(
                    course, stage_name, profile
                ).items()
                if key != "min_pass_rate"
            }
            for profile in stage_profiles(stage, profile_set="all")
        },
        "training_objectives": course.objectives,
        "gate": {
            key: stage["gate"][key]
            for key in sorted(_TRAINING_GATE_KEYS & set(stage["gate"]))
        },
        "reward_defaults": reward_defaults,
        "penalties": penalties,
        "randomization": course.randomization,
        "ppo": ppo,
        "source_sha256": {
            path.name: sha256(path)
            for path in (
                PACKAGE_DIR / "model.py",
                PACKAGE_DIR / "control.py",
                PACKAGE_DIR / "policies.py",
                PACKAGE_DIR / "mjlab_tasks.py",
                PACKAGE_DIR / "upstream.lock.json",
            )
        },
    }
    return {
        "version": content["version"],
        "stage": stage_name,
        "sha256": hashlib.sha256(
            json.dumps(content, sort_keys=True).encode()
        ).hexdigest(),
    }


def policy_interface_contract(course: Course) -> dict[str, Any]:
    return {
        "version": 9,
        "backend": "mjlab-rsl-rl",
        "policy_architecture": "temporal_residual_v1",
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
        "target_standing_height_m": TARGET_STANDING_HEIGHT_M,
        "reset_base_height_m": RESET_BASE_HEIGHT_M,
        "target_whole_height_m": TARGET_WHOLE_HEIGHT_M,
    }


def interface_contract(course: Course) -> dict[str, Any]:
    contract = policy_interface_contract(course)
    contract.update(
        {
            "stage_order": list(course.order),
            "turning_stage_order": list(course.turn_order),
            "adaptive_stage_order": list(course.adaptive_order),
            "task_ids": stage_task_ids(course),
            "source_sha256": {
                path.name: sha256(path)
                for path in (
                    PACKAGE_DIR / "model.py",
                    PACKAGE_DIR / "control.py",
                    PACKAGE_DIR / "policies.py",
                    PACKAGE_DIR / "mjlab_tasks.py",
                    PACKAGE_DIR / "upstream.lock.json",
                    CURRICULUM_PATH,
                    SPEED_STAGES_PATH,
                    TURN_STAGES_PATH,
                    ADAPTIVE_STAGES_PATH,
                    OBJECTIVES_PATH,
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
    _verify_contract_seal(content)
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


def _verify_contract_seal(content: dict[str, Any]) -> None:
    expected = content.get("sha256")
    unsigned = {key: value for key, value in content.items() if key != "sha256"}
    actual = hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest()
    if expected != actual:
        raise ValueError("MJLab run contract is missing or inconsistent")


def verify_warm_start_contract(content: dict[str, Any]) -> None:
    """Validate actor I/O only, allowing the task and reward to change."""
    _verify_contract_seal(content)
    current = policy_interface_contract(load_course())
    mismatched = [
        field
        for field in sorted(_WARM_START_FIELDS)
        if content.get(field) != current[field]
    ]
    if mismatched:
        raise ValueError(
            "Checkpoint actor interface is incompatible; mismatched fields: "
            + ", ".join(mismatched)
        )


def verify_run_contract(
    payload: dict[str, Any],
    stage_name: str | None = None,
    require_current_sources: bool = False,
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
    if require_current_sources:
        if actual_stage is None:
            raise ValueError("Resume comparison requires a saved stage")
        saved_training = payload.get("training_contract")
        if saved_training is None:
            current_sources = interface_contract(load_course())["source_sha256"]
            if contract.get("source_sha256") != current_sources:
                raise ValueError(
                    "Legacy checkpoint has no stage training fingerprint; "
                    "start a new run"
                )
        else:
            current_training = stage_training_contract(
                load_course(),
                actual_stage,
            )
            if saved_training != current_training:
                raise ValueError(
                    f"Checkpoint training contract differs for stage "
                    f"{actual_stage}; start a new run"
                )
    return contract


def load_run_contract(
    path: Path,
    stage_name: str | None = None,
    require_current_sources: bool = False,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing run contract: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return verify_run_contract(
        payload,
        stage_name,
        require_current_sources=require_current_sources,
    )


def load_warm_start_contract(path: Path) -> dict[str, Any]:
    """Load a run contract using actor-only compatibility rules."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing run contract: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    contract = payload.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("Run contract is missing the nested 'contract' object")
    verify_warm_start_contract(contract)
    return payload


if __name__ == "__main__":
    loaded = load_course()
    print(json.dumps(interface_contract(loaded), indent=2, ensure_ascii=False))
