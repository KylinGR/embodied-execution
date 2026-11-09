"""Adaptive velocity/yaw grid used by the speed-extension course."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any


@dataclass(frozen=True)
class GridCell:
    vx_index: int
    wz_index: int
    mode: str
    vx_bounds: tuple[float, float]
    wz_bounds: tuple[float, float]
    duration_s: float

    @property
    def key(self) -> tuple[int, int]:
        return self.vx_index, self.wz_index

    @property
    def cell_id(self) -> int:
        return (self.vx_index + 16) * 64 + self.wz_index + 16

    @property
    def vx(self) -> float:
        return self.vx_index * 0.2

    @property
    def wz(self) -> float:
        return self.wz_index * 0.2


class AdaptiveGrid:
    """Finite 0.2 m/s by 0.2 rad/s grid with protected replay cells."""

    MODE_ORDER = ("balance", "straight", "in_place_turn", "moving_turn")

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.vx_step = float(cfg["vx_resolution_mps"])
        self.wz_step = float(cfg["wz_resolution_rad_s"])
        if not math.isclose(self.vx_step, 0.2) or not math.isclose(
            self.wz_step, 0.2
        ):
            raise ValueError("adaptive grid resolution must be 0.2")
        self.mode_durations = {
            "balance": float(cfg["balance_hold_s"]),
            "straight": float(
                cfg["straight_settle_s"]
                + cfg["straight_accel_s"]
                + cfg["straight_cruise_s"]
                + cfg["straight_decel_s"]
                + cfg["straight_settle_s"]
            ),
            "in_place_turn": float(
                cfg["in_place_settle_s"]
                + cfg["in_place_yaw_ramp_s"]
                + cfg["in_place_hold_s"]
                + cfg["in_place_yaw_ramp_s"]
                + cfg["in_place_end_hold_s"]
            ),
            "moving_turn": 30.0,
        }
        fractions = cfg["mode_minimum_fractions"]
        if not isinstance(fractions, dict) or set(fractions) != set(self.MODE_ORDER):
            raise ValueError(
                "mode_minimum_fractions must define exactly "
                + ", ".join(self.MODE_ORDER)
            )
        self.mode_fractions = {
            mode: float(fractions[mode]) for mode in self.MODE_ORDER
        }
        if any(value <= 0.0 for value in self.mode_fractions.values()):
            raise ValueError("mode minimum fractions must be positive")
        if sum(self.mode_fractions.values()) > 1.0 + 1e-9:
            raise ValueError("mode minimum fractions must not sum above 1")
        self._mode_credit = {mode: 0.0 for mode in self.MODE_ORDER}
        self._direction_phase = {mode: 0 for mode in self.MODE_ORDER}
        self._protected_credit: dict[
            tuple[str, tuple[int, int]], float
        ] = {}
        self.cells = self._build_cells()
        self.protected: set[tuple[int, int]] = set()
        self.active: set[tuple[int, int]] = set()
        for key in self._seed_keys():
            mirrored = self._mirror_keys(key)
            missing = mirrored - self.cells.keys()
            if missing:
                continue
            self.protected.update(mirrored)
        self.active.update(self.protected)
        self.history: dict[tuple[int, int], list[bool]] = {
            key: [] for key in self.cells
        }
        self.health_history: dict[tuple[int, int], list[bool]] = {
            key: [] for key in self.cells
        }
        self.last_success: dict[tuple[int, int], float] = {}
        self.last_health: dict[tuple[int, int], float] = {}
        self.health_pass_streak: dict[tuple[int, int], int] = {}
        self.pass_streak: dict[tuple[int, int], int] = {}
        self.fail_streak: dict[tuple[int, int], int] = {}
        self.expansions = 0

    def _seed_keys(self) -> set[tuple[int, int]]:
        return {
            (int(vx_index), int(yaw_index))
            for cells in self.cfg["protected_cells"].values()
            for vx_index, yaw_index in cells
        }

    def _nearest_index(self, value: float, step: float) -> int:
        return int(round(float(value) / step))

    def _build_cells(self) -> dict[tuple[int, int], GridCell]:
        cells: dict[tuple[int, int], GridCell] = {}
        vx_min = self._nearest_index(self.cfg["straight_speed_min_mps"], self.vx_step)
        vx_max = self._nearest_index(
            self.cfg["target_straight_speed_max_mps"], self.vx_step
        )
        yaw_max = self._nearest_index(
            self.cfg["moving_turn_yaw_max_rad_s"], self.wz_step
        )
        moving_vx_max = int(
            math.floor(
                float(self.cfg["target_moving_turn_speed_max_mps"])
                / self.vx_step
                + 1e-9
            )
        )
        in_place_min = self._nearest_index(
            self.cfg["in_place_yaw_min_rad_s"], self.wz_step
        )
        in_place_max = self._nearest_index(
            self.cfg["in_place_yaw_max_rad_s"], self.wz_step
        )
        yaw_deadband = float(self.cfg["straight_yaw_deadband_rad_s"])
        radius_min = float(self.cfg["min_turn_radius_m"])
        radius_max = float(self.cfg["max_turn_radius_m"])
        for vi in range(-vx_max, vx_max + 1):
            for wi in range(-yaw_max, yaw_max + 1):
                if abs(vi) < vx_min and vi != 0:
                    continue
                if vi == 0 and wi == 0:
                    mode = "balance"
                    vx_bounds = (-yaw_deadband, yaw_deadband)
                    wz_bounds = (-yaw_deadband, yaw_deadband)
                elif vi == 0:
                    if abs(wi) < in_place_min or abs(wi) > in_place_max:
                        continue
                    mode = "in_place_turn"
                    vx_bounds = (-yaw_deadband, yaw_deadband)
                    wz_bounds = self._biased_bounds(wi, self.wz_step)
                elif wi == 0:
                    mode = "straight"
                    vx_bounds = self._biased_bounds(vi, self.vx_step)
                    wz_bounds = (-yaw_deadband, yaw_deadband)
                else:
                    if abs(vi) > moving_vx_max:
                        continue
                    vx = abs(vi) * self.vx_step
                    wz = abs(wi) * self.wz_step
                    radius = vx / wz
                    if not radius_min <= radius <= radius_max:
                        continue
                    mode = "moving_turn"
                    vx_bounds = self._biased_bounds(vi, self.vx_step)
                    wz_bounds = self._biased_bounds(wi, self.wz_step)
                cells[(vi, wi)] = GridCell(
                    vx_index=vi,
                    wz_index=wi,
                    mode=mode,
                    vx_bounds=vx_bounds,
                    wz_bounds=wz_bounds,
                    duration_s=self.mode_durations[mode],
                )
        return cells

    def _biased_bounds(self, index: int, step: float) -> tuple[float, float]:
        center = index * step
        half = 0.5 * step
        return center - half, center + half

    def _mirror_keys(self, key: tuple[int, int]) -> set[tuple[int, int]]:
        vi, wi = key
        vx_signs = (0,) if vi == 0 else (-1, 1)
        wz_signs = (0,) if wi == 0 else (-1, 1)
        return {
            (abs(vi) * vs, abs(wi) * ws)
            for vs in vx_signs
            for ws in wz_signs
        }

    def _neighbors(self, key: tuple[int, int]) -> set[tuple[int, int]]:
        vi, wi = key
        return self._mirror_keys((vi + 1, wi)) | self._mirror_keys((vi, wi + 1))

    def sample_cell_keys(self, count: int) -> list[tuple[int, int]]:
        if count <= 0:
            return []
        groups: dict[str, dict[tuple[int, int], list[tuple[int, int]]]] = {
            mode: {} for mode in self.MODE_ORDER
        }
        for key in sorted(self.active):
            cell = self.cells[key]
            sign = (
                int(cell.vx_index > 0) - int(cell.vx_index < 0),
                int(cell.wz_index > 0) - int(cell.wz_index < 0),
            )
            groups[cell.mode].setdefault(sign, []).append(key)
        active_modes = [mode for mode in self.MODE_ORDER if groups[mode]]
        if not active_modes:
            raise RuntimeError("adaptive grid has no active cells")

        weights = {
            mode: self.mode_fractions[mode]
            for mode in active_modes
        }
        weight_sum = sum(weights.values())
        for mode in active_modes:
            self._mode_credit[mode] += count * weights[mode] / weight_sum
        mode_counts = {mode: 0 for mode in active_modes}
        for _ in range(count):
            mode = max(
                active_modes,
                key=lambda value: (
                    self._mode_credit[value],
                    -self.MODE_ORDER.index(value),
                ),
            )
            mode_counts[mode] += 1
            self._mode_credit[mode] -= 1.0

        result: list[tuple[int, int]] = []
        for mode in active_modes:
            protected_fraction = float(self.cfg["protected_fraction"])
            if self._protected_unhealthy(mode):
                protected_fraction = float(
                    self.cfg["protected_recovery_fraction"]
                )
            strata = [
                (sign, groups[mode][sign]) for sign in sorted(groups[mode])
            ]
            base, extra = divmod(mode_counts[mode], len(strata))
            start = self._direction_phase[mode] % len(strata)
            counts = [base] * len(strata)
            for offset in range(extra):
                counts[(start + offset) % len(strata)] += 1
            self._direction_phase[mode] = (
                start + mode_counts[mode]
            ) % len(strata)
            for stratum_index, stratum_count in enumerate(counts):
                if not stratum_count:
                    continue
                sign, stratum = strata[stratum_index]
                key = (mode, sign)
                protected = sorted(set(stratum) & self.protected)
                active = stratum
                self._protected_credit[key] = (
                    self._protected_credit.get(key, 0.0)
                    + stratum_count * protected_fraction
                )
                protected_count = min(
                    len(protected),
                    int(self._protected_credit[key]),
                    stratum_count,
                )
                self._protected_credit[key] -= protected_count
                if protected_count:
                    result.extend(random.choices(protected, k=protected_count))
                remaining = stratum_count - protected_count
                if remaining:
                    result.extend(random.choices(active, k=remaining))
        random.shuffle(result)
        if len(result) != count:
            raise RuntimeError(
                f"adaptive sampler returned {len(result)} cells for {count} envs"
            )
        return result

    def cells_for_keys(self, keys: list[tuple[int, int]]) -> list[GridCell]:
        return [self.cells[key] for key in keys]

    def _protected_unhealthy(self, mode: str | None = None) -> bool:
        floor = float(self.cfg["protected_health_floor"])
        return any(
            self.last_health.get(key, 1.0) < floor
            for key in self.protected & self.active
            if mode is None or self.cells[key].mode == mode
        )

    def _protected_health_ready(self, mode: str) -> bool:
        keys = {
            key
            for key in self.protected & self.active
            if self.cells[key].mode == mode
        }
        windows = int(self.cfg["protected_health_windows"])
        return bool(keys) and all(
            key in self.last_health
            and self.last_health[key]
            >= float(self.cfg["protected_health_floor"])
            and self.health_pass_streak.get(key, 0) >= windows
            for key in keys
        )

    def record(
        self,
        episodes: list[tuple[tuple[int, int], bool, bool]],
    ) -> None:
        window = int(self.cfg["window_size"])
        consecutive = int(self.cfg["consecutive_windows"])
        expand_rate = float(self.cfg["expand_success_rate"])
        deactivate_rate = float(self.cfg["deactivate_success_rate"])
        for key, success, health in episodes:
            if key not in self.active:
                continue
            history = self.history[key]
            history.append(bool(success))
            if key in self.protected:
                self.health_history[key].append(bool(health))
            if len(history) < window:
                continue
            sample = history[:window]
            del history[:window]
            rate = sum(sample) / window
            self.last_success[key] = rate
            if key in self.protected:
                health_sample = self.health_history[key][:window]
                del self.health_history[key][:window]
                health_rate = sum(health_sample) / window
                self.last_health[key] = health_rate
                if health_rate >= float(self.cfg["protected_health_floor"]):
                    self.health_pass_streak[key] = (
                        self.health_pass_streak.get(key, 0) + 1
                    )
                else:
                    self.health_pass_streak[key] = 0
            if rate >= expand_rate:
                self.pass_streak[key] = self.pass_streak.get(key, 0) + 1
                self.fail_streak[key] = 0
            elif rate <= deactivate_rate:
                self.fail_streak[key] = self.fail_streak.get(key, 0) + 1
                self.pass_streak[key] = 0
            else:
                self.pass_streak[key] = 0
                self.fail_streak[key] = 0
            if (
                self.pass_streak[key] >= consecutive
                and self._protected_health_ready(self.cells[key].mode)
            ):
                for neighbor in self._neighbors(key):
                    if neighbor in self.cells and neighbor not in self.active:
                        self.active.add(neighbor)
                        self.expansions += 1
                self.pass_streak[key] = 0
            if (
                key not in self.protected
                and self.fail_streak[key] >= consecutive
            ):
                self.active.discard(key)
                self.fail_streak[key] = 0

    def state(self) -> dict[str, Any]:
        cells = []
        for key in sorted(self.active):
            cell = self.cells[key]
            cells.append(
                {
                    "cell": list(key),
                    "cell_id": cell.cell_id,
                    "mode": cell.mode,
                    "vx_bounds_mps": list(cell.vx_bounds),
                    "wz_bounds_rad_s": list(cell.wz_bounds),
                    "protected": key in self.protected,
                    "success_rate": self.last_success.get(key),
                    "health_rate": self.last_health.get(key),
                    "health_pass_streak": self.health_pass_streak.get(key, 0),
                    "samples": len(self.history[key]),
                }
            )
        return {
            "active_cells": len(self.active),
            "protected_cells": len(self.protected),
            "expansions": self.expansions,
            "protected_unhealthy": self._protected_unhealthy(),
            "protected_health_ready": {
                mode: self._protected_health_ready(mode)
                for mode in self.MODE_ORDER
            },
            "cells": cells,
        }
