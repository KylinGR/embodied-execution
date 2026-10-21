"""MuJoCo terrain worlds for the B/C/D XINGTIAN curriculum stages."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .model import _finish_model, _robot_spec
from .xingtian import TERRAIN_LEVELS, TerrainLevel, fractal_heightfield


STAGES = ("B", "C", "D")


@dataclass(frozen=True)
class TerrainProfile:
    stage: str
    level: TerrainLevel
    seed: int
    track_length_m: float
    track_width_m: float
    heightfield: np.ndarray
    obstacles: tuple[dict, ...]

    @property
    def start_x_range(self) -> tuple[float, float]:
        return 1.0, self.track_length_m - 3.0


def _level_for_stage(stage: str, level_index: int) -> TerrainLevel:
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    if not 0 <= level_index < len(TERRAIN_LEVELS):
        raise ValueError(f"terrain level must be 0..{len(TERRAIN_LEVELS) - 1}")
    if stage == "C":
        base = TERRAIN_LEVELS[2]
        return TerrainLevel(
            name="obstacle",
            rms_height=0.0,
            peak_height=0.0,
            feature_scale=1.0,
            max_slope_deg=0.0,
            obstacle_height=0.01 + 0.01 * level_index,
            friction_min=base.friction_min,
            friction_max=base.friction_max,
        )
    return TERRAIN_LEVELS[level_index]


def _height_at(
    heightfield: np.ndarray,
    *,
    x: float,
    y: float,
    track_length_m: float,
    track_width_m: float,
) -> float:
    nx, ny = heightfield.shape
    fx = np.clip(x / track_length_m, 0.0, 1.0) * (nx - 1)
    fy = np.clip((y + track_width_m / 2) / track_width_m, 0.0, 1.0) * (ny - 1)
    x0, y0 = int(np.floor(fx)), int(np.floor(fy))
    x1, y1 = min(x0 + 1, nx - 1), min(y0 + 1, ny - 1)
    tx, ty = fx - x0, fy - y0
    low = heightfield[x0, y0] * (1 - tx) + heightfield[x1, y0] * tx
    high = heightfield[x0, y1] * (1 - tx) + heightfield[x1, y1] * tx
    return float(low * (1 - ty) + high * ty)


def build_terrain_profile(
    *,
    stage: str,
    level_index: int,
    seed: int,
    track_length_m: float = 40.0,
    track_width_m: float = 6.0,
    obstacle_height_m: float | None = None,
    obstacle_spacing_m: float = 4.0,
) -> TerrainProfile:
    level = _level_for_stage(stage, level_index)
    if obstacle_height_m is not None:
        level = TerrainLevel(
            name=level.name,
            rms_height=level.rms_height,
            peak_height=level.peak_height,
            feature_scale=level.feature_scale,
            max_slope_deg=level.max_slope_deg,
            obstacle_height=float(obstacle_height_m),
            friction_min=level.friction_min,
            friction_max=level.friction_max,
        )
    if stage in {"B", "D"}:
        heightfield = fractal_heightfield(
            level,
            seed=seed,
            length_m=track_length_m,
            width_m=track_width_m,
            horizontal_resolution_m=0.05,
        )
        if stage == "D":
            x = np.linspace(0.0, track_length_m, heightfield.shape[0])
            flat = np.zeros_like(x)
            for start, end in (
                (5.0, 13.0),
                (0.45 * track_length_m, 0.55 * track_length_m),
                (0.80 * track_length_m, 0.88 * track_length_m),
            ):
                flat[(x >= start) & (x <= end)] = 1.0
            width = max(3, int(round(1.0 / 0.05)))
            kernel = np.ones(width, dtype=float) / width
            blend = np.convolve(flat, kernel, mode="same")
            heightfield = heightfield * (1.0 - blend[:, None])
    else:
        heightfield = np.zeros(
            (
                int(round(track_length_m / 0.05)) + 1,
                int(round(track_width_m / 0.05)) + 1,
            ),
            dtype=np.float32,
        )

    obstacles = []
    if stage in {"C", "D"} and level.obstacle_height > 0:
        rng = np.random.default_rng(seed + 91_337)
        x = 3.0
        index = 0
        while x < track_length_m - 3.0:
            height = level.obstacle_height * float(rng.uniform(0.55, 1.0))
            ground = _height_at(
                heightfield,
                x=x,
                y=0.0,
                track_length_m=track_length_m,
                track_width_m=track_width_m,
            )
            obstacles.append(
                {
                    "name": f"obstacle_{index:03d}",
                    "x": float(x),
                    "y": 0.0,
                    "height": float(height),
                    "length": float(rng.uniform(0.12, 0.25)),
                    "width": float(rng.uniform(2.0, 2.8)),
                    "ground": float(ground),
                }
            )
            x += obstacle_spacing_m * float(rng.uniform(0.85, 1.15))
            index += 1
    return TerrainProfile(
        stage=stage,
        level=level,
        seed=seed,
        track_length_m=track_length_m,
        track_width_m=track_width_m,
        heightfield=heightfield,
        obstacles=tuple(obstacles),
    )


def make_terrain_model(profile: TerrainProfile):
    spec = _robot_spec()
    minimum = float(profile.heightfield.min())
    maximum = float(profile.heightfield.max())
    elevation = max(maximum - minimum, 1e-6)
    spec.add_hfield(
        name="terrain",
        # MuJoCo hfield rows follow y and columns follow x.
        nrow=profile.heightfield.shape[1],
        ncol=profile.heightfield.shape[0],
        # MuJoCo normalizes these values to [0, 1] internally.
        userdata=profile.heightfield.T.astype(float).reshape(-1).tolist(),
        size=[
            profile.track_length_m / 2,
            profile.track_width_m / 2,
            elevation,
            0.02,
        ],
    )
    spec.worldbody.add_geom(
        name="terrain_surface",
        type=mujoco.mjtGeom.mjGEOM_HFIELD,
        hfieldname="terrain",
        pos=[profile.track_length_m / 2, 0, minimum],
        rgba=[0.30, 0.34, 0.31, 1],
        contype=2,
        conaffinity=1,
        friction=[0.8, 0.005, 0.0001],
    )
    for obstacle in profile.obstacles:
        spec.worldbody.add_geom(
            name=obstacle["name"],
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[
                obstacle["x"],
                obstacle["y"],
                obstacle["ground"] + obstacle["height"] / 2 - 0.01,
            ],
            size=[
                obstacle["length"] / 2,
                obstacle["width"] / 2,
                obstacle["height"] / 2 + 0.01,
            ],
            rgba=[0.68, 0.28, 0.12, 1],
            contype=2,
            conaffinity=1,
            friction=[0.85, 0.005, 0.0001],
        )
    if profile.stage in {"C", "D"}:
        half_lane = 1.55
        for side_name, y in (("left", -half_lane), ("right", half_lane)):
            spec.worldbody.add_geom(
                name=f"lane_wall_{side_name}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[profile.track_length_m / 2, y, 0.35],
                size=[profile.track_length_m / 2, 0.05, 0.35],
                rgba=[0.18, 0.20, 0.22, 0.65],
                contype=2,
                conaffinity=1,
                friction=[0.8, 0.005, 0.0001],
            )
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=[0, 0, -0.6],
        size=[80, 40, 0.1],
        rgba=[0.12, 0.14, 0.16, 1],
        contype=2,
        conaffinity=1,
        friction=[0.8, 0.005, 0.0001],
    )
    return _finish_model(spec)


def profile_height(profile: TerrainProfile, x: float, y: float) -> float:
    return _height_at(
        profile.heightfield,
        x=x,
        y=y,
        track_length_m=profile.track_length_m,
        track_width_m=profile.track_width_m,
    )
