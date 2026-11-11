"""Deterministic command-profile evaluation for the XINGTIAN reference task."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import time

import numpy as np

from .cli import fresh_directory, load_policy, write_json


class ProfileXingtianEnvMixin:
    """Add scripted command profiles to ``XingtianFlatEnv`` without changing physics."""

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        self.profile = options.pop("profile", None)
        self.profile_times = None
        self.profile_values = None
        if self.profile is not None:
            self.profile_times = np.asarray(
                [item["time"] for item in self.profile], dtype=float
            )
            self.profile_values = np.asarray(
                [item["command"] for item in self.profile], dtype=float
            )
            if (
                self.profile_times.ndim != 1
                or self.profile_values.shape != (len(self.profile_times), 3)
                or not np.isfinite(self.profile_times).all()
                or not np.isfinite(self.profile_values).all()
                or np.any(np.diff(self.profile_times) <= 0)
            ):
                raise ValueError("profile must contain increasing times and 3D commands")
        return super().reset(seed=seed, options=options)

    def get_command(self, sim_seconds):
        if self.profile is None:
            return super().get_command(sim_seconds)
        return np.array(
            [
                np.interp(sim_seconds, self.profile_times, self.profile_values[:, axis])
                for axis in range(3)
            ]
        )


STAGES = {
    "stand": {
        "duration": 10.0,
        "profile": [{"time": 0.0, "command": [0.0, 0.0, 0.25]}],
    },
    "forward": {
        "duration": 12.0,
        "profile": [
            {"time": 0.0, "command": [0.0, 0.0, 0.25]},
            {"time": 1.0, "command": [0.5, 0.0, 0.25]},
            {"time": 10.0, "command": [0.5, 0.0, 0.25]},
        ],
    },
    "backward": {
        "duration": 12.0,
        "profile": [
            {"time": 0.0, "command": [0.0, 0.0, 0.25]},
            {"time": 1.0, "command": [-0.3, 0.0, 0.25]},
            {"time": 10.0, "command": [-0.3, 0.0, 0.25]},
        ],
    },
    "turn": {
        "duration": 12.0,
        "profile": [
            {"time": 0.0, "command": [0.0, 0.0, 0.25]},
            {"time": 1.0, "command": [0.0, 0.8, 0.25]},
            {"time": 10.0, "command": [0.0, 0.8, 0.25]},
        ],
    },
    "start_stop": {
        "duration": 16.0,
        "profile": [
            {"time": 0.0, "command": [0.0, 0.0, 0.25]},
            {"time": 1.0, "command": [0.4, 0.0, 0.25]},
            {"time": 4.0, "command": [0.4, 0.0, 0.25]},
            {"time": 5.0, "command": [0.0, 0.0, 0.25]},
            {"time": 7.0, "command": [0.0, 0.0, 0.25]},
            {"time": 8.0, "command": [0.4, 0.0, 0.25]},
            {"time": 11.0, "command": [0.4, 0.0, 0.25]},
            {"time": 12.0, "command": [0.0, 0.0, 0.25]},
        ],
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=9001)
    parser.add_argument("--stages", nargs="+", choices=tuple(STAGES), default=list(STAGES))
    parser.add_argument("--video", action="store_true")
    parser.add_argument(
        "--terrain-seed",
        type=int,
        help="Override terrain seed for B/C/D evaluation",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    import imageio.v2 as imageio
    from .contract import environment_class
    from .xingtian_env import XingtianFlatEnv, XingtianStageConfig

    class ProfileEnv(ProfileXingtianEnvMixin, XingtianFlatEnv):
        pass

    model, config, resolved = load_policy(args.model)
    if args.terrain_seed is not None and isinstance(config, XingtianStageConfig):
        config = replace(config, terrain_seed=args.terrain_seed)
    env_type = environment_class(resolved["environment_class"])
    if not issubclass(env_type, XingtianFlatEnv):
        raise TypeError("xingtian-eval requires an XingtianFlatEnv-derived task")

    class LoadedProfileEnv(ProfileXingtianEnvMixin, env_type):
        pass

    env = LoadedProfileEnv(config)
    out = fresh_directory(args.output, "xingtian-eval")
    writer = imageio.get_writer(out / "rollout.mp4", fps=25) if args.video else None
    reports = []
    began = time.monotonic()
    try:
        for stage_name in args.stages:
            stage = STAGES[stage_name]
            obs, _ = env.reset(
                seed=args.seed,
                options={"profile": stage["profile"]},
            )
            observations, actions = [], []
            speed_errors, yaw_errors = [], []
            upright_values, heights = [], []
            terminated = truncated = False
            info = {}
            for index in range(round(stage["duration"] / 0.01)):
                action, _ = model.predict(obs, deterministic=True)
                observations.append(obs.copy())
                actions.append(action.copy())
                obs, reward, terminated, truncated, info = env.step(action)
                command = info["command"]
                speed_errors.append(abs(info["forward_mps"] - command[0]))
                yaw_errors.append(
                    abs(float(env.data.sensor("gyro").data[2]) - command[1])
                )
                upright_values.append(abs(info["upright"]))
                heights.append(float(env.data.xpos[env.base_id, 2]))
                if writer is not None and index % 4 == 0:
                    writer.append_data(env.render())
                if terminated or truncated:
                    break
            np.savez_compressed(
                out / f"{stage_name}.npz",
                observations=np.asarray(observations),
                actions=np.asarray(actions),
            )
            reports.append(
                {
                    "stage": stage_name,
                    "duration_s": float(info.get("sim_seconds", 0.0)),
                    "success": bool(not terminated),
                    "terminated": bool(terminated),
                    "time_limit_reached": bool(truncated),
                    "speed_mae_mps": float(np.mean(speed_errors)),
                    "yaw_mae_rad_s": float(np.mean(yaw_errors)),
                    "min_upright": float(np.min(upright_values)),
                    "height_mean_m": float(np.mean(heights)),
                }
            )
    finally:
        if writer is not None:
            writer.close()
        env.close()

    result = {
        "seed": args.seed,
        "wall_seconds": time.monotonic() - began,
        "reports": reports,
        "model": resolved["model"],
        "contract": resolved["contract"],
    }
    write_json(out / "result.json", result)
    print(result)
    return 0 if all(report["success"] for report in reports) else 2


if __name__ == "__main__":
    raise SystemExit(main())
