"""Fetch, check, train, evaluate and export a simulation-only DIABLO policy."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

from .resources import ROOT, fetch, source_path


def write_json(path, content):
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n")


def fresh_directory(requested, kind):
    path = (
        requested
        or ROOT
        / "runs"
        / f"{kind}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{time.time_ns() % 1000000000}"
    )
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def versions():
    return {
        name: importlib.metadata.version(name)
        for name in ("mujoco", "gymnasium", "stable-baselines3", "torch", "numpy")
    }


def doctor(args):
    from stable_baselines3.common.env_checker import check_env
    from .env import DiabloEnv
    from .model import metadata

    env = DiabloEnv()
    try:
        check_env(env, warn=True)
        result = {
            "python": platform.python_version(),
            "platform": platform.system(),
            "versions": versions(),
            "model": metadata(env.model),
            "observation_shape": list(env.observation_space.shape),
            "action_shape": list(env.action_space.shape),
            "gymnasium_check": "passed",
            "training_verified": False,
        }
        if args.render:
            env.reset(seed=0)
            result["render_shape"] = list(env.render().shape)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        env.close()


def smoke(args):
    from .env import DiabloEnv

    out = fresh_directory(args.output, "smoke")
    env = DiabloEnv()
    began = time.monotonic()
    episodes = []
    try:
        obs, _ = env.reset(seed=args.seed)
        env.action_space.seed(args.seed)
        with (out / "trace.jsonl").open("x") as log:
            for i in range(args.steps):
                obs, _, terminated, truncated, info = env.step(
                    env.action_space.sample() * 0.1
                )
                if not np.isfinite(obs).all():
                    raise FloatingPointError("Nonfinite observation")
                log.write(json.dumps(info) + "\n")
                if terminated or truncated:
                    episodes.append(
                        {**info, "terminated": terminated, "truncated": truncated}
                    )
                    env.reset()
        result = {
            "control_steps": args.steps,
            "integrated_sim_seconds": args.steps * 0.01,
            "wall_seconds": time.monotonic() - began,
            "episodes": episodes,
            "physics_pipeline": "passed",
            "policy_training": "not run",
            "note": "Normal training episode resets are recorded; this is not a locomotion qualification.",
        }
        write_json(out / "result.json", result)
        print(json.dumps({"output": str(out), **result}, ensure_ascii=False))
    finally:
        env.close()


def parameter_digest(policy):
    state = b"".join(p.detach().cpu().numpy().tobytes() for p in policy.parameters())
    return hashlib.sha256(state).hexdigest()


def train(args):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    from .env import Config, DiabloEnv
    from .model import metadata

    torch.set_num_threads(1)
    out = fresh_directory(args.output, "train")
    if args.resume:
        previous = json.loads(
            configuration_path(Path(args.resume).resolve()).read_text()
        )
        config = Config(**previous["environment"])
        if (args.max_speed is not None and args.max_speed != config.max_speed) or (
            args.noise and not config.observation_noise
        ):
            raise ValueError(
                "Resume keeps the saved environment; start a new experiment to change it"
            )
    else:
        config = Config(
            max_speed=0.5 if args.max_speed is None else args.max_speed,
            observation_noise=args.noise,
        )
    env = VecMonitor(
        DummyVecEnv([lambda: DiabloEnv(config) for _ in range(args.n_envs)]),
        filename=str(out / "episodes.csv"),
    )
    model = None
    began = time.monotonic()
    try:
        settings = {
            "n_steps": 128,
            "batch_size": 64,
            "n_epochs": 4,
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.01,
            "policy_kwargs": {"net_arch": [128, 128]},
        }
        model = (
            PPO.load(
                args.resume,
                env=env,
                device=args.device,
                tensorboard_log=str(out / "tensorboard"),
            )
            if args.resume
            else PPO(
                "MlpPolicy",
                env,
                **settings,
                device=args.device,
                seed=args.seed,
                verbose=1,
                tensorboard_log=str(out / "tensorboard"),
            )
        )
        before = parameter_digest(model.policy)
        starting_timesteps = model.num_timesteps
        _, lock = source_path()
        resolved = {
            "environment": asdict(config),
            "ppo": settings,
            "model": metadata(env.venv.envs[0].model),
            "upstream_commit": lock["commit"],
            "versions": versions(),
            "requested_timesteps": args.steps,
            "n_envs": args.n_envs,
            "seed": args.seed,
            "device": args.device,
            "resume": None if args.resume is None else str(Path(args.resume).resolve()),
            "parameter_sha256_before": before,
        }
        write_json(out / "resolved.json", resolved)
        callback = CheckpointCallback(
            save_freq=max(1, 100000 // args.n_envs), save_path=str(out / "checkpoints")
        )
        model.learn(
            total_timesteps=args.steps,
            callback=callback,
            reset_num_timesteps=not bool(args.resume),
        )
        model.save(out / "policy.zip")
        after = parameter_digest(model.policy)
        result = {
            "actual_total_timesteps": model.num_timesteps,
            "actual_additional_timesteps": model.num_timesteps - starting_timesteps,
            "requested_additional_timesteps": args.steps,
            "wall_seconds": time.monotonic() - began,
            "parameters_changed": before != after,
            "parameter_sha256_after": after,
            "policy": "policy.zip",
            "scope": "PPO training completed; evaluate locomotion separately",
        }
        write_json(out / "result.json", result)
        print(json.dumps({"output": str(out), **result}))
    except BaseException as error:
        if model is not None:
            model.save(out / "failure-policy.zip")
        write_json(
            out / "failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "wall_seconds": time.monotonic() - began,
            },
        )
        raise
    finally:
        env.close()


def configuration_path(path):
    candidates = [path.parent / "resolved.json"]
    if path.parent.name == "checkpoints":
        candidates.append(path.parent.parent / "resolved.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("resolved.json must accompany the training checkpoint")


def load_policy(path):
    from stable_baselines3 import PPO
    from .env import Config

    path = Path(path).resolve()
    resolved = json.loads(configuration_path(path).read_text())
    return PPO.load(path, device="cpu"), Config(**resolved["environment"]), resolved


def evaluate(args):
    import imageio.v2 as imageio
    from .env import DiabloEnv

    model, config, resolved = load_policy(args.model)
    env = DiabloEnv(config)
    out = fresh_directory(args.output, "evaluate")
    observations, actions, reports = [], [], []
    writer = imageio.get_writer(out / "rollout.mp4", fps=25) if args.video else None
    viewer = None
    began = time.monotonic()
    try:
        if args.viewer:
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(env.model, env.data)
        with (out / "trace.jsonl").open("x") as log:
            for task in range(4):
                obs, _ = env.reset(seed=args.seed + task, options={"task": task})
                errors, stopped_speeds = [], []
                terminated = truncated = cancelled = False
                for i in range(round(config.episode_seconds / 0.01)):
                    tick = time.monotonic()
                    action, _ = model.predict(obs, deterministic=True)
                    observations.append(obs.copy())
                    actions.append(action.copy())
                    obs, reward, terminated, truncated, info = env.step(action)
                    log.write(json.dumps({**info, "reward": reward}) + "\n")
                    errors.append(info["speed_error_mps"])
                    if info["sim_seconds"] >= config.episode_seconds - 0.5:
                        stopped_speeds.append(abs(info["forward_mps"]))
                    if writer is not None and i % 4 == 0:
                        writer.append_data(env.render())
                    if viewer is not None:
                        viewer.sync()
                        if not viewer.is_running():
                            cancelled = True
                            break
                        time.sleep(max(0.0, 0.01 - (time.monotonic() - tick)))
                    if terminated or truncated:
                        break
                reports.append(
                    {
                        "task": task,
                        "sim_seconds": info["sim_seconds"],
                        "terminated": terminated,
                        "time_limit_reached": truncated,
                        "cancelled": cancelled,
                        "speed_mae_mps": float(np.mean(errors)),
                        "final_forward_mps": info["forward_mps"],
                        "final_half_second_max_speed_mps": max(
                            stopped_speeds, default=None
                        ),
                        "hold_drift_m": info.get("hold_drift_m"),
                    }
                )
                if cancelled:
                    break
        np.savez_compressed(
            out / "trajectory.npz",
            observations=np.asarray(observations),
            actions=np.asarray(actions),
        )
        result = {
            "episodes": reports,
            "wall_seconds": time.monotonic() - began,
            "integrated_sim_seconds": sum(r["sim_seconds"] for r in reports),
            "model": resolved["model"],
            "note": "Each task is a separate first episode; failures are retained without retry.",
        }
        write_json(out / "result.json", result)
        print(json.dumps({"output": str(out), **result}, ensure_ascii=False))
        return 2 if any(r["terminated"] or r["cancelled"] for r in reports) else 0
    finally:
        if writer is not None:
            writer.close()
        if viewer is not None:
            viewer.close()
        env.close()


def export(args):
    import torch
    import onnxruntime as ort
    from .env import DiabloEnv

    model, config, resolved = load_policy(args.model)
    out = fresh_directory(args.output, "export")

    class Actor(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, observation):
            return self.policy._predict(observation, deterministic=True).clamp(-1, 1)

    actor = Actor(model.policy).eval()
    if args.observations:
        samples = np.load(args.observations, allow_pickle=False)["observations"][:256]
    else:
        env = DiabloEnv(config)
        frames = []
        try:
            obs, _ = env.reset(seed=0)
            for _ in range(128):
                frames.append(obs.copy())
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, truncated, _ = env.step(action)
                if done or truncated:
                    obs, _ = env.reset()
            samples = np.asarray(frames)
        finally:
            env.close()
    if (
        samples.ndim != 2
        or samples.shape[1] != 125
        or not len(samples)
        or not np.isfinite(samples).all()
    ):
        raise ValueError("Export observations must be nonempty finite [N,125]")
    samples = samples.astype(np.float32)
    example = torch.from_numpy(samples[:1])
    torch.jit.trace(actor, example).save(str(out / "policy.pt"))
    torch.onnx.export(
        actor,
        example,
        str(out / "policy.onnx"),
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    session = ort.InferenceSession(
        str(out / "policy.onnx"), providers=["CPUExecutionProvider"]
    )
    actual = session.run(None, {"observation": samples})[0]
    expected, _ = model.predict(samples, deterministic=True)
    scripted = torch.jit.load(str(out / "policy.pt"))
    with torch.no_grad():
        jit = scripted(torch.from_numpy(samples)).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(jit, expected, atol=1e-5, rtol=1e-5)
    contract = {
        "input": {"observation": ["N", 125]},
        "output": {"action": ["N", 6]},
        "dtype": "float32",
        "history": "oldest to newest, five 25-value frames including current",
        "control_dt_s": 0.01,
        "model": resolved["model"],
        "environment": resolved["environment"],
        "interface_document": "docs/model-interface.md",
        "simulation_only": True,
        "hardware_authorized": False,
        "samples_checked": len(samples),
        "onnx_max_abs_error": float(np.max(np.abs(actual - expected))),
        "jit_max_abs_error": float(np.max(np.abs(jit - expected))),
    }
    write_json(out / "contract.json", contract)
    print(json.dumps({"directory": str(out), **contract}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(
        description="DIABLO MuJoCo training — simulation only"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("fetch", help="Fetch the pinned upstream robot assets")
    check = commands.add_parser(
        "doctor", help="Validate model, sensors and Gymnasium API"
    )
    check.add_argument("--render", action="store_true")
    for name in ("smoke", "train"):
        command = commands.add_parser(name)
        command.add_argument(
            "--steps", type=int, default=200 if name == "smoke" else 200000
        )
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--output", type=Path)
        if name == "train":
            command.add_argument("--n-envs", type=int, default=4)
            command.add_argument("--max-speed", type=float)
            command.add_argument("--noise", action="store_true")
            command.add_argument("--device", default="cpu")
            command.add_argument("--resume", type=Path)
    check = commands.add_parser("evaluate", aliases=["play"])
    check.add_argument("--model", type=Path, required=True)
    check.add_argument("--output", type=Path)
    check.add_argument("--seed", type=int, default=314)
    check.add_argument("--video", action="store_true")
    check.add_argument("--viewer", action="store_true")
    check = commands.add_parser("export")
    check.add_argument("--model", type=Path, required=True)
    check.add_argument("--output", type=Path)
    check.add_argument("--observations", type=Path)
    args = parser.parse_args()
    if hasattr(args, "steps") and args.steps <= 0:
        parser.error("--steps must be positive")
    if hasattr(args, "n_envs") and not 1 <= args.n_envs <= 64:
        parser.error("--n-envs must be 1..64")
    functions = {
        "fetch": lambda _: fetch(),
        "doctor": doctor,
        "smoke": smoke,
        "train": train,
        "evaluate": evaluate,
        "play": evaluate,
        "export": export,
    }
    return functions[args.command](args) or 0


if __name__ == "__main__":
    sys.exit(main())
