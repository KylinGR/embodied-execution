"""Fetch, check, train, evaluate and export a simulation-only DIABLO policy."""

import argparse
from dataclasses import asdict, is_dataclass
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
    from .validation import validate_model

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
            "mechanics": validate_model(),
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


def benchmark(args):
    """Measure MuJoCo throughput before committing a long training budget."""
    from .env import DiabloEnv
    from .parallel import make_vector_env

    out = fresh_directory(args.output, "benchmark")
    env, resolved_kind = make_vector_env(
        DiabloEnv,
        None,
        n_envs=args.n_envs,
        kind=args.vec_env,
        monitor_path=str(out / "episodes.csv"),
        start_method=args.subproc_start_method,
    )
    began = time.monotonic()
    try:
        env.seed(args.seed)
        env.reset()
        actions = np.zeros((args.n_envs, 6), dtype=np.float32)
        for _ in range(args.warmup_steps):
            env.step(actions)
        measured_began = time.monotonic()
        for _ in range(args.steps):
            env.step(actions)
        wall_seconds = time.monotonic() - measured_began
        transitions = args.steps * args.n_envs
        result = {
            "vec_env": resolved_kind,
            "n_envs": args.n_envs,
            "control_steps_per_env": args.steps,
            "transitions": transitions,
            "measured_wall_seconds": wall_seconds,
            "transitions_per_second": transitions / max(wall_seconds, 1e-12),
            "total_wall_seconds": time.monotonic() - began,
            "note": (
                "This measures MuJoCo data collection only. Compare configurations "
                "on the same machine and load; PPO update speed is separate."
            ),
        }
        write_json(out / "result.json", result)
        print(json.dumps({"output": str(out), **result}, ensure_ascii=False))
    finally:
        env.close()


def select_best(args):
    from .select_best import run_selection

    return run_selection(args)


def parameter_digest(policy):
    state = b"".join(p.detach().cpu().numpy().tobytes() for p in policy.parameters())
    return hashlib.sha256(state).hexdigest()


def train(args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from .model import metadata
    from .parallel import configure_torch, make_vector_env, resolve_device
    from .contract import (
        environment_class,
        import_class,
        policy_contract,
        require_compatible,
        require_checkpoint,
    )

    if args.resume and args.warm_start:
        raise ValueError("--resume and --warm-start are different workflows")
    device = resolve_device(args.device)
    configure_torch(device, args.cpu_threads, args.gpu_fraction)
    out = fresh_directory(args.output, "train")
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
    task_class = "diablo_training.env:DiabloEnv"
    config_class = "diablo_training.env:Config"
    if args.resume:
        previous = json.loads(
            configuration_path(Path(args.resume).resolve()).read_text()
        )
        config_class = previous.get("config_class", config_class)
        config_type = import_class(config_class)
        config = config_type(**previous["environment"])
        task_class = previous.get("environment_class", task_class)
        settings = previous["ppo"]
        if args.config:
            raise ValueError("Resume inherits its saved config; do not supply --config")
        if (args.max_speed is not None and args.max_speed != config.max_speed) or (
            args.noise and not config.observation_noise
        ):
            raise ValueError(
                "Resume keeps the saved environment; start a new experiment to change it"
            )
    else:
        supplied = json.loads(args.config.read_text()) if args.config else {}
        if set(supplied) - {
            "environment",
            "ppo",
            "environment_class",
            "config_class",
        }:
            raise ValueError(
                "Config keys: environment, ppo, environment_class, config_class"
            )
        parameters = supplied.get("environment", {}).copy()
        if args.max_speed is not None:
            parameters["max_speed"] = args.max_speed
        if args.noise:
            parameters["observation_noise"] = True
        config_class = supplied.get("config_class", config_class)
        config_type = import_class(config_class)
        config = config_type(**parameters)
        settings.update(supplied.get("ppo", {}))
        if set(settings) & {
            "env",
            "policy",
            "device",
            "seed",
            "verbose",
            "tensorboard_log",
        }:
            raise ValueError("Use CLI options for environment, seed and device")
        task_class = supplied.get("environment_class", task_class)
    if not is_dataclass(config):
        raise TypeError("Environment configuration must be a dataclass")
    cls = environment_class(task_class)
    env, resolved_vec_env = make_vector_env(
        cls,
        config,
        n_envs=args.n_envs,
        kind=args.vec_env,
        monitor_path=str(out / "episodes.csv"),
        start_method=args.subproc_start_method,
    )
    contract_env = cls(config)
    model = None
    began = time.monotonic()
    try:
        contract = policy_contract(contract_env)
        if args.resume:
            require_compatible(previous, contract_env)
        if args.resume:
            model = PPO.load(
                args.resume,
                env=env,
                device=device,
                tensorboard_log=str(out / "tensorboard"),
            )
        else:
            model = PPO(
                "MlpPolicy",
                env,
                **settings,
                device=device,
                seed=args.seed,
                verbose=1,
                tensorboard_log=str(out / "tensorboard"),
            )
            if args.warm_start:
                source_resolved = json.loads(
                    configuration_path(Path(args.warm_start).resolve()).read_text()
                )
                source_contract = source_resolved["contract"]
                if source_contract["action_mode"] != contract["action_mode"]:
                    raise ValueError(
                        "Warm-start action semantics changed between stages"
                    )
                if (
                    source_contract["observation"]["shape"]
                    != contract["observation"]["shape"]
                ):
                    raise ValueError(
                        "Warm-start observation contract changed between stages"
                    )
                if (
                    source_contract["model"]["control_dt_s"]
                    != contract["model"]["control_dt_s"]
                ):
                    raise ValueError("Warm-start control period changed between stages")
                for key in ("joints", "actuators", "closure_constraints"):
                    if source_contract["model"][key] != contract["model"][key]:
                        raise ValueError(
                            f"Warm-start robot contract changed for {key}"
                        )
                source = PPO.load(args.warm_start, device=device)
                if source.observation_space.shape != env.observation_space.shape:
                    raise ValueError("Warm-start observation shape is incompatible")
                if source.action_space.shape != env.action_space.shape:
                    raise ValueError("Warm-start action shape is incompatible")
                model.policy.load_state_dict(source.policy.state_dict(), strict=True)
                model.warm_start_from = str(Path(args.warm_start).resolve())
        if args.resume:
            require_checkpoint(model, previous)
        model.diablo_contract_sha256 = contract["sha256"]
        before = parameter_digest(model.policy)
        starting_timesteps = model.num_timesteps
        _, lock = source_path()
        resolved = {
            "environment": asdict(config),
            "environment_class": task_class,
            "config_class": config_class,
            "contract": contract,
            "ppo": settings,
            "model": metadata(contract_env.model),
            "upstream_commit": lock["commit"],
            "versions": versions(),
            "requested_timesteps": args.steps,
            "n_envs": args.n_envs,
            "vec_env": resolved_vec_env,
            "vec_env_requested": args.vec_env,
            "subproc_start_method": args.subproc_start_method,
            "seed": model.seed,
            "device": str(model.device),
            "cpu_threads": args.cpu_threads,
            "gpu_memory_fraction": args.gpu_fraction,
            "resume": None if args.resume is None else str(Path(args.resume).resolve()),
            "warm_start": (
                None if args.warm_start is None else str(Path(args.warm_start).resolve())
            ),
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
            "warm_start": (
                None if args.warm_start is None else str(Path(args.warm_start).resolve())
            ),
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
        contract_env.close()
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
    from .contract import (
        environment_class,
        import_class,
        require_compatible,
        require_checkpoint,
    )

    path = Path(path).resolve()
    resolved = json.loads(configuration_path(path).read_text())
    config_class = import_class(
        resolved.get("config_class", "diablo_training.env:Config")
    )
    config = config_class(**resolved["environment"])
    env = environment_class(
        resolved.get("environment_class", "diablo_training.env:DiabloEnv")
    )(config)
    try:
        require_compatible(resolved, env)
    finally:
        env.close()
    policy = PPO.load(path, device="cpu")
    require_checkpoint(policy, resolved)
    return policy, config, resolved


def evaluate(args):
    import imageio.v2 as imageio
    from .contract import environment_class

    model, config, resolved = load_policy(args.model)
    env = environment_class(resolved["environment_class"])(config)
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
                    if (
                        info["sim_seconds"] >= config.episode_seconds - 0.5
                        and np.max(np.abs(info["command"][:2])) < 0.01
                    ):
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
            "contract": resolved["contract"],
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
    from .contract import environment_class

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
        env = environment_class(resolved["environment_class"])(config)
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
        "policy_contract": resolved["contract"],
        "environment_class": resolved["environment_class"],
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
            command.add_argument("--n-envs", type=int, default=16)
            command.add_argument("--max-speed", type=float)
            command.add_argument("--noise", action="store_true")
            command.add_argument(
                "--device",
                default="auto",
                help="auto, cpu, cuda, or a CUDA index such as cuda:3",
            )
            command.add_argument(
                "--vec-env",
                choices=("auto", "dummy", "subproc"),
                default="auto",
                help="auto uses one process per environment on Linux",
            )
            command.add_argument(
                "--subproc-start-method",
                choices=("fork", "forkserver", "spawn"),
                default="forkserver",
            )
            command.add_argument("--cpu-threads", type=int, default=1)
            command.add_argument(
                "--gpu-fraction",
                type=float,
                default=0.50,
                help="PyTorch CUDA allocator fraction in (0, 1]",
            )
            command.add_argument("--resume", type=Path)
            command.add_argument(
                "--warm-start",
                type=Path,
                help="Load policy weights across a changed environment/reward",
            )
            command.add_argument(
                "--config",
                type=Path,
                help="JSON environment / PPO / environment_class settings",
            )
    check = commands.add_parser(
        "benchmark",
        help="Compare dummy and subprocess MuJoCo collection throughput",
    )
    check.add_argument("--n-envs", type=int, default=16)
    check.add_argument("--steps", type=int, default=200)
    check.add_argument("--warmup-steps", type=int, default=20)
    check.add_argument("--seed", type=int, default=42)
    check.add_argument("--output", type=Path)
    check.add_argument(
        "--vec-env", choices=("auto", "dummy", "subproc"), default="subproc"
    )
    check.add_argument(
        "--subproc-start-method",
        choices=("fork", "forkserver", "spawn"),
        default="forkserver",
    )
    check = commands.add_parser(
        "select-best",
        help="Evaluate checkpoints and copy the best one",
    )
    check.add_argument("--run", type=Path, required=True)
    check.add_argument("--output", type=Path)
    check.add_argument("--seed", type=int, default=2001)
    check.add_argument(
        "--stages",
        nargs="+",
        choices=("stand", "forward", "backward", "turn", "start_stop"),
        default=(
            "stand",
            "forward",
            "backward",
            "turn",
            "start_stop",
        ),
    )
    check.add_argument("--min-success", type=float, default=1.0)
    check.add_argument("--strict", action="store_true")
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
    if hasattr(args, "n_envs") and not 1 <= args.n_envs <= 256:
        parser.error("--n-envs must be 1..256")
    if hasattr(args, "warmup_steps") and args.warmup_steps < 0:
        parser.error("--warmup-steps must be non-negative")
    if hasattr(args, "cpu_threads") and args.cpu_threads <= 0:
        parser.error("--cpu-threads must be positive")
    if hasattr(args, "gpu_fraction") and not 0 < args.gpu_fraction <= 1:
        parser.error("--gpu-fraction must be in (0, 1]")
    if hasattr(args, "min_success") and not 0 <= args.min_success <= 1:
        parser.error("--min-success must be in [0, 1]")
    functions = {
        "fetch": lambda _: fetch(),
        "doctor": doctor,
        "smoke": smoke,
        "benchmark": benchmark,
        "select-best": select_best,
        "train": train,
        "evaluate": evaluate,
        "play": evaluate,
        "export": export,
    }
    return functions[args.command](args) or 0


if __name__ == "__main__":
    sys.exit(main())
