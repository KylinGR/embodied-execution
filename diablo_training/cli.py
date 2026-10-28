"""Fetch assets, validate physics, train, evaluate, select, and export policies."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

from .contract import SCREEN_SEEDS, budget_iterations, load_course, task_id
from .resources import ROOT, fetch


def write_json(path: Path, content: object) -> None:
    path.write_text(
        json.dumps(content, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def fresh_directory(requested: Path | None, kind: str) -> Path:
    path = (
        requested
        or ROOT
        / "runs"
        / f"{kind}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{time.time_ns() % 1000000000}"
    )
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def versions() -> dict[str, str]:
    names = ("mujoco", "mjlab", "mujoco-warp", "torch", "rsl-rl-lib")
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def doctor(args) -> None:
    from gymnasium.utils.env_checker import check_env
    from .env import DiabloEnv
    from .model import metadata
    from .validation import validate_model

    env = DiabloEnv()
    try:
        check_env(env, skip_render_check=True)
        result = {
            "python": platform.python_version(),
            "platform": platform.system(),
            "versions": versions(),
            "model": metadata(env.model),
            "observation_shape": list(env.observation_space.shape),
            "action_shape": list(env.action_space.shape),
            "gymnasium_check": "passed",
            "mechanics": validate_model(),
            "training_backend": "mjlab",
            "scope": "CPU physics fixture only; no CPU training fallback",
        }
        if args.render:
            env.reset(seed=0)
            result["render_shape"] = list(env.render().shape)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        env.close()


def gpu_doctor(args) -> None:
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from .mjlab_tasks import COURSE, get_task_cfgs, require_cuda

    device = require_cuda(args.device)
    env_cfg, _ = get_task_cfgs("flat_balance", play=True)
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    try:
        obs, _ = env.reset(seed=2001)
        action = env.action_manager.action.new_zeros((1, 6))
        obs, reward, terminated, truncated, _ = env.step(action)
        result = {
            "device": device,
            "gpu": torch.cuda.get_device_name(torch.device(device)),
            "versions": versions(),
            "task_count": len(COURSE.order),
            "observation_shapes": {
                name: list(value.shape) for name, value in obs.items()
            },
            "action_shape": list(action.shape),
            "finite_reward": bool(reward.isfinite().all().item()),
            "terminated": bool(terminated.item()),
            "truncated": bool(truncated.item()),
            "scope": "one environment, one zero-action physics step",
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        env.close()


def list_stages(args) -> None:
    course = load_course()
    stages = []
    for order, name in enumerate(course.order):
        stage = course.stage(name)
        stages.append(
            {
                "order": order,
                "stage": name,
                "task_id": task_id(name),
                "budget": int(stage["budget"]),
                "iterations": budget_iterations(
                    int(stage["budget"]), args.n_envs
                ),
                "profiles": [profile["name"] for profile in stage["profiles"]],
                "terrain": stage["terrain"]["kind"],
            }
        )
    print(json.dumps({"n_envs": args.n_envs, "stages": stages}, indent=2))


def smoke(args) -> None:
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from .mjlab_tasks import get_task_cfgs, require_cuda

    device = require_cuda(args.device)
    env_cfg, _ = get_task_cfgs(args.stage, play=True)
    env_cfg.scene.num_envs = args.n_envs
    env_cfg.episode_length_s = min(env_cfg.episode_length_s, 2.0)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    out = fresh_directory(args.output, "smoke")
    try:
        env.reset(seed=args.seed)
        trace = []
        for step in range(args.steps):
            action = torch.rand(
                (args.n_envs, 6), device=device, dtype=torch.float32
            )
            obs, reward, terminated, truncated, _ = env.step(action * 0.1)
            record = {
                "step": step,
                "finite_obs": bool(torch.isfinite(obs["actor"]).all().item()),
                "finite_reward": bool(torch.isfinite(reward).all().item()),
                "terminated": terminated.int().tolist(),
                "truncated": truncated.int().tolist(),
            }
            trace.append(record)
            if bool((terminated | truncated).any().item()):
                break
        result = {
            "device": device,
            "stage": args.stage,
            "n_envs": args.n_envs,
            "steps": len(trace),
            "trace": trace,
            "physics_pipeline": "passed",
            "training": "not run",
        }
        write_json(out / "result.json", result)
        print(json.dumps({"output": str(out), **result}, ensure_ascii=False))
    finally:
        env.close()


def _experiment_root(log_root: Path) -> Path:
    return (log_root / "diablo_xingtian_course").resolve()


def _resume_args(checkpoint: Path | None, log_root: Path) -> list[str]:
    if checkpoint is None:
        return []
    checkpoint = checkpoint.resolve()
    expected_parent = _experiment_root(log_root)
    if checkpoint.parent.parent != expected_parent:
        raise ValueError(
            "Resume checkpoint must be inside "
            f"{expected_parent}/<run>/<checkpoint>"
        )
    return [
        "--agent.resume",
        "True",
        "--agent.load-run",
        checkpoint.parent.name,
        "--agent.load-checkpoint",
        checkpoint.name,
    ]


def train_stage(
    stage: str,
    n_envs: int,
    log_root: Path,
    resume: Path | None,
    gpu_ids: list[int] | str,
    iterations_override: int | None = None,
) -> Path:
    from .mjlab_tasks import latest_run, require_cuda, write_run_contract

    require_cuda()
    course = load_course()
    if stage not in course.stages:
        raise ValueError(f"Unknown stage '{stage}'")
    iterations = (
        iterations_override
        if iterations_override is not None
        else budget_iterations(int(course.stage(stage)["budget"]), n_envs, 24)
    )
    gpu_value = "all" if gpu_ids == "all" else str(list(gpu_ids))
    command = [
        sys.executable,
        "-m",
        "mjlab.scripts.train",
        task_id(stage),
        "--env.scene.num-envs",
        str(n_envs),
        "--agent.max-iterations",
        str(iterations),
        "--agent.save-interval",
        "10" if stage == "composite" else "50",
        "--agent.experiment-name",
        "diablo_xingtian_course",
        "--agent.run-name",
        stage,
        "--agent.logger",
        "tensorboard",
        "--agent.upload-model",
        "False",
        "--log-root",
        str(log_root.resolve()),
        "--gpu-ids",
        gpu_value,
        *_resume_args(resume, log_root),
    ]
    subprocess.run(command, check=True)
    run_dir = latest_run(log_root.resolve(), stage)
    write_run_contract(run_dir, stage, n_envs, iterations)
    return run_dir


def _latest_checkpoint(run_dir: Path) -> Path:
    from .mjlab_tasks import candidate_checkpoints

    return candidate_checkpoints(run_dir)[-1]


def evaluate_policy(args) -> None:
    from .mjlab_tasks import evaluate_checkpoint, require_cuda

    device = require_cuda(args.device)
    seeds = tuple(
        int(value.strip()) for value in args.seeds.split(",") if value.strip()
    )
    if not seeds:
        raise ValueError("At least one evaluation seed is required")
    result = evaluate_checkpoint(
        args.stage, Path(args.checkpoint), seeds, device=device
    )
    out = fresh_directory(args.output, "evaluate")
    write_json(out / "result.json", result)
    print(json.dumps({"output": str(out), **result}, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(2)


def train(args) -> None:
    run_dir = train_stage(
        args.stage,
        args.n_envs,
        args.log_root,
        args.resume,
        args.gpu_ids,
        iterations_override=args.iterations,
    )
    print(
        json.dumps(
            {
                "run": str(run_dir),
                "checkpoint": str(_latest_checkpoint(run_dir)),
            },
            ensure_ascii=False,
        )
    )


def course_train(args) -> None:
    course = load_course()
    resume = args.resume
    final_run = None
    for stage in course.order:
        run_dir = train_stage(
            stage,
            args.n_envs,
            args.log_root,
            resume,
            args.gpu_ids,
        )
        checkpoint = _latest_checkpoint(run_dir)
        if stage != "composite":
            result = evaluate_policy_result(stage, checkpoint, SCREEN_SEEDS, args)
            gate_out = fresh_directory(
                args.output / stage if args.output else None, "gate"
            )
            write_json(gate_out / "result.json", result)
            if not result["passed"]:
                raise RuntimeError(
                    f"Stage {stage} failed its all-seed gate; see {gate_out}"
                )
        resume = checkpoint
        final_run = run_dir
        print(
            f"[course] {stage}: "
            f"{'TRAINED' if stage == 'composite' else 'PASS'} -> {run_dir}"
        )
    if final_run is None:
        raise RuntimeError("No final composite run was produced")
    select(args, run_dir=final_run)


def evaluate_policy_result(
    stage: str, checkpoint: Path, seeds: tuple[int, ...], args
) -> dict:
    from .mjlab_tasks import evaluate_checkpoint

    return evaluate_checkpoint(stage, checkpoint, seeds, device=args.device)


def select(args, run_dir: Path | None = None) -> None:
    from .mjlab_tasks import require_cuda, select_policy_best

    require_cuda(args.device)
    result = select_policy_best(run_dir or args.run_dir, device=args.device)
    print(
        json.dumps(
            {
                "selected": result["selected"],
                "policy_best_pt": result["policy_best_pt"],
                "policy_best_onnx": result["policy_best_onnx"],
                "blind_passed": result["blind"]["passed"],
                "limit_passed": result["limit"]["passed"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if not result["blind"]["passed"] or not result["limit"]["passed"]:
        raise SystemExit(2)


def export(args) -> None:
    from .mjlab_tasks import export_checkpoint, require_cuda

    device = require_cuda(args.device)
    out = fresh_directory(args.output, "export")
    path = export_checkpoint(
        args.stage,
        Path(args.checkpoint),
        out,
        filename="policy.onnx",
        device=device,
    )
    print(json.dumps({"onnx": str(path)}, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Xingtian DIABLO MJLab GPU locomotion training"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("fetch", help="Fetch pinned CAD sources")
    check = commands.add_parser("doctor", help="CPU model and API checks")
    check.add_argument("--render", action="store_true")
    gpu_check = commands.add_parser("gpu-doctor", help="CUDA/MJLab smoke check")
    gpu_check.add_argument("--device", default="cuda:0")

    stages = commands.add_parser("list-stages")
    stages.add_argument("--n-envs", type=int, default=4096)

    smoke_parser = commands.add_parser("smoke")
    smoke_parser.add_argument("--stage", default="flat_balance")
    smoke_parser.add_argument("--n-envs", type=int, default=2)
    smoke_parser.add_argument("--steps", type=int, default=20)
    smoke_parser.add_argument("--seed", type=int, default=42)
    smoke_parser.add_argument("--device", default="cuda:0")
    smoke_parser.add_argument("--output", type=Path)

    train_parser = commands.add_parser("train")
    train_parser.add_argument("--stage", required=True)
    train_parser.add_argument("--n-envs", type=int, default=4096)
    train_parser.add_argument("--log-root", type=Path, default=Path("logs/rsl_rl"))
    train_parser.add_argument("--resume", type=Path)
    train_parser.add_argument("--gpu-ids", default="0")
    train_parser.add_argument("--iterations", type=int)

    course_parser = commands.add_parser("course-train")
    course_parser.add_argument("--n-envs", type=int, default=4096)
    course_parser.add_argument("--log-root", type=Path, default=Path("logs/rsl_rl"))
    course_parser.add_argument("--resume", type=Path)
    course_parser.add_argument("--gpu-ids", default="0")
    course_parser.add_argument("--device", default="cuda:0")
    course_parser.add_argument("--output", type=Path)

    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--stage", required=True)
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--seeds", required=True)
    evaluate_parser.add_argument("--device", default="cuda:0")
    evaluate_parser.add_argument("--output", type=Path)

    select_parser = commands.add_parser("select-best")
    select_parser.add_argument("--run-dir", type=Path, required=True)
    select_parser.add_argument("--device", default="cuda:0")

    export_parser = commands.add_parser("export")
    export_parser.add_argument("--stage", required=True)
    export_parser.add_argument("--checkpoint", required=True)
    export_parser.add_argument("--device", default="cuda:0")
    export_parser.add_argument("--output", type=Path)

    args = parser.parse_args()
    if hasattr(args, "n_envs") and args.n_envs <= 0:
        parser.error("--n-envs must be positive")
    if hasattr(args, "steps") and args.steps <= 0:
        parser.error("--steps must be positive")
    if hasattr(args, "iterations") and args.iterations is not None:
        if args.iterations <= 0:
            parser.error("--iterations must be positive")

    if hasattr(args, "gpu_ids") and args.gpu_ids != "all":
        try:
            args.gpu_ids = [
                int(value.strip())
                for value in str(args.gpu_ids).strip("[]").split(",")
                if value.strip()
            ]
        except ValueError as error:
            parser.error(f"Invalid --gpu-ids: {error}")

    functions = {
        "fetch": lambda _: fetch(),
        "doctor": doctor,
        "gpu-doctor": gpu_doctor,
        "list-stages": list_stages,
        "smoke": smoke,
        "train": train,
        "course-train": course_train,
        "evaluate": evaluate_policy,
        "select-best": select,
        "export": export,
    }
    functions[args.command](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
