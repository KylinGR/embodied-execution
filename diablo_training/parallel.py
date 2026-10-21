"""Training-process helpers for parallel MuJoCo physics and CUDA PPO."""

from __future__ import annotations

from functools import partial
import re
import sys
from typing import Literal, Type

import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor


VecEnvKind = Literal["auto", "dummy", "subproc"]


def resolve_device(requested: str) -> str:
    """Resolve ``auto`` without silently falling back from an explicit CUDA request."""
    if requested not in {"auto", "cpu", "cuda"} and not re.fullmatch(
        r"cuda:\d+", requested
    ):
        raise ValueError(
            "device must be auto, cpu, cuda, or a CUDA index such as cuda:3"
        )
    if requested == "cpu":
        return "cpu"

    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is false; "
            "install a CUDA-enabled PyTorch build or use --device cpu"
        )
    index = 0 if requested == "cuda" else int(requested.split(":")[1])
    if index >= torch.cuda.device_count():
        raise ValueError(
            f"Requested {requested}, but only {torch.cuda.device_count()} CUDA device(s) exist"
        )
    return f"cuda:{index}"


def configure_torch(device: str, cpu_threads: int, gpu_fraction: float) -> None:
    """Keep vector workers from oversubscribing CPU and optionally cap allocator memory."""
    if cpu_threads <= 0:
        raise ValueError("cpu_threads must be positive")
    if not 0.0 < gpu_fraction <= 1.0:
        raise ValueError("gpu_fraction must be in (0, 1]")

    import torch

    torch.set_num_threads(cpu_threads)
    if device.startswith("cuda"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # PyTorch only permits setting this before inter-op work starts.
            pass
        index = int(device.split(":")[1]) if ":" in device else 0
        torch.cuda.set_device(index)
        if gpu_fraction < 1.0:
            torch.cuda.set_per_process_memory_fraction(gpu_fraction, index)


def make_vector_env(
    env_type: Type[gym.Env],
    config,
    *,
    n_envs: int,
    kind: VecEnvKind,
    monitor_path: str,
    start_method: str = "forkserver",
) -> tuple[VecMonitor, str]:
    """Construct independent MuJoCo environments.

    ``subproc`` runs each MuJoCo instance in its own process, which is essential
    because moving only the small policy network to CUDA does not parallelise
    MuJoCo's CPU contact solver.
    """
    if n_envs <= 0:
        raise ValueError("n_envs must be positive")
    if kind not in {"auto", "dummy", "subproc"}:
        raise ValueError("kind must be 'auto', 'dummy' or 'subproc'")
    resolved_kind = kind
    if resolved_kind == "auto":
        resolved_kind = (
            "subproc" if n_envs > 1 and sys.platform != "win32" else "dummy"
        )
    if resolved_kind == "subproc" and n_envs == 1:
        raise ValueError("subproc vectorisation requires n_envs >= 2")
    if start_method not in {"fork", "forkserver", "spawn"}:
        raise ValueError("start_method must be fork, forkserver, or spawn")

    constructor = partial(env_type, config)
    if resolved_kind == "dummy":
        venv = DummyVecEnv([constructor for _ in range(n_envs)])
    else:
        venv = SubprocVecEnv(
            [constructor for _ in range(n_envs)],
            start_method=start_method,
        )
    return VecMonitor(venv, filename=monitor_path), resolved_kind
