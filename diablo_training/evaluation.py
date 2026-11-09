"""Vectorized MJLab checkpoint evaluation helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from .contract import (
    gate_failures,
    load_course,
    profile_task_type,
    resolved_profile_gate,
    stage_profiles,
)


def run_vector_seed(
    stage_name: str,
    profiles: list[dict[str, Any]],
    checkpoint: Path,
    seed: int,
    device: str,
    eval_envs: int,
) -> list[dict[str, Any]]:
    """Run all profiles in one MJLab vector environment for one seed."""
    from .mjlab_tasks import _load_runner, validate_policy_io, verify_reset_state

    if not profiles:
        return []
    course = load_course()
    stage = course.stage(stage_name)
    all_profiles = list(stage["profiles"])
    global_indices = [all_profiles.index(profile) for profile in profiles]
    replicates = max(1, math.ceil(eval_envs / len(profiles)))
    sequence = tuple(
        global_indices[index]
        for _ in range(replicates)
        for index in range(len(profiles))
    )
    env, wrapped, runner = _load_runner(
        stage_name,
        seed,
        None,
        checkpoint,
        device,
        profile_sequence=sequence,
        num_envs=len(sequence),
    )
    try:
        wrapped.seed(seed)
        obs, _ = wrapped.reset()
        validate_policy_io(obs)
        verify_reset_state(env)
        policy = runner.get_inference_policy(device=device)
        reward_term = env.reward_manager.cfg["course"].func
        done_envs = torch.zeros(len(sequence), dtype=torch.bool, device=device)
        episode_returns = torch.zeros(len(sequence), device=device)
        captured: dict[int, dict[str, float | bool]] = {}
        for _ in range(env.max_episode_length + 1):
            with torch.no_grad():
                action = policy(obs)
                validate_policy_io(obs, action)
            obs, reward, done, _ = wrapped.step(action)
            if not torch.isfinite(reward).all():
                raise FloatingPointError("Vector evaluation produced non-finite reward")
            episode_returns += reward
            done_now = done.bool()
            pending_reset = getattr(env, "_manual_reset_pending", None)
            if pending_reset is not None:
                # MJLab rejects the next step for every pending env. Treat its
                # state as authoritative in case the wrapper masks a reset flag.
                done_now = done_now | pending_reset
            newly_done = done_now & ~done_envs
            if bool(newly_done.any().item()):
                capture_ids = newly_done.nonzero().flatten()
                metrics = reward_term.metrics()
                for env_index in capture_ids.tolist():
                    captured[env_index] = {
                        name: float(values[env_index].item())
                        for name, values in metrics.items()
                    }
                    captured[env_index]["episode_return"] = float(
                        episode_returns[env_index].item()
                    )
                    captured[env_index]["terminated"] = bool(
                        env.reset_terminated[env_index].item()
                    )
                    captured[env_index]["truncated"] = bool(
                        env.reset_time_outs[env_index].item()
                    )
                done_envs[capture_ids] = True
            reset_ids = done_now.nonzero().flatten()
            if len(reset_ids):
                episode_returns[reset_ids] = 0.0
                reset_obs, _ = env.reset(env_ids=reset_ids)
                for key in obs.keys():
                    values = reset_obs[key]
                    obs[key][reset_ids] = (
                        values[reset_ids]
                        if values.shape[0] == obs[key].shape[0]
                        else values
                    )
            if bool(done_envs.all().item()):
                break
        if not bool(done_envs.all().item()):
            metrics = reward_term.metrics()
            for env_index in (~done_envs).nonzero().flatten().tolist():
                captured[env_index] = {
                    name: float(values[env_index].item())
                    for name, values in metrics.items()
                }
                captured[env_index]["episode_return"] = float(
                    episode_returns[env_index].item()
                )
                captured[env_index]["terminated"] = bool(
                    env.reset_terminated[env_index].item()
                )
                captured[env_index]["truncated"] = False
        runs = []
        for env_index in range(len(sequence)):
            profile = profiles[env_index % len(profiles)]
            gate = resolved_profile_gate(course, stage_name, profile)
            metrics = captured[env_index]
            runs.append(
                {
                    "seed": seed,
                    "profile": profile["name"],
                    "task_type": profile_task_type(
                        stage_name, stage, profile
                    ),
                    "resolved_gate": gate,
                    "metrics": metrics,
                    "failures": gate_failures(metrics, gate),
                }
            )
        return runs
    finally:
        env.close()


def run_exact_seed(
    stage_name: str,
    profiles: list[dict[str, Any]],
    checkpoint: Path,
    seed: int,
    device: str,
    include_training_profiles: bool = False,
) -> list[dict[str, Any]]:
    """Run paired profiles sequentially while reusing one env and one policy."""
    from .mjlab_tasks import (
        _load_runner,
        validate_policy_io,
        verify_reset_state,
    )

    if not profiles:
        return []
    course = load_course()
    stage = course.stage(stage_name)
    all_profiles = stage_profiles(
        stage,
        include_training_profiles=include_training_profiles,
        profile_set="all",
    )
    env, wrapped, runner = _load_runner(
        stage_name,
        seed,
        None,
        checkpoint,
        device,
        include_training_profiles=include_training_profiles,
        profile_sequence=(0,),
        num_envs=1,
    )
    try:
        policy = runner.get_inference_policy(device=device)
        command_term = env.command_manager.get_term("course")
        reward_term = env.reward_manager.cfg["course"].func
        runs = []
        for profile in profiles:
            profile_index = all_profiles.index(profile)
            command_term._profile_sequence[0] = profile_index
            wrapped.seed(seed)
            obs, _ = wrapped.reset()
            validate_policy_io(obs)
            verify_reset_state(env)
            episode_return = 0.0
            for _ in range(env.max_episode_length):
                with torch.no_grad():
                    action = policy(obs)
                    validate_policy_io(obs, action)
                obs, reward, done, _ = wrapped.step(action)
                if not torch.isfinite(reward).all():
                    raise FloatingPointError(
                        "Evaluation produced a non-finite reward"
                    )
                episode_return += float(reward[0].item())
                if bool(done[0].item()):
                    break
            metrics = {
                name: float(values[0].item())
                for name, values in reward_term.metrics().items()
            }
            metrics["episode_return"] = episode_return
            metrics["terminated"] = bool(env.reset_terminated[0].item())
            metrics["truncated"] = bool(env.reset_time_outs[0].item())
            gate = (
                course.rehearsal_gate
                if include_training_profiles
                else resolved_profile_gate(course, stage_name, profile)
            )
            if gate is None:
                raise ValueError("Training profiles require a rehearsal gate")
            runs.append(
                {
                    "seed": seed,
                    "profile": profile["name"],
                    "task_type": profile_task_type(
                        stage_name, stage, profile
                    ),
                    "resolved_gate": gate,
                    "metrics": metrics,
                    "failures": gate_failures(metrics, gate),
                }
            )
        return runs
    finally:
        env.close()
