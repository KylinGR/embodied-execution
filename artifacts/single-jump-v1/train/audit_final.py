"""Independent repeatable audit; never edits frozen control/qualification code.

Usage: PYTHONPATH=examples python runs/single-jump-dev-20261027/audit_final.py
  --training runs/... --qualification runs/... [--output path.json]
"""
import argparse
import gzip
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from diablo_training.cli import load_policy, parameter_digest
from diablo_training.contract import environment_class, policy_contract
from diablo_training.control import decode_action
from diablo_training.env import Config
from jump_metrics import summarize


SCALES = np.array([.03, .03, .02, .03, .03, .02])


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compare(actual, expected, location):
    if isinstance(actual, dict):
        for key, value in actual.items():
            require(key in expected, f"{location}: missing {key}")
            compare(value, expected[key], f"{location}.{key}")
    elif isinstance(actual, list):
        require(len(actual) == len(expected), f"{location}: different lengths")
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, f"{location}[{i}]")
    elif isinstance(actual, (float, np.floating)):
        require(np.isclose(actual, expected, rtol=1e-9, atol=1e-9), f"{location}: {actual} != {expected}")
    else:
        require(actual == expected, f"{location}: {actual} != {expected}")


def parameter_changes(initial, final):
    first = dict(initial.named_parameters())
    groups = {name: [] for name in ("actor", "critic", "shared_or_other")}
    for name, tensor in final.named_parameters():
        require(name in first, f"Unexpected parameter {name}")
        difference = tensor.detach().cpu().numpy()-first[name].detach().cpu().numpy()
        group = "actor" if (name.startswith("action_net.") or name.startswith("mlp_extractor.policy_net.") or name == "log_std") else (
            "critic" if name.startswith("value_net.") or name.startswith("mlp_extractor.value_net.") else "shared_or_other")
        groups[group].append({"name": name, "changed": bool(np.any(difference != 0)),
                              "maximum_absolute_change": float(np.max(np.abs(difference))),
                              "l2_change": float(np.linalg.norm(difference.astype(np.float64)))})
    for key in ("actor", "critic"):
        require(any(item["changed"] for item in groups[key]), f"No {key} parameter update")
    return groups


def audit_episode(rows, reported, env):
    seed = int(reported["seed"])
    computed = summarize(rows)
    compare(computed, reported, f"seed={seed}")
    require(computed["passed"] and not reported["fell"] and reported["truncated"], f"Seed {seed} failed qualification")
    env.reset(seed=seed)
    require(np.array_equal(env.data.qvel, np.zeros_like(env.data.qvel)), "Reset velocity is not exactly zero")
    reset_external = max(float(np.max(np.abs(env.data.qfrc_applied))), float(np.max(np.abs(env.data.xfrc_applied))))
    require(reset_external == 0., "Reset external forces are nonzero")
    compare(list(env.episode_pulse), reported["pulse"], f"seed={seed}.pulse")
    previous_qpos = env.data.qpos.copy()
    previous_qvel = env.data.qvel.copy()
    maximum_formula_error = 0.
    maximum_torque_error = 0.
    maximum_residual = 0.
    maximum_action_effect = 0.
    maximum_torque_effect = 0.
    peak_torque = 0.
    nonzero_residual_samples = nonzero_action_samples = nonzero_torque_samples = 0
    body_positive_work = 0.
    actual_rows = 0
    for row in rows:
        if abs(row["time"]) < 1e-9:
            # Frozen qualifier's initial row lacks full qpos/qvel/actions.
            # Reconstruct the documented zero-speed grounded reset by this seed.
            if "qpos" in row:
                require(np.allclose(previous_qpos, row["qpos"], atol=1e-12, rtol=0), "Initial qpos mismatch")
            if "qvel" in row:
                require(np.allclose(previous_qvel, row["qvel"], atol=1e-12, rtol=0), "Initial qvel mismatch")
            continue
        residual = np.asarray(row["residual_action"], dtype=float)
        feedback = np.asarray(row["feedback_action"], dtype=float)
        applied = np.asarray(row["applied_action"], dtype=float)
        torque = np.asarray(row["motor_torque_nm"], dtype=float)
        require(all(np.isfinite(x).all() for x in (residual, feedback, applied, torque)), "Nonfinite action or torque")
        require(np.max(np.abs(residual)) <= 1.+1e-12, "Residual outside clipping interval")
        expected = np.clip(feedback+residual*SCALES, -1., 1.)
        error = float(np.max(np.abs(expected-applied)))
        maximum_formula_error = max(maximum_formula_error, error)
        require(error < 1e-10, f"Seed {seed}, t={row['time']}: applied action mismatch")
        q = previous_qpos[env.qi]
        dq = previous_qvel[env.vi]
        _, expected_torque = decode_action(applied, q, dq, "torque")
        torque_error = float(np.max(np.abs(expected_torque-torque)))
        maximum_torque_error = max(maximum_torque_error, torque_error)
        require(torque_error < 1e-9, f"Seed {seed}, t={row['time']}: torque/speed-envelope mismatch {torque_error}")
        require(np.max(np.abs(torque)) <= 17.+1e-10, "17 Nm exceeded")
        _, zero_residual_torque = decode_action(np.clip(feedback, -1, 1), q, dq, "torque")
        residual_norm = float(np.linalg.norm(residual))
        action_effect = float(np.max(np.abs(applied-feedback)))
        torque_effect = float(np.max(np.abs(torque-zero_residual_torque)))
        maximum_residual = max(maximum_residual, residual_norm)
        maximum_action_effect = max(maximum_action_effect, action_effect)
        maximum_torque_effect = max(maximum_torque_effect, torque_effect)
        nonzero_residual_samples += residual_norm > 1e-12
        nonzero_action_samples += action_effect > 1e-12
        nonzero_torque_samples += torque_effect > 1e-12
        peak_torque = max(peak_torque, float(np.max(np.abs(torque))))
        # Mechanical power is an additional diagnostic, not a qualification gate.
        body_positive_work += max(0., float(row["joint_power_w"]))*0.002
        previous_qpos = np.asarray(row["qpos"], dtype=float)
        previous_qvel = np.asarray(row["qvel"], dtype=float)
        actual_rows += 1
    require(actual_rows == 10000, f"Seed {seed}: incomplete physical trace")
    return {"seed": seed, "metrics_match": True, "passed": computed["passed"],
            "control_formula_samples": actual_rows, "maximum_action_formula_error": maximum_formula_error,
            "maximum_torque_formula_error_nm": maximum_torque_error, "peak_torque_nm": peak_torque,
            "maximum_residual_norm": maximum_residual, "maximum_action_effect": maximum_action_effect,
            "maximum_torque_effect_nm": maximum_torque_effect,
            "nonzero_residual_samples": int(nonzero_residual_samples),
            "nonzero_action_effect_samples": int(nonzero_action_samples),
            "nonzero_torque_effect_samples": int(nonzero_torque_samples),
            "positive_joint_work_j": body_positive_work, "reset_external_force_maximum": reset_external,
            "first_substep_initial_state_reconstructed_from_seed": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    saved = json.loads((args.training/"resolved.json").read_text())
    training_result = json.loads((args.training/"result.json").read_text())
    qualification = json.loads((args.qualification/"result.json").read_text())
    require(saved["resume"] is None, "Initial-policy reconstruction supports fresh training only")
    require(qualification["source_unchanged_during_run"], "Qualification source changed during run")
    require(qualification["qualified"], "Qualification did not pass")
    require(Path(qualification["model"]).resolve() == (args.training/"policy.zip").resolve(), "Qualification refers to another model path")
    for path, digest in qualification["source_hashes"].items():
        require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, f"Changed qualification source {path}")
    final, config, loaded = load_policy(args.training/"policy.zip")
    require(loaded["contract"]["sha256"] == qualification["contract"]["sha256"], "Policy/qualification contract mismatch")
    require(final.num_timesteps == training_result["actual_total_timesteps"], "Saved policy timestep mismatch")
    after = parameter_digest(final.policy)
    require(after == training_result["parameter_sha256_after"], "Final parameter digest mismatch")
    cls = environment_class(saved["environment_class"])
    vector = VecMonitor(DummyVecEnv([lambda: cls(Config(**saved["environment"])) for _ in range(saved["n_envs"])]))
    env = cls(config)
    try:
        initial = PPO("MlpPolicy", vector, **saved["ppo"], device=saved["device"], seed=saved["seed"], verbose=0)
        before = parameter_digest(initial.policy)
        require(before == saved["parameter_sha256_before"], "Reconstructed initial parameter digest mismatch")
        changes = parameter_changes(initial.policy, final.policy)
        require(before != after, "Policy did not change")
        require(policy_contract(env)["sha256"] == saved["contract"]["sha256"], "Current environment contract mismatch")
        reports = qualification["episodes"]
        require(len(reports) == 30 and len({r["seed"] for r in reports}) == 30, "Need 30 unique episodes")
        expected = {int(report["seed"]): report for report in reports}
        audited = []
        seen = set()
        rows = []
        current_seed = None
        with gzip.open(args.qualification/"trace.jsonl.gz", "rt") as stream:
            for line in stream:
                row = json.loads(line)
                seed = int(row["seed"])
                if current_seed is not None and seed != current_seed:
                    require(current_seed not in seen, "Noncontiguous or repeated seed group")
                    audited.append(audit_episode(rows, expected[current_seed], env))
                    seen.add(current_seed)
                    print(json.dumps({"audited_seed": current_seed}), flush=True)
                    rows = []
                require(seed in expected, f"Unexpected seed {seed}")
                current_seed = seed
                rows.append(row)
        if current_seed is not None:
            require(current_seed not in seen, "Repeated final seed")
            audited.append(audit_episode(rows, expected[current_seed], env))
            seen.add(current_seed)
        require(seen == set(expected), "Missing qualification seeds")
        require(any(r["nonzero_torque_effect_samples"] > 0 for r in audited), "No PPO residual effect on motor torque")
        # Absence of applied external forces is a code/reset audit. Frozen trace
        # does not log xfrc_applied/qfrc_applied, so do not claim direct measurement.
        source_paths = {inspect.getsourcefile(c) for c in cls.__mro__ if inspect.isclass(c) and hasattr(c, "step")}
        source_paths.discard(None)
        force_references = [str(p) for p in source_paths if "qfrc_applied" in Path(p).read_text() or "xfrc_applied" in Path(p).read_text()]
        result = {"audit_passed": True, "before_parameter_sha256": before,
                  "after_parameter_sha256": after, "parameter_changes": changes,
                  "contract_sha256": saved["contract"]["sha256"],
                  "actual_training_timesteps": final.num_timesteps,
                  "policy_file_sha256": file_sha256(args.training/"policy.zip"),
                  "qualification_source_hashes": qualification["source_hashes"],
                  "trace_file_sha256": file_sha256(args.qualification/"trace.jsonl.gz"),
                  "episodes": audited,
                  "total_checked_physics_steps": sum(r["control_formula_samples"] for r in audited),
                  "peak_torque_nm": max(r["peak_torque_nm"] for r in audited),
                  "external_force_boundary": {"reset_forces_zero": True,
                      "source_files_referencing_external_force_arrays": force_references,
                      "full_trace_external_forces_recorded": False,
                      "scope": "No external force setters in reviewed task source; reset arrays checked zero. Full-run absence is source-derived, not measured in frozen trace."},
                  "claim_boundary": "Searched active-jump pulse and sensor feedback plus trained bounded PPO residual; no PPO superiority or discovery claim."}
        require(not force_references, "External force references require manual review")
        destination = args.output or args.training/"independent_audit.json"
        destination.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        print(json.dumps({"audit_passed": True, "output": str(destination), "checked_steps": result["total_checked_physics_steps"]}), flush=True)
    finally:
        vector.close()
        env.close()


if __name__ == "__main__":
    main()
