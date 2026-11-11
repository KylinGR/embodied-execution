"""Fail closed when a checkpoint belongs to another physical/interface contract."""

import hashlib
import importlib
import inspect
import json
from pathlib import Path

import mujoco

from .env import DiabloEnv
from .model import metadata


def environment_class(name="diablo_training.env:DiabloEnv"):
    cls = import_class(name)
    if not issubclass(cls, DiabloEnv):
        raise TypeError("Environment class must inherit DiabloEnv")
    return cls


def import_class(name):
    module, separator, attribute = name.partition(":")
    if not separator:
        raise ValueError("Class path must be module:Class")
    cls = getattr(importlib.import_module(module), attribute)
    if not isinstance(cls, type):
        raise TypeError("Import target must be a class")
    return cls


def policy_contract(env):
    implementation = {}
    for filename in ("model.py", "control.py", "env.py"):
        implementation[filename] = hashlib.sha256(
            Path(__file__).with_name(filename).read_bytes()
        ).hexdigest()
    implementation["task_module"] = hashlib.sha256(
        Path(inspect.getsourcefile(type(env))).read_bytes()
    ).hexdigest()
    if type(env).__module__.startswith("diablo_training.xingtian"):
        for filename in ("xingtian.py", "xingtian_env.py", "xingtian_terrain.py"):
            implementation[filename] = hashlib.sha256(
                Path(__file__).with_name(filename).read_bytes()
            ).hexdigest()
    content = {
        "interface_version": 2,
        "implementation_sha256": implementation,
        "mujoco_version": mujoco.__version__,
        "model": metadata(env.model),
        "action_mode": env.config.action_mode,
        "observation": {
            "shape": [125],
            "frames": 5,
            "frame_width": 25,
            "order": "oldest_to_current",
        },
    }
    content["sha256"] = hashlib.sha256(
        json.dumps(content, sort_keys=True).encode()
    ).hexdigest()
    return content


def require_compatible(saved, env):
    current = policy_contract(env)
    if saved.get("contract", {}).get("sha256") != current["sha256"]:
        raise ValueError(
            "Checkpoint model/controller/observation contract is incompatible (including v0.1). Use the matching code and pinned assets, or start fresh training; do not force-load by tensor shape."
        )
    return current


def require_checkpoint(model, saved):
    if getattr(model, "diablo_contract_sha256", None) != saved["contract"]["sha256"]:
        raise ValueError(
            "Checkpoint and resolved.json do not belong to the same contract"
        )
