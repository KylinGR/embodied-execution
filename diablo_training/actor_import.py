"""Load SB3 or RSL-RL actor weights into a current-format actor bundle."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile


FORMAT = "diablo-rsl-actor-v1"
SOURCE_KEYS = (
    "mlp_extractor.policy_net.0.weight",
    "mlp_extractor.policy_net.0.bias",
    "mlp_extractor.policy_net.2.weight",
    "mlp_extractor.policy_net.2.bias",
    "action_net.weight",
    "action_net.bias",
    "log_std",
)
TARGET_KEYS = (
    "mlp.0.weight",
    "mlp.0.bias",
    "mlp.2.weight",
    "mlp.2.bias",
    "mlp.4.weight",
    "mlp.4.bias",
    "distribution.std_param",
)


def _native_actor_state(checkpoint: dict) -> dict:
    """Extract the actor tensors from an RSL-RL runner checkpoint."""
    current_actor = checkpoint.get("actor_state_dict")
    if isinstance(current_actor, dict):
        missing = set(TARGET_KEYS) - set(current_actor)
        if missing:
            raise ValueError(f"RSL checkpoint is missing actor keys: {sorted(missing)}")
        return {
            key: current_actor[key].detach().clone()
            for key in TARGET_KEYS
        }
    model_state = checkpoint.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise ValueError("RSL checkpoint is missing model_state_dict")
    actor_state = {}
    missing = []
    for target in TARGET_KEYS:
        source = next(
            (
                key
                for key in (f"actor.{target}", target)
                if key in model_state
            ),
            None,
        )
        if source is None:
            missing.append(target)
        else:
            actor_state[target] = model_state[source].detach().clone()
    if missing:
        raise ValueError(f"RSL checkpoint is missing actor keys: {missing}")
    return actor_state


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_sb3_actor(source: Path, output: Path) -> dict:
    """Convert the actor portion of an SB3 PPO zip to RSL-RL 5.x names."""
    import torch

    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with zipfile.ZipFile(source) as archive:
        state = torch.load(
            io.BytesIO(archive.read("policy.pth")),
            map_location="cpu",
            weights_only=True,
        )

    missing = set(SOURCE_KEYS) - set(state)
    if missing:
        raise ValueError(f"SB3 actor is missing keys: {sorted(missing)}")

    expected_shapes = {
        SOURCE_KEYS[0]: (256, 125),
        SOURCE_KEYS[1]: (256,),
        SOURCE_KEYS[2]: (256, 256),
        SOURCE_KEYS[3]: (256,),
        SOURCE_KEYS[4]: (6, 256),
        SOURCE_KEYS[5]: (6,),
        SOURCE_KEYS[6]: (6,),
    }
    for key, shape in expected_shapes.items():
        if tuple(state[key].shape) != shape:
            raise ValueError(
                f"{key} has shape {tuple(state[key].shape)}, expected {shape}"
            )

    actor_state_dict = {
        target: state[source_key].detach().to(dtype=torch.float32).clone()
        for target, source_key in zip(TARGET_KEYS, SOURCE_KEYS, strict=True)
    }
    # RSL-RL 5.4 uses scalar std throughout, while SB3 stores log(std).
    actor_state_dict["distribution.std_param"] = torch.exp(
        actor_state_dict["distribution.std_param"]
    )

    bundle = {
        "format": FORMAT,
        "source": str(source),
        "source_sha256": _sha256(source),
        "source_state_keys": list(SOURCE_KEYS),
        "actor_activation": "tanh",
        "actor_state_dict": actor_state_dict,
        "notes": {
            "critic": "not imported; initialize against current observations",
            "optimizer": "not imported",
            "observation_normalizer": "not imported; current observation semantics differ",
            "log_std": "converted to std_param with exp()",
        },
    }
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output)
    return {
        "output": str(output),
        "sha256": _sha256(output),
        "source_sha256": bundle["source_sha256"],
        "actor_state_keys": sorted(actor_state_dict),
    }


def load_actor_bundle(path: Path) -> dict:
    """Load an actor bundle, RSL checkpoint, or convert an SB3 zip."""
    import torch

    path = path.resolve()
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            state = torch.load(
                io.BytesIO(archive.read("policy.pth")),
                map_location="cpu",
                weights_only=True,
            )
        if set(SOURCE_KEYS) <= set(state):
            cached = path.with_suffix(".actor.pt")
            convert_sb3_actor(path, cached)
            path = cached

    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(bundle, dict):
        raise ValueError(f"{path} does not contain a checkpoint dictionary")
    if bundle.get("format") == FORMAT:
        actor_state_dict = bundle.get("actor_state_dict")
    elif "actor_state_dict" in bundle or "model_state_dict" in bundle:
        actor_state_dict = _native_actor_state(bundle)
        bundle = {
            "format": FORMAT,
            "source": str(path),
            "source_sha256": _sha256(path),
            "actor_activation": "unknown",
            "actor_state_dict": actor_state_dict,
            "notes": {
                "critic": "not imported; initialize against current observations",
                "optimizer": "not imported",
                "observation_normalizer": "not imported",
            },
        }
    else:
        raise ValueError(
            f"{path} is neither a {FORMAT} actor bundle nor an RSL checkpoint"
        )
    if not isinstance(actor_state_dict, dict):
        raise ValueError("Actor bundle is missing actor_state_dict")
    missing = set(TARGET_KEYS) - set(actor_state_dict)
    if missing:
        raise ValueError(f"Actor bundle is missing keys: {sorted(missing)}")
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(convert_sb3_actor(args.source, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
