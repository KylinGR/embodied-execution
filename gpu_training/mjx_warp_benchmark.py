"""Reference GPU physics gate for the same DIABLO MuJoCo model.

This is deliberately a benchmark rather than a second training implementation.
It verifies that the compiled model, free joint, two equality constraints and
wheel contacts can be batched on a GPU before porting the PPO environment.

Install the optional runtime first:

    pip install 'mujoco-mjx[warp]'

Then run on a Linux CUDA host:

    python -m gpu_training.mjx_warp_benchmark \
        --num-envs 1024 --steps 100
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from diablo_training.model import DEFAULT_Q, JOINTS, make_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--impl", choices=("jax", "warp"), default="warp")
    parser.add_argument(
        "--naconmax",
        type=int,
        default=16,
        help="contacts per world for the MJX-Warp data buffer",
    )
    parser.add_argument("--njmax", type=int, default=32)
    args = parser.parse_args()
    if args.num_envs <= 0 or args.steps <= 0 or args.warmup < 0:
        parser.error("num-envs/steps must be positive and warmup non-negative")
    return args


def initial_qpos(cpu_model: mujoco.MjModel) -> jax.Array:
    qpos = np.array(cpu_model.qpos0, copy=True)
    qpos[2] = 0.4
    qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for i, name in enumerate(JOINTS):
        address = cpu_model.jnt_qposadr[cpu_model.joint(name).id]
        qpos[address] = DEFAULT_Q[i]
    for side_index, side in enumerate(("left", "right")):
        hip, knee, wheel = DEFAULT_Q[3 * side_index : 3 * side_index + 3]
        for name, value in (
            ("hip_motor", hip),
            ("knee_motor", knee),
            ("wheel_motor", wheel),
            ("return_passive", knee),
            ("branch_passive", -knee),
        ):
            address = cpu_model.jnt_qposadr[cpu_model.joint(f"{side}_{name}").id]
            qpos[address] = value
    return jnp.asarray(qpos, dtype=jnp.float32)


def main() -> int:
    args = parse_args()
    cpu_model = make_model()
    qpos0 = initial_qpos(cpu_model)

    model_kwargs = {"impl": args.impl}
    data_kwargs = {"impl": args.impl}
    if args.impl == "warp":
        data_kwargs.update(naconmax=args.naconmax, njmax=args.njmax)
    gpu_model = mjx.put_model(cpu_model, **model_kwargs)

    def make_world(key):
        del key
        data = mjx.make_data(cpu_model, **data_kwargs)
        return data.replace(
            qpos=qpos0,
            qvel=jnp.zeros_like(data.qvel),
            ctrl=jnp.zeros_like(data.ctrl),
        )

    data = jax.vmap(make_world)(
        jax.random.split(jax.random.PRNGKey(0), args.num_envs)
    )

    @jax.jit
    def batched_step(batch):
        return jax.vmap(lambda world: mjx.step(gpu_model, world))(batch)

    for _ in range(args.warmup):
        data = batched_step(data)
    jax.block_until_ready(data.qpos)

    began = time.monotonic()
    for _ in range(args.steps):
        data = batched_step(data)
    jax.block_until_ready(data.qpos)
    wall = time.monotonic() - began

    finite = bool(jnp.isfinite(data.qpos).all())
    transitions = args.steps * args.num_envs
    print(
        {
            "device": str(jax.devices()[0]),
            "impl": args.impl,
            "num_envs": args.num_envs,
            "physics_steps_per_env": args.steps,
            "world_steps_per_second": transitions / max(wall, 1e-12),
            "wall_seconds": wall,
            "finite": finite,
            "note": "CPU MuJoCo remains the independent reference simulator.",
        }
    )
    return 0 if finite else 2


if __name__ == "__main__":
    raise SystemExit(main())
