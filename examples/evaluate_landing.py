"""Legacy ever-stable diagnostic, NOT terminal-stability qualification.

Use qualify_landing.py for current acceptance. Historical result files retain
their original definitions and must not be interpreted as sustained stability.
"""
import argparse
import json
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
from diablo_training.cli import load_policy
from diablo_training.contract import environment_class


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", type=Path)
    source.add_argument("--zero-action", action="store_true")
    source.add_argument("--feedback", action="store_true")
    source.add_argument("--cushion", action="store_true")
    source.add_argument("--hold", action="store_true")
    parser.add_argument("--heights", type=float, nargs="+", default=[.022, .035, .048])
    parser.add_argument("--jitter", type=float, default=.002)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--no-extra-damping", action="store_true")
    parser.add_argument("--seconds", type=float, default=4.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.zero_action or args.feedback or args.cushion or args.hold:
        from diablo_training.env import Config
        policy, config = None, Config(episode_seconds=args.seconds)
        resolved = {"environment_class": "landing_hold:HoldLandingEnv" if args.hold else "landing_cushion:CushionLandingEnv" if args.cushion else "landing_residual:ResidualLandingEnv" if args.feedback else "landing_task:LandingEnv"}
    else:
        policy, config, resolved = load_policy(args.model)
        if args.seconds != config.episode_seconds:
            raise ValueError("Model evaluation must keep its saved episode duration")
    if resolved["environment_class"] not in ("landing_task:LandingEnv", "landing_residual:ResidualLandingEnv", "landing_cushion:CushionLandingEnv", "landing_hold:HoldLandingEnv"):
        raise ValueError("Expected a landing-task checkpoint")
    args.output.mkdir(parents=True, exist_ok=False)
    env = environment_class(resolved["environment_class"])(config)
    if args.no_extra_damping:
        if not args.cushion:
            raise ValueError("Damping ablation requires --cushion")
        env.extra_damping[:] = 0.
    reports = []
    try:
        with imageio.get_writer(args.output / "landing.mp4", fps=25) as video:
            with (args.output / "trace.jsonl").open("w") as trace:
                for nominal_height in args.heights:
                    for seed in range(args.seed_start, args.seed_start+10):
                        height = float(nominal_height + np.random.default_rng(seed).uniform(-args.jitter, args.jitter))
                        obs, _ = env.reset(seed=seed, options={"drop_height": height})
                        success = False
                        max_stable = 0.
                        landing_window_force = 0.
                        for step in range(round(config.episode_seconds / .01)):
                            action = np.zeros(6) if policy is None else policy.predict(obs, deterministic=True)[0]
                            obs, reward, done, truncated, info = env.step(action)
                            success |= info["landing_success"]
                            max_stable = max(max_stable, info["stable_seconds"])
                            if (info["landing_time_s"] is not None and
                                    info["sim_seconds"] <= info["landing_time_s"] + .15 and
                                    not info["body_ground_contact"]):
                                landing_window_force = max(landing_window_force, info["sampled_contact_force_n"])
                            trace.write(json.dumps({"seed": seed, **info}) + "\n")
                            if seed == args.seed_start and step % 4 == 0:
                                frame = Image.fromarray(env.render())
                                draw = ImageDraw.Draw(frame)
                                draw.rectangle((0, 0, 640, 45), fill="black")
                                draw.text((8, 8), f"FREE DROP {height:.3f} m | t={info['sim_seconds']:.2f}s | stable={info['stable_seconds']:.2f}s", fill="white")
                                video.append_data(np.asarray(frame))
                            if done or truncated:
                                break
                        reports.append({"height_m": height, "seed": seed, "success": bool(success and not done),
                                        "fell": bool(done), "duration_s": info["sim_seconds"],
                                        "max_stable_seconds": max_stable,
                                        "final_stable_seconds": info["stable_seconds"],
                                        "landing_time_s": info["landing_time_s"],
                                        "landing_window_peak_sampled_force_n": landing_window_force,
                                        "peak_sampled_force_n": info["peak_sampled_contact_force_n"]})
        rate = sum(r["success"] for r in reports) / len(reports)
        result = {"controller": ("long-horizon sensor feedback" if args.hold else "sensor feedback with extra leg damping" if args.cushion else "sensor feedback with zero PPO residual" if args.feedback else "zero action with default PD") if policy is None else str(args.model),
                  "feedback_gains": env.gains.tolist() if hasattr(env, "gains") else None,
                  "extra_damping": env.extra_damping.tolist() if hasattr(env, "extra_damping") else None,
                  "episodes": reports, "success_rate": rate,
                  "legacy_gate_passed": rate >= .9 and all(not r["fell"] for r in reports),
                  "metric_version": "legacy-v1-ever-stable-not-qualification",
                  "gate": f"at least 90% one-second stable landings; no falls during {config.episode_seconds:g}-second evaluation",
                  "scope": "free-drop surrogate only; no active takeoff, platform departure or obstacles",
                  "force_note": "100 Hz samples; landing window is first 0.15 s after detected wheel contact, excluding body-floor contact; not physics-rate impact peaks",
                  "force_sampling_hz": 100}
        (args.output / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
