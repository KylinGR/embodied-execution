# record_headless.py
import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

from diablo_training.contract import (
    load_course,
    load_run_contract,
    reward_defaults_for_stage,
)
from diablo_training.mjlab_tasks import (
    GAIT_JOINTS,
    WHEEL_JOINTS,
    WHEEL_RADIUS,
    _eval_env_cfg,
    require_cuda,
    validate_policy_io,
    verify_reset_state,
)
from diablo_training.model import JOINTS


def has_contact(env, sensor_name: str) -> bool:
    sensor = env.scene[sensor_name]
    return bool(sensor.data.found[0].reshape(-1).any().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage", default="flat_balance")
    parser.add_argument("--seed", type=int, default=2001)
    parser.add_argument("--profile-index", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/headless-rollout"),
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    load_run_contract(checkpoint.parent / "contract.json", args.stage)
    course = load_course()
    stage = course.stage(args.stage)
    profile = stage["profiles"][args.profile_index]
    reward_defaults = reward_defaults_for_stage(course, args.stage)
    zero_tolerance = float(
        reward_defaults.get("zero_speed_tolerance_mps", 0.03)
    )
    straight_yaw_tolerance = float(
        reward_defaults.get("straight_yaw_threshold_rad_s", 0.05)
    )

    args.out.mkdir(parents=True, exist_ok=True)
    video_path = args.out / "rollout.mp4"
    csv_path = args.out / "rollout.csv"
    summary_path = args.out / "summary.json"

    device = require_cuda(args.device)
    env_cfg, agent_cfg = _eval_env_cfg(
        args.stage,
        args.seed,
        args.profile_index,
    )

    # 使用 MJLab 的离屏渲染，不设置 MUJOCO_GL 等环境变量。
    env = ManagerBasedRlEnv(
        cfg=env_cfg,
        device=device,
        render_mode="rgb_array",
    )
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    runner = MjlabOnPolicyRunner(
        wrapped,
        asdict(agent_cfg),
        device=device,
    )
    runner.load(
        str(checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )

    writer = None
    step = -1
    stop_reason = "max_episode_length"

    try:
        wrapped.seed(args.seed)
        obs, _ = wrapped.reset()
        validate_policy_io(obs)
        reset_info = verify_reset_state(env)
        print("reset_check:", reset_info)

        robot = env.scene["robot"]
        wheel_ids, _ = robot.find_joints(WHEEL_JOINTS, preserve_order=True)
        wheel_ids = torch.tensor(wheel_ids, device=device, dtype=torch.long)
        wheel_action_ids = [
            JOINTS.index(name) for name in WHEEL_JOINTS
        ]
        leg_ids, _ = robot.find_joints(GAIT_JOINTS, preserve_order=True)
        leg_ids = torch.tensor(leg_ids, device=device, dtype=torch.long)
        wheel_body_ids = torch.tensor(
            robot.find_bodies(("left_wheel", "right_wheel"))[0],
            device=device,
            dtype=torch.long,
        )
        action_term = env.action_manager.get_term("joints")
        command_term = env.command_manager.get_term("course")
        holding = False
        hold_origin = None
        hold_drift_max = 0.0
        hold_steps = 0
        hold_drift_after_settle_max = 0.0
        wheel_difference_sum = 0.0
        wheel_target_difference_sum = 0.0
        wheel_common_action_sum = 0.0
        wheel_common_action_max = 0.0
        wheel_differential_action_sum = 0.0
        wheel_center_offset_sum = 0.0
        wheel_center_offset_max = 0.0
        hip_difference_sum = 0.0
        hip_difference_max = 0.0
        knee_difference_sum = 0.0
        knee_difference_max = 0.0

        policy = runner.get_inference_policy(device=device)

        writer = imageio.get_writer(
            str(video_path),
            fps=round(1.0 / env.step_dt),
            codec="libx264",
            quality=8,
            macro_block_size=2,
        )
        writer.append_data(env.render())

        fieldnames = [
            "step",
            "time_s",
            "reward",
            "base_height_m",
            "tilt_deg",
            "vx_mps",
            "yaw_rate_rad_s",
            "command_vx",
            "command_yaw",
            "command_yaw_effective",
            "world_heading_deg",
            "target_heading_deg",
            "heading_error_deg",
            "left_wheel_target_rad_s",
            "right_wheel_target_rad_s",
            "left_wheel_target_mps",
            "right_wheel_target_mps",
            "left_wheel_actual_rad_s",
            "right_wheel_actual_rad_s",
            "left_wheel_action",
            "right_wheel_action",
            "wheel_common_action",
            "wheel_differential_action",
            "left_wheel_linear_mps",
            "right_wheel_linear_mps",
            "wheel_speed_difference_mps",
            "wheel_center_offset_signed_m",
            "left_hip_offset_rad",
            "right_hip_offset_rad",
            "left_knee_offset_rad",
            "right_knee_offset_rad",
            "hip_difference_rad",
            "knee_difference_rad",
            "hold_drift_m",
            "hold_drift_after_0.5s_m",
            "body_ground",
            "leg_ground",
            "terminated",
            "truncated",
            "termination_reason",
        ]

        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()

            for step in range(env.max_episode_length):
                with torch.inference_mode():
                    action = policy(obs)
                    validate_policy_io(obs, action)
                    obs, reward, done, _ = wrapped.step(action)

                # _eval_env_cfg 使用 auto_reset=False，因此这里仍是终止时的状态。
                writer.append_data(env.render())

                body_ground = has_contact(env, "body_ground_contact")
                leg_ground = has_contact(env, "leg_ground_contact")

                gravity_z = float(robot.data.projected_gravity_b[0, 2].item())
                tilt_deg = math.degrees(
                    math.acos(max(-1.0, min(1.0, -gravity_z)))
                )

                command = env.command_manager.get_command("course")[0]
                command_vx = float(command[0].item())
                command_yaw = float(command[1].item())
                command_yaw_effective = float(
                    command_term.yaw_target[0].item()
                )
                position_xy = robot.data.root_link_pos_w[0, :2]
                standing = (
                    abs(command_vx) <= zero_tolerance
                    and abs(command_yaw) <= straight_yaw_tolerance
                )
                if standing:
                    if not holding:
                        hold_origin = position_xy.clone()
                        holding = True
                        hold_steps = 0
                    hold_steps += 1
                    hold_drift = float(
                        torch.linalg.vector_norm(position_xy - hold_origin).item()
                    )
                    hold_drift_max = max(hold_drift_max, hold_drift)
                    if hold_steps * env.step_dt >= 0.5:
                        hold_drift_after_settle = hold_drift
                        hold_drift_after_settle_max = max(
                            hold_drift_after_settle_max,
                            hold_drift_after_settle,
                        )
                    else:
                        hold_drift_after_settle = 0.0
                else:
                    holding = False
                    hold_steps = 0
                    hold_drift = 0.0
                    hold_drift_after_settle = 0.0

                world_heading_deg = math.degrees(
                    float(robot.data.heading_w[0].item())
                )
                heading_rad = float(robot.data.heading_w[0].item())
                target_heading_rad = float(
                    command_term.target_heading[0].item()
                )
                target_heading_deg = math.degrees(target_heading_rad)
                wheel_positions = robot.data.body_link_pos_w[
                    0, wheel_body_ids
                ]
                wheel_center_offset_signed = float(
                    (
                        (wheel_positions[0, 0] - wheel_positions[1, 0])
                        * math.cos(heading_rad)
                        + (wheel_positions[0, 1] - wheel_positions[1, 1])
                        * math.sin(heading_rad)
                    ).item()
                )
                leg_offset = (
                    robot.data.joint_pos[0, leg_ids]
                    - robot.data.default_joint_pos[0, leg_ids]
                )
                left_hip_offset = float(leg_offset[0].item())
                left_knee_offset = float(leg_offset[1].item())
                right_hip_offset = float(leg_offset[2].item())
                right_knee_offset = float(leg_offset[3].item())
                hip_difference = left_hip_offset - right_hip_offset
                knee_difference = left_knee_offset - right_knee_offset
                wheel_center_offset_sum += wheel_center_offset_signed
                wheel_center_offset_max = max(
                    wheel_center_offset_max,
                    abs(wheel_center_offset_signed),
                )
                hip_difference_sum += hip_difference
                hip_difference_max = max(
                    hip_difference_max,
                    abs(hip_difference),
                )
                knee_difference_sum += knee_difference
                knee_difference_max = max(
                    knee_difference_max,
                    abs(knee_difference),
                )
                heading_error_deg = math.degrees(
                    math.atan2(
                        math.sin(
                            heading_rad - target_heading_rad
                        ),
                        math.cos(
                            heading_rad - target_heading_rad
                        ),
                    )
                )
                wheel_target_rad_s = action_term.wheel_velocity_target[0]
                wheel_actual_rad_s = robot.data.joint_vel[0, wheel_ids]
                raw_wheel_action = action_term.raw_action[
                    0, wheel_action_ids
                ]
                wheel_common_action = float(
                    raw_wheel_action.mean().item()
                )
                wheel_differential_action = float(
                    (
                        0.5
                        * (raw_wheel_action[1] - raw_wheel_action[0])
                    ).item()
                )
                wheel_common_action_sum += wheel_common_action
                wheel_common_action_max = max(
                    wheel_common_action_max,
                    abs(wheel_common_action),
                )
                wheel_differential_action_sum += wheel_differential_action
                wheel_target_linear_mps = -wheel_target_rad_s * WHEEL_RADIUS
                wheel_actual_linear_mps = -wheel_actual_rad_s * WHEEL_RADIUS
                wheel_speed_difference = float(
                    torch.abs(
                        wheel_actual_linear_mps[0]
                        - wheel_actual_linear_mps[1]
                    ).item()
                )
                wheel_target_difference = float(
                    torch.abs(
                        wheel_target_rad_s[0] - wheel_target_rad_s[1]
                    ).item()
                )
                wheel_difference_sum += wheel_speed_difference
                wheel_target_difference_sum += wheel_target_difference

                is_terminated = bool(env.reset_terminated[0].item())
                is_truncated = bool(env.reset_time_outs[0].item())
                bad_orientation = bool(
                    env.termination_manager.get_term("bad_orientation")[0].item()
                )

                reason = ""
                if is_terminated:
                    if body_ground:
                        reason = "body_ground"
                    elif leg_ground:
                        reason = "leg_ground"
                    elif bad_orientation:
                        reason = "bad_orientation"
                    else:
                        reason = "other_termination"
                elif is_truncated:
                    reason = "time_out"

                csv_writer.writerow(
                    {
                        "step": step,
                        "time_s": step * env.step_dt,
                        "reward": float(reward[0].item()),
                        "base_height_m": float(
                            robot.data.root_link_pos_w[0, 2].item()
                        ),
                        "tilt_deg": tilt_deg,
                        "vx_mps": float(
                            robot.data.root_link_lin_vel_b[0, 0].item()
                        ),
                        "yaw_rate_rad_s": float(
                            robot.data.root_link_ang_vel_b[0, 2].item()
                        ),
                        "command_vx": command_vx,
                        "command_yaw": command_yaw,
                        "command_yaw_effective": command_yaw_effective,
                        "world_heading_deg": world_heading_deg,
                        "target_heading_deg": target_heading_deg,
                        "heading_error_deg": heading_error_deg,
                        "left_wheel_target_rad_s": float(
                            wheel_target_rad_s[0].item()
                        ),
                        "right_wheel_target_rad_s": float(
                            wheel_target_rad_s[1].item()
                        ),
                        "left_wheel_target_mps": float(
                            wheel_target_linear_mps[0].item()
                        ),
                        "right_wheel_target_mps": float(
                            wheel_target_linear_mps[1].item()
                        ),
                        "left_wheel_actual_rad_s": float(
                            wheel_actual_rad_s[0].item()
                        ),
                        "right_wheel_actual_rad_s": float(
                            wheel_actual_rad_s[1].item()
                        ),
                        "left_wheel_action": float(
                            raw_wheel_action[0].item()
                        ),
                        "right_wheel_action": float(
                            raw_wheel_action[1].item()
                        ),
                        "wheel_common_action": wheel_common_action,
                        "wheel_differential_action": (
                            wheel_differential_action
                        ),
                        "left_wheel_linear_mps": float(
                            wheel_actual_linear_mps[0].item()
                        ),
                        "right_wheel_linear_mps": float(
                            wheel_actual_linear_mps[1].item()
                        ),
                        "wheel_speed_difference_mps": wheel_speed_difference,
                        "wheel_center_offset_signed_m": (
                            wheel_center_offset_signed
                        ),
                        "left_hip_offset_rad": left_hip_offset,
                        "right_hip_offset_rad": right_hip_offset,
                        "left_knee_offset_rad": left_knee_offset,
                        "right_knee_offset_rad": right_knee_offset,
                        "hip_difference_rad": hip_difference,
                        "knee_difference_rad": knee_difference,
                        "hold_drift_m": hold_drift,
                        "hold_drift_after_0.5s_m": hold_drift_after_settle,
                        "body_ground": int(body_ground),
                        "leg_ground": int(leg_ground),
                        "terminated": int(is_terminated),
                        "truncated": int(is_truncated),
                        "termination_reason": reason,
                    }
                )

                if is_terminated or is_truncated:
                    stop_reason = reason or "done"
                    print(
                        f"stopped step={step}, "
                        f"time={step * env.step_dt:.3f}s, "
                        f"reason={stop_reason}, "
                        f"tilt={tilt_deg:.2f}deg, "
                        f"body_ground={body_ground}, "
                        f"leg_ground={leg_ground}"
                    )
                    break

        summary = {
            "checkpoint": str(checkpoint),
            "stage": args.stage,
            "profile": profile.get("name", str(args.profile_index)),
            "seed": args.seed,
            "profile_index": args.profile_index,
            "steps": step + 1,
            "duration_s": (step + 1) * env.step_dt,
            "stop_reason": stop_reason,
            "reset_check": reset_info,
            "final_heading_deg": world_heading_deg,
            "final_heading_error_deg": heading_error_deg,
            "final_position_m": [
                float(robot.data.root_link_pos_w[0, 0].item()),
                float(robot.data.root_link_pos_w[0, 1].item()),
            ],
            "hold_drift_max_m": hold_drift_max,
            "hold_drift_after_0.5s_max_m": hold_drift_after_settle_max,
            "mean_wheel_speed_difference_mps": wheel_difference_sum
            / (step + 1),
            "mean_wheel_target_difference_rad_s": wheel_target_difference_sum
            / (step + 1),
            "mean_wheel_common_action": wheel_common_action_sum
            / (step + 1),
            "max_abs_wheel_common_action": wheel_common_action_max,
            "mean_wheel_differential_action": wheel_differential_action_sum
            / (step + 1),
            "mean_wheel_center_offset_signed_m": wheel_center_offset_sum
            / (step + 1),
            "max_abs_wheel_center_offset_m": wheel_center_offset_max,
            "mean_hip_difference_rad": hip_difference_sum / (step + 1),
            "max_abs_hip_difference_rad": hip_difference_max,
            "mean_knee_difference_rad": knee_difference_sum / (step + 1),
            "max_abs_knee_difference_rad": knee_difference_max,
            "video": str(video_path),
            "csv": str(csv_path),
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    finally:
        if writer is not None:
            writer.close()
        env.close()


if __name__ == "__main__":
    main()
