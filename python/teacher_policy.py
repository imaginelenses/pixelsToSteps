#!/usr/bin/env python3
"""Render the teacher-policy cart-pole rollout and save binary capture data.

This script reuses the same plant, LQR design, and rollout physics from
python/cartpole_lqr_gain.py. It captures exactly one rendered frame per control
action, stops capture when the cart-pole reaches the upright goal, and writes
the result at the control rate so video time matches the simulated control loop
time.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - optional fallback for older setups.
    import gym

from cartpole_lqr_gain import (
    PhysicalCartPoleParams,
    choose_cost_matrices,
    delayed_command_steps_per_second,
    linearize_physical_cartpole,
    render_hold_loop,
    rollout_episode,
    rollout_nominal_linear_episode,
    sample_initial_angle_deg,
    solve_discrete_lqr,
    verify_sign_conventions,
    write_generated_teacher_lqr_json,
)
from cartpole_frames_to_video import assemble_png_frames_to_video


GOAL_ANGLE_RAD = math.radians(1.0)
GOAL_ANGLE_RATE_RAD_S = math.radians(10.0)
GOAL_HOLD_TIME_S = 0.25
GOAL_POSITION_STEPS = 10.0
GOAL_VELOCITY_STEPS_S = 20.0
DEFAULT_PREVIEW_RATE_HZ = 25.0


def default_video_output_path() -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return Path(__file__).resolve().parent / "captures" / f"cartpole_binary_{timestamp}.mkv"


def default_frame_directory(output_path: Path) -> Path:
    return output_path.with_suffix("").with_name(f"{output_path.stem}_frames")


def capture_metadata_path(frames_dir: Path) -> Path:
    return frames_dir / "capture_metadata.json"


def capture_trace_path(frames_dir: Path) -> Path:
    return frames_dir / "capture_trace.csv"


def capture_controller_path(frames_dir: Path) -> Path:
    return frames_dir / "teacher_lqr_controller.json"


def map_state_to_gym(state: np.ndarray, params: PhysicalCartPoleParams) -> np.ndarray:
    return np.array(
        [
            state[0] * params.meters_per_step,
            state[1] * params.meters_per_step,
            state[2],
            state[3],
        ],
        dtype=np.float64,
    )


def configure_cartpole_renderer(env, params: PhysicalCartPoleParams) -> None:
    env.reset(seed=0)
    env.unwrapped.length = params.pole_length_m / 2.0
    env.unwrapped.x_threshold = params.track_half_length_m * 1.05
    env.unwrapped.theta_threshold_radians = math.radians(35.0)


def extract_binary_pole_frame(
    rgb_frame: np.ndarray,
    frame_width_px: int,
    frame_height_px: int,
    pole_rgb: tuple[int, int, int] = (202, 152, 101),
    color_tolerance: int = 90,
) -> np.ndarray:
    rgb_int = rgb_frame.astype(np.int16)
    pole_color = np.array(pole_rgb, dtype=np.int16)
    color_distance_sq = np.sum((rgb_int - pole_color) ** 2, axis=2)
    pole_mask = np.where(color_distance_sq <= (color_tolerance**2), 255, 0).astype(np.uint8)
    return cv2.resize(pole_mask, (frame_width_px, frame_height_px), interpolation=cv2.INTER_NEAREST)


def pace_realtime_loop(next_deadline_s: float) -> float:
    sleep_duration_s = next_deadline_s - time.monotonic()
    if sleep_duration_s > 0.0:
        time.sleep(sleep_duration_s)
    return next_deadline_s


def effective_render_every(control_rate_hz: float, requested_render_every: int | None) -> int:
    if requested_render_every is not None:
        return requested_render_every
    return max(1, int(math.ceil(control_rate_hz / DEFAULT_PREVIEW_RATE_HZ)))


def is_goal_state(state: np.ndarray, params: PhysicalCartPoleParams) -> bool:
    return (
        abs(float(state[0])) <= GOAL_POSITION_STEPS
        and abs(float(state[1])) <= GOAL_VELOCITY_STEPS_S
        and abs(float(state[2])) <= GOAL_ANGLE_RAD
        and abs(float(state[3])) <= GOAL_ANGLE_RATE_RAD_S
    )


def capture_goal_index(
    states_after_actions: list[np.ndarray],
    params: PhysicalCartPoleParams,
) -> int | None:
    required_consecutive_steps = max(1, int(math.ceil(GOAL_HOLD_TIME_S / params.sample_time_s)))
    consecutive_goal_steps = 0

    for state_index, state in enumerate(states_after_actions):
        if is_goal_state(state, params):
            consecutive_goal_steps += 1
            if consecutive_goal_steps >= required_consecutive_steps:
                return state_index
        else:
            consecutive_goal_steps = 0
    return None


def build_capture_trace_rows(
    captured_states_with_initial: list[np.ndarray],
    gain: np.ndarray,
    params: PhysicalCartPoleParams,
    rollout_model: str,
) -> list[dict[str, float | int | str]]:
    max_delay_samples = int(math.ceil(max(params.command_delay_s, 0.0) / params.sample_time_s))
    command_history = [0.0] * (max_delay_samples + 2)
    trace_rows: list[dict[str, float | int | str]] = []

    for state_index in range(len(captured_states_with_initial) - 1):
        state_before = captured_states_with_initial[state_index]
        state_after = captured_states_with_initial[state_index + 1]
        raw_control_steps_s = float(-(gain @ state_before)[0])
        if rollout_model == "nominal_linear":
            target_control_steps_s = raw_control_steps_s
            applied_control_steps_s = raw_control_steps_s
        else:
            target_control_steps_s = float(
                np.clip(raw_control_steps_s, -params.max_step_rate_steps_s, params.max_step_rate_steps_s)
            )
            command_history.append(target_control_steps_s)
            applied_control_steps_s = delayed_command_steps_per_second(
                command_history,
                params.command_delay_s,
                params.sample_time_s,
            )
        trace_rows.append(
            {
                "frame_index": state_index,
                "frame_filename": f"frame_{state_index:06d}.png",
                "control_step": state_index + 1,
                "simulated_time_s": (state_index + 1) * params.sample_time_s,
                "cart_position_steps": float(state_after[0]),
                "cart_velocity_steps_s": float(state_after[1]),
                "pole_angle_rad": float(state_after[2]),
                "pole_angle_deg": math.degrees(float(state_after[2])),
                "pole_angle_rate_rad_s": float(state_after[3]),
                "pole_angle_rate_deg_s": math.degrees(float(state_after[3])),
                "raw_control_steps_s": raw_control_steps_s,
                "target_control_steps_s": target_control_steps_s,
                "applied_control_steps_s": float(applied_control_steps_s),
            }
        )

    return trace_rows


def write_capture_trace_csv(
    frames_dir: Path,
    trace_rows: list[dict[str, float | int | str]],
) -> Path:
    trace_path = capture_trace_path(frames_dir)
    fieldnames = [
        "frame_index",
        "frame_filename",
        "control_step",
        "simulated_time_s",
        "cart_position_steps",
        "cart_velocity_steps_s",
        "pole_angle_rad",
        "pole_angle_deg",
        "pole_angle_rate_rad_s",
        "pole_angle_rate_deg_s",
        "raw_control_steps_s",
        "target_control_steps_s",
        "applied_control_steps_s",
    ]
    with trace_path.open("w", newline="", encoding="ascii") as trace_file:
        writer = csv.DictWriter(trace_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trace_rows)
    return trace_path


def write_capture_metadata(
    frames_dir: Path,
    params: PhysicalCartPoleParams,
    frame_rate_hz: float,
    frame_width_px: int,
    frame_height_px: int,
    initial_angle_deg: float,
    steps_requested: int,
    frames_captured: int,
    capture_end_reason: str,
    goal_reached_step: int | None,
    capture_wall_clock_duration_s: float,
    teacher_controller_json: str | None,
    rollout_model: str,
) -> Path:
    metadata = {
        "capture_trace_csv": capture_trace_path(frames_dir).name,
        "capture_end_reason": capture_end_reason,
        "capture_wall_clock_duration_s": capture_wall_clock_duration_s,
        "frame_rate_hz": frame_rate_hz,
        "sample_time_s": 1.0 / frame_rate_hz,
        "frame_width_px": frame_width_px,
        "frame_height_px": frame_height_px,
        "goal_position_steps": GOAL_POSITION_STEPS,
        "goal_velocity_steps_s": GOAL_VELOCITY_STEPS_S,
        "goal_angle_deg": math.degrees(GOAL_ANGLE_RAD),
        "goal_angle_rate_deg_s": math.degrees(GOAL_ANGLE_RATE_RAD_S),
        "goal_definition": "abs(x) <= goal_position_steps and abs(x_dot) <= goal_velocity_steps_s and abs(theta) <= goal_angle_deg and abs(theta_dot) <= goal_angle_rate_deg_s for goal_hold_time_s",
        "goal_hold_time_s": GOAL_HOLD_TIME_S,
        "initial_angle_deg": initial_angle_deg,
        "max_step_rate_steps_s": params.max_step_rate_steps_s,
        "rollout_model": rollout_model,
        "simulated_capture_duration_s": frames_captured * (1.0 / frame_rate_hz),
        "steps_requested": steps_requested,
        "frames_captured": frames_captured,
    }
    metadata["goal_reached_step"] = goal_reached_step
    if teacher_controller_json is not None:
        metadata["teacher_controller_json"] = teacher_controller_json
    metadata_path = capture_metadata_path(frames_dir)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="ascii")
    return metadata_path


def capture_binary_png_sequence(
    states_after_actions: list[np.ndarray],
    params: PhysicalCartPoleParams,
    frame_width_px: int,
    frame_height_px: int,
    frames_dir: Path,
    render_preview: bool,
    render_every: int,
) -> tuple[int, Path, float]:
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    capture_env = gym.make("CartPole-v1", render_mode="rgb_array")
    preview_env = gym.make("CartPole-v1", render_mode="human") if render_preview else None
    try:
        configure_cartpole_renderer(capture_env, params)
        if preview_env is not None:
            configure_cartpole_renderer(preview_env, params)

        capture_start_time_s = time.monotonic()
        next_deadline_s = time.monotonic() + params.sample_time_s
        for index, state in enumerate(states_after_actions):
            mapped_state = map_state_to_gym(state, params)
            capture_env.unwrapped.state = mapped_state
            rendered = capture_env.render()
            if rendered is None:
                raise RuntimeError("Gym did not return a frame for rgb_array rendering.")
            binary_frame = extract_binary_pole_frame(
                np.asarray(rendered, dtype=np.uint8),
                frame_width_px,
                frame_height_px,
            )
            frame_path = frames_dir / f"frame_{index:06d}.png"
            if not cv2.imwrite(str(frame_path), binary_frame):
                raise RuntimeError(f"Failed to write frame PNG: {frame_path}")

            if preview_env is not None and index % max(render_every, 1) == 0:
                preview_env.unwrapped.state = mapped_state
                preview_env.render()
            if preview_env is not None:
                next_deadline_s = pace_realtime_loop(next_deadline_s)
                next_deadline_s += params.sample_time_s

        capture_wall_clock_duration_s = time.monotonic() - capture_start_time_s
        if preview_env is not None and states_after_actions:
            simulated_capture_duration_s = len(states_after_actions) * params.sample_time_s
            print(
                "Capture finished: "
                f"{len(states_after_actions)} frames, "
                f"{simulated_capture_duration_s:.3f} s simulated, "
                f"{capture_wall_clock_duration_s:.3f} s wall-clock before hold. "
                f"Preview cadence: every {render_every} control steps. "
                "Holding final viewer state until Ctrl+C.",
                flush=True,
            )
            render_hold_loop(preview_env, states_after_actions[-1], params, -1.0)
        return len(states_after_actions), frames_dir, capture_wall_clock_duration_s
    finally:
        capture_env.close()
        if preview_env is not None:
            preview_env.close()


def write_binary_video(
    frames_dir: Path,
    frame_rate_hz: float,
    output_path: Path,
) -> Path:
    return assemble_png_frames_to_video(
        frames_dir=frames_dir,
        output_path=output_path,
        frame_rate_hz=frame_rate_hz,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1500, help="Simulation steps to run.")
    parser.add_argument(
        "--theta0-deg",
        type=float,
        default=None,
        help="Fixed initial pole angle in degrees from upright. If omitted, a random offset is used.",
    )
    parser.add_argument(
        "--theta0-range-deg",
        type=float,
        default=8.0,
        help="Random initial angle range sampled uniformly from [-range, +range] degrees.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for the random initial angle.")
    parser.add_argument(
        "--control-rate-hz",
        type=float,
        default=250.0,
        help="Discrete control rate in Hz. One output video frame is written per control step.",
    )
    parser.add_argument(
        "--horizon-steps",
        type=int,
        default=None,
        help="Alias for --steps. If set, overrides --steps.",
    )
    parser.add_argument(
        "--max-step-rate-steps-s",
        type=float,
        default=12000.0,
        help="Maximum control command magnitude in steps/s used for design and saturation.",
    )
    parser.add_argument(
        "--actuator-time-constant-s",
        type=float,
        default=0.03,
        help="First-order cart velocity time constant used in the LQR design plant.",
    )
    parser.add_argument(
        "--max-cart-accel-steps-s2",
        type=float,
        default=150000.0,
        help="Validation-only cart acceleration magnitude limit in steps/s^2.",
    )
    parser.add_argument(
        "--r-weight",
        type=float,
        default=1e-6,
        help="Direct scalar control penalty R for the discrete LQR design.",
    )
    parser.add_argument(
        "--command-delay-s",
        type=float,
        default=0.0,
        help="Pure command-to-motion delay used only in simulation validation.",
    )
    parser.add_argument(
        "--frame-height-px",
        type=int,
        default=125,
        help="Output frame height in pixels.",
    )
    parser.add_argument(
        "--frame-width-px",
        type=int,
        default=160,
        help="Output frame width in pixels.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Show Gym's human viewer while frames are being captured.",
    )
    parser.add_argument(
        "--render-every",
        type=int,
        default=None,
        help="Update the live preview every N captured control steps. Defaults to an automatic cadence near 25 Hz.",
    )
    parser.add_argument(
        "--video-output",
        type=Path,
        default=default_video_output_path(),
        help="Lossless output video path. Defaults to python/captures/cartpole_binary_TIMESTAMP.mkv.",
    )
    parser.add_argument(
        "--frame-png-dir",
        type=Path,
        default=None,
        help="Directory to store the exact binary PNG frames used to assemble the video. Defaults to <video_stem>_frames.",
    )
    parser.add_argument(
        "--skip-video-assembly",
        action="store_true",
        help="Capture the binary PNG frames only and skip the final video assembly step.",
    )
    parser.add_argument(
        "--nominal-linear-rollout",
        action="store_true",
        help="Use the unconstrained nominal discrete closed-loop rollout x[k+1] = A x[k] + B u[k] with u = -Kx instead of the constrained validation rollout.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.horizon_steps is not None:
        args.steps = args.horizon_steps
    if args.steps <= 0:
        raise SystemExit("Simulation steps must be positive.")
    if args.control_rate_hz <= 0.0:
        raise SystemExit("Control rate must be positive.")
    if args.max_step_rate_steps_s <= 0.0:
        raise SystemExit("Maximum step rate must be positive.")
    if args.actuator_time_constant_s <= 0.0:
        raise SystemExit("Actuator time constant must be positive.")
    if args.max_cart_accel_steps_s2 <= 0.0:
        raise SystemExit("Maximum cart acceleration must be positive.")
    if args.r_weight <= 0.0:
        raise SystemExit("R weight must be positive.")
    if args.command_delay_s < 0.0:
        raise SystemExit("Command delay must be non-negative.")
    if args.theta0_range_deg < 0.0:
        raise SystemExit("Initial angle range must be non-negative.")
    if args.frame_height_px <= 0 or args.frame_width_px <= 0:
        raise SystemExit("Frame dimensions must be positive.")
    if args.render_every is not None and args.render_every <= 0:
        raise SystemExit("Render cadence must be positive.")


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_args(args)

    params = PhysicalCartPoleParams(
        sample_time_s=1.0 / args.control_rate_hz,
        max_step_rate_steps_s=args.max_step_rate_steps_s,
        actuator_time_constant_s=args.actuator_time_constant_s,
        max_cart_accel_steps_s2=args.max_cart_accel_steps_s2,
        control_penalty_r=args.r_weight,
        command_delay_s=args.command_delay_s,
    )

    rng = np.random.default_rng(args.seed)
    initial_angle_deg = sample_initial_angle_deg(rng, args.theta0_deg, args.theta0_range_deg)
    a_discrete, b_discrete, derived = linearize_physical_cartpole(params)
    q_matrix, r_matrix = choose_cost_matrices(params)
    gain, _riccati_solution, closed_loop_poles = solve_discrete_lqr(a_discrete, b_discrete, q_matrix, r_matrix)
    sign_report = verify_sign_conventions(gain)

    if args.nominal_linear_rollout:
        rollout_model = "nominal_linear"
        survived_steps, final_state, last_target_command_steps_s, states, rollout_stats = rollout_nominal_linear_episode(
            gain,
            args.steps,
            initial_angle_deg,
            a_discrete,
            b_discrete,
            params,
        )
    else:
        rollout_model = "constrained_physical"
        survived_steps, final_state, last_target_command_steps_s, states, rollout_stats = rollout_episode(
            gain,
            args.steps,
            initial_angle_deg,
            params,
            derived,
        )
    states_after_actions = states[1:]
    goal_reached_state_index = capture_goal_index(states_after_actions, params)
    if goal_reached_state_index is not None:
        capture_end_reason = "goal"
        captured_states = states_after_actions[: goal_reached_state_index + 1]
    elif survived_steps < args.steps:
        capture_end_reason = "failure"
        captured_states = states_after_actions
    else:
        capture_end_reason = "horizon"
        captured_states = states_after_actions
    capture_stop_state = captured_states[-1] if captured_states else states[0]
    captured_states_with_initial = states[: len(captured_states) + 1]

    frame_png_dir = args.frame_png_dir or default_frame_directory(args.video_output)
    render_every = effective_render_every(args.control_rate_hz, args.render_every)
    frames_captured, written_frames_dir, capture_wall_clock_duration_s = capture_binary_png_sequence(
        captured_states,
        params,
        args.frame_width_px,
        args.frame_height_px,
        frame_png_dir,
        args.render,
        render_every,
    )
    capture_trace_path_value = write_capture_trace_csv(
        written_frames_dir,
        build_capture_trace_rows(captured_states_with_initial, gain, params, rollout_model),
    )
    controller_json_path = write_generated_teacher_lqr_json(
        params,
        q_matrix,
        r_matrix,
        gain,
        closed_loop_poles,
        sign_report,
        a_discrete=a_discrete,
        b_discrete=b_discrete,
        output_path=capture_controller_path(written_frames_dir),
    )
    metadata_path = write_capture_metadata(
        written_frames_dir,
        params,
        args.control_rate_hz,
        args.frame_width_px,
        args.frame_height_px,
        initial_angle_deg,
        args.steps,
        frames_captured,
        capture_end_reason,
        None if goal_reached_state_index is None else goal_reached_state_index + 1,
        capture_wall_clock_duration_s,
        controller_json_path.name,
        rollout_model,
    )
    output_path = None
    if not args.skip_video_assembly:
        output_path = write_binary_video(
            written_frames_dir,
            args.control_rate_hz,
            args.video_output,
        )

    print(f"Frame dir   : {written_frames_dir}")
    print(f"Metadata    : {metadata_path}")
    print(f"Trace csv   : {capture_trace_path_value}")
    print(f"Teacher K   : {controller_json_path}")
    if output_path is not None:
        print(f"Video path  : {output_path}")
    else:
        print("Video path  : skipped (--skip-video-assembly)")
    print(f"Frames      : {frames_captured}")
    print(f"Frame size  : {args.frame_height_px}x{args.frame_width_px}")
    print(f"Frame rate  : {args.control_rate_hz:.3f} fps")
    print(f"Sample time : {params.sample_time_s:.6f} s")
    print(f"Rollout mdl : {rollout_model}")
    print(f"Sim time    : {frames_captured * params.sample_time_s:.6f} s")
    print(f"Wall time   : {capture_wall_clock_duration_s:.6f} s before viewer hold")
    if args.render:
        print(f"Preview     : every {render_every} control steps (~{args.control_rate_hz / render_every:.3f} Hz)")
    print(f"Init angle  : {initial_angle_deg:.3f} deg")
    print(f"Capture end : {capture_end_reason}")
    if goal_reached_state_index is not None:
        print(f"Goal step   : {goal_reached_state_index + 1}")
    print(f"Stop state  : {capture_stop_state}")
    print(f"Rollout     : survived {survived_steps}/{args.steps} control steps")
    print(f"Rollout end : {final_state}")
    print(f"Rollout u   : {last_target_command_steps_s:.5f} steps/s")
    print(
        "Peaks       : "
        f"raw_u={rollout_stats['peak_requested_command_steps_s']:.2f} steps/s, "
        f"target_u={rollout_stats['peak_target_command_steps_s']:.2f} steps/s, "
        f"|v|={rollout_stats['peak_velocity_steps_s']:.2f} steps/s, "
        f"|a|={rollout_stats['peak_cart_acceleration_m_s2']:.4f} m/s^2"
    )
    print(
        "Saturation  : "
        f"command clips={int(rollout_stats['command_clip_count'])}, "
        f"accel clips={int(rollout_stats['accel_clip_count'])}"
    )


if __name__ == "__main__":
    main()