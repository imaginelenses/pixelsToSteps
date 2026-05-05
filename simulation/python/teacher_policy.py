#!/usr/bin/env python3
"""Render nominal Gym CartPole rollouts and save binary capture data.

This script uses Gym's built-in CartPole dynamics internally and exports the
teacher and student interface in native cartpole units

    x = [cart_position_m, cart_velocity_m_s, pole_angle_rad, pole_angle_rate_rad_s]
    u = signed_force_n

When an observer JSON is supplied, the plant still evolves in its hidden true
state, but the control law uses only the rendered binary images through the
learned observer recurrence

    x_hat[k+1] = A_L x_hat[k] + B_L u[k] + L z[k+1] + d.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from cartpole_frames_to_video import assemble_png_frames_to_video
from nominal_cartpole import (
    NominalCartPoleParams,
    choose_cost_matrices,
    create_cartpole_env,
    linearize_nominal_cartpole,
    perturb_cartpole_params,
    render_hold_loop,
    sample_initial_angle_deg,
    scale_cartpole_params,
    set_cartpole_state,
    solve_discrete_lqr,
    step_cartpole_env,
    verify_sign_conventions,
    write_teacher_controller_json,
)


GOAL_POSITION_M = 0.05
GOAL_VELOCITY_M_S = 0.05
GOAL_ANGLE_RAD = math.radians(1.0)
GOAL_ANGLE_RATE_RAD_S = math.radians(10.0)
GOAL_HOLD_TIME_S = 0.25
DEFAULT_PREVIEW_RATE_HZ = 25.0
DEFAULT_CAPTURE_ROOT = Path(__file__).resolve().parents[1] / "captures"


def default_video_output_path() -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return DEFAULT_CAPTURE_ROOT / f"cartpole_nominal_{timestamp}.mkv"


def default_frame_directory(output_path: Path) -> Path:
    return output_path.with_suffix("").with_name(f"{output_path.stem}_frames")


def capture_metadata_path(frames_dir: Path) -> Path:
    return frames_dir / "capture_metadata.json"


def capture_trace_path(frames_dir: Path) -> Path:
    return frames_dir / "capture_trace.csv"


def capture_controller_path(frames_dir: Path) -> Path:
    return frames_dir / "teacher_lqr_controller.json"


def extract_binary_pole_frame(
    rgb_frame: np.ndarray,
    frame_width_px: int | None,
    frame_height_px: int | None,
    background_white_threshold: int = 245,
    dilation_kernel_size: int = 5,
    top_padding_px: int = 2,
    track_black_fraction: float = 0.9,
) -> np.ndarray:
    foreground_mask = np.where(np.any(rgb_frame < background_white_threshold, axis=2), 255, 0).astype(np.uint8)

    black_row_counts = np.count_nonzero(np.all(rgb_frame == 0, axis=2), axis=1)
    track_row_candidates = np.flatnonzero(black_row_counts >= track_black_fraction * rgb_frame.shape[1])
    crop_bottom = int(track_row_candidates[0]) if track_row_candidates.size else rgb_frame.shape[0]

    foreground_rows = np.flatnonzero(np.any(foreground_mask > 0, axis=1))
    crop_top = max(0, int(foreground_rows[0]) - top_padding_px) if foreground_rows.size else 0
    if crop_bottom <= crop_top:
        crop_top = 0
        crop_bottom = rgb_frame.shape[0]

    foreground_mask = foreground_mask[crop_top:crop_bottom, :]
    if frame_width_px is not None and frame_height_px is not None:
        foreground_mask = cv2.resize(foreground_mask, (frame_width_px, frame_height_px), interpolation=cv2.INTER_NEAREST)
    if dilation_kernel_size > 1:
        dilation_kernel = np.ones((dilation_kernel_size, dilation_kernel_size), dtype=np.uint8)
        foreground_mask = cv2.dilate(foreground_mask, dilation_kernel, iterations=1)
    return foreground_mask


def pad_binary_frame_to_shape(
    binary_frame: np.ndarray,
    target_width_px: int,
    target_height_px: int,
) -> np.ndarray:
    frame_height_px, frame_width_px = binary_frame.shape
    if frame_height_px > target_height_px or frame_width_px > target_width_px:
        raise ValueError(
            "Binary frame is larger than the target observer geometry. "
            f"Got {frame_height_px}x{frame_width_px}, target {target_height_px}x{target_width_px}."
        )
    pad_top = max(0, target_height_px - frame_height_px)
    pad_bottom = 0
    pad_left = max(0, (target_width_px - frame_width_px) // 2)
    pad_right = max(0, target_width_px - frame_width_px - pad_left)
    return cv2.copyMakeBorder(
        binary_frame,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )


def write_binary_frame(frame_path: Path, binary_frame: np.ndarray) -> None:
    if not cv2.imwrite(str(frame_path), binary_frame):
        raise RuntimeError(f"Failed to write frame PNG: {frame_path}")


# CartPole-v1 rendering constants (screen_width=600, x_threshold=2.4 m).
_CARTPOLE_RENDER_SCALE_PX_PER_M: float = 125.0  # 600 / (2 * 2.4)
_CARTPOLE_RENDER_CENTER_COL_PX: float = 300.0   # 600 / 2
# How many rows from the bottom of the binary frame to search for the cart body.
_CART_CENTROID_BOTTOM_ROWS: int = 20


def estimate_cart_position_from_binary_frame(binary_frame: np.ndarray) -> float | None:
    """Return cart position in metres estimated from the geometric centroid of white
    pixels in the bottom rows of a binary frame.

    Returns None if no foreground pixels are found (safe to skip blending).
    The frame width must equal the original CartPole render width (600 px) – i.e.
    no side-cropping or rescaling is applied before this call.
    """
    bottom = binary_frame[-_CART_CENTROID_BOTTOM_ROWS:, :]
    cols = np.flatnonzero(np.any(bottom > 128, axis=0))
    if cols.size == 0:
        return None
    mean_col = float(np.mean(cols))
    return (mean_col - _CARTPOLE_RENDER_CENTER_COL_PX) / _CARTPOLE_RENDER_SCALE_PX_PER_M


def pace_realtime_loop(next_deadline_s: float) -> float:
    sleep_duration_s = next_deadline_s - time.monotonic()
    if sleep_duration_s > 0.0:
        time.sleep(sleep_duration_s)
    return next_deadline_s


def effective_render_every(control_rate_hz: float, requested_render_every: int | None) -> int:
    if requested_render_every is not None:
        return requested_render_every
    return max(1, int(math.ceil(control_rate_hz / DEFAULT_PREVIEW_RATE_HZ)))


def load_observer_policy(observer_json_path: Path) -> dict[str, object]:
    observer_path = observer_json_path.expanduser().resolve()
    payload = json.loads(observer_path.read_text(encoding="utf-8"))
    unit_system = str(payload.get("unit_system", "cartpole"))
    if unit_system != "cartpole":
        raise ValueError(
            "Observer JSON does not target nominal cartpole units. "
            f"Expected 'cartpole', got {unit_system!r}."
        )

    a_l = np.asarray(payload["A_L"], dtype=np.float64)
    b_l = np.asarray(payload["B_L"], dtype=np.float64)
    l_gain = np.asarray(payload["L"], dtype=np.float64)
    d_bias = np.asarray(payload["d"], dtype=np.float64).reshape(-1)
    teacher_gain = np.asarray(payload["teacher_gain_K"], dtype=np.float64).reshape(1, -1)
    frame_height_px = int(payload["frame_height_px"])
    frame_width_px = int(payload["frame_width_px"])
    pixel_count = frame_height_px * frame_width_px
    theta_pixel_coefficients_payload = payload.get("theta_pixel_coefficients")
    theta_pixel_coefficients = None
    if theta_pixel_coefficients_payload is not None:
        theta_pixel_coefficients = np.asarray(theta_pixel_coefficients_payload, dtype=np.float64).reshape(-1)
    theta_pixel_bias = payload.get("theta_pixel_bias")
    theta_pixel_blend_weight = payload.get("theta_pixel_blend_weight", 0.0)

    if a_l.shape != (4, 4):
        raise ValueError(f"Observer A_L must have shape (4, 4), got {a_l.shape!r}")
    if b_l.shape == (4,):
        b_l = b_l.reshape(4, 1)
    if b_l.shape != (4, 1):
        raise ValueError(f"Observer B_L must have shape (4, 1), got {b_l.shape!r}")
    if d_bias.shape != (4,):
        raise ValueError(f"Observer d must have shape (4,), got {d_bias.shape!r}")
    if teacher_gain.shape != (1, 4):
        raise ValueError(f"Teacher gain K must have shape (1, 4), got {teacher_gain.shape!r}")
    if l_gain.shape != (4, pixel_count):
        raise ValueError(
            "Observer L does not match the configured frame geometry. "
            f"Expected (4, {pixel_count}), got {l_gain.shape!r}."
        )
    if theta_pixel_coefficients is not None and theta_pixel_coefficients.shape != (pixel_count,):
        raise ValueError(
            "theta_pixel_coefficients does not match the configured frame geometry. "
            f"Expected ({pixel_count},), got {theta_pixel_coefficients.shape!r}."
        )
    if theta_pixel_bias is not None:
        theta_pixel_bias = float(theta_pixel_bias)
    theta_pixel_blend_weight = float(theta_pixel_blend_weight)
    theta_dot_blend_weight = float(payload.get("theta_dot_blend_weight", 0.0))
    if not 0.0 <= theta_pixel_blend_weight <= 1.0:
        raise ValueError(
            "theta_pixel_blend_weight must lie in [0, 1]. "
            f"Got {theta_pixel_blend_weight!r}."
        )
    if not 0.0 <= theta_dot_blend_weight <= 1.0:
        raise ValueError(
            "theta_dot_blend_weight must lie in [0, 1]. "
            f"Got {theta_dot_blend_weight!r}."
        )
    cart_pixel_blend_weight = float(payload.get("cart_pixel_blend_weight", 0.0))
    if not 0.0 <= cart_pixel_blend_weight <= 1.0:
        raise ValueError(
            "cart_pixel_blend_weight must lie in [0, 1]. "
            f"Got {cart_pixel_blend_weight!r}."
        )
    nonzero_measurement_coefficients = int(np.count_nonzero(np.abs(l_gain) > 1e-12))
    nonzero_theta_pixel_coefficients = 0
    if theta_pixel_coefficients is not None:
        nonzero_theta_pixel_coefficients = int(np.count_nonzero(np.abs(theta_pixel_coefficients) > 1e-12))

    return {
        "path": observer_path,
        "unit_system": unit_system,
        "A_L": a_l,
        "B_L": b_l,
        "L": l_gain,
        "d": d_bias,
        "teacher_gain": teacher_gain,
        "frame_height_px": frame_height_px,
        "frame_width_px": frame_width_px,
        "observer_target": str(payload.get("observer_target", "pixels-to-cartpole-state")),
        "nonzero_measurement_coefficients": nonzero_measurement_coefficients,
        "theta_pixel_coefficients": theta_pixel_coefficients,
        "theta_pixel_bias": theta_pixel_bias,
        "theta_pixel_blend_weight": theta_pixel_blend_weight,
        "theta_dot_blend_weight": theta_dot_blend_weight,
        "cart_pixel_blend_weight": cart_pixel_blend_weight,
        "nonzero_theta_pixel_coefficients": nonzero_theta_pixel_coefficients,
    }


def is_goal_state(state_meters: np.ndarray) -> bool:
    return (
        abs(float(state_meters[0])) <= GOAL_POSITION_M
        and abs(float(state_meters[1])) <= GOAL_VELOCITY_M_S
        and abs(float(state_meters[2])) <= GOAL_ANGLE_RAD
        and abs(float(state_meters[3])) <= GOAL_ANGLE_RATE_RAD_S
    )


def write_capture_trace_csv(
    frames_dir: Path,
    trace_rows: list[dict[str, float | int | str]],
) -> Path:
    if not trace_rows:
        raise ValueError("No trace rows were generated for this capture.")
    trace_path = capture_trace_path(frames_dir)
    fieldnames = list(trace_rows[0].keys())
    with trace_path.open("w", newline="", encoding="ascii") as trace_file:
        writer = csv.DictWriter(trace_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trace_rows)
    return trace_path


def write_capture_metadata(
    frames_dir: Path,
    nominal_params: NominalCartPoleParams,
    true_params: NominalCartPoleParams,
    logged_initial_state: np.ndarray,
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
    control_policy: str,
    true_model_scale_factors: dict[str, float],
    true_model_perturbation_frac: float,
    true_model_seed: int | None,
    process_noise_std_n: float,
    process_noise_seed: int | None,
    observation_noise_std: np.ndarray,
    observation_noise_seed: int | None,
    collection_id: str | None = None,
    demo_name: str | None = None,
    student_observer_json: str | None = None,
    observer_target: str | None = None,
) -> Path:
    metadata = {
        "capture_trace_csv": capture_trace_path(frames_dir).name,
        "capture_end_reason": capture_end_reason,
        "control_policy": control_policy,
        "capture_wall_clock_duration_s": capture_wall_clock_duration_s,
        "frame_rate_hz": float(1.0 / true_params.sample_time_s),
        "sample_time_s": float(true_params.sample_time_s),
        "frame_width_px": frame_width_px,
        "frame_height_px": frame_height_px,
        "unit_system": "cartpole",
        "position_units": "meters",
        "control_units": "newtons",
        "goal_position_m": GOAL_POSITION_M,
        "goal_velocity_m_s": GOAL_VELOCITY_M_S,
        "goal_angle_deg": math.degrees(GOAL_ANGLE_RAD),
        "goal_angle_rate_deg_s": math.degrees(GOAL_ANGLE_RATE_RAD_S),
        "goal_definition": "abs(x_m) <= goal_position_m and abs(x_dot_m_s) <= goal_velocity_m_s and abs(theta) <= goal_angle_deg and abs(theta_dot) <= goal_angle_rate_deg_s for goal_hold_time_s",
        "goal_hold_time_s": GOAL_HOLD_TIME_S,
        "initial_angle_deg": initial_angle_deg,
        "max_force_n": true_params.max_force_n,
        "x_threshold_m": true_params.x_threshold_m,
        "theta_threshold_deg": math.degrees(true_params.theta_threshold_rad),
        "rollout_model": rollout_model,
        "simulated_capture_duration_s": frames_captured * true_params.sample_time_s,
        "steps_requested": steps_requested,
        "frames_captured": frames_captured,
        "goal_reached_step": goal_reached_step,
        "nominal_model_params": asdict(nominal_params),
        "true_model_params": asdict(true_params),
        "true_model_scale_factors": true_model_scale_factors,
        "logged_initial_state": np.asarray(logged_initial_state, dtype=np.float64).astype(float).tolist(),
        "true_model_perturbation_frac": float(true_model_perturbation_frac),
        "true_model_seed": true_model_seed,
        "process_noise_std_n": float(process_noise_std_n),
        "process_noise_seed": process_noise_seed,
        "observation_noise_std": {
            "cart_position_m": float(observation_noise_std[0]),
            "cart_velocity_m_s": float(observation_noise_std[1]),
            "pole_angle_rad": float(observation_noise_std[2]),
            "pole_angle_rate_rad_s": float(observation_noise_std[3]),
            "pole_angle_deg": math.degrees(float(observation_noise_std[2])),
            "pole_angle_rate_deg_s": math.degrees(float(observation_noise_std[3])),
        },
        "observation_noise_seed": observation_noise_seed,
    }
    if collection_id is not None:
        metadata["collection_id"] = collection_id
    if demo_name is not None:
        metadata["demo_name"] = demo_name
    if teacher_controller_json is not None:
        metadata["teacher_controller_json"] = teacher_controller_json
    if student_observer_json is not None:
        metadata["student_observer_json"] = student_observer_json
    if observer_target is not None:
        metadata["observer_target"] = observer_target
    metadata_path = capture_metadata_path(frames_dir)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="ascii")
    return metadata_path


def rollout_with_capture(
    controller_gain: np.ndarray,
    observer_policy: dict[str, object] | None,
    steps: int,
    initial_angle_deg: float,
    true_params: NominalCartPoleParams,
    process_noise_std_n: float,
    process_noise_rng: np.random.Generator,
    observation_noise_std: np.ndarray,
    observation_noise_rng: np.random.Generator,
    frame_width_px: int,
    frame_height_px: int,
    frames_dir: Path,
    render_preview: bool,
    render_every: int,
    stop_on_goal: bool,
) -> dict[str, object]:
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    capture_env = create_cartpole_env("rgb_array", true_params)
    preview_env = create_cartpole_env("human", true_params) if render_preview else None
    try:
        initial_state_m = np.array([0.0, 0.0, math.radians(initial_angle_deg), 0.0], dtype=np.float64)
        logged_initial_state_m = initial_state_m + observation_noise_rng.normal(0.0, observation_noise_std)
        set_cartpole_state(capture_env, initial_state_m, true_params)
        if preview_env is not None:
            set_cartpole_state(preview_env, initial_state_m, true_params)

        observer_a = observer_b = observer_l = observer_d = None
        observer_theta_pixel_coefficients = None
        observer_theta_pixel_bias = None
        observer_theta_pixel_blend_weight = 0.0
        observer_theta_dot_blend_weight = 0.0
        observer_cart_pixel_blend_weight = 0.0
        estimated_state_m = None
        if observer_policy is not None:
            observer_a = np.asarray(observer_policy["A_L"], dtype=np.float64)
            observer_b = np.asarray(observer_policy["B_L"], dtype=np.float64).reshape(4, 1)
            observer_l = np.asarray(observer_policy["L"], dtype=np.float64)
            observer_d = np.asarray(observer_policy["d"], dtype=np.float64).reshape(4)
            observer_theta_pixel_coefficients = observer_policy.get("theta_pixel_coefficients")
            observer_theta_pixel_bias = observer_policy.get("theta_pixel_bias")
            observer_theta_pixel_blend_weight = float(observer_policy.get("theta_pixel_blend_weight", 0.0))
            observer_theta_dot_blend_weight = float(observer_policy.get("theta_dot_blend_weight", 0.0))
            observer_cart_pixel_blend_weight = float(observer_policy.get("cart_pixel_blend_weight", 0.0))
            estimated_state_m = initial_state_m.copy()
        state_m = initial_state_m.copy()
        states = [state_m.copy()]
        trace_rows: list[dict[str, float | int | str]] = []
        survived_steps = 0
        last_applied_force_n = 0.0
        peak_requested_force_n = 0.0
        peak_applied_force_n = 0.0
        peak_velocity_m_s = 0.0
        peak_abs_position_m = 0.0
        peak_abs_angle_deg = 0.0
        force_clip_count = 0
        capture_end_reason = "horizon"
        goal_reached_step = None
        output_frame_width_px = None
        output_frame_height_px = None
        required_consecutive_goal_steps = max(1, int(math.ceil(GOAL_HOLD_TIME_S / true_params.sample_time_s)))
        consecutive_goal_steps = 0

        capture_start_time_s = time.monotonic()
        next_deadline_s = time.monotonic() + true_params.sample_time_s

        for step_index in range(steps):
            control_state_m = estimated_state_m if estimated_state_m is not None else state_m
            raw_force_n = float(-(controller_gain @ control_state_m)[0])
            state_m, terminated, step_stats = step_cartpole_env(
                capture_env,
                raw_force_n,
                true_params,
                process_noise_std_n=process_noise_std_n,
                rng=process_noise_rng,
            )
            states.append(state_m.copy())
            survived_steps = step_index + 1
            last_applied_force_n = float(step_stats["applied_force_n"])
            peak_requested_force_n = max(peak_requested_force_n, abs(raw_force_n))
            peak_applied_force_n = max(peak_applied_force_n, abs(last_applied_force_n))
            peak_velocity_m_s = max(peak_velocity_m_s, abs(float(state_m[1])))
            peak_abs_position_m = max(peak_abs_position_m, abs(float(state_m[0])))
            peak_abs_angle_deg = max(peak_abs_angle_deg, abs(math.degrees(float(state_m[2]))))
            if bool(step_stats["force_clipped"]):
                force_clip_count += 1

            if preview_env is not None and step_index % max(render_every, 1) == 0:
                set_cartpole_state(preview_env, state_m, true_params)
                preview_env.render()
            if preview_env is not None:
                next_deadline_s = pace_realtime_loop(next_deadline_s)
                next_deadline_s += true_params.sample_time_s

            rendered = capture_env.render()
            if rendered is None:
                raise RuntimeError("Gym did not return a frame for rgb_array rendering.")
            binary_frame = extract_binary_pole_frame(
                np.asarray(rendered, dtype=np.uint8),
                None,
                None,
            )
            if observer_policy is not None:
                binary_frame = pad_binary_frame_to_shape(binary_frame, frame_width_px, frame_height_px)
            output_frame_height_px, output_frame_width_px = binary_frame.shape
            frame_path = frames_dir / f"frame_{step_index:06d}.png"
            write_binary_frame(frame_path, binary_frame)
            logged_state_m = state_m + observation_noise_rng.normal(0.0, observation_noise_std)

            trace_row: dict[str, float | int | str] = {
                "frame_index": step_index,
                "frame_filename": frame_path.name,
                "control_step": step_index + 1,
                "simulated_time_s": (step_index + 1) * true_params.sample_time_s,
                "cart_position_m": float(logged_state_m[0]),
                "cart_velocity_m_s": float(logged_state_m[1]),
                "pole_angle_rad": float(logged_state_m[2]),
                "pole_angle_deg": math.degrees(float(logged_state_m[2])),
                "pole_angle_rate_rad_s": float(logged_state_m[3]),
                "pole_angle_rate_deg_s": math.degrees(float(logged_state_m[3])),
                "true_cart_position_m": float(state_m[0]),
                "true_cart_velocity_m_s": float(state_m[1]),
                "true_pole_angle_rad": float(state_m[2]),
                "true_pole_angle_deg": math.degrees(float(state_m[2])),
                "true_pole_angle_rate_rad_s": float(state_m[3]),
                "true_pole_angle_rate_deg_s": math.degrees(float(state_m[3])),
                "raw_control_force_n": raw_force_n,
                "commanded_control_force_n": float(step_stats["commanded_control_force_n"]),
                "process_noise_force_n": float(step_stats["process_noise_force_n"]),
                "noisy_control_force_n": float(step_stats["noisy_control_force_n"]),
                "applied_control_force_n": last_applied_force_n,
            }

            if estimated_state_m is not None:
                measurement_vector = binary_frame.astype(np.float64).reshape(-1) / 255.0
                previous_estimated_state_m = estimated_state_m.copy()
                predicted_state_m = (
                    (observer_a @ estimated_state_m)
                    + (observer_b[:, 0] * float(step_stats["commanded_control_force_n"]))
                    + (observer_l @ measurement_vector)
                    + observer_d
                )
                theta_pixel_deg_text = "n/a"
                if observer_theta_pixel_coefficients is not None and observer_theta_pixel_bias is not None:
                    theta_pixel = float(
                        observer_theta_pixel_coefficients @ measurement_vector + observer_theta_pixel_bias
                    )
                    predicted_state_m[2] = (
                        (1.0 - observer_theta_pixel_blend_weight) * predicted_state_m[2]
                        + observer_theta_pixel_blend_weight * theta_pixel
                    )
                    if observer_theta_dot_blend_weight > 0.0:
                        theta_dot_corrected = (
                            predicted_state_m[2] - previous_estimated_state_m[2]
                        ) / true_params.sample_time_s
                        predicted_state_m[3] = (
                            (1.0 - observer_theta_dot_blend_weight) * predicted_state_m[3]
                            + observer_theta_dot_blend_weight * theta_dot_corrected
                        )
                    theta_pixel_deg_text = f"{math.degrees(theta_pixel):+.3f} deg"
                    # --- Cart position blend from pixel geometry ---
                    if observer_cart_pixel_blend_weight > 0.0:
                        cart_pos_pixel = estimate_cart_position_from_binary_frame(binary_frame)
                        if cart_pos_pixel is not None:
                            predicted_state_m[0] = (
                                (1.0 - observer_cart_pixel_blend_weight) * predicted_state_m[0]
                                + observer_cart_pixel_blend_weight * cart_pos_pixel
                            )
                    estimated_state_m = predicted_state_m
                if step_index < 10:
                    print(
                        f"Step {step_index:3d} | "
                        f"true: pos={state_m[0]:+.4f} vel={state_m[1]:+.4f} "
                        f"theta={math.degrees(state_m[2]):+.3f} deg theta_dot={math.degrees(state_m[3]):+.3f} deg/s | "
                        f"est: pos={estimated_state_m[0]:+.4f} vel={estimated_state_m[1]:+.4f} "
                        f"theta={math.degrees(estimated_state_m[2]):+.3f} deg theta_dot={math.degrees(estimated_state_m[3]):+.3f} deg/s | "
                        f"theta_pixel={theta_pixel_deg_text} | "
                        f"u={raw_force_n:+.4f} N",
                        flush=True,
                    )
                trace_row.update(
                    {
                        "estimated_cart_position_m": float(estimated_state_m[0]),
                        "estimated_cart_velocity_m_s": float(estimated_state_m[1]),
                        "estimated_pole_angle_rad": float(estimated_state_m[2]),
                        "estimated_pole_angle_deg": math.degrees(float(estimated_state_m[2])),
                        "estimated_pole_angle_rate_rad_s": float(estimated_state_m[3]),
                        "estimated_pole_angle_rate_deg_s": math.degrees(float(estimated_state_m[3])),
                    }
                )

            trace_rows.append(trace_row)

            if is_goal_state(state_m):
                consecutive_goal_steps += 1
                if consecutive_goal_steps >= required_consecutive_goal_steps and goal_reached_step is None:
                    goal_reached_step = step_index + 1
                    if stop_on_goal:
                        capture_end_reason = "goal"
                        break
            else:
                consecutive_goal_steps = 0

            if terminated:
                capture_end_reason = "failure"
                break

        capture_wall_clock_duration_s = time.monotonic() - capture_start_time_s
        return {
            "survived_steps": survived_steps,
            "final_state": state_m,
            "logged_initial_state": logged_initial_state_m,
            "last_applied_force_n": last_applied_force_n,
            "states": states,
            "trace_rows": trace_rows,
            "frames_captured": len(trace_rows),
            "frames_dir": frames_dir,
            "capture_wall_clock_duration_s": capture_wall_clock_duration_s,
            "capture_end_reason": capture_end_reason,
            "goal_reached_step": goal_reached_step,
            "output_frame_width_px": output_frame_width_px,
            "output_frame_height_px": output_frame_height_px,
            "rollout_stats": {
                "force_clip_count": float(force_clip_count),
                "peak_requested_force_n": peak_requested_force_n,
                "peak_applied_force_n": peak_applied_force_n,
                "peak_velocity_m_s": peak_velocity_m_s,
                "peak_abs_position_m": peak_abs_position_m,
                "peak_abs_angle_deg": peak_abs_angle_deg,
            },
        }
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
    parser.add_argument("--steps", type=int, default=500, help="Simulation steps to run.")
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
        default=60.0,
        help="Discrete control rate in Hz. Defaults to 60 Hz for the clean simulation-only capture path.",
    )
    parser.add_argument(
        "--max-force-n",
        type=float,
        default=10.0,
        help="Maximum CartPole force magnitude in newtons.",
    )
    parser.add_argument(
        "--r-weight",
        type=float,
        default=1e-6,
        help="Direct scalar control penalty R for the discrete LQR design.",
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
        help="Lossless output video path. Defaults to simulation/captures/cartpole_nominal_TIMESTAMP.mkv.",
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
        "--observer-json",
        type=Path,
        default=None,
        help="Learned observer JSON exported from student_policy.ipynb. When set, control uses rendered binary images only instead of the true cart state.",
    )
    parser.add_argument(
        "--collection-id",
        type=str,
        default=None,
        help="Optional data-collection run identifier stored in capture metadata.",
    )
    parser.add_argument(
        "--demo-name",
        type=str,
        default=None,
        help="Optional demo label stored in capture metadata.",
    )
    parser.add_argument(
        "--true-gravity-scale",
        type=float,
        default=1.0,
        help="Deterministic gravity scale applied to the true rollout model relative to the nominal model.",
    )
    parser.add_argument(
        "--true-masscart-scale",
        type=float,
        default=1.0,
        help="Deterministic cart-mass scale applied to the true rollout model relative to the nominal model.",
    )
    parser.add_argument(
        "--true-masspole-scale",
        type=float,
        default=1.25,
        help="Deterministic pole-mass scale applied to the true rollout model relative to the nominal model.",
    )
    parser.add_argument(
        "--true-half-pole-length-scale",
        type=float,
        default=0.8,
        help="Deterministic pole half-length scale applied to the true rollout model relative to the nominal model.",
    )
    parser.add_argument(
        "--true-model-perturbation-frac",
        type=float,
        default=0.0,
        help="Additional uniform relative perturbation magnitude applied after the deterministic true-model scaling. Set to 0 to use the clean-path mismatch only.",
    )
    parser.add_argument(
        "--true-model-seed",
        type=int,
        default=0,
        help="Seed used to draw the optional extra true-model parameter perturbations.",
    )
    parser.add_argument(
        "--process-noise-std-n",
        type=float,
        default=0.0,
        help="Gaussian process-noise standard deviation, in newtons, added to the true plant input after the nominal controller command.",
    )
    parser.add_argument(
        "--process-noise-seed",
        type=int,
        default=0,
        help="Seed for the per-step process-noise sequence.",
    )
    parser.add_argument(
        "--observation-noise-position-m",
        type=float,
        default=0.0,
        help="Logged cart-position observation noise standard deviation in meters.",
    )
    parser.add_argument(
        "--observation-noise-velocity-m-s",
        type=float,
        default=0.0,
        help="Logged cart-velocity observation noise standard deviation in meters per second.",
    )
    parser.add_argument(
        "--observation-noise-angle-deg",
        type=float,
        default=0.0,
        help="Logged pole-angle observation noise standard deviation in degrees.",
    )
    parser.add_argument(
        "--observation-noise-angle-rate-deg-s",
        type=float,
        default=0.0,
        help="Logged pole angular-rate observation noise standard deviation in degrees per second.",
    )
    parser.add_argument(
        "--observation-noise-seed",
        type=int,
        default=2000,
        help="Seed for the logged-state observation-noise sequence.",
    )
    parser.add_argument(
        "--stop-on-goal",
        action="store_true",
        help="Stop the rollout once the state has remained inside the goal band for the required hold time.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.steps <= 0:
        raise SystemExit("Simulation steps must be positive.")
    if args.control_rate_hz <= 0.0:
        raise SystemExit("Control rate must be positive.")
    if args.max_force_n <= 0.0:
        raise SystemExit("Maximum force must be positive.")
    if args.r_weight <= 0.0:
        raise SystemExit("R weight must be positive.")
    if args.theta0_range_deg < 0.0:
        raise SystemExit("Initial angle range must be non-negative.")
    if args.frame_height_px <= 0 or args.frame_width_px <= 0:
        raise SystemExit("Frame dimensions must be positive.")
    if args.render_every is not None and args.render_every <= 0:
        raise SystemExit("Render cadence must be positive.")
    if args.true_gravity_scale <= 0.0:
        raise SystemExit("True-model gravity scale must be positive.")
    if args.true_masscart_scale <= 0.0:
        raise SystemExit("True-model cart-mass scale must be positive.")
    if args.true_masspole_scale <= 0.0:
        raise SystemExit("True-model pole-mass scale must be positive.")
    if args.true_half_pole_length_scale <= 0.0:
        raise SystemExit("True-model pole half-length scale must be positive.")
    if args.true_model_perturbation_frac < 0.0:
        raise SystemExit("True-model perturbation fraction must be non-negative.")
    if args.process_noise_std_n < 0.0:
        raise SystemExit("Process-noise standard deviation must be non-negative.")
    if args.observation_noise_position_m < 0.0:
        raise SystemExit("Observation-noise position standard deviation must be non-negative.")
    if args.observation_noise_velocity_m_s < 0.0:
        raise SystemExit("Observation-noise velocity standard deviation must be non-negative.")
    if args.observation_noise_angle_deg < 0.0:
        raise SystemExit("Observation-noise angle standard deviation must be non-negative.")
    if args.observation_noise_angle_rate_deg_s < 0.0:
        raise SystemExit("Observation-noise angle-rate standard deviation must be non-negative.")


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_args(args)

    nominal_params = NominalCartPoleParams(
        sample_time_s=1.0 / args.control_rate_hz,
        max_force_n=args.max_force_n,
        control_penalty_r=args.r_weight,
    )
    rng = np.random.default_rng(args.seed)
    initial_angle_deg = sample_initial_angle_deg(rng, args.theta0_deg, args.theta0_range_deg)
    true_model_rng = np.random.default_rng(args.true_model_seed)
    true_params = scale_cartpole_params(
        nominal_params,
        gravity_scale=args.true_gravity_scale,
        masscart_scale=args.true_masscart_scale,
        masspole_scale=args.true_masspole_scale,
        half_pole_length_scale=args.true_half_pole_length_scale,
    )
    true_params = perturb_cartpole_params(
        true_params,
        args.true_model_perturbation_frac,
        true_model_rng,
    )
    process_noise_rng = np.random.default_rng(args.process_noise_seed)
    observation_noise_rng = np.random.default_rng(args.observation_noise_seed)
    observation_noise_std = np.array(
        [
            args.observation_noise_position_m,
            args.observation_noise_velocity_m_s,
            math.radians(args.observation_noise_angle_deg),
            math.radians(args.observation_noise_angle_rate_deg_s),
        ],
        dtype=np.float64,
    )

    a_discrete, b_discrete = linearize_nominal_cartpole(nominal_params)
    q_matrix, r_matrix = choose_cost_matrices(nominal_params)
    gain, _riccati_solution, closed_loop_poles = solve_discrete_lqr(a_discrete, b_discrete, q_matrix, r_matrix)

    observer_policy = None
    controller_gain = gain
    if args.observer_json is not None:
        observer_policy = load_observer_policy(args.observer_json)
        controller_gain = np.asarray(observer_policy["teacher_gain"], dtype=np.float64).reshape(1, 4)
        if (
            int(observer_policy["nonzero_measurement_coefficients"]) == 0
            and int(observer_policy.get("nonzero_theta_pixel_coefficients", 0)) == 0
        ):
            print(
                "Warning: observer JSON has zero nonzero image-gain coefficients. "
                "This fit does not use pixel measurements and cannot bootstrap image-only control from frames.",
                flush=True,
            )
        if not np.allclose(controller_gain, gain, rtol=0.0, atol=1e-9):
            print(
                "Observer JSON teacher K differs from the controller generated from the current nominal CartPole parameters. "
                "Using the K stored in the observer JSON.",
                flush=True,
            )
        closed_loop_poles = np.linalg.eigvals(a_discrete - b_discrete @ controller_gain)
    sign_report = verify_sign_conventions(controller_gain)

    frame_png_dir = args.frame_png_dir or default_frame_directory(args.video_output)
    render_every = effective_render_every(args.control_rate_hz, args.render_every)
    rollout_model = "gym_cartpole_nominal"
    rollout_frame_width_px = args.frame_width_px if observer_policy is None else int(observer_policy["frame_width_px"])
    rollout_frame_height_px = args.frame_height_px if observer_policy is None else int(observer_policy["frame_height_px"])
    rollout = rollout_with_capture(
        controller_gain,
        observer_policy,
        args.steps,
        initial_angle_deg,
        true_params,
        args.process_noise_std_n,
        process_noise_rng,
        observation_noise_std,
        observation_noise_rng,
        rollout_frame_width_px,
        rollout_frame_height_px,
        frame_png_dir,
        args.render,
        render_every,
        args.stop_on_goal,
    )

    survived_steps = int(rollout["survived_steps"])
    final_state = np.asarray(rollout["final_state"], dtype=np.float64)
    logged_initial_state = np.asarray(rollout["logged_initial_state"], dtype=np.float64)
    last_applied_force_n = float(rollout["last_applied_force_n"])
    states = [np.asarray(state, dtype=np.float64) for state in rollout["states"]]
    captured_states = states[1:]
    frames_captured = int(rollout["frames_captured"])
    written_frames_dir = Path(rollout["frames_dir"])
    capture_wall_clock_duration_s = float(rollout["capture_wall_clock_duration_s"])
    capture_end_reason = str(rollout["capture_end_reason"])
    goal_reached_step = rollout["goal_reached_step"]
    rollout_stats = dict(rollout["rollout_stats"])

    capture_trace_path_value = write_capture_trace_csv(written_frames_dir, list(rollout["trace_rows"]))
    controller_json_path = write_teacher_controller_json(
        capture_controller_path(written_frames_dir),
        nominal_params,
        q_matrix,
        r_matrix,
        controller_gain,
        closed_loop_poles,
        sign_report,
        a_discrete,
        b_discrete,
    )
    metadata_path = write_capture_metadata(
        written_frames_dir,
        nominal_params,
        true_params,
        logged_initial_state,
        int(rollout["output_frame_width_px"]),
        int(rollout["output_frame_height_px"]),
        initial_angle_deg,
        args.steps,
        frames_captured,
        capture_end_reason,
        goal_reached_step,
        capture_wall_clock_duration_s,
        controller_json_path.name,
        rollout_model,
        control_policy="image_observer_feedback" if observer_policy is not None else "teacher_state_feedback",
        true_model_scale_factors={
            "gravity_scale": float(args.true_gravity_scale),
            "masscart_scale": float(args.true_masscart_scale),
            "masspole_scale": float(args.true_masspole_scale),
            "half_pole_length_scale": float(args.true_half_pole_length_scale),
        },
        true_model_perturbation_frac=args.true_model_perturbation_frac,
        true_model_seed=args.true_model_seed,
        process_noise_std_n=args.process_noise_std_n,
        process_noise_seed=args.process_noise_seed,
        observation_noise_std=observation_noise_std,
        observation_noise_seed=args.observation_noise_seed,
        collection_id=args.collection_id,
        demo_name=args.demo_name,
        student_observer_json=None if observer_policy is None else Path(observer_policy["path"]).name,
        observer_target=None if observer_policy is None else str(observer_policy["observer_target"]),
    )

    output_path = None
    if not args.skip_video_assembly:
        output_path = write_binary_video(
            written_frames_dir,
            args.control_rate_hz,
            args.video_output,
        )

    capture_stop_state = captured_states[-1] if captured_states else states[0]
    print(f"Frame dir   : {written_frames_dir}")
    print(f"Metadata    : {metadata_path}")
    print(f"Trace csv   : {capture_trace_path_value}")
    print(f"Teacher K   : {controller_json_path}")
    if output_path is not None:
        print(f"Video path  : {output_path}")
    else:
        print("Video path  : skipped (--skip-video-assembly)")
    print(f"Frames      : {frames_captured}")
    print(
        f"Frame size  : {int(rollout['output_frame_height_px'])}x{int(rollout['output_frame_width_px'])}"
    )
    print(f"Frame rate  : {args.control_rate_hz:.3f} fps")
    print(f"Sample time : {nominal_params.sample_time_s:.6f} s")
    print(f"Max force   : {nominal_params.max_force_n:.3f} N")
    print(f"Rollout mdl : {rollout_model}")
    print(f"Sim time    : {frames_captured * true_params.sample_time_s:.6f} s")
    print(f"Wall time   : {capture_wall_clock_duration_s:.6f} s before viewer hold")
    if args.collection_id is not None:
        print(f"Collection  : {args.collection_id}")
    if args.demo_name is not None:
        print(f"Demo name   : {args.demo_name}")
    if args.render:
        print(f"Preview     : every {render_every} control steps (~{args.control_rate_hz / render_every:.3f} Hz)")
    print(f"Init angle  : {initial_angle_deg:.3f} deg")
    print(
        "True model : "
        f"perturb={args.true_model_perturbation_frac:.3f}, "
        f"seed={args.true_model_seed}, "
        f"masscart={true_params.masscart_kg:.5f} kg, "
        f"masspole={true_params.masspole_kg:.5f} kg, "
        f"half_length={true_params.half_pole_length_m:.5f} m"
    )
    print(
        "Proc noise : "
        f"std={args.process_noise_std_n:.5f} N, "
        f"seed={args.process_noise_seed}"
    )
    print(
        "Obs noise  : "
        f"pos={args.observation_noise_position_m:.5f} m, "
        f"vel={args.observation_noise_velocity_m_s:.5f} m/s, "
        f"angle={args.observation_noise_angle_deg:.3f} deg, "
        f"rate={args.observation_noise_angle_rate_deg_s:.3f} deg/s, "
        f"seed={args.observation_noise_seed}"
    )
    print(f"Capture end : {capture_end_reason}")
    if goal_reached_step is not None:
        print(f"Goal step   : {goal_reached_step}")
    print(f"Stop state  : {capture_stop_state}")
    print(f"Rollout     : survived {survived_steps}/{args.steps} control steps")
    print(f"Rollout end : {final_state}")
    print(f"Rollout u   : {last_applied_force_n:.5f} N")
    if observer_policy is not None:
        print(f"Observer    : {observer_policy['path']}")
        print(f"Obs target  : {observer_policy['observer_target']}")
        if observer_policy.get("theta_pixel_coefficients") is not None:
            print(
                "Theta blend : "
                f"w={float(observer_policy['theta_pixel_blend_weight']):.3f}, "
                f"nonzero={int(observer_policy['nonzero_theta_pixel_coefficients'])}"
            )
    print(
        "Peaks       : "
        f"raw_u={rollout_stats['peak_requested_force_n']:.5f} N, "
        f"applied_u={rollout_stats['peak_applied_force_n']:.5f} N, "
        f"|x|={rollout_stats['peak_abs_position_m']:.5f} m, "
        f"|v|={rollout_stats['peak_velocity_m_s']:.5f} m/s, "
        f"|theta|={rollout_stats['peak_abs_angle_deg']:.5f} deg"
    )
    print(f"Saturation  : force clips={int(rollout_stats['force_clip_count'])}")

    if args.render and captured_states:
        simulated_capture_duration_s = len(captured_states) * true_params.sample_time_s
        print(
            "Capture finished: "
            f"{len(captured_states)} frames, "
            f"{simulated_capture_duration_s:.3f} s simulated, "
            f"{capture_wall_clock_duration_s:.3f} s wall-clock before hold. "
            f"Preview cadence: every {render_every} control steps. "
            "Holding final viewer state until Ctrl+C.",
            flush=True,
        )
        preview_env = create_cartpole_env("human", true_params)
        try:
            set_cartpole_state(preview_env, captured_states[-1], true_params)
            render_hold_loop(preview_env, -1.0)
        finally:
            preview_env.close()


if __name__ == "__main__":
    main()
