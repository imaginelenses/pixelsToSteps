#!/usr/bin/env python3
"""Collect multiple nominal Gym CartPole teacher demos with one shared configuration."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_CAPTURE_ROOT = Path(__file__).resolve().parents[1] / "captures" / "collections"
DEFAULT_THETA0_DEG_LIST = [
    -12.0,
    -10.0,
    -9.0,
    -8.0,
    -7.0,
    -6.0,
    -5.0,
    -4.0,
    -3.0,
    -2.0,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    9.0,
    10.0,
    12.0,
]


def default_collection_id() -> str:
    return f"nominal_cartpole_collection_{time.strftime('%Y%m%d_%H%M%S')}"


def sanitize_angle_label(theta0_deg: float) -> str:
    label = f"{theta0_deg:+.2f}".replace("+", "p").replace("-", "m").replace(".", "p")
    return label


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--theta0-deg-list",
        type=float,
        nargs="+",
        default=DEFAULT_THETA0_DEG_LIST,
        help="Initial pole angles, in degrees, for the demos collected in one run. Defaults to 20 varied demos that stop on steady-state or at the step cap.",
    )
    parser.add_argument("--steps", type=int, default=600, help="Maximum simulation steps per demo.")
    parser.add_argument(
        "--control-rate-hz",
        type=float,
        default=60.0,
        help="Common control rate for all demos in the collection.",
    )
    parser.add_argument(
        "--max-force-n",
        type=float,
        default=10.0,
        help="Common maximum CartPole force magnitude in newtons.",
    )
    parser.add_argument(
        "--r-weight",
        type=float,
        default=1e-6,
        help="Common scalar control penalty for the LQR design.",
    )
    parser.add_argument(
        "--frame-height-px",
        type=int,
        default=125,
        help="Common output frame height in pixels.",
    )
    parser.add_argument(
        "--frame-width-px",
        type=int,
        default=160,
        help="Common output frame width in pixels.",
    )
    parser.add_argument(
        "--collection-id",
        type=str,
        default=None,
        help="Optional explicit collection identifier. Defaults to a timestamped name.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_CAPTURE_ROOT,
        help="Directory under which the collection folder will be created.",
    )
    parser.add_argument(
        "--assemble-videos",
        action="store_true",
        help="Also assemble a lossless video for each demo. By default only PNG frames and metadata are written.",
    )
    parser.add_argument(
        "--stop-on-goal",
        dest="stop_on_goal",
        action="store_true",
        help="Stop each demo once the state has remained inside the goal band for the required hold time.",
    )
    parser.add_argument(
        "--no-stop-on-goal",
        dest="stop_on_goal",
        action="store_false",
        help="Always run each demo to the full step horizon, even if it has already stabilized.",
    )
    parser.set_defaults(stop_on_goal=True)
    parser.add_argument(
        "--true-gravity-scale",
        type=float,
        default=1.0,
        help="Deterministic gravity scale applied to the shared true rollout model for the collection.",
    )
    parser.add_argument(
        "--true-masscart-scale",
        type=float,
        default=1.0,
        help="Deterministic cart-mass scale applied to the shared true rollout model for the collection.",
    )
    parser.add_argument(
        "--true-masspole-scale",
        type=float,
        default=1.25,
        help="Deterministic pole-mass scale applied to the shared true rollout model for the collection.",
    )
    parser.add_argument(
        "--true-half-pole-length-scale",
        type=float,
        default=0.8,
        help="Deterministic pole half-length scale applied to the shared true rollout model for the collection.",
    )
    parser.add_argument(
        "--true-model-perturbation-frac",
        type=float,
        default=0.0,
        help="Additional uniform relative perturbation magnitude applied once to the shared true rollout model after the deterministic clean-path scaling.",
    )
    parser.add_argument(
        "--true-model-seed",
        type=int,
        default=0,
        help="Seed used to draw the shared true-model perturbations for the collection.",
    )
    parser.add_argument(
        "--process-noise-std-n",
        type=float,
        default=0.0,
        help="Gaussian process-noise standard deviation, in newtons, added during rollout collection.",
    )
    parser.add_argument(
        "--process-noise-seed-base",
        type=int,
        default=1000,
        help="Base seed for per-demo process-noise sequences. Demo i uses seed_base + i - 1.",
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
        "--observation-noise-seed-base",
        type=int,
        default=2000,
        help="Base seed for per-demo logged-state observation-noise sequences. Demo i uses seed_base + i - 1.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.steps <= 0:
        raise SystemExit("Simulation steps must be positive.")
    if not args.theta0_deg_list:
        raise SystemExit("At least one initial angle must be provided.")

    collection_id = args.collection_id or default_collection_id()
    collection_root = args.output_root.expanduser().resolve() / collection_id
    collection_root.mkdir(parents=True, exist_ok=True)

    teacher_policy_path = Path(__file__).resolve().parent / "teacher_policy.py"
    manifest = {
        "collection_id": collection_id,
        "collection_root": str(collection_root),
        "teacher_policy": str(teacher_policy_path),
        "common_args": {
            "steps": args.steps,
            "control_rate_hz": args.control_rate_hz,
            "max_force_n": args.max_force_n,
            "r_weight": args.r_weight,
            "frame_height_px": args.frame_height_px,
            "frame_width_px": args.frame_width_px,
            "assemble_videos": bool(args.assemble_videos),
            "stop_on_goal": bool(args.stop_on_goal),
            "true_gravity_scale": args.true_gravity_scale,
            "true_masscart_scale": args.true_masscart_scale,
            "true_masspole_scale": args.true_masspole_scale,
            "true_half_pole_length_scale": args.true_half_pole_length_scale,
            "true_model_perturbation_frac": args.true_model_perturbation_frac,
            "true_model_seed": args.true_model_seed,
            "process_noise_std_n": args.process_noise_std_n,
            "process_noise_seed_base": args.process_noise_seed_base,
            "observation_noise_position_m": args.observation_noise_position_m,
            "observation_noise_velocity_m_s": args.observation_noise_velocity_m_s,
            "observation_noise_angle_deg": args.observation_noise_angle_deg,
            "observation_noise_angle_rate_deg_s": args.observation_noise_angle_rate_deg_s,
            "observation_noise_seed_base": args.observation_noise_seed_base,
        },
        "demo_specs": [],
    }

    for demo_index, theta0_deg in enumerate(args.theta0_deg_list, start=1):
        angle_label = sanitize_angle_label(theta0_deg)
        demo_name = f"demo_{demo_index:02d}_theta_{angle_label}"
        frame_dir = collection_root / f"{demo_name}_frames"
        command = [
            sys.executable,
            str(teacher_policy_path),
            "--steps",
            str(args.steps),
            "--theta0-deg",
            str(theta0_deg),
            "--control-rate-hz",
            str(args.control_rate_hz),
            "--max-force-n",
            str(args.max_force_n),
            "--r-weight",
            str(args.r_weight),
            "--frame-height-px",
            str(args.frame_height_px),
            "--frame-width-px",
            str(args.frame_width_px),
            "--true-gravity-scale",
            str(args.true_gravity_scale),
            "--true-masscart-scale",
            str(args.true_masscart_scale),
            "--true-masspole-scale",
            str(args.true_masspole_scale),
            "--true-half-pole-length-scale",
            str(args.true_half_pole_length_scale),
            "--true-model-perturbation-frac",
            str(args.true_model_perturbation_frac),
            "--true-model-seed",
            str(args.true_model_seed),
            "--process-noise-std-n",
            str(args.process_noise_std_n),
            "--process-noise-seed",
            str(args.process_noise_seed_base + demo_index - 1),
            "--observation-noise-position-m",
            str(args.observation_noise_position_m),
            "--observation-noise-velocity-m-s",
            str(args.observation_noise_velocity_m_s),
            "--observation-noise-angle-deg",
            str(args.observation_noise_angle_deg),
            "--observation-noise-angle-rate-deg-s",
            str(args.observation_noise_angle_rate_deg_s),
            "--observation-noise-seed",
            str(args.observation_noise_seed_base + demo_index - 1),
            "--collection-id",
            collection_id,
            "--demo-name",
            demo_name,
            "--frame-png-dir",
            str(frame_dir),
        ]
        if args.assemble_videos:
            command.extend(
                [
                    "--video-output",
                    str(collection_root / f"{demo_name}.mkv"),
                ]
            )
        else:
            command.append("--skip-video-assembly")
        if args.stop_on_goal:
            command.append("--stop-on-goal")

        print(f"[{demo_index}/{len(args.theta0_deg_list)}] theta0={theta0_deg:.3f} deg -> {frame_dir}", flush=True)
        subprocess.run(command, check=True)
        manifest["demo_specs"].append(
            {
                "demo_index": demo_index,
                "demo_name": demo_name,
                "theta0_deg": theta0_deg,
                "frame_dir": str(frame_dir),
                "process_noise_seed": args.process_noise_seed_base + demo_index - 1,
                "observation_noise_seed": args.observation_noise_seed_base + demo_index - 1,
            }
        )

    manifest_path = collection_root / "collection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Collection root : {collection_root}")
    print(f"Manifest       : {manifest_path}")
    print(f"Demo count     : {len(manifest['demo_specs'])}")


if __name__ == "__main__":
    main()
