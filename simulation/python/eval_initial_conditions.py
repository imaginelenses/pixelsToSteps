#!/usr/bin/env python3
"""Sweep N initial conditions: verify teacher LQR, then evaluate student pixel observer.

For each trial:
  1. Teacher (full-state LQR) rollout is run first; if it fails, the IC is skipped.
  2. Student (hybrid pixel observer) rollout runs on the same IC.
  3. e_stab = ||x̂_final - x_goal||_2, x_goal = 0 is recorded.

Reports median and 5th/95th percentile of e_stab across all student trials.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nominal_cartpole import (
    NominalCartPoleParams,
    choose_cost_matrices,
    create_cartpole_env,
    linearize_nominal_cartpole,
    scale_cartpole_params,
    set_cartpole_state,
    solve_discrete_lqr,
    step_cartpole_env,
)
from teacher_policy import (
    extract_binary_pole_frame,
    load_observer_policy,
    pad_binary_frame_to_shape,
)

DEFAULT_OBSERVER = SCRIPT_DIR / "hybrid_pixels_to_cartpole_observer_theta_blend_0p7.json"

# Paper evaluation protocol (Lee et al. 2024)
PAPER_SUCCESS_X_M     = 0.176   # m
PAPER_SUCCESS_THETA_RAD = 0.035  # rad ≈ 2°


def run_teacher(
    initial_state: np.ndarray,
    true_params: NominalCartPoleParams,
    gain: np.ndarray,
    steps: int,
) -> tuple[int, np.ndarray, bool]:
    env = create_cartpole_env(None, true_params)
    set_cartpole_state(env, initial_state, true_params)
    state = initial_state.copy()
    for step in range(steps):
        force = float(-(gain @ state)[0])
        state, terminated, _ = step_cartpole_env(env, force, true_params)
        if terminated:
            env.close()
            return step + 1, state, True
    env.close()
    return steps, state, False


def run_student(
    initial_state: np.ndarray,
    true_params: NominalCartPoleParams,
    gain: np.ndarray,
    observer: dict,
    steps: int,
    checkpoints: list[int],
) -> tuple[int, dict[int, np.ndarray], bool]:
    """Run student rollout and snapshot x̂ at each checkpoint step.

    Returns (survived_steps, {step: x_hat_snapshot}, terminated).
    Snapshots are only recorded for steps where the trial is still alive.
    """
    A_L = np.asarray(observer["A_L"], dtype=np.float64)
    B_L = np.asarray(observer["B_L"], dtype=np.float64).reshape(4, 1)
    L_gain = np.asarray(observer["L"], dtype=np.float64)
    d_bias = np.asarray(observer["d"], dtype=np.float64).reshape(4)
    theta_coeff = observer.get("theta_pixel_coefficients")
    theta_bias = observer.get("theta_pixel_bias")
    theta_blend = float(observer.get("theta_pixel_blend_weight", 0.0))
    frame_w = int(observer["frame_width_px"])
    frame_h = int(observer["frame_height_px"])
    checkpoint_set = set(checkpoints)

    env = create_cartpole_env("rgb_array", true_params)
    set_cartpole_state(env, initial_state, true_params)
    x_hat = initial_state.copy()
    snapshots: dict[int, np.ndarray] = {}

    for step in range(steps):
        force = float(-(gain @ x_hat)[0])
        state, terminated, stats = step_cartpole_env(env, force, true_params)

        rendered = env.render()
        rgb = np.asarray(rendered, dtype=np.uint8)
        binary = extract_binary_pole_frame(rgb, None, None)
        binary = pad_binary_frame_to_shape(binary, frame_w, frame_h)
        y = binary.astype(np.float64).reshape(-1) / 255.0

        cmd = float(stats["commanded_control_force_n"])
        x_hat_pred = A_L @ x_hat + B_L[:, 0] * cmd + L_gain @ y + d_bias

        if theta_coeff is not None and theta_bias is not None:
            theta_pixel = float(theta_coeff @ y + theta_bias)
            x_hat_pred[2] = (1.0 - theta_blend) * x_hat_pred[2] + theta_blend * theta_pixel

        x_hat = x_hat_pred
        completed_step = step + 1  # 1-indexed: step N completes after index step N-1

        if completed_step in checkpoint_set:
            snapshots[completed_step] = x_hat.copy()

        if terminated:
            env.close()
            return completed_step, snapshots, True

    env.close()
    return steps, snapshots, False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=100, help="Number of initial conditions to sample.")
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[150, 200, 250],
                        help="Steps at which to snapshot x̂ and evaluate paper success. "
                             "Rollout runs to max(checkpoints). Default: 150 200 250.")
    parser.add_argument("--theta-range-deg", type=float, default=12.0, help="Sample theta0 uniformly in [-range, +range] deg (paper uses full ±12°).")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for initial condition sampling.")
    parser.add_argument("--observer-json", type=Path, default=DEFAULT_OBSERVER)
    parser.add_argument("--true-masspole-scale", type=float, default=1.1)
    parser.add_argument("--true-half-pole-length-scale", type=float, default=0.9)
    return parser


def paper_ok_at(x_hat: np.ndarray) -> bool:
    return (abs(float(x_hat[0])) <= PAPER_SUCCESS_X_M
            and abs(float(x_hat[2])) <= PAPER_SUCCESS_THETA_RAD)


def main() -> None:
    args = build_parser().parse_args()
    checkpoints = sorted(set(args.checkpoints))
    max_steps = max(checkpoints)

    nominal_params = NominalCartPoleParams(
        sample_time_s=1.0 / 60.0,
        max_force_n=10.0,
        control_penalty_r=1e-6,
    )
    true_params = scale_cartpole_params(
        nominal_params,
        masspole_scale=args.true_masspole_scale,
        half_pole_length_scale=args.true_half_pole_length_scale,
    )
    A, B = linearize_nominal_cartpole(nominal_params)
    Q, R = choose_cost_matrices(nominal_params)
    teacher_gain, _, _ = solve_discrete_lqr(A, B, Q, R)

    observer = load_observer_policy(args.observer_json)
    student_gain = np.asarray(observer["teacher_gain"], dtype=np.float64).reshape(1, 4)

    # Sample ICs from the same distribution as training (paper protocol: uniform ±12°).
    rng = np.random.default_rng(args.seed)
    theta0_rad_list = rng.uniform(
        -args.theta_range_deg * math.pi / 180.0,
        args.theta_range_deg * math.pi / 180.0,
        size=args.n_trials,
    )
    theta0_list = np.degrees(theta0_rad_list)

    ckpt_header = "  ".join(f"@{c}" for c in checkpoints)
    print(f"Observer     : {args.observer_json.name}")
    print(f"Mismatch     : masspole×{args.true_masspole_scale}, half_length×{args.true_half_pole_length_scale}")
    print(f"Trials       : {args.n_trials}  |  Max steps : {max_steps} ({max_steps/60:.1f} s)  |  "
          f"Checkpoints : {checkpoints}  |  seed={args.seed}")
    print(f"Paper success: |x̂| ≤ {PAPER_SUCCESS_X_M} m  AND  |θ̂| ≤ {math.degrees(PAPER_SUCCESS_THETA_RAD):.1f}°")
    print()
    ckpt_cols = "  ".join(f"{'@'+str(c):>6}" for c in checkpoints)
    print(f"{'#':>4}  {'theta0':>8}  {'teacher':>8}  {'survived':>9}  {ckpt_cols}")
    print("-" * (4 + 2 + 8 + 2 + 8 + 2 + 9 + 2 + 8 * len(checkpoints)))

    teacher_fail_count = 0
    student_results: list[dict] = []

    for idx, theta0_deg in enumerate(theta0_list):
        initial_state = np.array([0.0, 0.0, math.radians(theta0_deg), 0.0], dtype=np.float64)

        t_survived, _, t_failed = run_teacher(initial_state, true_params, teacher_gain, max_steps)
        if t_failed:
            teacher_fail_count += 1
            skip_cols = "  ".join(f"{'---':>6}" for _ in checkpoints)
            print(f"{idx+1:4d}  {theta0_deg:+8.3f}  {'FAIL@'+str(t_survived):>8}  {'skip':>9}  {skip_cols}")
            continue

        s_survived, snapshots, s_failed = run_student(
            initial_state, true_params, student_gain, observer, max_steps, checkpoints
        )

        # paper_ok[c] = True iff survived to step c AND in goal zone at that snapshot
        ckpt_ok: dict[int, bool] = {}
        for c in checkpoints:
            snap = snapshots.get(c)
            ckpt_ok[c] = (snap is not None) and paper_ok_at(snap)

        final_snap = snapshots.get(max_steps) if snapshots else None
        if final_snap is None and snapshots:
            final_snap = snapshots[max(snapshots)]
        e_stab = float(np.linalg.norm(final_snap)) if final_snap is not None else float("nan")

        student_results.append({
            "theta0_deg": theta0_deg,
            "survived": s_survived,
            "failed": s_failed,
            "e_stab": e_stab,
            "ckpt_ok": ckpt_ok,
            "x_hat_final": final_snap.copy() if final_snap is not None else np.zeros(4),
        })

        survived_str = "FAIL@"+str(s_survived) if s_failed else f"all{max_steps}"
        ok_cols = "  ".join(f"{'YES' if ckpt_ok[c] else 'no':>6}" for c in checkpoints)
        print(f"{idx+1:4d}  {theta0_deg:+8.3f}  {'OK':>8}  {survived_str:>9}  {ok_cols}", flush=True)

    # ── Statistics ─────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    n_evaluated = len(student_results)

    if n_evaluated == 0:
        print("No valid trials (all teacher failures).")
        return

    n_survived = sum(1 for r in student_results if not r["failed"])
    n_failed   = n_evaluated - n_survived

    print(f"Teacher failures : {teacher_fail_count}/{args.n_trials}")
    print(f"Student survival : {n_survived}/{n_evaluated}  ({100*n_survived/n_evaluated:.1f}%)")
    if n_failed:
        print(f"Student failed   : {n_failed}/{n_evaluated}  ({100*n_failed/n_evaluated:.1f}%)")
    print()

    # Convergence-by-deadline table
    goal_def = f"|x̂|≤{PAPER_SUCCESS_X_M}m, |θ̂|≤{math.degrees(PAPER_SUCCESS_THETA_RAD):.0f}°"
    print(f"  Convergence by deadline  ({goal_def}):")
    print(f"    {'Deadline':>12}  {'n OK':>6}  {'%':>6}  {'seconds':>8}")
    print(f"    {'-'*40}")
    for c in checkpoints:
        n_ok = sum(1 for r in student_results if r["ckpt_ok"].get(c, False))
        secs = c / 60.0
        print(f"    {f'step {c}':>12}  {n_ok:>6}  {100*n_ok/n_evaluated:>5.1f}%  {secs:>6.2f} s")

    # e_stab at final checkpoint
    e_vals = np.array([r["e_stab"] for r in student_results if not math.isnan(r["e_stab"])])
    if len(e_vals):
        print()
        print(f"  e_stab = ||x̂||₂ at step {max_steps}  (all {n_evaluated} trials):")
        print(f"    median   : {np.median(e_vals):.4f}")
        print(f"    5th pct  : {np.percentile(e_vals, 5):.4f}")
        print(f"    95th pct : {np.percentile(e_vals, 95):.4f}")
        print(f"    min/max  : {np.min(e_vals):.4f} / {np.max(e_vals):.4f}")

    # Component breakdown at final checkpoint
    finals = np.stack([r["x_hat_final"] for r in student_results])
    print()
    labels = ["x (m)", "ẋ (m/s)", "θ (rad)", "θ̇ (rad/s)"]
    print(f"  |component| median at step {max_steps}  (all {n_evaluated} trials):")
    for i, lbl in enumerate(labels):
        vals = np.abs(finals[:, i])
        print(f"    {lbl:12s}: median={np.median(vals):.4f}  p95={np.percentile(vals, 95):.4f}")


if __name__ == "__main__":
    main()
