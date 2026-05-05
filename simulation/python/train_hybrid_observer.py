#!/usr/bin/env python3
"""Train a hybrid pixel-angle observer from a teacher demo collection.

Replicates the hybrid observer training path from student_policy.ipynb:
  - A_L, B_L fixed from nominal dynamics (not fitted)
  - L = 0, d = 0
  - theta_pixel_coefficients fitted via Ridge regression from pixels → pole angle
  - theta_pixel_blend_weight fixed at 0.7

Writes the observer JSON to --output-json.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

STATE_COLUMNS = [
    "cart_position_m",
    "cart_velocity_m_s",
    "pole_angle_rad",
    "pole_angle_rate_rad_s",
]
ANGLE_IDX = STATE_COLUMNS.index("pole_angle_rad")

THETA_PIXEL_BLEND_WEIGHT = 0.7
THETA_DOT_BLEND_WEIGHT   = 0.1
CART_PIXEL_BLEND_WEIGHT  = 0.8
DEFAULT_RIDGE_ALPHA      = 0.1


def load_raw_frame(frame_path: Path) -> np.ndarray:
    img = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read frame: {frame_path}")
    return img


def pad_to_target(frame: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = frame.shape
    pad_top  = max(0, target_h - h)
    pad_left = max(0, (target_w - w) // 2)
    pad_right = max(0, target_w - w - pad_left)
    return cv2.copyMakeBorder(frame, pad_top, 0, pad_left, pad_right,
                              cv2.BORDER_CONSTANT, value=0)


def frame_vector(path: Path, target_h: int, target_w: int) -> np.ndarray:
    return pad_to_target(load_raw_frame(path), target_h, target_w).astype(np.float64).reshape(-1) / 255.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-dir", type=Path, required=True,
                        help="Directory containing demo frame dirs and collection_manifest.json.")
    parser.add_argument("--output-json", type=Path, required=True,
                        help="Path to write the observer JSON.")
    parser.add_argument("--theta-blend", type=float, default=THETA_PIXEL_BLEND_WEIGHT,
                        help="Blend weight for pixel angle estimate (default 0.7).")
    parser.add_argument("--ridge-alpha", type=float, default=DEFAULT_RIDGE_ALPHA,
                        help="Ridge regularisation alpha for pixel→angle fit (default 0.1).")
    args = parser.parse_args()

    collection_dir = args.collection_dir.resolve()
    manifest_path = collection_dir / "collection_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No collection_manifest.json found in {collection_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    demo_specs = manifest["demo_specs"]

    print(f"Collection : {collection_dir.name}")
    print(f"Demos      : {len(demo_specs)}")

    # ── Discover frame geometry across all demos ──────────────────────────────
    heights, widths = [], []
    for spec in demo_specs:
        sample = next(Path(spec["frame_dir"]).glob("frame_000000.png"), None)
        if sample:
            img = cv2.imread(str(sample), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                heights.append(img.shape[0])
                widths.append(img.shape[1])
    if not heights:
        raise RuntimeError("No frame_000000.png found in any demo dir.")
    target_h = max(heights)
    target_w = max(widths)
    print(f"Frame geo  : {target_h}×{target_w}  (max across demos, pad only)")

    # ── Load all demo data ────────────────────────────────────────────────────
    x_prev_segs, u_prev_segs, x_next_segs, z_segs = [], [], [], []
    teacher_gain = nominal_a = nominal_b = None
    demo_transition_counts: list[int] = []
    source_frame_dirs: list[str] = []
    latest_frame_dir = latest_trace_path = latest_meta = None

    for i, spec in enumerate(demo_specs):
        frame_dir   = Path(spec["frame_dir"])
        meta_path   = frame_dir / "capture_metadata.json"
        trace_path  = frame_dir / "capture_trace.csv"
        ctrl_path   = frame_dir / "teacher_lqr_controller.json"

        meta  = json.loads(meta_path.read_text(encoding="ascii"))
        trace = pd.read_csv(trace_path)
        ctrl  = json.loads(ctrl_path.read_text(encoding="utf-8"))

        K = np.asarray(ctrl["K"], dtype=np.float64).reshape(1, 4)
        A = np.asarray(ctrl["A"], dtype=np.float64)
        B = np.asarray(ctrl["B"], dtype=np.float64).reshape(4, 1)

        if teacher_gain is None:
            teacher_gain, nominal_a, nominal_b = K, A, B
        elif not np.allclose(K, teacher_gain, atol=1e-9):
            raise ValueError(f"Demo {i+1} has a different teacher gain K — demos must share parameters.")

        ctrl_col = ("commanded_control_force_n" if "commanded_control_force_n" in trace.columns
                    else "applied_control_force_n")
        x_next = trace[STATE_COLUMNS].to_numpy(dtype=np.float64)
        logged_init = meta.get("logged_initial_state")
        init_state = (np.asarray(logged_init, dtype=np.float64)
                      if (logged_init and len(logged_init) == 4)
                      else np.array([0.0, 0.0, math.radians(meta["initial_angle_deg"]), 0.0]))
        x_prev = np.vstack([init_state, x_next[:-1]])
        u_prev = trace[[ctrl_col]].to_numpy(dtype=np.float64)

        frame_paths = [frame_dir / fn for fn in trace["frame_filename"]]
        print(f"  [{i+1:2d}/{len(demo_specs)}] θ₀={meta['initial_angle_deg']:+6.1f}°  "
              f"frames={len(trace)}", flush=True)
        z = np.stack([frame_vector(fp, target_h, target_w) for fp in frame_paths])

        x_prev_segs.append(x_prev)
        u_prev_segs.append(u_prev)
        x_next_segs.append(x_next)
        z_segs.append(z)
        demo_transition_counts.append(len(trace))
        source_frame_dirs.append(str(frame_dir))
        latest_frame_dir  = frame_dir
        latest_trace_path = trace_path
        latest_meta       = meta

    x_prev   = np.vstack(x_prev_segs)
    u_prev   = np.vstack(u_prev_segs)
    x_next   = np.vstack(x_next_segs)
    z_matrix = np.vstack(z_segs)
    n_transitions = len(x_next)
    print(f"Total transitions : {n_transitions}")

    # ── Ridge regression: pixels → pole angle ────────────────────────────────
    angle_targets = x_next[:, ANGLE_IDX]
    ridge = Ridge(alpha=args.ridge_alpha, fit_intercept=True)
    ridge.fit(z_matrix, angle_targets)

    r2       = float(ridge.score(z_matrix, angle_targets))
    preds    = ridge.predict(z_matrix)
    rmse_deg = float(np.degrees(np.sqrt(np.mean((preds - angle_targets) ** 2))))
    print(f"Pixel→angle Ridge : R²={r2:.6f}  RMSE={rmse_deg:.4f}°  alpha={args.ridge_alpha}")

    theta_pixel_coeff = np.asarray(ridge.coef_, dtype=np.float64).reshape(-1)
    theta_pixel_bias  = float(ridge.intercept_)

    # Bootstrap on first frame of latest demo
    first_trace = pd.read_csv(Path(source_frame_dirs[-1]) / "capture_trace.csv")
    first_fp    = Path(source_frame_dirs[-1]) / first_trace["frame_filename"].iloc[0]
    boot_vec    = frame_vector(first_fp, target_h, target_w)
    boot_pred_rad = float(theta_pixel_coeff @ boot_vec + theta_pixel_bias)
    boot_true_deg = float(np.degrees(float(first_trace["pole_angle_rad"].iloc[0])))
    boot_pred_deg = float(np.degrees(boot_pred_rad))
    print(f"Bootstrap         : true={boot_true_deg:+.3f}°  predicted={boot_pred_deg:+.3f}°")

    # ── Build and save observer JSON ──────────────────────────────────────────
    pixel_count = target_h * target_w
    payload = {
        "anchor_frame_dir"              : str(latest_frame_dir),
        "source_frame_dirs"             : source_frame_dirs,
        "source_trace_csv"              : latest_trace_path.name,
        "rollout_model"                 : str(latest_meta.get("rollout_model", "gym_cartpole_nominal")),
        "demo_count"                    : len(demo_specs),
        "demo_transition_counts"        : demo_transition_counts,
        "total_transition_pairs"        : n_transitions,
        "teacher_controller_json"       : str(latest_frame_dir / "teacher_lqr_controller.json"),
        "teacher_gain_K"                : teacher_gain.reshape(-1).astype(float).tolist(),
        "unit_system"                   : "cartpole",
        "observer_target"               : "hybrid_nominal_dynamics_plus_direct_pixel_angle",
        "frame_height_px"               : int(target_h),
        "frame_width_px"                : int(target_w),
        "frame_rate_hz"                 : float(latest_meta["frame_rate_hz"]),
        "state_order"                   : STATE_COLUMNS,
        "measurement_description"       : "flattened padded binary frame z_{k+1}",
        "control_units"                 : "newtons",
        "preprocessing"                 : (
            "Captured PNG frames are padded to a shared training geometry "
            "using top-only and centered-horizontal zero padding. No resizing applied."
        ),
        "selected_model"                : "hybrid_nominal_plus_ridge_theta_blend",
        "A_L"                           : nominal_a.tolist(),
        "B_L"                           : nominal_b.tolist(),
        "L"                             : np.zeros((4, pixel_count)).tolist(),
        "d"                             : np.zeros(4).tolist(),
        "theta_pixel_model"             : "ridge_direct_angle",
        "theta_pixel_coefficients"      : theta_pixel_coeff.astype(float).tolist(),
        "theta_pixel_bias"              : theta_pixel_bias,
        "theta_pixel_blend_weight"      : float(args.theta_blend),
        "theta_dot_blend_weight"        : THETA_DOT_BLEND_WEIGHT,
        "cart_pixel_blend_weight"       : CART_PIXEL_BLEND_WEIGHT,
        "theta_pixel_ridge_alpha"       : float(args.ridge_alpha),
        "theta_pixel_r2"                : r2,
        "theta_pixel_rmse_deg"          : rmse_deg,
        "theta_pixel_bootstrap_true_angle_deg"      : boot_true_deg,
        "theta_pixel_bootstrap_predicted_angle_deg" : boot_pred_deg,
        "nonzero_image_coefficients"        : 0,
        "nonzero_theta_pixel_coefficients"  : int(np.count_nonzero(np.abs(theta_pixel_coeff) > 1e-12)),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved → {args.output_json}")


if __name__ == "__main__":
    main()
