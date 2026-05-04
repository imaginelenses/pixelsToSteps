#!/usr/bin/env python3
"""Assemble captured cart-pole PNG frames into a lossless fixed-rate video."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


def capture_metadata_path(frames_dir: Path) -> Path:
    return frames_dir / "capture_metadata.json"


def default_video_output_path(frames_dir: Path) -> Path:
    frames_name = frames_dir.name.removesuffix("_frames")
    return frames_dir.parent / f"{frames_name}.mkv"


def load_capture_metadata(frames_dir: Path) -> dict[str, float | int] | None:
    metadata_path = capture_metadata_path(frames_dir)
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="ascii"))


def resolve_frame_rate_hz(frames_dir: Path, explicit_frame_rate_hz: float | None) -> float:
    if explicit_frame_rate_hz is not None:
        if explicit_frame_rate_hz <= 0.0:
            raise SystemExit("Frame rate must be positive.")
        return explicit_frame_rate_hz

    metadata = load_capture_metadata(frames_dir)
    if metadata is None or "frame_rate_hz" not in metadata:
        raise SystemExit("Frame rate was not provided and no capture_metadata.json with frame_rate_hz was found.")
    frame_rate_hz = float(metadata["frame_rate_hz"])
    if frame_rate_hz <= 0.0:
        raise SystemExit("Capture metadata contained a non-positive frame rate.")
    return frame_rate_hz


def validate_frame_sequence(frames_dir: Path) -> tuple[int, tuple[int, ...], list[int]]:
    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    if not frame_paths:
        raise SystemExit(f"No frame_*.png files found in {frames_dir}")

    sample_frame = cv2.imread(str(frame_paths[0]), cv2.IMREAD_UNCHANGED)
    if sample_frame is None:
        raise SystemExit(f"Failed to read frame {frame_paths[0]}")
    unique_values = np.unique(sample_frame).tolist()
    return len(frame_paths), sample_frame.shape, unique_values


def assemble_png_frames_to_video(
    frames_dir: Path,
    output_path: Path,
    frame_rate_hz: float,
) -> Path:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg is required to assemble the captured PNG frames into a video.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_path,
        "-y",
        "-framerate",
        f"{frame_rate_hz:.9f}",
        "-i",
        str(frames_dir / "frame_%06d.png"),
        "-c:v",
        "ffv1",
        "-pix_fmt",
        "gray",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(exc.stderr.strip() or "ffmpeg failed while assembling the output video.") from exc
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frame-png-dir",
        type=Path,
        required=True,
        help="Directory containing frame_*.png files and optional capture_metadata.json.",
    )
    parser.add_argument(
        "--video-output",
        type=Path,
        default=None,
        help="Output video path. Defaults to <frame_dir_name_without__frames>.mkv.",
    )
    parser.add_argument(
        "--control-rate-hz",
        type=float,
        default=None,
        help="Output video frame rate. If omitted, uses capture_metadata.json.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    frame_rate_hz = resolve_frame_rate_hz(args.frame_png_dir, args.control_rate_hz)
    frame_count, frame_shape, unique_values = validate_frame_sequence(args.frame_png_dir)
    output_path = args.video_output or default_video_output_path(args.frame_png_dir)
    written_video = assemble_png_frames_to_video(args.frame_png_dir, output_path, frame_rate_hz)

    print(f"Frame dir   : {args.frame_png_dir}")
    print(f"Video path  : {written_video}")
    print(f"Frames      : {frame_count}")
    print(f"Frame shape : {frame_shape}")
    print(f"Frame rate  : {frame_rate_hz:.3f} fps")
    print(f"Unique vals : {unique_values}")


if __name__ == "__main__":
    main()