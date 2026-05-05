#!/usr/bin/env python3
"""
High-FPS video capture utility for the Arducam VGA global shutter USB camera.

Usage:
	python camera.py
	python camera.py --source 1 --fps 800 --duration 10 --output ../captures/run.avi
	python camera.py --camera-fourcc NONE --video-fourcc MJPG --frames 5000

The script requests a small frame size and uses grab()/retrieve() so the capture
loop stays as close as possible to the freshest frame. A background writer
thread records the video and a sidecar CSV with monotonic timestamps for each
captured frame.
"""

from __future__ import annotations

import argparse
import csv
import queue
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

try:
	import cv2
except ImportError as exc:  # pragma: no cover - import error is user-environment specific.
	raise SystemExit(
		"OpenCV is required. Install dependencies with `pip install -r python/requirements.txt`."
	) from exc


DEFAULT_WIDTH = 160
DEFAULT_HEIGHT = 120
DEFAULT_FPS = 800.0
DEFAULT_CAMERA_FOURCC = "MJPG"
DEFAULT_VIDEO_FOURCC = "MJPG"
DEFAULT_QUEUE_SIZE = 512
DEFAULT_WARMUP_S = 0.5
STATUS_INTERVAL_S = 1.0
GRAB_RETRY_S = 0.001

BACKENDS = {
	"any": cv2.CAP_ANY,
	"avfoundation": getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY),
	"v4l2": getattr(cv2, "CAP_V4L2", cv2.CAP_ANY),
	"dshow": getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
}


@dataclass
class FramePacket:
	index: int
	timestamp_ns: int
	frame: Any


class AsyncVideoWriter:
	def __init__(
		self,
		output_path: Path,
		container_fps: float,
		video_fourcc: str,
		queue_size: int,
		drop_if_lagging: bool,
	) -> None:
		self.output_path = output_path
		self.container_fps = container_fps
		self.video_fourcc = video_fourcc
		self.drop_if_lagging = drop_if_lagging
		self.frame_queue: queue.Queue[object] = queue.Queue(maxsize=queue_size)
		self.timestamps_path = output_path.with_name(f"{output_path.stem}_timestamps.csv")
		self._thread = threading.Thread(target=self._run, name="video-writer", daemon=True)
		self._started = False
		self._writer: Optional[cv2.VideoWriter] = None
		self._csv_file: Optional[Any] = None
		self._csv_writer: Optional[csv.writer] = None
		self._stop_sentinel = object()
		self._exception: Optional[BaseException] = None
		self._last_timestamp_ns: Optional[int] = None
		self.written_frames = 0
		self.dropped_frames = 0

	def start(self) -> None:
		self.output_path.parent.mkdir(parents=True, exist_ok=True)
		self._csv_file = self.timestamps_path.open("w", newline="")
		self._csv_writer = csv.writer(self._csv_file)
		self._csv_writer.writerow(("frame_index", "timestamp_ns", "delta_us"))
		self._thread.start()
		self._started = True

	def enqueue(self, packet: FramePacket) -> bool:
		self._raise_if_failed()
		item = (packet.index, packet.timestamp_ns, packet.frame)

		if self.drop_if_lagging:
			try:
				self.frame_queue.put_nowait(item)
				return True
			except queue.Full:
				self.dropped_frames += 1
				return False

		self.frame_queue.put(item)
		return True

	def queue_depth(self) -> int:
		return self.frame_queue.qsize()

	def close(self) -> None:
		if self._started and self._thread.is_alive():
			self.frame_queue.put(self._stop_sentinel)
			self._thread.join()

		if self._writer is not None:
			self._writer.release()
			self._writer = None

		if self._csv_file is not None:
			self._csv_file.flush()
			self._csv_file.close()
			self._csv_file = None

		self._raise_if_failed()

	def _run(self) -> None:
		try:
			while True:
				item = self.frame_queue.get()
				if item is self._stop_sentinel:
					break

				frame_index, timestamp_ns, frame = item
				self._write_frame(frame_index, timestamp_ns, frame)
		except BaseException as exc:  # pragma: no cover - surfaced to caller in close().
			self._exception = exc

	def _write_frame(self, frame_index: int, timestamp_ns: int, frame: Any) -> None:
		if self._writer is None:
			self._writer = self._open_writer(frame)

		if frame.ndim == 2:
			frame_to_write = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
		else:
			frame_to_write = frame

		self._writer.write(frame_to_write)

		delta_us = 0.0
		if self._last_timestamp_ns is not None:
			delta_us = (timestamp_ns - self._last_timestamp_ns) / 1_000.0
		self._last_timestamp_ns = timestamp_ns

		if self._csv_writer is not None:
			self._csv_writer.writerow((frame_index, timestamp_ns, f"{delta_us:.3f}"))

		self.written_frames += 1

	def _open_writer(self, frame: Any) -> cv2.VideoWriter:
		height, width = frame.shape[:2]
		fourcc = cv2.VideoWriter_fourcc(*self.video_fourcc)
		writer = cv2.VideoWriter(
			str(self.output_path),
			fourcc,
			self.container_fps,
			(width, height),
			True,
		)
		if not writer.isOpened():
			raise RuntimeError(
				f"Could not open video writer for {self.output_path} with codec {self.video_fourcc}."
			)
		return writer

	def _raise_if_failed(self) -> None:
		if self._exception is not None:
			raise RuntimeError(f"Background writer failed: {self._exception}") from self._exception


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Capture the fastest video stream OpenCV can get from a UVC camera.")
	parser.add_argument("--source", default="0", help="Camera index or device path. Defaults to 0.")
	parser.add_argument(
		"--backend",
		choices=sorted(BACKENDS.keys()),
		default="avfoundation" if sys.platform == "darwin" else "any",
		help="OpenCV capture backend.",
	)
	parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Requested capture width in pixels.")
	parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Requested capture height in pixels.")
	parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Requested camera FPS.")
	parser.add_argument(
		"--camera-fourcc",
		default=DEFAULT_CAMERA_FOURCC,
		help="Requested camera stream FOURCC, for example MJPG, YUYV, GREY, Y800, or NONE.",
	)
	parser.add_argument(
		"--video-fourcc",
		default=DEFAULT_VIDEO_FOURCC,
		help="Output video FOURCC. MJPG is the safest default for high-rate capture.",
	)
	parser.add_argument(
		"--output",
		default=None,
		help="Video output path. Defaults to hardware/captures/capture_YYYYmmdd_HHMMSS.avi.",
	)
	parser.add_argument("--duration", type=float, default=0.0, help="Capture duration in seconds. 0 means until Ctrl+C.")
	parser.add_argument("--frames", type=int, default=0, help="Stop after this many captured frames. 0 means unlimited.")
	parser.add_argument("--warmup", type=float, default=DEFAULT_WARMUP_S, help="Seconds to warm up the stream before recording.")
	parser.add_argument(
		"--queue-size",
		type=int,
		default=DEFAULT_QUEUE_SIZE,
		help="Background writer queue depth in frames.",
	)
	parser.add_argument(
		"--drop-if-lagging",
		action="store_true",
		help="Drop frames if the writer queue fills instead of blocking capture.",
	)
	parser.add_argument(
		"--preview",
		action="store_true",
		help="Show a live preview window. This can reduce maximum capture FPS.",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Overwrite an existing output path if it already exists.",
	)
	return parser.parse_args()


def parse_source(source: str) -> Union[int, str]:
	try:
		return int(source)
	except ValueError:
		return source


def resolve_output_path(explicit_path: Optional[str]) -> Path:
	if explicit_path:
		return Path(explicit_path).expanduser().resolve()

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	return (Path(__file__).resolve().parents[1] / "captures" / f"capture_{timestamp}.avi").resolve()


def fourcc_to_string(raw_value: float) -> str:
	value = int(raw_value)
	if value <= 0:
		return "unknown"

	chars = [chr((value >> (8 * offset)) & 0xFF) for offset in range(4)]
	if all(32 <= ord(char) <= 126 for char in chars):
		return "".join(chars)
	return f"0x{value:08x}"


def normalize_frame(frame: Any) -> Any:
	if frame.ndim == 2:
		return frame

	if frame.ndim == 3 and frame.shape[2] == 1:
		return frame[:, :, 0]

	if frame.ndim == 3 and frame.shape[2] == 2:
		return cv2.cvtColor(frame, cv2.COLOR_YUV2GRAY_YUY2)

	return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def configure_capture(cap: cv2.VideoCapture, width: int, height: int, fps: float, camera_fourcc: str) -> None:
	if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
		cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
	if hasattr(cv2, "CAP_PROP_CONVERT_RGB"):
		cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
	if hasattr(cv2, "CAP_PROP_MONOCHROME"):
		cap.set(cv2.CAP_PROP_MONOCHROME, 1)

	normalized_fourcc = camera_fourcc.strip().upper()
	if normalized_fourcc and normalized_fourcc != "NONE":
		cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*normalized_fourcc))

	cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
	cap.set(cv2.CAP_PROP_FPS, fps)


def warmup_capture(cap: cv2.VideoCapture, warmup_s: float) -> None:
	if warmup_s <= 0.0:
		return

	deadline = time.perf_counter() + warmup_s
	while time.perf_counter() < deadline:
		if not cap.grab():
			time.sleep(GRAB_RETRY_S)
			continue
		cap.retrieve()


def print_stream_summary(cap: cv2.VideoCapture, fallback_fps: float) -> float:
	actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	reported_fps = cap.get(cv2.CAP_PROP_FPS)
	actual_fourcc = fourcc_to_string(cap.get(cv2.CAP_PROP_FOURCC))

	print(
		"Opened stream: "
		f"{actual_width}x{actual_height}, "
		f"reported_fps={reported_fps:.2f}, "
		f"fourcc={actual_fourcc}",
		flush=True,
	)
	print("Measured FPS below is based on frame timestamps, not the reported backend FPS.", flush=True)
	return reported_fps if reported_fps > 0.0 else fallback_fps


def run_capture(args: argparse.Namespace) -> int:
	source = parse_source(args.source)
	output_path = resolve_output_path(args.output)
	if output_path.exists() and not args.overwrite:
		raise RuntimeError(f"Refusing to overwrite existing file: {output_path}")

	backend = BACKENDS[args.backend]
	cap = cv2.VideoCapture(source, backend)
	if not cap.isOpened():
		raise RuntimeError(
			f"Could not open camera source {source!r} with backend {args.backend}."
		)

	try:
		configure_capture(cap, args.width, args.height, args.fps, args.camera_fourcc)
		warmup_capture(cap, args.warmup)
		container_fps = print_stream_summary(cap, args.fps)

		writer = AsyncVideoWriter(
			output_path=output_path,
			container_fps=container_fps,
			video_fourcc=args.video_fourcc.strip().upper(),
			queue_size=max(args.queue_size, 1),
			drop_if_lagging=args.drop_if_lagging,
		)
		writer.start()

		print(f"Writing video to {output_path}", flush=True)
		print(f"Writing timestamps to {writer.timestamps_path}", flush=True)
		print("Press Ctrl+C to stop capture.", flush=True)

		frame_count = 0
		failed_grabs = 0
		capture_start_ns: Optional[int] = None
		last_status_t = time.perf_counter()
		last_status_frame_count = 0

		while True:
			if not cap.grab():
				failed_grabs += 1
				if failed_grabs >= 100:
					raise RuntimeError("Too many failed frame grabs from the camera.")
				time.sleep(GRAB_RETRY_S)
				continue

			failed_grabs = 0
			ok, frame = cap.retrieve()
			if not ok:
				continue

			timestamp_ns = time.perf_counter_ns()
			if capture_start_ns is None:
				capture_start_ns = timestamp_ns

			mono_frame = normalize_frame(frame)
			writer.enqueue(FramePacket(index=frame_count, timestamp_ns=timestamp_ns, frame=mono_frame.copy()))
			frame_count += 1

			if args.preview:
				cv2.imshow("camera", mono_frame)
				if cv2.waitKey(1) & 0xFF == 27:
					break

			now_t = time.perf_counter()
			if now_t - last_status_t >= STATUS_INTERVAL_S:
				assert capture_start_ns is not None
				elapsed_s = (timestamp_ns - capture_start_ns) / 1_000_000_000.0
				interval_s = max(now_t - last_status_t, 1e-6)
				interval_frames = frame_count - last_status_frame_count
				print(
					f"Captured={frame_count} avg_fps={frame_count / max(elapsed_s, 1e-6):.1f} "
					f"interval_fps={interval_frames / interval_s:.1f} "
					f"writer_queue={writer.queue_depth()} dropped={writer.dropped_frames}",
					flush=True,
				)
				last_status_t = now_t
				last_status_frame_count = frame_count

			if args.frames > 0 and frame_count >= args.frames:
				break
			if args.duration > 0.0 and capture_start_ns is not None:
				elapsed_s = (timestamp_ns - capture_start_ns) / 1_000_000_000.0
				if elapsed_s >= args.duration:
					break

	except KeyboardInterrupt:
		print("Stopping capture on Ctrl+C.", flush=True)
	finally:
		if "writer" in locals():
			writer.close()
		cap.release()
		if args.preview:
			cv2.destroyAllWindows()

	if capture_start_ns is None:
		raise RuntimeError("No frames were captured.")

	capture_end_ns = time.perf_counter_ns()
	total_elapsed_s = (capture_end_ns - capture_start_ns) / 1_000_000_000.0
	measured_fps = frame_count / max(total_elapsed_s, 1e-6)

	print(
		f"Finished: captured={frame_count} written={writer.written_frames} "
		f"dropped={writer.dropped_frames} measured_fps={measured_fps:.1f}",
		flush=True,
	)
	return 0


def main() -> None:
	args = parse_args()
	try:
		raise SystemExit(run_capture(args))
	except RuntimeError as exc:
		print(f"ERROR: {exc}", file=sys.stderr)
		raise SystemExit(1) from exc


if __name__ == "__main__":
	main()
