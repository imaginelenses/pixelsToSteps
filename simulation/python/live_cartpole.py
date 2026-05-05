from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

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
from teacher_policy import extract_binary_pole_frame, load_observer_policy, pad_binary_frame_to_shape
from teacher_policy import estimate_cart_position_from_binary_frame


@dataclass
class SimulationConfig:
    sample_time_s: float = 1.0 / 60.0
    max_force_n: float = 10.0
    control_penalty_r: float = 1e-6
    frame_width_px: int = 160
    frame_height_px: int = 125
    true_gravity_scale: float = 1.0
    true_masscart_scale: float = 1.0
    true_masspole_scale: float = 1.1
    true_half_pole_length_scale: float = 0.9
    process_noise_std_n: float = 0.0
    seed_truth_from_initial_state: bool = True


@dataclass
class ResetRequest:
    cart_position_m: float = 0.0
    cart_velocity_m_s: float = 0.0
    pole_angle_deg: float = 12.0
    pole_angle_rate_deg_s: float = 0.0
    use_image_controller: bool = True
    observer_json_path: str | None = None
    auto_start: bool = False


class LiveCartPoleSimulator:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.python_dir = base_dir / "python"
        self.config = SimulationConfig()
        self.lock = threading.RLock()
        self.frame_condition = threading.Condition(self.lock)
        self.thread: threading.Thread | None = None
        self.shutdown_requested = False
        self.running = False
        self.frame_counter = 0
        self.nominal_params: NominalCartPoleParams | None = None
        self.true_params: NominalCartPoleParams | None = None
        self.controller_gain: np.ndarray | None = None
        self.observer_policy: dict[str, object] | None = None
        self.capture_env = None
        self.state_m = np.zeros(4, dtype=np.float64)
        self.estimated_state_m: np.ndarray | None = None
        self.last_raw_force_n = 0.0
        self.last_applied_force_n = 0.0
        self.last_commanded_force_n = 0.0
        self.last_theta_pixel_rad: float | None = None
        self.last_step_index = 0
        self.last_simulated_time_s = 0.0
        self.capture_end_reason = "idle"
        self.terminated = False
        self.latest_rgb_jpeg = b""
        self.latest_binary_jpeg = b""
        self.process_noise_rng = np.random.default_rng(0)
        self.last_reset_request = ResetRequest()
        self._configure_locked()
        self.reset(ResetRequest())
        self.start_background_loop()

    def available_observers(self) -> list[str]:
        return sorted(path.name for path in self.python_dir.glob("*.json"))

    def start_background_loop(self) -> None:
        with self.lock:
            if self.thread is not None:
                return
            self.thread = threading.Thread(target=self._run_loop, name="live-cartpole-sim", daemon=True)
            self.thread.start()

    def close(self) -> None:
        with self.lock:
            self.shutdown_requested = True
            self.running = False
            self.frame_condition.notify_all()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        with self.lock:
            if self.capture_env is not None:
                self.capture_env.close()
                self.capture_env = None

    def reconfigure(self, config: SimulationConfig) -> None:
        with self.lock:
            self.config = config
            current_state = self.state_m.copy()
            use_image_controller = self.observer_policy is not None
            observer_json_path = None if self.observer_policy is None else str(self.observer_policy["path"])
            self._configure_locked()
            self._reset_locked(current_state, use_image_controller, observer_json_path)

    def reset(self, request: ResetRequest) -> None:
        initial_state = np.array(
            [
                request.cart_position_m,
                request.cart_velocity_m_s,
                math.radians(request.pole_angle_deg),
                math.radians(request.pole_angle_rate_deg_s),
            ],
            dtype=np.float64,
        )
        with self.lock:
            self.last_reset_request = request
            self._reset_locked(initial_state, request.use_image_controller, request.observer_json_path)
            self.running = bool(request.auto_start)
            self.frame_condition.notify_all()

    def restart(self) -> None:
        request = replace(self.last_reset_request, auto_start=True)
        self.reset(request)

    def start(self) -> None:
        with self.lock:
            self.running = True
            self.capture_end_reason = "running"
            self.frame_condition.notify_all()

    def stop(self) -> None:
        with self.lock:
            self.running = False
            if not self.terminated:
                self.capture_end_reason = "paused"
            self.frame_condition.notify_all()

    def step_once(self) -> None:
        with self.lock:
            self._step_locked()
            self.frame_condition.notify_all()

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            estimated = None
            if self.estimated_state_m is not None:
                estimated = self._state_dict(self.estimated_state_m)
            return {
                "running": self.running,
                "terminated": self.terminated,
                "capture_end_reason": self.capture_end_reason,
                "frame_index": self.last_step_index,
                "simulated_time_s": self.last_simulated_time_s,
                "control": {
                    "raw_force_n": self.last_raw_force_n,
                    "commanded_force_n": self.last_commanded_force_n,
                    "applied_force_n": self.last_applied_force_n,
                },
                "ground_truth": self._state_dict(self.state_m),
                "estimated": estimated,
                "theta_pixel_deg": None if self.last_theta_pixel_rad is None else math.degrees(self.last_theta_pixel_rad),
                "observer": None if self.observer_policy is None else {
                    "path": Path(self.observer_policy["path"]).name,
                    "target": self.observer_policy["observer_target"],
                    "theta_pixel_blend_weight": self.observer_policy.get("theta_pixel_blend_weight", 0.0),
                },
                "config": {
                    "sample_time_s": self.config.sample_time_s,
                    "frame_width_px": self.config.frame_width_px,
                    "frame_height_px": self.config.frame_height_px,
                    "true_gravity_scale": self.config.true_gravity_scale,
                    "true_masscart_scale": self.config.true_masscart_scale,
                    "true_masspole_scale": self.config.true_masspole_scale,
                    "true_half_pole_length_scale": self.config.true_half_pole_length_scale,
                    "process_noise_std_n": self.config.process_noise_std_n,
                },
            }

    def wait_for_frame(self, last_frame_counter: int, timeout_s: float = 1.0) -> int:
        with self.lock:
            self.frame_condition.wait_for(lambda: self.frame_counter != last_frame_counter or self.shutdown_requested, timeout=timeout_s)
            return self.frame_counter

    def get_rgb_jpeg(self) -> bytes:
        with self.lock:
            return self.latest_rgb_jpeg

    def get_binary_jpeg(self) -> bytes:
        with self.lock:
            return self.latest_binary_jpeg

    def _configure_locked(self) -> None:
        nominal_params = NominalCartPoleParams(
            sample_time_s=self.config.sample_time_s,
            max_force_n=self.config.max_force_n,
            control_penalty_r=self.config.control_penalty_r,
        )
        true_params = scale_cartpole_params(
            nominal_params,
            gravity_scale=self.config.true_gravity_scale,
            masscart_scale=self.config.true_masscart_scale,
            masspole_scale=self.config.true_masspole_scale,
            half_pole_length_scale=self.config.true_half_pole_length_scale,
        )
        a_discrete, b_discrete = linearize_nominal_cartpole(nominal_params)
        q_matrix, r_matrix = choose_cost_matrices(nominal_params)
        gain, _riccati_solution, _poles = solve_discrete_lqr(a_discrete, b_discrete, q_matrix, r_matrix)
        self.nominal_params = nominal_params
        self.true_params = true_params
        self.controller_gain = gain
        self.process_noise_rng = np.random.default_rng(0)
        if self.capture_env is not None:
            self.capture_env.close()
        self.capture_env = create_cartpole_env("rgb_array", true_params)

    def _default_observer_path(self) -> str | None:
        candidate = self.python_dir / "hybrid_pixels_to_cartpole_observer_theta_blend_0p7.json"
        if candidate.exists():
            return str(candidate)
        observers = self.available_observers()
        if not observers:
            return None
        return str(self.python_dir / observers[0])

    def _reset_locked(self, initial_state: np.ndarray, use_image_controller: bool, observer_json_path: str | None) -> None:
        assert self.capture_env is not None
        assert self.nominal_params is not None
        assert self.true_params is not None
        set_cartpole_state(self.capture_env, initial_state, self.true_params)
        self.state_m = initial_state.copy()
        self.last_raw_force_n = 0.0
        self.last_applied_force_n = 0.0
        self.last_commanded_force_n = 0.0
        self.last_theta_pixel_rad = None
        self.last_step_index = 0
        self.last_simulated_time_s = 0.0
        self.terminated = False
        self.capture_end_reason = "ready"
        observer_path_to_load = observer_json_path or self._default_observer_path()
        self.observer_policy = None
        self.estimated_state_m = None
        if use_image_controller and observer_path_to_load is not None:
            self.observer_policy = load_observer_policy(Path(observer_path_to_load))
            if self.config.seed_truth_from_initial_state:
                self.estimated_state_m = initial_state.copy()
            else:
                self.estimated_state_m = np.zeros(4, dtype=np.float64)
            self.controller_gain = np.asarray(self.observer_policy["teacher_gain"], dtype=np.float64).reshape(1, 4)
        else:
            a_discrete, b_discrete = linearize_nominal_cartpole(self.nominal_params)
            q_matrix, r_matrix = choose_cost_matrices(self.nominal_params)
            self.controller_gain, _riccati_solution, _poles = solve_discrete_lqr(a_discrete, b_discrete, q_matrix, r_matrix)
        self._render_locked()

    def _run_loop(self) -> None:
        while True:
            with self.lock:
                if self.shutdown_requested:
                    return
                if not self.running:
                    self.frame_condition.wait(timeout=0.1)
                    continue
                step_started_at = time.monotonic()
                self._step_locked()
                self.frame_condition.notify_all()
                sample_time_s = self.config.sample_time_s
            elapsed_s = time.monotonic() - step_started_at
            remaining_s = sample_time_s - elapsed_s
            if remaining_s > 0.0:
                time.sleep(remaining_s)

    def _step_locked(self) -> None:
        assert self.capture_env is not None
        assert self.true_params is not None
        assert self.controller_gain is not None
        if self.terminated:
            self.running = False
            return
        control_state = self.estimated_state_m if self.estimated_state_m is not None else self.state_m
        raw_force_n = float(-(self.controller_gain @ control_state)[0])
        state_m, terminated, step_stats = step_cartpole_env(
            self.capture_env,
            raw_force_n,
            self.true_params,
            process_noise_std_n=self.config.process_noise_std_n,
            rng=self.process_noise_rng,
        )
        self.state_m = state_m.copy()
        self.last_raw_force_n = raw_force_n
        self.last_commanded_force_n = float(step_stats["commanded_control_force_n"])
        self.last_applied_force_n = float(step_stats["applied_force_n"])
        self.last_step_index += 1
        self.last_simulated_time_s = self.last_step_index * self.true_params.sample_time_s
        self.terminated = bool(terminated)
        self.capture_end_reason = "failure" if self.terminated else "running"
        self._render_locked()
        if self.estimated_state_m is not None and self.observer_policy is not None:
            measurement_vector = self._binary_measurement_vector()
            observer_a = np.asarray(self.observer_policy["A_L"], dtype=np.float64)
            observer_b = np.asarray(self.observer_policy["B_L"], dtype=np.float64).reshape(4, 1)
            observer_l = np.asarray(self.observer_policy["L"], dtype=np.float64)
            observer_d = np.asarray(self.observer_policy["d"], dtype=np.float64).reshape(4)
            predicted_state_m = (
                observer_a @ self.estimated_state_m
                + (observer_b[:, 0] * self.last_commanded_force_n)
                + (observer_l @ measurement_vector)
                + observer_d
            )
            theta_pixel_coefficients = self.observer_policy.get("theta_pixel_coefficients")
            theta_pixel_bias = self.observer_policy.get("theta_pixel_bias")
            theta_pixel_blend_weight = float(self.observer_policy.get("theta_pixel_blend_weight", 0.0))
            theta_dot_blend_weight = float(self.observer_policy.get("theta_dot_blend_weight", 0.0))
            cart_pixel_blend_weight = float(self.observer_policy.get("cart_pixel_blend_weight", 0.0))
            self.last_theta_pixel_rad = None
            if theta_pixel_coefficients is not None and theta_pixel_bias is not None:
                previous_estimated_state_m = self.estimated_state_m.copy()
                theta_pixel = float(theta_pixel_coefficients @ measurement_vector + theta_pixel_bias)
                predicted_state_m[2] = (
                    (1.0 - theta_pixel_blend_weight) * predicted_state_m[2]
                    + theta_pixel_blend_weight * theta_pixel
                )
                if theta_dot_blend_weight > 0.0:
                    theta_dot_corrected = (
                        predicted_state_m[2] - previous_estimated_state_m[2]
                    ) / self.true_params.sample_time_s
                    predicted_state_m[3] = (
                        (1.0 - theta_dot_blend_weight) * predicted_state_m[3]
                        + theta_dot_blend_weight * theta_dot_corrected
                    )
                self.last_theta_pixel_rad = theta_pixel
                if cart_pixel_blend_weight > 0.0:
                    cart_pos_pixel = estimate_cart_position_from_binary_frame(self._last_binary_frame)
                    if cart_pos_pixel is not None:
                        predicted_state_m[0] = (
                            (1.0 - cart_pixel_blend_weight) * predicted_state_m[0]
                            + cart_pixel_blend_weight * cart_pos_pixel
                        )
            self.estimated_state_m = predicted_state_m
        self.frame_counter += 1
        if self.terminated:
            self.running = False

    def _render_locked(self) -> None:
        rendered = self.capture_env.render()
        if rendered is None:
            raise RuntimeError("Gym did not return a frame for rgb_array rendering.")
        rgb_frame = np.asarray(rendered, dtype=np.uint8)
        binary_frame = extract_binary_pole_frame(rgb_frame, None, None)
        if self.observer_policy is not None:
            binary_frame = pad_binary_frame_to_shape(
                binary_frame,
                int(self.observer_policy["frame_width_px"]),
                int(self.observer_policy["frame_height_px"]),
            )
            self._last_binary_frame = binary_frame
        rgb_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        rgb_ok, rgb_encoded = cv2.imencode(".jpg", rgb_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        binary_ok, binary_encoded = cv2.imencode(".jpg", binary_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not rgb_ok or not binary_ok:
            raise RuntimeError("Failed to encode simulation frames for streaming.")
        self.latest_rgb_jpeg = rgb_encoded.tobytes()
        self.latest_binary_jpeg = binary_encoded.tobytes()
        self._latest_binary_frame = binary_frame

    def _binary_measurement_vector(self) -> np.ndarray:
        return self._latest_binary_frame.astype(np.float64).reshape(-1) / 255.0

    @staticmethod
    def _state_dict(state: np.ndarray) -> dict[str, float]:
        return {
            "cart_position_m": float(state[0]),
            "cart_velocity_m_s": float(state[1]),
            "pole_angle_rad": float(state[2]),
            "pole_angle_deg": math.degrees(float(state[2])),
            "pole_angle_rate_rad_s": float(state[3]),
            "pole_angle_rate_deg_s": math.degrees(float(state[3])),
        }
