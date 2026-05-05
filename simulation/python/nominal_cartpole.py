#!/usr/bin/env python3
"""Nominal Gym CartPole helpers exposed in native cartpole units.

The underlying plant is Gym's built-in CartPole, and the exported teacher and
student interfaces now use the same nominal unit system directly

    x = [cart_position_m, cart_velocity_m_s, pole_angle_rad, pole_angle_rate_rad_s]
    u = signed_force_n

This keeps the controller, teacher traces, and student observer in the same
meter/newton coordinates as the Gym CartPole dynamics that are already working.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_are


def load_gym_module():
    try:
        import gymnasium as gym_module
    except ImportError:
        try:
            import gym as gym_module
        except ImportError as exc:  # pragma: no cover - optional runtime dependency.
            raise ModuleNotFoundError(
                "gymnasium or gym is required for nominal CartPole rollouts"
            ) from exc
    return gym_module


@dataclass(frozen=True)
class NominalCartPoleParams:
    sample_time_s: float = 0.02
    gravity_m_s2: float = 9.8
    masscart_kg: float = 1.0
    masspole_kg: float = 0.1
    half_pole_length_m: float = 0.5
    max_force_n: float = 10.0
    x_threshold_m: float = 2.4
    theta_threshold_rad: float = math.radians(12.0)
    control_penalty_r: float = 1e-6

    @property
    def total_mass_kg(self) -> float:
        return self.masscart_kg + self.masspole_kg

    @property
    def polemass_length(self) -> float:
        return self.masspole_kg * self.half_pole_length_m


def scale_cartpole_params(
    params: NominalCartPoleParams,
    *,
    gravity_scale: float = 1.0,
    masscart_scale: float = 1.0,
    masspole_scale: float = 1.0,
    half_pole_length_scale: float = 1.0,
) -> NominalCartPoleParams:
    scale_factors = {
        "gravity_scale": gravity_scale,
        "masscart_scale": masscart_scale,
        "masspole_scale": masspole_scale,
        "half_pole_length_scale": half_pole_length_scale,
    }
    for scale_name, scale_value in scale_factors.items():
        if scale_value <= 0.0:
            raise ValueError(f"{scale_name} must be positive")

    return replace(
        params,
        gravity_m_s2=float(params.gravity_m_s2 * gravity_scale),
        masscart_kg=float(params.masscart_kg * masscart_scale),
        masspole_kg=float(params.masspole_kg * masspole_scale),
        half_pole_length_m=float(params.half_pole_length_m * half_pole_length_scale),
    )


def perturb_cartpole_params(
    params: NominalCartPoleParams,
    relative_fraction: float,
    rng: np.random.Generator,
) -> NominalCartPoleParams:
    if relative_fraction < 0.0:
        raise ValueError("relative_fraction must be non-negative")
    if relative_fraction == 0.0:
        return params

    def perturb(value: float) -> float:
        return float(value * (1.0 + rng.uniform(-relative_fraction, relative_fraction)))

    return replace(
        params,
        gravity_m_s2=perturb(params.gravity_m_s2),
        masscart_kg=perturb(params.masscart_kg),
        masspole_kg=perturb(params.masspole_kg),
        half_pole_length_m=perturb(params.half_pole_length_m),
    )


def _nominal_cartpole_discrete_step_meters(
    state_meters: np.ndarray,
    applied_force_n: float,
    params: NominalCartPoleParams,
) -> np.ndarray:
    x_m, x_dot_m_s, theta_rad, theta_dot_rad_s = np.asarray(state_meters, dtype=np.float64)
    force_n = float(np.clip(applied_force_n, -params.max_force_n, params.max_force_n))
    costheta = math.cos(theta_rad)
    sintheta = math.sin(theta_rad)

    temp = (force_n + params.polemass_length * (theta_dot_rad_s**2) * sintheta) / params.total_mass_kg
    thetaacc_rad_s2 = (params.gravity_m_s2 * sintheta - costheta * temp) / (
        params.half_pole_length_m
        * (4.0 / 3.0 - params.masspole_kg * (costheta**2) / params.total_mass_kg)
    )
    xacc_m_s2 = temp - params.polemass_length * thetaacc_rad_s2 * costheta / params.total_mass_kg

    next_x_m = x_m + (params.sample_time_s * x_dot_m_s)
    next_x_dot_m_s = x_dot_m_s + (params.sample_time_s * xacc_m_s2)
    next_theta_rad = theta_rad + (params.sample_time_s * theta_dot_rad_s)
    next_theta_dot_rad_s = theta_dot_rad_s + (params.sample_time_s * thetaacc_rad_s2)
    return np.array([next_x_m, next_x_dot_m_s, next_theta_rad, next_theta_dot_rad_s], dtype=np.float64)


def linearize_nominal_cartpole(
    params: NominalCartPoleParams,
    state_epsilon: float = 1e-6,
    force_epsilon: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    equilibrium_state_meters = np.zeros(4, dtype=np.float64)
    a_discrete_meters = np.zeros((4, 4), dtype=np.float64)
    b_discrete_meters = np.zeros((4, 1), dtype=np.float64)

    for state_index in range(4):
        perturbation = np.zeros(4, dtype=np.float64)
        perturbation[state_index] = state_epsilon
        forward = _nominal_cartpole_discrete_step_meters(
            equilibrium_state_meters + perturbation,
            0.0,
            params,
        )
        backward = _nominal_cartpole_discrete_step_meters(
            equilibrium_state_meters - perturbation,
            0.0,
            params,
        )
        a_discrete_meters[:, state_index] = (forward - backward) / (2.0 * state_epsilon)

    forward_force = _nominal_cartpole_discrete_step_meters(equilibrium_state_meters, force_epsilon, params)
    backward_force = _nominal_cartpole_discrete_step_meters(equilibrium_state_meters, -force_epsilon, params)
    b_discrete_meters[:, 0] = (forward_force - backward_force) / (2.0 * force_epsilon)

    return a_discrete_meters, b_discrete_meters


def choose_cost_matrices(params: NominalCartPoleParams) -> tuple[np.ndarray, np.ndarray]:
    q_matrix_meters = np.diag(
        [
            2.0 / (params.x_threshold_m**2),
            0.5 / (3.0**2),
            12.0 / (params.theta_threshold_rad**2),
            1.5 / (math.radians(120.0) ** 2),
        ]
    )
    r_matrix_newtons = np.array([[params.control_penalty_r]], dtype=np.float64)
    return q_matrix_meters, r_matrix_newtons


def solve_discrete_lqr(
    a_discrete: np.ndarray,
    b_discrete: np.ndarray,
    q_matrix: np.ndarray,
    r_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    riccati_solution = solve_discrete_are(a_discrete, b_discrete, q_matrix, r_matrix)
    gain = np.linalg.solve(
        r_matrix + b_discrete.T @ riccati_solution @ b_discrete,
        b_discrete.T @ riccati_solution @ a_discrete,
    )
    closed_loop_poles = np.linalg.eigvals(a_discrete - b_discrete @ gain)
    return gain, riccati_solution, closed_loop_poles


def verify_sign_conventions(gain: np.ndarray) -> dict[str, float | bool]:
    sign_report: dict[str, float | bool] = {
        "positive_theta_commands_positive_force": float(-gain[0, 2]) > 0.0,
        "positive_theta_rate_commands_positive_force": float(-gain[0, 3]) > 0.0,
        "theta_only_command_force_n_per_rad": float(-gain[0, 2]),
        "theta_rate_only_command_force_n_per_rad_s": float(-gain[0, 3]),
    }
    if not sign_report["positive_theta_commands_positive_force"]:
        raise SystemExit(
            "Sign check failed: positive theta would command negative force under u = -Kx."
        )
    if not sign_report["positive_theta_rate_commands_positive_force"]:
        raise SystemExit(
            "Sign check failed: positive theta_dot would command negative force under u = -Kx."
        )
    return sign_report


def sample_initial_angle_deg(
    rng: np.random.Generator,
    fixed_angle_deg: float | None,
    angle_range_deg: float,
) -> float:
    if fixed_angle_deg is not None:
        return ((fixed_angle_deg + 180.0) % 360.0) - 180.0

    sampled = 0.0
    while abs(sampled) < 0.25:
        sampled = float(rng.uniform(-angle_range_deg, angle_range_deg))
    return sampled


def create_cartpole_env(render_mode: str | None, params: NominalCartPoleParams):
    gym_module = load_gym_module()
    env = gym_module.make("CartPole-v1", render_mode=render_mode)
    base_env = env.unwrapped
    base_env.gravity = params.gravity_m_s2
    base_env.masscart = params.masscart_kg
    base_env.masspole = params.masspole_kg
    base_env.total_mass = params.total_mass_kg
    base_env.length = params.half_pole_length_m
    base_env.polemass_length = params.polemass_length
    base_env.tau = params.sample_time_s
    base_env.force_mag = params.max_force_n
    base_env.theta_threshold_radians = params.theta_threshold_rad
    base_env.x_threshold = params.x_threshold_m
    env.reset(seed=0)
    base_env.steps_beyond_terminated = None
    return env


def set_cartpole_state(env, state_meters: np.ndarray, params: NominalCartPoleParams) -> None:
    base_env = env.unwrapped
    base_env.state = np.asarray(state_meters, dtype=np.float64).reshape(4)
    base_env.steps_beyond_terminated = None


def step_cartpole_env(
    env,
    commanded_force_n: float,
    params: NominalCartPoleParams,
    process_noise_std_n: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, bool, dict[str, float | bool]]:
    base_env = env.unwrapped
    commanded_control_force_n = float(np.clip(commanded_force_n, -params.max_force_n, params.max_force_n))
    process_noise_force_n = 0.0
    if process_noise_std_n > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        process_noise_force_n = float(rng.normal(0.0, process_noise_std_n))
    noisy_control_force_n = commanded_control_force_n + process_noise_force_n
    applied_force_n = float(np.clip(noisy_control_force_n, -params.max_force_n, params.max_force_n))
    base_env.force_mag = abs(applied_force_n)
    action = 1 if applied_force_n >= 0.0 else 0
    _observation, _reward, terminated, truncated, _info = base_env.step(action)
    next_state_meters = np.asarray(base_env.state, dtype=np.float64).reshape(4)
    command_clipped = abs(commanded_control_force_n - float(commanded_force_n)) > 1e-9
    applied_force_clipped = abs(applied_force_n - noisy_control_force_n) > 1e-9
    return next_state_meters, bool(terminated or truncated), {
        "force_clipped": bool(command_clipped or applied_force_clipped),
        "command_clipped": command_clipped,
        "applied_force_clipped": applied_force_clipped,
        "commanded_control_force_n": commanded_control_force_n,
        "process_noise_force_n": process_noise_force_n,
        "noisy_control_force_n": noisy_control_force_n,
        "applied_force_n": applied_force_n,
    }


def render_hold_loop(env, hold_open_seconds: float) -> None:
    end_time = None if hold_open_seconds < 0.0 else time.monotonic() + hold_open_seconds
    try:
        while end_time is None or time.monotonic() < end_time:
            env.render()
            time.sleep(1.0 / 30.0)
    except KeyboardInterrupt:
        pass


def build_teacher_controller_payload(
    params: NominalCartPoleParams,
    q_matrix: np.ndarray,
    r_matrix: np.ndarray,
    gain: np.ndarray,
    closed_loop_poles: np.ndarray,
    sign_report: dict[str, float | bool],
    a_discrete: np.ndarray,
    b_discrete: np.ndarray,
) -> dict[str, object]:
    return {
        "model": "gym_cartpole_v1_nominal_cartpole_units",
        "unit_system": "cartpole",
        "state_order": [
            "cart_position_m",
            "cart_velocity_m_s",
            "pole_angle_rad",
            "pole_angle_rate_rad_s",
        ],
        "input_units": "newtons",
        "control_law": "u = -Kx",
        "control_mode": "signed_force_command",
        "control_rate_hz": float(1.0 / params.sample_time_s),
        "sample_time_s": float(params.sample_time_s),
        "params": asdict(params),
        "Q_diag": np.diag(q_matrix).astype(float).tolist(),
        "R": float(r_matrix[0, 0]),
        "K": np.asarray(gain, dtype=np.float64).reshape(-1).astype(float).tolist(),
        "A": np.asarray(a_discrete, dtype=np.float64).astype(float).tolist(),
        "B": np.asarray(b_discrete, dtype=np.float64).astype(float).tolist(),
        "closed_loop_poles": [
            {
                "real": float(np.real(pole)),
                "imag": float(np.imag(pole)),
                "magnitude": float(abs(pole)),
            }
            for pole in closed_loop_poles
        ],
        "closed_loop_pole_magnitudes": np.abs(closed_loop_poles).astype(float).tolist(),
        "sign_report": {
            key: (bool(value) if isinstance(value, (bool, np.bool_)) else float(value))
            for key, value in sign_report.items()
        },
    }


def write_teacher_controller_json(
    output_path: Path,
    params: NominalCartPoleParams,
    q_matrix: np.ndarray,
    r_matrix: np.ndarray,
    gain: np.ndarray,
    closed_loop_poles: np.ndarray,
    sign_report: dict[str, float | bool],
    a_discrete: np.ndarray,
    b_discrete: np.ndarray,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_teacher_controller_payload(
        params,
        q_matrix,
        r_matrix,
        gain,
        closed_loop_poles,
        sign_report,
        a_discrete,
        b_discrete,
    )
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path
