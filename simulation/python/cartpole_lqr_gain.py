#!/usr/bin/env python3
"""Compute a step-domain LQR gain for the cart-pole and visualize it with Gym.

The state matches the current firmware teacher controller:
    x = [cart_position_steps, cart_velocity_steps_s, pole_angle_rad, pole_angle_rate_rad_s]
    u = signed_step_rate_steps_s

The plant uses a first-order actuator model for the cart velocity target and a
validation-only acceleration limit. This keeps the design in the current
firmware state coordinates while accounting for the stepper's inability to jump
to arbitrary step rates instantaneously.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete


def load_gym_module():
    try:
        import gymnasium as gym_module
    except ImportError:
        try:
            import gym as gym_module
        except ImportError as exc:  # pragma: no cover - optional runtime dependency.
            raise ModuleNotFoundError(
                "gymnasium or gym is required only for visualization in cartpole_lqr_gain.py"
            ) from exc
    return gym_module


@dataclass(frozen=True)
class PhysicalCartPoleParams:
    sample_time_s: float = 1.0 / 250.0
    gravity_m_s2: float = 9.81
    track_total_length_m: float = 0.30
    max_step_rate_steps_s: float = 20000.0
    actuator_time_constant_s: float = 0.03
    max_cart_accel_steps_s2: float = 300000.0
    meters_per_step: float = 0.0000125
    pole_length_m: float = 0.13
    pole_rod_mass_kg: float = 0.0095
    pole_tip_mass_kg: float = 0.005
    control_penalty_r: float = 1e-6
    command_delay_s: float = 0.01
    teacher_enable_angle_rad: float = math.radians(3.0)
    teacher_enable_angle_rate_rad_s: float = math.radians(45.0)
    teacher_disable_angle_rad: float = math.radians(10.0)
    teacher_disable_angle_rate_rad_s: float = math.radians(150.0)

    @property
    def track_half_length_m(self) -> float:
        return self.track_total_length_m / 2.0

    @property
    def track_half_length_steps(self) -> float:
        return self.track_half_length_m / self.meters_per_step

    @property
    def actuator_alpha(self) -> float:
        return 1.0 - math.exp(-self.sample_time_s / self.actuator_time_constant_s)

    @property
    def max_velocity_delta_steps_s(self) -> float:
        return self.max_cart_accel_steps_s2 * self.sample_time_s


def generated_header_path() -> Path:
    return Path(__file__).resolve().parents[2] / "hardware" / "firmware" / "include" / "generated_teacher_lqr_gains.h"


def generated_json_path() -> Path:
    return Path(__file__).resolve().parent / "generated_teacher_lqr_controller.json"


def cpp_float(value: float) -> str:
    return f"{value:.9f}f"


def cpp_bool(value: bool) -> str:
    return "true" if value else "false"


def validated_steps_per_unit(steps_per_unit: float) -> float:
    validated_value = float(steps_per_unit)
    if validated_value <= 0.0:
        raise ValueError(f"steps_per_unit must be positive, got {steps_per_unit!r}")
    return validated_value


def state_steps_to_scaled_steps(state: np.ndarray, steps_per_unit: float) -> np.ndarray:
    validated_value = validated_steps_per_unit(steps_per_unit)
    state_array = np.asarray(state, dtype=np.float64)
    scale = np.array([1.0 / validated_value, 1.0 / validated_value, 1.0, 1.0], dtype=np.float64)
    return state_array * scale


def control_steps_to_scaled_steps(command_steps_s: float, steps_per_unit: float) -> float:
    validated_value = validated_steps_per_unit(steps_per_unit)
    return float(command_steps_s) / validated_value


def gain_steps_to_scaled_steps(gain: np.ndarray, steps_per_unit: float) -> np.ndarray:
    validated_value = validated_steps_per_unit(steps_per_unit)
    gain_array = np.asarray(gain, dtype=np.float64)
    gain_scale = np.array([1.0, 1.0, 1.0 / validated_value, 1.0 / validated_value], dtype=np.float64)
    return gain_array * gain_scale


def discrete_system_steps_to_scaled_steps(
    a_discrete: np.ndarray,
    b_discrete: np.ndarray,
    steps_per_unit: float,
) -> tuple[np.ndarray, np.ndarray]:
    validated_value = validated_steps_per_unit(steps_per_unit)
    state_scale = np.diag([1.0 / validated_value, 1.0 / validated_value, 1.0, 1.0]).astype(np.float64)
    inverse_state_scale = np.diag([validated_value, validated_value, 1.0, 1.0]).astype(np.float64)
    a_scaled = state_scale @ np.asarray(a_discrete, dtype=np.float64) @ inverse_state_scale
    b_scaled = (state_scale @ np.asarray(b_discrete, dtype=np.float64)) * validated_value
    return a_scaled, b_scaled


def q_matrix_steps_to_scaled_steps(q_matrix: np.ndarray, steps_per_unit: float) -> np.ndarray:
    validated_value = validated_steps_per_unit(steps_per_unit)
    inverse_state_scale = np.diag([validated_value, validated_value, 1.0, 1.0]).astype(np.float64)
    return inverse_state_scale.T @ np.asarray(q_matrix, dtype=np.float64) @ inverse_state_scale


def r_matrix_steps_to_scaled_steps(r_matrix: np.ndarray, steps_per_unit: float) -> np.ndarray:
    validated_value = validated_steps_per_unit(steps_per_unit)
    return np.asarray(r_matrix, dtype=np.float64) * (validated_value**2)


def sign_report_steps_to_scaled_steps(
    sign_report: dict[str, float | bool],
    steps_per_unit: float,
) -> dict[str, float | bool]:
    return {
        "positive_theta_commands_positive_scaled_step_rate": bool(
            sign_report["positive_theta_commands_positive_step_rate"]
        ),
        "positive_theta_rate_commands_positive_scaled_step_rate": bool(
            sign_report["positive_theta_rate_commands_positive_step_rate"]
        ),
        "theta_only_command_scaled_steps_s_per_rad": control_steps_to_scaled_steps(
            float(sign_report["theta_only_command_steps_s_per_rad"]),
            steps_per_unit,
        ),
        "theta_rate_only_command_scaled_steps_s_per_rad_s": control_steps_to_scaled_steps(
            float(sign_report["theta_rate_only_command_steps_s_per_rad_s"]),
            steps_per_unit,
        ),
    }


def build_observation_matrix() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )


def choose_kalman_noise_matrices(
    sigma_v_steps_s: float,
    sigma_omega_rad_s: float,
    sigma_position_steps: float,
    sigma_angle_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    if sigma_v_steps_s <= 0.0:
        raise ValueError("sigma_v_steps_s must be positive.")
    if sigma_omega_rad_s <= 0.0:
        raise ValueError("sigma_omega_rad_s must be positive.")
    if sigma_position_steps <= 0.0:
        raise ValueError("sigma_position_steps must be positive.")
    if sigma_angle_rad <= 0.0:
        raise ValueError("sigma_angle_rad must be positive.")

    q_process = np.diag(
        [
            0.0,
            float(sigma_v_steps_s**2),
            0.0,
            float(sigma_omega_rad_s**2),
        ]
    )
    r_observation = np.diag(
        [
            float(sigma_position_steps**2),
            float(sigma_angle_rad**2),
        ]
    )
    return q_process, r_observation


def solve_steady_state_kalman_gain(
    a_discrete: np.ndarray,
    observation_matrix: np.ndarray,
    q_process: np.ndarray,
    r_observation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    estimator_covariance = solve_discrete_are(
        a_discrete.T,
        observation_matrix.T,
        q_process,
        r_observation,
    )
    innovation_covariance = (
        observation_matrix @ estimator_covariance @ observation_matrix.T
    ) + r_observation
    kalman_gain = estimator_covariance @ observation_matrix.T @ np.linalg.inv(innovation_covariance)
    estimator_poles = np.linalg.eigvals(a_discrete - kalman_gain @ observation_matrix)
    return kalman_gain, estimator_covariance, estimator_poles


def build_teacher_lqr_payload(
    params: PhysicalCartPoleParams,
    q_matrix: np.ndarray,
    r_matrix: np.ndarray,
    gain: np.ndarray,
    closed_loop_poles: np.ndarray,
    sign_report: dict[str, float | bool],
    observation_matrix: np.ndarray | None = None,
    q_process: np.ndarray | None = None,
    r_observation: np.ndarray | None = None,
    kalman_gain: np.ndarray | None = None,
    estimator_poles: np.ndarray | None = None,
    a_discrete: np.ndarray | None = None,
    b_discrete: np.ndarray | None = None,
    unit_system: str = "steps",
    steps_per_unit: float = 1.0,
) -> dict[str, object]:
    if unit_system not in {"steps", "scaled_steps"}:
        raise ValueError(f"Unsupported unit system: {unit_system}")

    q_payload = np.asarray(q_matrix, dtype=np.float64)
    r_payload = np.asarray(r_matrix, dtype=np.float64)
    gain_payload = np.asarray(gain, dtype=np.float64)
    observation_payload = np.asarray(observation_matrix, dtype=np.float64) if observation_matrix is not None else None
    q_process_payload = np.asarray(q_process, dtype=np.float64) if q_process is not None else None
    r_observation_payload = np.asarray(r_observation, dtype=np.float64) if r_observation is not None else None
    kalman_gain_payload = np.asarray(kalman_gain, dtype=np.float64) if kalman_gain is not None else None
    estimator_poles_payload = np.asarray(estimator_poles, dtype=np.complex128) if estimator_poles is not None else None
    a_payload = np.asarray(a_discrete, dtype=np.float64) if a_discrete is not None else None
    b_payload = np.asarray(b_discrete, dtype=np.float64) if b_discrete is not None else None
    sign_report_payload = sign_report
    state_order = [
        "cart_position_steps",
        "cart_velocity_steps_s",
        "pole_angle_rad",
        "pole_angle_rate_rad_s",
    ]
    input_units = "steps_per_second"
    max_velocity_delta_key = "max_velocity_delta_steps_s"
    max_velocity_delta_value = float(params.max_velocity_delta_steps_s)

    if unit_system == "scaled_steps":
        validated_value = validated_steps_per_unit(steps_per_unit)
        q_payload = q_matrix_steps_to_scaled_steps(q_matrix, validated_value)
        r_payload = r_matrix_steps_to_scaled_steps(r_matrix, validated_value)
        gain_payload = gain_steps_to_scaled_steps(gain, validated_value)
        if a_payload is not None and b_payload is not None:
            a_payload, b_payload = discrete_system_steps_to_scaled_steps(a_payload, b_payload, validated_value)
        sign_report_payload = sign_report_steps_to_scaled_steps(sign_report, validated_value)
        state_order = [
            "cart_position_scaled_steps",
            "cart_velocity_scaled_steps_s",
            "pole_angle_rad",
            "pole_angle_rate_rad_s",
        ]
        input_units = "scaled_steps_per_second"
        max_velocity_delta_key = "max_velocity_delta_scaled_steps_s"
        max_velocity_delta_value = control_steps_to_scaled_steps(params.max_velocity_delta_steps_s, validated_value)

    q_diag = np.diag(q_payload)
    pole_magnitudes = np.abs(closed_loop_poles)
    payload: dict[str, object] = {
        "unit_system": unit_system,
        "steps_per_unit": float(validated_steps_per_unit(steps_per_unit)),
        "state_order": [
            *state_order,
        ],
        "input_units": input_units,
        "control_law": "u = -Kx",
        "control_rate_hz": float(1.0 / params.sample_time_s),
        "sample_time_s": float(params.sample_time_s),
        "params": asdict(params),
        "actuator_alpha": float(params.actuator_alpha),
        max_velocity_delta_key: max_velocity_delta_value,
        "Q_diag": q_diag.astype(float).tolist(),
        "R": float(r_payload[0, 0]),
        "K": gain_payload.reshape(-1).astype(float).tolist(),
        "closed_loop_pole_magnitudes": pole_magnitudes.astype(float).tolist(),
        "closed_loop_poles": [
            {
                "real": float(np.real(pole)),
                "imag": float(np.imag(pole)),
                "magnitude": float(abs(pole)),
            }
            for pole in closed_loop_poles
        ],
        "sign_report": {
            key: (bool(value) if isinstance(value, (bool, np.bool_)) else float(value))
            for key, value in sign_report_payload.items()
        },
    }
    if a_payload is not None:
        payload["A"] = a_payload.astype(float).tolist()
    if b_payload is not None:
        payload["B"] = b_payload.astype(float).tolist()
    if observation_payload is not None:
        payload["C"] = observation_payload.astype(float).tolist()
    if q_process_payload is not None:
        payload["Q_process_diag"] = np.diag(q_process_payload).astype(float).tolist()
    if r_observation_payload is not None:
        payload["R_observation_diag"] = np.diag(r_observation_payload).astype(float).tolist()
    if kalman_gain_payload is not None:
        payload["Kf"] = kalman_gain_payload.astype(float).tolist()
    if estimator_poles_payload is not None:
        estimator_pole_magnitudes = np.abs(estimator_poles_payload)
        payload["estimator_pole_magnitudes"] = estimator_pole_magnitudes.astype(float).tolist()
        payload["estimator_poles"] = [
            {
                "real": float(np.real(pole)),
                "imag": float(np.imag(pole)),
                "magnitude": float(abs(pole)),
            }
            for pole in estimator_poles_payload
        ]
    return payload


def write_generated_teacher_lqr_json(
    params: PhysicalCartPoleParams,
    q_matrix: np.ndarray,
    r_matrix: np.ndarray,
    gain: np.ndarray,
    closed_loop_poles: np.ndarray,
    sign_report: dict[str, float | bool],
    observation_matrix: np.ndarray | None = None,
    q_process: np.ndarray | None = None,
    r_observation: np.ndarray | None = None,
    kalman_gain: np.ndarray | None = None,
    estimator_poles: np.ndarray | None = None,
    a_discrete: np.ndarray | None = None,
    b_discrete: np.ndarray | None = None,
    output_path: Path | None = None,
    unit_system: str = "steps",
    steps_per_unit: float = 1.0,
) -> Path:
    json_path = output_path or generated_json_path()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_teacher_lqr_payload(
        params,
        q_matrix,
        r_matrix,
        gain,
        closed_loop_poles,
        sign_report,
        observation_matrix=observation_matrix,
        q_process=q_process,
        r_observation=r_observation,
        kalman_gain=kalman_gain,
        estimator_poles=estimator_poles,
        a_discrete=a_discrete,
        b_discrete=b_discrete,
        unit_system=unit_system,
        steps_per_unit=steps_per_unit,
    )
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return json_path


def write_generated_teacher_lqr_header(
    params: PhysicalCartPoleParams,
    q_matrix: np.ndarray,
    r_matrix: np.ndarray,
    gain: np.ndarray,
    closed_loop_poles: np.ndarray,
    sign_report: dict[str, float | bool],
    derived: dict[str, float],
    q_process: np.ndarray,
    r_observation: np.ndarray,
    kalman_gain: np.ndarray,
    estimator_poles: np.ndarray,
) -> Path:
    header_path = generated_header_path()
    header_path.parent.mkdir(parents=True, exist_ok=True)
    q_diag = np.diag(q_matrix)
    q_process_diag = np.diag(q_process)
    r_observation_diag = np.diag(r_observation)
    pole_magnitudes = np.abs(closed_loop_poles)
    estimator_pole_magnitudes = np.abs(estimator_poles)
    control_rate_hz = int(round(1.0 / params.sample_time_s))
    kalman_gain_rows = ", ".join(
        "{" + ", ".join(cpp_float(float(value)) for value in row) + "}"
        for row in kalman_gain
    )
    header_text = f"""#pragma once
#include <stdint.h>

// Generated by python/cartpole_lqr_gain.py.
// State order: [cart_position_steps, cart_velocity_steps_s, pole_angle_rad, pole_angle_rate_rad_s]
// Input: signed_step_rate_steps_s target for the cart velocity state.
// Nominal plant: first-order actuator lag; validation also clips cart acceleration.
// Control rate: {control_rate_hz} Hz
// Sample time: {params.sample_time_s:.9f} s
// Max command step rate: {params.max_step_rate_steps_s:.3f} steps/s
// Actuator time constant: {params.actuator_time_constant_s:.9f} s
// Actuator alpha: {params.actuator_alpha:.9f}
// Max cart acceleration: {params.max_cart_accel_steps_s2:.3f} steps/s^2
// Max velocity delta/sample: {params.max_velocity_delta_steps_s:.3f} steps/s
// Validation command delay: {params.command_delay_s:.9f} s
// Track total length: {params.track_total_length_m:.6f} m
// Pole length: {params.pole_length_m:.6f} m
// Pole rod mass: {params.pole_rod_mass_kg:.6f} kg
// Pole tip mass: {params.pole_tip_mass_kg:.6f} kg
// Meters per step: {params.meters_per_step:.9f} m/step
// Teacher enable gate: |theta| < {math.degrees(params.teacher_enable_angle_rad):.3f} deg and |theta_dot| < {math.degrees(params.teacher_enable_angle_rate_rad_s):.3f} deg/s
// Teacher disable gate: |theta| < {math.degrees(params.teacher_disable_angle_rad):.3f} deg and |theta_dot| < {math.degrees(params.teacher_disable_angle_rate_rad_s):.3f} deg/s
// Q diag: [{q_diag[0]:.9e}, {q_diag[1]:.9e}, {q_diag[2]:.9e}, {q_diag[3]:.9e}]
// R: {r_matrix[0, 0]:.9e}
// KF Q_proc diag: [{q_process_diag[0]:.9e}, {q_process_diag[1]:.9e}, {q_process_diag[2]:.9e}, {q_process_diag[3]:.9e}]
// KF R_obs diag: [{r_observation_diag[0]:.9e}, {r_observation_diag[1]:.9e}]
// KF estimator pole magnitudes: [{estimator_pole_magnitudes[0]:.9f}, {estimator_pole_magnitudes[1]:.9f}, {estimator_pole_magnitudes[2]:.9f}, {estimator_pole_magnitudes[3]:.9f}]
// Closed-loop pole magnitudes: [{pole_magnitudes[0]:.9f}, {pole_magnitudes[1]:.9f}, {pole_magnitudes[2]:.9f}, {pole_magnitudes[3]:.9f}]
// Gains: [{gain[0, 0]:.9f}, {gain[0, 1]:.9f}, {gain[0, 2]:.9f}, {gain[0, 3]:.9f}]
// Sign check: +theta -> +u is {sign_report['positive_theta_commands_positive_step_rate']}, +theta_dot -> +u is {sign_report['positive_theta_rate_commands_positive_step_rate']}

namespace generated_teacher_lqr {{
constexpr uint32_t kControlRateHz = {control_rate_hz}U;
constexpr float kSampleTimeSeconds = {cpp_float(params.sample_time_s)};
constexpr float kMaxCommandStepRateStepsPerSecond = {cpp_float(params.max_step_rate_steps_s)};
constexpr float kActuatorTimeConstantSeconds = {cpp_float(params.actuator_time_constant_s)};
constexpr float kActuatorVelocityAlpha = {cpp_float(params.actuator_alpha)};
constexpr float kMaxAccelerationStepsPerSecondSquared = {cpp_float(params.max_cart_accel_steps_s2)};
constexpr float kMaxVelocityDeltaPerSampleStepsPerSecond = {cpp_float(params.max_velocity_delta_steps_s)};
constexpr float kValidationCommandDelaySeconds = {cpp_float(params.command_delay_s)};
constexpr float kTrackTotalLengthMeters = {cpp_float(params.track_total_length_m)};
constexpr float kPoleLengthMeters = {cpp_float(params.pole_length_m)};
constexpr float kPoleRodMassKg = {cpp_float(params.pole_rod_mass_kg)};
constexpr float kPoleTipMassKg = {cpp_float(params.pole_tip_mass_kg)};
constexpr float kPoleRodCenterOfMassMeters = {cpp_float(params.pole_length_m / 2.0)};
constexpr float kPoleRodInertiaKgMetersSquared = {cpp_float(params.pole_rod_mass_kg * (params.pole_length_m**2) / 3.0)};
constexpr float kPoleCenterOfMassMeters = {cpp_float(derived['center_of_mass_m'])};
constexpr float kPolePivotInertiaKgMetersSquared = {cpp_float(derived['pivot_inertia_kg_m2'])};
constexpr float kAngleGainRadPerSecSquared = {cpp_float(derived['angle_gain'])};
constexpr float kAccelerationCoupling = {cpp_float(derived['accel_coupling'])};
constexpr float kMetersPerStep = {cpp_float(params.meters_per_step)};
constexpr float kEnableAngleThresholdRad = {cpp_float(params.teacher_enable_angle_rad)};
constexpr float kEnableAngleRateThresholdRadPerSec = {cpp_float(params.teacher_enable_angle_rate_rad_s)};
constexpr float kDisableAngleThresholdRad = {cpp_float(params.teacher_disable_angle_rad)};
constexpr float kDisableAngleRateThresholdRadPerSec = {cpp_float(params.teacher_disable_angle_rate_rad_s)};
constexpr float kQDiag[4] = {{{cpp_float(float(q_diag[0]))}, {cpp_float(float(q_diag[1]))}, {cpp_float(float(q_diag[2]))}, {cpp_float(float(q_diag[3]))}}};
constexpr float kR = {cpp_float(float(r_matrix[0, 0]))};
constexpr float kKalmanProcessNoiseDiag[4] = {{{cpp_float(float(q_process_diag[0]))}, {cpp_float(float(q_process_diag[1]))}, {cpp_float(float(q_process_diag[2]))}, {cpp_float(float(q_process_diag[3]))}}};
constexpr float kKalmanObservationNoiseDiag[2] = {{{cpp_float(float(r_observation_diag[0]))}, {cpp_float(float(r_observation_diag[1]))}}};
constexpr float kKalmanGain[4][2] = {{{kalman_gain_rows}}};
constexpr float kKalmanEstimatorPoleMagnitudes[4] = {{{cpp_float(float(estimator_pole_magnitudes[0]))}, {cpp_float(float(estimator_pole_magnitudes[1]))}, {cpp_float(float(estimator_pole_magnitudes[2]))}, {cpp_float(float(estimator_pole_magnitudes[3]))}}};
constexpr float kClosedLoopPoleMagnitudes[4] = {{{cpp_float(float(pole_magnitudes[0]))}, {cpp_float(float(pole_magnitudes[1]))}, {cpp_float(float(pole_magnitudes[2]))}, {cpp_float(float(pole_magnitudes[3]))}}};
constexpr float kGains[4] = {{{cpp_float(float(gain[0, 0]))}, {cpp_float(float(gain[0, 1]))}, {cpp_float(float(gain[0, 2]))}, {cpp_float(float(gain[0, 3]))}}};
constexpr bool kPositiveThetaCommandsPositiveStepRate = {cpp_bool(bool(sign_report['positive_theta_commands_positive_step_rate']))};
constexpr bool kPositiveThetaRateCommandsPositiveStepRate = {cpp_bool(bool(sign_report['positive_theta_rate_commands_positive_step_rate']))};
}}  // namespace generated_teacher_lqr
"""
    header_path.write_text(header_text, encoding="ascii")
    return header_path


def physical_pole_properties(params: PhysicalCartPoleParams) -> tuple[float, float, float]:
    total_pole_mass = params.pole_rod_mass_kg + params.pole_tip_mass_kg
    center_of_mass_m = (
        params.pole_rod_mass_kg * (params.pole_length_m / 2.0)
        + params.pole_tip_mass_kg * params.pole_length_m
    ) / total_pole_mass
    pivot_inertia_kg_m2 = (
        params.pole_rod_mass_kg * (params.pole_length_m**2) / 3.0
        + params.pole_tip_mass_kg * (params.pole_length_m**2)
    )
    return total_pole_mass, center_of_mass_m, pivot_inertia_kg_m2


def linearize_physical_cartpole(params: PhysicalCartPoleParams) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    total_pole_mass, center_of_mass_m, pivot_inertia_kg_m2 = physical_pole_properties(params)
    angle_gain = total_pole_mass * params.gravity_m_s2 * center_of_mass_m / pivot_inertia_kg_m2
    accel_coupling = total_pole_mass * center_of_mass_m / pivot_inertia_kg_m2
    actuator_lag_rate_s = 1.0 / params.actuator_time_constant_s
    cart_accel_per_velocity_error_m_s2 = params.meters_per_step * actuator_lag_rate_s
    theta_accel_per_velocity_error = accel_coupling * cart_accel_per_velocity_error_m_s2
    a_continuous = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -actuator_lag_rate_s, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, theta_accel_per_velocity_error, angle_gain, 0.0],
        ],
        dtype=np.float64,
    )
    b_continuous = np.array(
        [
            [0.0],
            [actuator_lag_rate_s],
            [0.0],
            [-theta_accel_per_velocity_error],
        ],
        dtype=np.float64,
    )
    a_discrete, b_discrete, _c_matrix, _d_matrix, _sample_time = cont2discrete(
        (a_continuous, b_continuous, np.eye(4, dtype=np.float64), np.zeros((4, 1), dtype=np.float64)),
        params.sample_time_s,
    )

    derived = {
        "pole_mass_kg": total_pole_mass,
        "center_of_mass_m": center_of_mass_m,
        "pivot_inertia_kg_m2": pivot_inertia_kg_m2,
        "natural_frequency_rad_s": math.sqrt(angle_gain),
        "angle_gain": angle_gain,
        "accel_coupling": accel_coupling,
        "actuator_lag_rate_s": actuator_lag_rate_s,
        "actuator_alpha": params.actuator_alpha,
        "cart_accel_per_velocity_error_m_s2": cart_accel_per_velocity_error_m_s2,
        "theta_accel_per_velocity_error": theta_accel_per_velocity_error,
        "max_velocity_delta_steps_s": params.max_velocity_delta_steps_s,
    }
    return a_discrete, b_discrete, derived


def choose_cost_matrices(params: PhysicalCartPoleParams) -> tuple[np.ndarray, np.ndarray]:
    max_position_steps = params.track_half_length_steps
    max_velocity_steps_s = params.max_step_rate_steps_s
    max_angle_rad = math.radians(15.0)
    max_angle_rate_rad_s = math.radians(90.0)

    # The physical rail is short enough that spending track to save angle is not acceptable.
    # Keep the angle terms dominant near upright, but weight cart position/velocity much more
    # strongly than the initial simulation defaults so late recovery stops pulling into the stop.
    q_matrix = np.diag(
        [
            768.0 / (max_position_steps**2),
            96.0 / (max_velocity_steps_s**2),
            8.0 / (max_angle_rad**2),
            4.0 / (max_angle_rate_rad_s**2),
        ]
    )
    r_matrix = np.array([[params.control_penalty_r]], dtype=np.float64)
    return q_matrix, r_matrix


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
        "positive_theta_commands_positive_step_rate": float(-gain[0, 2]) > 0.0,
        "positive_theta_rate_commands_positive_step_rate": float(-gain[0, 3]) > 0.0,
        "theta_only_command_steps_s_per_rad": float(-gain[0, 2]),
        "theta_rate_only_command_steps_s_per_rad_s": float(-gain[0, 3]),
    }
    if not sign_report["positive_theta_commands_positive_step_rate"]:
        raise SystemExit(
            "Sign check failed: positive theta would command negative cart motion under u = -Kx."
        )
    if not sign_report["positive_theta_rate_commands_positive_step_rate"]:
        raise SystemExit(
            "Sign check failed: positive theta_dot would command negative cart motion under u = -Kx."
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


def delayed_command_steps_per_second(
    command_history: list[float],
    command_delay_s: float,
    sample_time_s: float,
) -> float:
    if command_delay_s <= 0.0:
        return command_history[-1]

    delay_in_samples = command_delay_s / sample_time_s
    whole_samples = int(math.floor(delay_in_samples))
    fractional_sample = delay_in_samples - whole_samples

    newer = command_history[-(whole_samples + 1)]
    older = command_history[-(whole_samples + 2)]
    return ((1.0 - fractional_sample) * newer) + (fractional_sample * older)


def propagate_cartpole_state(
    state: np.ndarray,
    target_command_steps_s: float,
    params: PhysicalCartPoleParams,
    derived: dict[str, float],
) -> tuple[np.ndarray, dict[str, float | bool]]:
    position_steps, velocity_steps_s, angle_rad, angle_rate_rad_s = state
    raw_velocity_delta_steps_s = params.actuator_alpha * (target_command_steps_s - velocity_steps_s)
    clipped_velocity_delta_steps_s = float(
        np.clip(
            raw_velocity_delta_steps_s,
            -params.max_velocity_delta_steps_s,
            params.max_velocity_delta_steps_s,
        )
    )
    cart_acceleration_m_s2 = (clipped_velocity_delta_steps_s / params.sample_time_s) * params.meters_per_step
    theta_acceleration_rad_s2 = derived["angle_gain"] * angle_rad - (
        derived["accel_coupling"] * cart_acceleration_m_s2
    )

    next_position_steps = (
        position_steps
        + (params.sample_time_s * velocity_steps_s)
        + (0.5 * params.sample_time_s * clipped_velocity_delta_steps_s)
    )
    next_velocity_steps_s = velocity_steps_s + clipped_velocity_delta_steps_s
    next_angle_rad = (
        angle_rad
        + (params.sample_time_s * angle_rate_rad_s)
        + (0.5 * params.sample_time_s * params.sample_time_s * theta_acceleration_rad_s2)
    )
    next_angle_rate_rad_s = angle_rate_rad_s + (params.sample_time_s * theta_acceleration_rad_s2)

    next_state = np.array(
        [next_position_steps, next_velocity_steps_s, next_angle_rad, next_angle_rate_rad_s],
        dtype=np.float64,
    )
    step_stats: dict[str, float | bool] = {
        "accel_clipped": abs(clipped_velocity_delta_steps_s - raw_velocity_delta_steps_s) > 1e-9,
        "cart_acceleration_m_s2": abs(cart_acceleration_m_s2),
    }
    return next_state, step_stats


def rollout_episode(
    gain: np.ndarray,
    steps: int,
    initial_angle_deg: float,
    params: PhysicalCartPoleParams,
    derived: dict[str, float],
) -> tuple[int, np.ndarray, float, list[np.ndarray], dict[str, float]]:
    state = np.array([0.0, 0.0, math.radians(initial_angle_deg), 0.0], dtype=np.float64)
    last_target_command_steps_s = 0.0
    survived_steps = 0
    states = [state.copy()]
    angle_failure_limit_rad = math.radians(30.0)
    max_delay_samples = int(math.ceil(max(params.command_delay_s, 0.0) / params.sample_time_s))
    command_history = [0.0] * (max_delay_samples + 2)
    command_clip_count = 0
    accel_clip_count = 0
    peak_requested_command_steps_s = 0.0
    peak_target_command_steps_s = 0.0
    peak_velocity_steps_s = 0.0
    peak_cart_acceleration_m_s2 = 0.0

    for step_index in range(steps):
        raw_command_steps_s = float(-(gain @ state)[0])
        peak_requested_command_steps_s = max(peak_requested_command_steps_s, abs(raw_command_steps_s))
        target_command_steps_s = float(
            np.clip(raw_command_steps_s, -params.max_step_rate_steps_s, params.max_step_rate_steps_s)
        )
        if abs(target_command_steps_s - raw_command_steps_s) > 1e-9:
            command_clip_count += 1

        command_history.append(target_command_steps_s)
        delayed_target_command_steps_s = delayed_command_steps_per_second(
            command_history,
            params.command_delay_s,
            params.sample_time_s,
        )
        next_state, step_stats = propagate_cartpole_state(
            state,
            delayed_target_command_steps_s,
            params,
            derived,
        )
        states.append(next_state.copy())
        state = next_state
        last_target_command_steps_s = delayed_target_command_steps_s
        survived_steps = step_index + 1
        peak_target_command_steps_s = max(peak_target_command_steps_s, abs(delayed_target_command_steps_s))
        peak_velocity_steps_s = max(peak_velocity_steps_s, abs(state[1]))
        peak_cart_acceleration_m_s2 = max(
            peak_cart_acceleration_m_s2,
            float(step_stats["cart_acceleration_m_s2"]),
        )
        if bool(step_stats["accel_clipped"]):
            accel_clip_count += 1

        if abs(state[0]) > params.track_half_length_steps or abs(state[2]) > angle_failure_limit_rad:
            break

    rollout_stats = {
        "command_clip_count": float(command_clip_count),
        "accel_clip_count": float(accel_clip_count),
        "peak_requested_command_steps_s": peak_requested_command_steps_s,
        "peak_target_command_steps_s": peak_target_command_steps_s,
        "peak_velocity_steps_s": peak_velocity_steps_s,
        "peak_cart_acceleration_m_s2": peak_cart_acceleration_m_s2,
    }
    return survived_steps, state, last_target_command_steps_s, states, rollout_stats


def rollout_nominal_linear_episode(
    gain: np.ndarray,
    steps: int,
    initial_angle_deg: float,
    a_discrete: np.ndarray,
    b_discrete: np.ndarray,
    params: PhysicalCartPoleParams,
) -> tuple[int, np.ndarray, float, list[np.ndarray], dict[str, float]]:
    state = np.array([0.0, 0.0, math.radians(initial_angle_deg), 0.0], dtype=np.float64)
    last_target_command_steps_s = 0.0
    states = [state.copy()]
    peak_requested_command_steps_s = 0.0
    peak_target_command_steps_s = 0.0
    peak_velocity_steps_s = 0.0
    peak_cart_acceleration_m_s2 = 0.0

    input_matrix = b_discrete.reshape(-1)
    for _step_index in range(steps):
        raw_command_steps_s = float(-(gain @ state)[0])
        next_state = (a_discrete @ state) + (input_matrix * raw_command_steps_s)
        cart_acceleration_m_s2 = ((next_state[1] - state[1]) / params.sample_time_s) * params.meters_per_step

        state = np.asarray(next_state, dtype=np.float64)
        states.append(state.copy())
        last_target_command_steps_s = raw_command_steps_s
        peak_requested_command_steps_s = max(peak_requested_command_steps_s, abs(raw_command_steps_s))
        peak_target_command_steps_s = max(peak_target_command_steps_s, abs(raw_command_steps_s))
        peak_velocity_steps_s = max(peak_velocity_steps_s, abs(state[1]))
        peak_cart_acceleration_m_s2 = max(peak_cart_acceleration_m_s2, abs(cart_acceleration_m_s2))

    rollout_stats = {
        "command_clip_count": 0.0,
        "accel_clip_count": 0.0,
        "peak_requested_command_steps_s": peak_requested_command_steps_s,
        "peak_target_command_steps_s": peak_target_command_steps_s,
        "peak_velocity_steps_s": peak_velocity_steps_s,
        "peak_cart_acceleration_m_s2": peak_cart_acceleration_m_s2,
    }
    return steps, state, last_target_command_steps_s, states, rollout_stats


def estimate_capture_angle_deg(
    gain: np.ndarray,
    steps: int,
    params: PhysicalCartPoleParams,
    derived: dict[str, float],
    require_no_clipping: bool,
    max_search_angle_deg: float = 20.0,
    iterations: int = 12,
) -> float:
    def qualifies(angle_deg: float) -> bool:
        survived_steps, _final_state, _last_target, _states, rollout_stats = rollout_episode(
            gain,
            steps,
            angle_deg,
            params,
            derived,
        )
        if survived_steps < steps:
            return False
        if not require_no_clipping:
            return True
        return (rollout_stats["command_clip_count"] == 0.0) and (rollout_stats["accel_clip_count"] == 0.0)

    lower_bound_deg = 0.0
    upper_bound_deg = max_search_angle_deg
    if qualifies(upper_bound_deg):
        return upper_bound_deg

    for _ in range(iterations):
        midpoint_deg = 0.5 * (lower_bound_deg + upper_bound_deg)
        if qualifies(midpoint_deg):
            lower_bound_deg = midpoint_deg
        else:
            upper_bound_deg = midpoint_deg

    return lower_bound_deg


def render_hold_loop(env, state: np.ndarray, params: PhysicalCartPoleParams, hold_open_seconds: float) -> None:
    mapped_state = np.array(
        [
            state[0] * params.meters_per_step,
            state[1] * params.meters_per_step,
            state[2],
            state[3],
        ],
        dtype=np.float64,
    )
    end_time = None if hold_open_seconds < 0.0 else time.monotonic() + hold_open_seconds

    try:
        while end_time is None or time.monotonic() < end_time:
            env.unwrapped.state = mapped_state
            env.render()
            time.sleep(1.0 / 30.0)
    except KeyboardInterrupt:
        pass


def visualize_with_gym(
    states: list[np.ndarray],
    params: PhysicalCartPoleParams,
    render_every: int,
    hold_open_seconds: float,
) -> None:
    gym_module = load_gym_module()
    env = gym_module.make("CartPole-v1", render_mode="human")
    try:
        reset_result = env.reset(seed=0)
        if isinstance(reset_result, tuple):
            _observation, _info = reset_result
        # Gym stores half the pole length in `length`, so set it explicitly to
        # match the 13 cm physical pole instead of the default long CartPole pole.
        env.unwrapped.length = params.pole_length_m / 2.0
        env.unwrapped.x_threshold = params.track_half_length_m * 1.05
        env.unwrapped.theta_threshold_radians = math.radians(35.0)

        for index, state in enumerate(states):
            if index % max(render_every, 1) != 0:
                continue
            env.unwrapped.state = np.array(
                [
                    state[0] * params.meters_per_step,
                    state[1] * params.meters_per_step,
                    state[2],
                    state[3],
                ],
                dtype=np.float64,
            )
            env.render()
            time.sleep(params.sample_time_s * render_every)

        if states:
            render_hold_loop(env, states[-1], params, hold_open_seconds)
    finally:
        env.close()


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
    parser.add_argument("--render", action="store_true", help="Visualize the simulated rollout with Gym's CartPole viewer.")
    parser.add_argument("--render-every", type=int, default=20, help="Render every N simulation steps to keep playback watchable.")
    parser.add_argument(
        "--hold-open-seconds",
        type=float,
        default=-1.0,
        help="How long to keep the viewer open after stabilization. Negative means until Ctrl+C.",
    )
    parser.add_argument(
        "--control-rate-hz",
        type=float,
        default=250.0,
        help="Discrete control rate in Hz. Defaults to the current hardware teacher loop rate.",
    )
    parser.add_argument(
        "--max-step-rate-steps-s",
        type=float,
        default=20000.0,
        help="Maximum teacher command magnitude in steps/s used for design and saturation.",
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
        default=300000.0,
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
        default=0.01,
        help="Pure command-to-motion delay used only in simulation validation.",
    )
    parser.add_argument(
        "--enable-angle-threshold-deg",
        type=float,
        default=3.0,
        help="Teacher enable gate for |theta| in degrees.",
    )
    parser.add_argument(
        "--enable-angle-rate-threshold-deg-s",
        type=float,
        default=45.0,
        help="Teacher enable gate for |theta_dot| in deg/s.",
    )
    parser.add_argument(
        "--disable-angle-threshold-deg",
        type=float,
        default=10.0,
        help="Teacher dropout gate for |theta| in degrees.",
    )
    parser.add_argument(
        "--disable-angle-rate-threshold-deg-s",
        type=float,
        default=150.0,
        help="Teacher dropout gate for |theta_dot| in deg/s.",
    )
    parser.add_argument(
        "--kf-sigma-v-steps-s",
        type=float,
        default=50.0,
        help="Steady-state KF process noise sigma for cart velocity in steps/s.",
    )
    parser.add_argument(
        "--kf-sigma-omega-rad-s",
        type=float,
        default=0.5,
        help="Steady-state KF process noise sigma for pole angle rate in rad/s.",
    )
    parser.add_argument(
        "--kf-sigma-position-steps",
        type=float,
        default=2.0,
        help="Steady-state KF observation noise sigma for cart position in steps.",
    )
    parser.add_argument(
        "--kf-sigma-angle-rad",
        type=float,
        default=0.002,
        help="Steady-state KF observation noise sigma for pole angle in radians.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
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
    if args.enable_angle_threshold_deg <= 0.0 or args.enable_angle_rate_threshold_deg_s <= 0.0:
        raise SystemExit("Teacher enable thresholds must be positive.")
    if args.disable_angle_threshold_deg < args.enable_angle_threshold_deg:
        raise SystemExit("Teacher disable angle threshold must be >= the enable angle threshold.")
    if args.disable_angle_rate_threshold_deg_s < args.enable_angle_rate_threshold_deg_s:
        raise SystemExit(
            "Teacher disable angle-rate threshold must be >= the enable angle-rate threshold."
        )

    params = PhysicalCartPoleParams(
        sample_time_s=1.0 / args.control_rate_hz,
        max_step_rate_steps_s=args.max_step_rate_steps_s,
        actuator_time_constant_s=args.actuator_time_constant_s,
        max_cart_accel_steps_s2=args.max_cart_accel_steps_s2,
        control_penalty_r=args.r_weight,
        command_delay_s=args.command_delay_s,
        teacher_enable_angle_rad=math.radians(args.enable_angle_threshold_deg),
        teacher_enable_angle_rate_rad_s=math.radians(args.enable_angle_rate_threshold_deg_s),
        teacher_disable_angle_rad=math.radians(args.disable_angle_threshold_deg),
        teacher_disable_angle_rate_rad_s=math.radians(args.disable_angle_rate_threshold_deg_s),
    )
    rng = np.random.default_rng(args.seed)
    initial_angle_deg = sample_initial_angle_deg(rng, args.theta0_deg, args.theta0_range_deg)

    a_discrete, b_discrete, derived = linearize_physical_cartpole(params)
    q_matrix, r_matrix = choose_cost_matrices(params)
    observation_matrix = build_observation_matrix()
    q_process, r_observation = choose_kalman_noise_matrices(
        args.kf_sigma_v_steps_s,
        args.kf_sigma_omega_rad_s,
        args.kf_sigma_position_steps,
        args.kf_sigma_angle_rad,
    )
    gain, _riccati_solution, closed_loop_poles = solve_discrete_lqr(a_discrete, b_discrete, q_matrix, r_matrix)
    kalman_gain, _estimator_covariance, estimator_poles = solve_steady_state_kalman_gain(
        a_discrete,
        observation_matrix,
        q_process,
        r_observation,
    )
    sign_report = verify_sign_conventions(gain)
    survived_steps, final_state, last_target_command_steps_s, states, rollout_stats = rollout_episode(
        gain,
        args.steps,
        initial_angle_deg,
        params,
        derived,
    )
    max_survivable_angle_deg = estimate_capture_angle_deg(
        gain,
        args.steps,
        params,
        derived,
        require_no_clipping=False,
    )
    max_no_clip_angle_deg = estimate_capture_angle_deg(
        gain,
        args.steps,
        params,
        derived,
        require_no_clipping=True,
    )
    output_header_path = write_generated_teacher_lqr_header(
        params,
        q_matrix,
        r_matrix,
        gain,
        closed_loop_poles,
        sign_report,
        derived,
        q_process,
        r_observation,
        kalman_gain,
        estimator_poles,
    )
    output_json_path = write_generated_teacher_lqr_json(
        params,
        q_matrix,
        r_matrix,
        gain,
        closed_loop_poles,
        sign_report,
        observation_matrix=observation_matrix,
        q_process=q_process,
        r_observation=r_observation,
        kalman_gain=kalman_gain,
        estimator_poles=estimator_poles,
        a_discrete=a_discrete,
        b_discrete=b_discrete,
    )

    np.set_printoptions(precision=5, suppress=True)
    print("Model       : cart-pole with actuator-lagged step-rate target input")
    print("State order : [cart_position_steps, cart_velocity_steps_s, pole_angle_rad, pole_angle_rate_rad_s]")
    print("Input       : signed_step_rate_steps_s target")
    print(f"Control rate: {args.control_rate_hz:.2f} Hz")
    print(f"Sample time : {params.sample_time_s:.4f} s")
    print(f"Max step rate: {params.max_step_rate_steps_s:.1f} steps/s")
    print(f"Actuator tau: {params.actuator_time_constant_s:.4f} s")
    print(f"Actuator a  : {params.actuator_alpha:.5f} per sample")
    print(f"Accel limit : {params.max_cart_accel_steps_s2:.1f} steps/s^2")
    print(f"A           :\n{np.array2string(a_discrete, precision=8, suppress_small=False)}")
    print(f"B           :\n{np.array2string(b_discrete, precision=8, suppress_small=False)}")
    print(f"Q diag      : {np.array2string(np.diag(q_matrix), precision=8, suppress_small=False)}")
    print(f"R           : {r_matrix[0, 0]:.5e}")
    print(f"K           : {gain[0]}")
    print(f"C           :\n{np.array2string(observation_matrix, precision=8, suppress_small=False)}")
    print(f"Q_proc diag : {np.array2string(np.diag(q_process), precision=8, suppress_small=False)}")
    print(f"R_obs diag  : {np.array2string(np.diag(r_observation), precision=8, suppress_small=False)}")
    print(f"Kf          :\n{np.array2string(kalman_gain, precision=8, suppress_small=False)}")
    print(f"SETK        : {' '.join(f'{value:.6f}' for value in gain[0])}")
    print(f"Gain file   : {output_header_path}")
    print(f"Gain json   : {output_json_path}")
    print(f"|kf poles|  : {np.abs(estimator_poles)}")
    print(f"|poles|     : {np.abs(closed_loop_poles)}")
    print(f"Track       : {params.track_total_length_m:.3f} m total ({params.track_half_length_steps:.0f} steps each side)")
    print(f"Pole length : {params.pole_length_m:.3f} m")
    print(f"Meters/step : {params.meters_per_step:.8f}")
    print(f"Pole mass   : {derived['pole_mass_kg']:.5f} kg")
    print(f"Pole COM    : {derived['center_of_mass_m']:.5f} m")
    print(f"Pole inertia: {derived['pivot_inertia_kg_m2']:.8f} kg*m^2")
    print(f"Nat. freq   : {derived['natural_frequency_rad_s']:.5f} rad/s")
    print(f"Cmd delay   : {params.command_delay_s:.6f} s ({params.command_delay_s / params.sample_time_s:.3f} samples)")
    print("Stepper     : open-loop pulses, but the design now assumes actuator lag and command saturation")
    print("Vel dyn     : v_dot = (u_target - v) / tau, then |v[k+1] - v[k]| <= a_max * Ts in validation")
    print("Pole accel  : theta_ddot = g_eff * theta - c * cart_accel")
    print(
        "Teacher gate: "
        f"enable if |theta|<{math.degrees(params.teacher_enable_angle_rad):.1f} deg and "
        f"|theta_dot|<{math.degrees(params.teacher_enable_angle_rate_rad_s):.1f} deg/s; "
        f"disable beyond {math.degrees(params.teacher_disable_angle_rad):.1f} deg or "
        f"{math.degrees(params.teacher_disable_angle_rate_rad_s):.1f} deg/s"
    )
    print(
        "Sign check  : "
        f"+theta -> {'+u' if sign_report['positive_theta_commands_positive_step_rate'] else '-u'}, "
        f"+theta_dot -> {'+u' if sign_report['positive_theta_rate_commands_positive_step_rate'] else '-u'}"
    )
    print(f"Init angle  : {initial_angle_deg:.3f} deg")
    print(f"Rollout     : survived {survived_steps}/{args.steps} steps")
    print(f"Final state : {final_state}")
    print(f"Last target : {last_target_command_steps_s:.5f} steps/s")
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
    print(
        "Capture     : "
        f"approx full-survival |theta0| <= {max_survivable_angle_deg:.2f} deg, "
        f"zero-clip |theta0| <= {max_no_clip_angle_deg:.2f} deg"
    )
    if abs(initial_angle_deg) > max_survivable_angle_deg:
        print(
            "Capture note: requested |theta0| exceeds the estimated survivable region for the current model and limits."
        )
    elif abs(initial_angle_deg) > max_no_clip_angle_deg:
        print(
            "Capture note: requested |theta0| survives only with heavy clipping, so it is outside the conservative teacher gate region."
        )

    if args.render:
        visualize_with_gym(states, params, args.render_every, args.hold_open_seconds)

    if survived_steps < args.steps:
        raise SystemExit("Closed-loop rollout did not stay balanced for the full test horizon.")


if __name__ == "__main__":
    main()