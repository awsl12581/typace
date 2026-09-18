"""Saturated quaternion feedback with integral anti-windup."""

from dataclasses import dataclass
from math import atan2, sqrt

import numpy as np

from typace.config.flight import (
    ATTITUDE_DERIVATIVE_GAIN,
    ATTITUDE_INTEGRAL_GAIN,
    ATTITUDE_INTEGRAL_LIMIT_RAD_S,
    ATTITUDE_PROPORTIONAL_GAIN,
)
from typace.config.vehicle import QUATERNION_NORM_TOLERANCE
from typace.vehicle.attitude import AttitudeState


@dataclass(frozen=True, slots=True)
class AttitudeControllerState:
    integral_error_rad_s: np.ndarray


@dataclass(frozen=True, slots=True)
class AttitudeControlOutput:
    wheel_torque_n_m: np.ndarray
    attitude_error_rad: float
    state: AttitudeControllerState


def control_attitude(
    state: AttitudeControllerState,
    attitude: AttitudeState,
    target_quaternion_wxyz: np.ndarray,
    maximum_torque_n_m: np.ndarray,
    duration_s: float,
) -> AttitudeControlOutput:
    if duration_s <= 0.0:
        raise ValueError("control duration must be positive")
    error_body_rad, attitude_error_rad = _attitude_error_body_rad(
        attitude.quaternion_wxyz, target_quaternion_wxyz
    )
    candidate_integral = np.clip(
        state.integral_error_rad_s + error_body_rad * duration_s,
        -ATTITUDE_INTEGRAL_LIMIT_RAD_S,
        ATTITUDE_INTEGRAL_LIMIT_RAD_S,
    )
    normalized_request = (
        ATTITUDE_PROPORTIONAL_GAIN * error_body_rad
        + ATTITUDE_INTEGRAL_GAIN * candidate_integral
        - ATTITUDE_DERIVATIVE_GAIN * attitude.angular_velocity_rad_s
    )
    requested_torque = maximum_torque_n_m * normalized_request
    torque = np.clip(requested_torque, -maximum_torque_n_m, maximum_torque_n_m)
    saturated = bool(np.any(torque != requested_torque))
    next_integral = state.integral_error_rad_s if saturated else candidate_integral
    return AttitudeControlOutput(
        torque,
        attitude_error_rad,
        AttitudeControllerState(next_integral),
    )


def _attitude_error_body_rad(
    current_quaternion_wxyz: np.ndarray,
    target_quaternion_wxyz: np.ndarray,
) -> tuple[np.ndarray, float]:
    current = _normalized_quaternion(current_quaternion_wxyz)
    target = _normalized_quaternion(target_quaternion_wxyz)
    conjugate = current * np.asarray((1.0, -1.0, -1.0, -1.0))
    relative = _quaternion_product(conjugate, target)
    if relative[0] < 0.0:
        relative = -relative
    vector = relative[1:]
    vector_norm = sqrt(float(np.dot(vector, vector)))
    if vector_norm == 0.0:
        return np.zeros(3), 0.0
    angle_rad = 2.0 * atan2(vector_norm, float(relative[0]))
    return vector / vector_norm * angle_rad, angle_rad


def _normalized_quaternion(quaternion_wxyz: np.ndarray) -> np.ndarray:
    norm_squared = float(np.dot(quaternion_wxyz, quaternion_wxyz))
    if abs(norm_squared - 1.0) <= QUATERNION_NORM_TOLERANCE:
        return quaternion_wxyz
    norm = sqrt(norm_squared)
    return quaternion_wxyz / norm


def _quaternion_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        )
    )
