"""Saturated quaternion feedback with integral anti-windup."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from typace.config.flight import (
    ATTITUDE_DERIVATIVE_GAIN,
    ATTITUDE_INTEGRAL_GAIN,
    ATTITUDE_INTEGRAL_LIMIT_RAD_S,
    ATTITUDE_PROPORTIONAL_GAIN,
)
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
    error_body_rad = _attitude_error_body_rad(
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
    saturated = not np.allclose(torque, requested_torque)
    next_integral = state.integral_error_rad_s if saturated else candidate_integral
    return AttitudeControlOutput(
        torque,
        float(np.linalg.norm(error_body_rad)),
        AttitudeControllerState(next_integral),
    )


def _attitude_error_body_rad(
    current_quaternion_wxyz: np.ndarray,
    target_quaternion_wxyz: np.ndarray,
) -> np.ndarray:
    current = Rotation.from_quat(_xyzw(current_quaternion_wxyz))
    target = Rotation.from_quat(_xyzw(target_quaternion_wxyz))
    error_inertial = (target * current.inv()).as_rotvec()
    return current.inv().apply(error_inertial)


def _xyzw(quaternion_wxyz: np.ndarray) -> np.ndarray:
    return quaternion_wxyz[np.asarray((1, 2, 3, 0))]
