"""Rigid-body attitude and reaction-wheel dynamics."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

type Vector = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AttitudeState:
    quaternion_wxyz: Vector
    angular_velocity_rad_s: Vector
    wheel_momentum_n_m_s: Vector


def _quaternion_product(left: Vector, right: Vector) -> Vector:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dtype=np.float64,
    )


def normalize_quaternion(quaternion_wxyz: Vector) -> Vector:
    norm = float(np.linalg.norm(quaternion_wxyz))
    if norm == 0.0:
        raise ValueError("quaternion must be non-zero")
    return quaternion_wxyz / norm


def rotate_body_to_inertial(quaternion_wxyz: Vector, body_vector: Vector) -> Vector:
    if np.array_equal(quaternion_wxyz, np.asarray((1.0, 0.0, 0.0, 0.0))):
        return body_vector
    quaternion = normalize_quaternion(quaternion_wxyz)
    conjugate = quaternion * np.asarray((1.0, -1.0, -1.0, -1.0))
    pure_vector = np.concatenate((np.asarray((0.0,)), body_vector))
    return _quaternion_product(_quaternion_product(quaternion, pure_vector), conjugate)[
        1:
    ]


def integrate_attitude(
    state: AttitudeState,
    wheel_torque_n_m: Vector,
    external_torque_n_m: Vector,
    inertia_diagonal_kg_m2: Vector,
    duration_s: float,
) -> AttitudeState:
    """Integrate rigid-body attitude through one classical RK4 step."""

    if duration_s <= 0.0:
        raise ValueError("duration must be positive")
    if np.any(inertia_diagonal_kg_m2 <= 0.0):
        raise ValueError("inertia must be positive")
    torque_n_m = wheel_torque_n_m + external_torque_n_m
    is_stationary = not np.any(state.angular_velocity_rad_s) and not np.any(torque_n_m)
    if is_stationary:
        return AttitudeState(
            state.quaternion_wxyz,
            state.angular_velocity_rad_s,
            state.wheel_momentum_n_m_s - wheel_torque_n_m * duration_s,
        )
    first = _attitude_derivative(
        state.quaternion_wxyz,
        state.angular_velocity_rad_s,
        torque_n_m,
        inertia_diagonal_kg_m2,
    )
    second = _attitude_derivative(
        state.quaternion_wxyz + first[0] * duration_s / 2.0,
        state.angular_velocity_rad_s + first[1] * duration_s / 2.0,
        torque_n_m,
        inertia_diagonal_kg_m2,
    )
    third = _attitude_derivative(
        state.quaternion_wxyz + second[0] * duration_s / 2.0,
        state.angular_velocity_rad_s + second[1] * duration_s / 2.0,
        torque_n_m,
        inertia_diagonal_kg_m2,
    )
    fourth = _attitude_derivative(
        state.quaternion_wxyz + third[0] * duration_s,
        state.angular_velocity_rad_s + third[1] * duration_s,
        torque_n_m,
        inertia_diagonal_kg_m2,
    )
    sixth_step = duration_s / 6.0
    updated_quaternion = normalize_quaternion(
        state.quaternion_wxyz
        + sixth_step * (first[0] + 2.0 * second[0] + 2.0 * third[0] + fourth[0])
    )
    updated_angular_velocity = state.angular_velocity_rad_s + sixth_step * (
        first[1] + 2.0 * second[1] + 2.0 * third[1] + fourth[1]
    )
    return AttitudeState(
        updated_quaternion,
        updated_angular_velocity,
        state.wheel_momentum_n_m_s - wheel_torque_n_m * duration_s,
    )


def _attitude_derivative(
    quaternion_wxyz: Vector,
    angular_velocity_rad_s: Vector,
    torque_n_m: Vector,
    inertia_diagonal_kg_m2: Vector,
) -> tuple[Vector, Vector]:
    spin_quaternion = np.concatenate((np.asarray((0.0,)), angular_velocity_rad_s))
    quaternion_rate = 0.5 * _quaternion_product(quaternion_wxyz, spin_quaternion)
    angular_momentum = inertia_diagonal_kg_m2 * angular_velocity_rad_s
    angular_acceleration = (
        torque_n_m - np.cross(angular_velocity_rad_s, angular_momentum)
    ) / inertia_diagonal_kg_m2
    return quaternion_rate, angular_acceleration
