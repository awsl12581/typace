"""Reaction-wheel and RCS actuator limits."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

type Vector = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ActuatorCommand:
    main_throttle: float
    wheel_torque_n_m: Vector
    rcs_torque_n_m: Vector


@dataclass(frozen=True, slots=True)
class ActuatorOutput:
    main_throttle: float
    wheel_torque_n_m: Vector
    rcs_torque_n_m: Vector
    main_mass_flow_kg_s: float
    rcs_mass_flow_kg_s: float
    main_active_duration_s: float
    rcs_active_duration_s: float


def limit_reaction_wheel_torque(
    requested_torque_n_m: Vector,
    wheel_momentum_n_m_s: Vector,
    maximum_torque_n_m: Vector,
    maximum_momentum_n_m_s: Vector,
    duration_s: float,
) -> Vector:
    if duration_s <= 0.0:
        raise ValueError("duration must be positive")
    if np.any(maximum_torque_n_m < 0.0) or np.any(maximum_momentum_n_m_s <= 0.0):
        raise ValueError("actuator limits must be non-negative")
    limited = np.clip(requested_torque_n_m, -maximum_torque_n_m, maximum_torque_n_m)
    minimum_torque = (wheel_momentum_n_m_s - maximum_momentum_n_m_s) / duration_s
    maximum_torque = (wheel_momentum_n_m_s + maximum_momentum_n_m_s) / duration_s
    return np.clip(limited, minimum_torque, maximum_torque)


def limit_rcs_torque(requested_torque_n_m: Vector, maximum_torque_n_m: float) -> Vector:
    if maximum_torque_n_m < 0.0:
        raise ValueError("RCS torque limit must be non-negative")
    norm = float(np.linalg.norm(requested_torque_n_m))
    if norm <= maximum_torque_n_m or norm == 0.0:
        return requested_torque_n_m.copy()
    return requested_torque_n_m * (maximum_torque_n_m / norm)


def momentum_unload_command(
    wheel_momentum_n_m_s: Vector,
    maximum_wheel_torque_n_m: Vector,
    maximum_rcs_torque_n_m: float,
    duration_s: float,
) -> ActuatorCommand:
    if duration_s <= 0.0:
        raise ValueError("duration must be positive")
    requested_wheel_torque = wheel_momentum_n_m_s / duration_s
    wheel_torque = np.clip(
        requested_wheel_torque,
        -maximum_wheel_torque_n_m,
        maximum_wheel_torque_n_m,
    )
    rcs_torque = limit_rcs_torque(-wheel_torque, maximum_rcs_torque_n_m)
    return ActuatorCommand(0.0, -rcs_torque, rcs_torque)
