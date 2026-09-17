"""Conversions between Cartesian state and classical elliptic elements."""

from dataclasses import dataclass, replace
from math import acos, atan2, cos, pi, sin, sqrt

import numpy as np
from numpy.typing import NDArray

from typace.physics.frames import perifocal_to_inertial_matrix
from typace.physics.kepler import TAU, normalize_angle, solve_eccentric_anomaly

type Vector = NDArray[np.float64]

SINGULARITY_TOLERANCE = 1.0e-10


@dataclass(frozen=True, slots=True)
class CartesianState:
    position_m: Vector
    velocity_m_s: Vector


@dataclass(frozen=True, slots=True)
class ClassicalElements:
    semi_major_axis_m: float
    eccentricity: float
    inclination_rad: float
    ascending_node_rad: float
    periapsis_argument_rad: float
    mean_anomaly_rad: float


def _clamp_unit(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _oriented_angle(first: Vector, second: Vector, normal: Vector) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    normal_norm = float(np.linalg.norm(normal))
    if denominator == 0.0 or normal_norm == 0.0:
        return 0.0
    cosine = float(np.dot(first, second)) / denominator
    sine = float(np.dot(np.cross(first, second), normal)) / (denominator * normal_norm)
    return normalize_angle(atan2(sine, cosine))


def elements_to_state(elements: ClassicalElements, mu_m3_s2: float) -> CartesianState:
    """Convert elliptic classical elements to an inertial Cartesian state."""

    if mu_m3_s2 <= 0.0:
        raise ValueError("gravitational parameter must be positive")
    if elements.semi_major_axis_m <= 0.0:
        raise ValueError("semi-major axis must be positive")
    if not 0.0 <= elements.eccentricity < 1.0:
        raise ValueError("eccentricity must be in [0, 1)")
    eccentric_anomaly = solve_eccentric_anomaly(
        elements.mean_anomaly_rad, elements.eccentricity
    )
    true_anomaly = 2.0 * atan2(
        sqrt(1.0 + elements.eccentricity) * sin(eccentric_anomaly / 2.0),
        sqrt(1.0 - elements.eccentricity) * cos(eccentric_anomaly / 2.0),
    )
    parameter_m = elements.semi_major_axis_m * (1.0 - elements.eccentricity**2)
    radius_m = parameter_m / (1.0 + elements.eccentricity * cos(true_anomaly))
    perifocal_position = np.asarray(
        (radius_m * cos(true_anomaly), radius_m * sin(true_anomaly), 0.0),
        dtype=np.float64,
    )
    speed_factor = sqrt(mu_m3_s2 / parameter_m)
    perifocal_velocity = np.asarray(
        (
            -speed_factor * sin(true_anomaly),
            speed_factor * (elements.eccentricity + cos(true_anomaly)),
            0.0,
        ),
        dtype=np.float64,
    )
    rotation = perifocal_to_inertial_matrix(
        elements.ascending_node_rad,
        elements.inclination_rad,
        elements.periapsis_argument_rad,
    )
    return CartesianState(rotation @ perifocal_position, rotation @ perifocal_velocity)


def state_to_elements(state: CartesianState, mu_m3_s2: float) -> ClassicalElements:
    """Convert a non-degenerate elliptic Cartesian state to stable elements."""

    position = np.asarray(state.position_m, dtype=np.float64)
    velocity = np.asarray(state.velocity_m_s, dtype=np.float64)
    radius_m = float(np.linalg.norm(position))
    if mu_m3_s2 <= 0.0 or radius_m == 0.0:
        raise ValueError("gravitational parameter and radius must be positive")
    angular_momentum = np.cross(position, velocity)
    angular_momentum_norm = float(np.linalg.norm(angular_momentum))
    if angular_momentum_norm == 0.0:
        raise ValueError("radial trajectories do not define classical elements")
    node = np.cross(np.asarray((0.0, 0.0, 1.0)), angular_momentum)
    node_norm = float(np.linalg.norm(node))
    eccentricity_vector = (
        np.cross(velocity, angular_momentum) / mu_m3_s2 - position / radius_m
    )
    eccentricity = float(np.linalg.norm(eccentricity_vector))
    speed_squared = float(np.dot(velocity, velocity))
    specific_energy = speed_squared / 2.0 - mu_m3_s2 / radius_m
    if specific_energy >= 0.0 or eccentricity >= 1.0:
        raise ValueError("only elliptic states are supported")
    semi_major_axis_m = -mu_m3_s2 / (2.0 * specific_energy)
    inclination_rad = acos(
        _clamp_unit(float(angular_momentum[2]) / angular_momentum_norm)
    )

    orbit_is_equatorial = node_norm <= SINGULARITY_TOLERANCE
    orbit_is_circular = eccentricity <= SINGULARITY_TOLERANCE
    ascending_node_rad = (
        0.0 if orbit_is_equatorial else normalize_angle(atan2(node[1], node[0]))
    )
    if orbit_is_circular:
        periapsis_argument_rad = 0.0
        true_anomaly = (
            normalize_angle(atan2(position[1], position[0]))
            if orbit_is_equatorial
            else _oriented_angle(node, position, angular_momentum)
        )
    else:
        periapsis_argument_rad = (
            normalize_angle(atan2(eccentricity_vector[1], eccentricity_vector[0]))
            if orbit_is_equatorial
            else _oriented_angle(node, eccentricity_vector, angular_momentum)
        )
        true_anomaly = _oriented_angle(eccentricity_vector, position, angular_momentum)
    eccentric_anomaly = 2.0 * atan2(
        sqrt(1.0 - eccentricity) * sin(true_anomaly / 2.0),
        sqrt(1.0 + eccentricity) * cos(true_anomaly / 2.0),
    )
    mean_anomaly_rad = normalize_angle(
        eccentric_anomaly - eccentricity * sin(eccentric_anomaly)
    )
    return ClassicalElements(
        semi_major_axis_m,
        eccentricity,
        inclination_rad,
        ascending_node_rad,
        periapsis_argument_rad,
        mean_anomaly_rad,
    )


def analytic_kepler_step(
    state: CartesianState, mu_m3_s2: float, duration_s: float
) -> CartesianState:
    """Advance an elliptic two-body state analytically by a duration."""

    elements = state_to_elements(state, mu_m3_s2)
    mean_motion_rad_s = sqrt(mu_m3_s2 / elements.semi_major_axis_m**3)
    advanced = replace(
        elements,
        mean_anomaly_rad=(elements.mean_anomaly_rad + mean_motion_rad_s * duration_s)
        % TAU,
    )
    return elements_to_state(advanced, mu_m3_s2)
