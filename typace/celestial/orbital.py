"""Keplerian positions for catalog bodies."""

from dataclasses import dataclass
from math import atan2, cos, pi, radians, sin, sqrt
from typing import Mapping

import numpy as np

from typace.celestial.model import CelestialBody, CelestialSystem, Vec3
from typace.physics.frames import perifocal_to_inertial_matrix
from typace.physics.kepler import solve_eccentric_anomaly

DAY_SECONDS = 86_400.0
TAU = 2.0 * pi


@dataclass(frozen=True, slots=True)
class OrbitElements:
    semimajor_axis_m: float
    eccentricity: float
    inclination_rad: float
    ascending_node_rad: float
    periapsis_argument_rad: float


@dataclass(frozen=True, slots=True)
class CelestialState:
    """Positions for every body at one instant in a celestial system."""

    system: CelestialSystem
    elapsed_seconds: float
    positions: Mapping[str, Vec3]


@dataclass(frozen=True, slots=True)
class BodyKinematics:
    position_m: np.ndarray
    velocity_m_s: np.ndarray


def elements_from_body(body: CelestialBody) -> OrbitElements:
    return OrbitElements(
        body.semimajor_axis_m,
        body.eccentricity,
        radians(body.inclination_deg),
        radians(body.ascending_node_deg),
        radians(body.periapsis_argument_deg),
    )


def mean_anomaly(body: CelestialBody, elapsed_seconds: float) -> float:
    if body.mean_anomaly_at_epoch_deg is None:
        seed_deg = (
            body.semimajor_axis_m / 1000.0
            + body.orbit_period_days
            + body.radius_m / 1000.0
        )
    else:
        seed_deg = body.mean_anomaly_at_epoch_deg
    initial = radians(seed_deg) % TAU
    if body.orbit_period_days <= 0:
        return initial
    return (
        initial + TAU * elapsed_seconds / (body.orbit_period_days * DAY_SECONDS)
    ) % TAU


def solve_kepler(mean: float, eccentricity: float) -> float:
    """Solve through the shared SciPy-backed production implementation."""

    return solve_eccentric_anomaly(mean, eccentricity)


def true_anomaly(eccentric_anomaly: float, eccentricity: float) -> float:
    return 2.0 * atan2(
        sqrt(1.0 + eccentricity) * sin(eccentric_anomaly / 2.0),
        sqrt(1.0 - eccentricity) * cos(eccentric_anomaly / 2.0),
    )


def position_at_true_anomaly(elements: OrbitElements, anomaly: float) -> Vec3:
    if elements.semimajor_axis_m == 0:
        return Vec3()
    parameter = elements.semimajor_axis_m * (1.0 - elements.eccentricity**2)
    radius = parameter / (1.0 + elements.eccentricity * cos(anomaly))
    perifocal_x = radius * cos(anomaly)
    perifocal_y = radius * sin(anomaly)

    rotation = perifocal_to_inertial_matrix(
        elements.ascending_node_rad,
        elements.inclination_rad,
        elements.periapsis_argument_rad,
    )
    position = rotation @ np.asarray((perifocal_x, perifocal_y, 0.0))
    return Vec3(float(position[0]), float(position[1]), float(position[2]))


def relative_position(body: CelestialBody, elapsed_seconds: float) -> Vec3:
    mean = mean_anomaly(body, elapsed_seconds)
    eccentric = solve_kepler(mean, body.eccentricity)
    return position_at_true_anomaly(
        elements_from_body(body), true_anomaly(eccentric, body.eccentricity)
    )


def relative_kinematics(body: CelestialBody, elapsed_seconds: float) -> BodyKinematics:
    elements = elements_from_body(body)
    eccentric_anomaly = solve_kepler(
        mean_anomaly(body, elapsed_seconds), elements.eccentricity
    )
    semimajor_axis_m = elements.semimajor_axis_m
    eccentricity = elements.eccentricity
    root = sqrt(1.0 - eccentricity**2)
    cosine = cos(eccentric_anomaly)
    sine = sin(eccentric_anomaly)
    position_perifocal_m = np.asarray(
        (
            semimajor_axis_m * (cosine - eccentricity),
            semimajor_axis_m * root * sine,
            0.0,
        )
    )
    velocity_perifocal_m_s = np.zeros(3)
    if body.orbit_period_days > 0.0 and semimajor_axis_m > 0.0:
        mean_motion_rad_s = TAU / (body.orbit_period_days * DAY_SECONDS)
        eccentric_rate_rad_s = mean_motion_rad_s / (1.0 - eccentricity * cosine)
        velocity_perifocal_m_s = np.asarray(
            (
                -semimajor_axis_m * sine * eccentric_rate_rad_s,
                semimajor_axis_m * root * cosine * eccentric_rate_rad_s,
                0.0,
            )
        )
    rotation = perifocal_to_inertial_matrix(
        elements.ascending_node_rad,
        elements.inclination_rad,
        elements.periapsis_argument_rad,
    )
    return BodyKinematics(
        rotation @ position_perifocal_m,
        rotation @ velocity_perifocal_m_s,
    )


def body_kinematics(
    system: CelestialSystem, elapsed_seconds: float
) -> dict[str, BodyKinematics]:
    """Return system-inertial position and velocity for every catalog body."""

    states: dict[str, BodyKinematics] = {}

    def resolve(body: CelestialBody) -> BodyKinematics:
        if body.id in states:
            return states[body.id]
        local = relative_kinematics(body, elapsed_seconds)
        parent = system.parent_of(body)
        if parent is None:
            states[body.id] = local
            return local
        parent_state = resolve(parent)
        combined = BodyKinematics(
            parent_state.position_m + local.position_m,
            parent_state.velocity_m_s + local.velocity_m_s,
        )
        states[body.id] = combined
        return combined

    for body in system.bodies:
        resolve(body)
    return states


def body_positions(system: CelestialSystem, elapsed_seconds: float) -> dict[str, Vec3]:
    """Return system-inertial positions, recursively offset by each parent."""
    positions: dict[str, Vec3] = {}

    def resolve(body: CelestialBody) -> Vec3:
        if body.id in positions:
            return positions[body.id]
        local = relative_position(body, elapsed_seconds)
        parent = system.parent_of(body)
        positions[body.id] = resolve(parent) + local if parent is not None else local
        return positions[body.id]

    for body in system.bodies:
        resolve(body)
    return positions


def state_at(system: CelestialSystem, elapsed_seconds: float) -> CelestialState:
    """Calculate a consistent system snapshot for rendering and phenomena."""
    return CelestialState(
        system, elapsed_seconds, body_positions(system, elapsed_seconds)
    )
