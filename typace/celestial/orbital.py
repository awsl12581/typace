"""Keplerian positions for catalog bodies."""

from dataclasses import dataclass
from math import atan2, cos, pi, radians, sin, sqrt
from typing import Mapping

from typace.celestial.model import CelestialBody, CelestialSystem, Vec3

DAY_SECONDS = 86_400.0
TAU = 2.0 * pi
KEPLER_TOLERANCE = 1e-12
KEPLER_MAX_ITERATIONS = 32


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
    """Solve M = E - e sin(E) with Newton-Raphson iteration."""
    normalized = (mean + pi) % TAU - pi
    anomaly = normalized + eccentricity * sin(normalized)
    for _ in range(KEPLER_MAX_ITERATIONS):
        derivative = 1.0 - eccentricity * cos(anomaly)
        if derivative == 0:
            break
        correction = (anomaly - eccentricity * sin(anomaly) - normalized) / derivative
        anomaly -= correction
        if abs(correction) < KEPLER_TOLERANCE:
            break
    return anomaly


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

    node_cos = cos(elements.ascending_node_rad)
    node_sin = sin(elements.ascending_node_rad)
    periapsis_cos = cos(elements.periapsis_argument_rad)
    periapsis_sin = sin(elements.periapsis_argument_rad)
    inclination_cos = cos(elements.inclination_rad)
    inclination_sin = sin(elements.inclination_rad)
    return Vec3(
        perifocal_x
        * (node_cos * periapsis_cos - node_sin * periapsis_sin * inclination_cos)
        + perifocal_y
        * (-node_cos * periapsis_sin - node_sin * periapsis_cos * inclination_cos),
        perifocal_x
        * (node_sin * periapsis_cos + node_cos * periapsis_sin * inclination_cos)
        + perifocal_y
        * (-node_sin * periapsis_sin + node_cos * periapsis_cos * inclination_cos),
        perifocal_x * periapsis_sin * inclination_sin
        + perifocal_y * periapsis_cos * inclination_sin,
    )


def relative_position(body: CelestialBody, elapsed_seconds: float) -> Vec3:
    mean = mean_anomaly(body, elapsed_seconds)
    eccentric = solve_kepler(mean, body.eccentricity)
    return position_at_true_anomaly(
        elements_from_body(body), true_anomaly(eccentric, body.eccentricity)
    )


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
