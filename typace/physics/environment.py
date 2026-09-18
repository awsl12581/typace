"""Atmosphere, relative flow, eclipse, and re-entry observables."""

from dataclasses import dataclass
from math import exp, sqrt

import numpy as np
from numpy.typing import NDArray

from typace.config.physics import (
    EARTH_ATMOSPHERE_LAYER_ALTITUDES_M,
    EARTH_ATMOSPHERE_SCALE_HEIGHTS_M,
    EARTH_ATMOSPHERE_SEA_LEVEL_DENSITY_KG_M3,
    SUTTON_GRAVES_EARTH_COEFFICIENT,
)

type Vector = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class OccludingBody:
    center_m: Vector
    radius_m: float


def _base_densities() -> tuple[float, ...]:
    densities = [EARTH_ATMOSPHERE_SEA_LEVEL_DENSITY_KG_M3]
    for index in range(1, len(EARTH_ATMOSPHERE_LAYER_ALTITUDES_M)):
        layer_thickness_m = (
            EARTH_ATMOSPHERE_LAYER_ALTITUDES_M[index]
            - EARTH_ATMOSPHERE_LAYER_ALTITUDES_M[index - 1]
        )
        densities.append(
            densities[-1]
            * exp(-layer_thickness_m / EARTH_ATMOSPHERE_SCALE_HEIGHTS_M[index - 1])
        )
    return tuple(densities)


_EARTH_ATMOSPHERE_BASE_DENSITIES_KG_M3 = _base_densities()


def atmospheric_density_kg_m3(altitude_m: float) -> float:
    """Return a continuous, non-increasing engineering density profile."""

    bounded_altitude_m = max(0.0, altitude_m)
    layer_index = len(EARTH_ATMOSPHERE_LAYER_ALTITUDES_M) - 1
    for index, base_altitude_m in enumerate(EARTH_ATMOSPHERE_LAYER_ALTITUDES_M):
        if bounded_altitude_m < base_altitude_m:
            layer_index = max(0, index - 1)
            break
    altitude_above_base_m = (
        bounded_altitude_m - EARTH_ATMOSPHERE_LAYER_ALTITUDES_M[layer_index]
    )
    return _EARTH_ATMOSPHERE_BASE_DENSITIES_KG_M3[layer_index] * exp(
        -altitude_above_base_m / EARTH_ATMOSPHERE_SCALE_HEIGHTS_M[layer_index]
    )


def atmosphere_relative_velocity_m_s(
    position_m: Vector, inertial_velocity_m_s: Vector, rotation_rate_rad_s: float
) -> Vector:
    atmosphere_velocity_m_s = np.asarray(
        (
            -rotation_rate_rad_s * position_m[1],
            rotation_rate_rad_s * position_m[0],
            0.0,
        ),
        dtype=np.float64,
    )
    return inertial_velocity_m_s - atmosphere_velocity_m_s


def dynamic_pressure_pa(density_kg_m3: float, relative_velocity_m_s: Vector) -> float:
    speed_m_s = float(np.linalg.norm(relative_velocity_m_s))
    return 0.5 * density_kg_m3 * speed_m_s**2


def stagnation_heat_flux_w_m2(
    density_kg_m3: float, nose_radius_m: float, relative_velocity_m_s: Vector
) -> float:
    if nose_radius_m <= 0.0:
        raise ValueError("nose radius must be positive")
    speed_m_s = float(np.linalg.norm(relative_velocity_m_s))
    return (
        SUTTON_GRAVES_EARTH_COEFFICIENT
        * sqrt(max(0.0, density_kg_m3) / nose_radius_m)
        * speed_m_s**3
    )


def segment_intersects_sphere(
    start_m: Vector, end_m: Vector, body: OccludingBody
) -> bool:
    """Return whether the open segment crosses an occluding body sphere."""

    segment = end_m - start_m
    segment_length_squared = float(np.dot(segment, segment))
    if segment_length_squared == 0.0:
        return False
    fraction = float(np.dot(body.center_m - start_m, segment)) / segment_length_squared
    clamped_fraction = max(0.0, min(1.0, fraction))
    closest = start_m + clamped_fraction * segment
    return float(np.linalg.norm(closest - body.center_m)) <= body.radius_m


def is_fully_eclipsed(
    satellite_position_m: Vector,
    sun_position_m: Vector,
    occluding_bodies: tuple[OccludingBody, ...],
) -> bool:
    return any(
        segment_intersects_sphere(satellite_position_m, sun_position_m, body)
        for body in occluding_bodies
    )
