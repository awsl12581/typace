"""Pure acceleration functions for translation propagation."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from typace.physics.environment import (
    atmosphere_relative_velocity_m_s,
    atmospheric_density_kg_m3,
)

type Vector = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ForceModel:
    primary_body_id: str
    gravitational_parameter_m3_s2: float
    body_radius_m: float
    j2: float = 0.0
    equatorial_radius_m: float = 0.0
    atmosphere_rotation_rate_rad_s: float = 0.0


@dataclass(frozen=True, slots=True)
class ForceInputs:
    mass_kg: float
    drag_area_m2: float
    drag_coefficient: float
    thrust_inertial_n: Vector


def has_j2_acceleration(model: ForceModel) -> bool:
    return model.primary_body_id == "earth" and model.j2 != 0.0


def has_drag_acceleration(
    model: ForceModel, drag_area_m2: float, drag_coefficient: float
) -> bool:
    return (
        model.primary_body_id == "earth"
        and drag_area_m2 > 0.0
        and drag_coefficient > 0.0
    )


def central_gravity_acceleration_m_s2(position_m: Vector, mu_m3_s2: float) -> Vector:
    radius_m = float(np.linalg.norm(position_m))
    if radius_m == 0.0 or mu_m3_s2 < 0.0:
        raise ValueError("radius must be non-zero and gravity cannot be negative")
    return -mu_m3_s2 * position_m / radius_m**3


def j2_acceleration_m_s2(
    position_m: Vector,
    mu_m3_s2: float,
    equatorial_radius_m: float,
    j2: float,
) -> Vector:
    radius_m = float(np.linalg.norm(position_m))
    if radius_m == 0.0:
        raise ValueError("radius must be non-zero")
    x_m, y_m, z_m = position_m
    z_ratio_squared = (z_m / radius_m) ** 2
    factor = 1.5 * j2 * mu_m3_s2 * equatorial_radius_m**2 / radius_m**5
    return factor * np.asarray(
        (
            x_m * (5.0 * z_ratio_squared - 1.0),
            y_m * (5.0 * z_ratio_squared - 1.0),
            z_m * (5.0 * z_ratio_squared - 3.0),
        ),
        dtype=np.float64,
    )


def drag_acceleration_m_s2(
    position_m: Vector,
    velocity_m_s: Vector,
    body_radius_m: float,
    rotation_rate_rad_s: float,
    mass_kg: float,
    drag_area_m2: float,
    drag_coefficient: float,
) -> Vector:
    if mass_kg <= 0.0:
        raise ValueError("mass must be positive")
    altitude_m = float(np.linalg.norm(position_m)) - body_radius_m
    density_kg_m3 = atmospheric_density_kg_m3(altitude_m)
    relative_velocity = atmosphere_relative_velocity_m_s(
        position_m, velocity_m_s, rotation_rate_rad_s
    )
    speed_m_s = float(np.linalg.norm(relative_velocity))
    return (
        -0.5
        * density_kg_m3
        * speed_m_s
        * drag_coefficient
        * drag_area_m2
        / mass_kg
        * relative_velocity
    )


def total_acceleration_m_s2(
    model: ForceModel,
    position_m: Vector,
    velocity_m_s: Vector,
    inputs: ForceInputs,
) -> Vector:
    if inputs.mass_kg <= 0.0:
        raise ValueError("mass must be positive")
    acceleration = central_gravity_acceleration_m_s2(
        position_m, model.gravitational_parameter_m3_s2
    )
    if has_j2_acceleration(model):
        acceleration = acceleration + j2_acceleration_m_s2(
            position_m,
            model.gravitational_parameter_m3_s2,
            model.equatorial_radius_m,
            model.j2,
        )
    if has_drag_acceleration(model, inputs.drag_area_m2, inputs.drag_coefficient):
        acceleration = acceleration + drag_acceleration_m_s2(
            position_m,
            velocity_m_s,
            model.body_radius_m,
            model.atmosphere_rotation_rate_rad_s,
            inputs.mass_kg,
            inputs.drag_area_m2,
            inputs.drag_coefficient,
        )
    return acceleration + inputs.thrust_inertial_n / inputs.mass_kg
