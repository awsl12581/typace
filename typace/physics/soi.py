"""Patched-conic sphere-of-influence geometry."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from typace.physics.frames import translate_relative_state

type Vector = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class OriginState:
    position_m: Vector
    velocity_m_s: Vector


def sphere_of_influence_radius_m(
    orbital_semi_major_axis_m: float,
    body_mass_kg: float,
    parent_mass_kg: float,
) -> float:
    """Return a(m/M)^(2/5), the classical patched-conic SOI radius."""

    if orbital_semi_major_axis_m <= 0.0:
        raise ValueError("orbital semi-major axis must be positive")
    if body_mass_kg <= 0.0 or parent_mass_kg <= body_mass_kg:
        raise ValueError("body masses must be positive and parent mass must be larger")
    return orbital_semi_major_axis_m * (body_mass_kg / parent_mass_kg) ** 0.4


def change_primary(
    relative_position_m: Vector,
    relative_velocity_m_s: Vector,
    old_origin: OriginState,
    new_origin: OriginState,
) -> tuple[Vector, Vector]:
    """Translate a state between primaries without an inertial discontinuity."""

    return translate_relative_state(
        np.asarray(relative_position_m, dtype=np.float64),
        np.asarray(relative_velocity_m_s, dtype=np.float64),
        old_origin.position_m,
        old_origin.velocity_m_s,
        new_origin.position_m,
        new_origin.velocity_m_s,
    )
