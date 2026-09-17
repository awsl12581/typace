"""Reference-frame transforms with explicit SI vectors."""

from math import cos, sin

import numpy as np
from numpy.typing import NDArray

type Vector = NDArray[np.float64]
type Matrix = NDArray[np.float64]


def perifocal_to_inertial_matrix(
    ascending_node_rad: float,
    inclination_rad: float,
    periapsis_argument_rad: float,
) -> Matrix:
    """Return the right-handed perifocal-to-inertial rotation matrix."""

    node_cos = cos(ascending_node_rad)
    node_sin = sin(ascending_node_rad)
    inclination_cos = cos(inclination_rad)
    inclination_sin = sin(inclination_rad)
    periapsis_cos = cos(periapsis_argument_rad)
    periapsis_sin = sin(periapsis_argument_rad)
    return np.asarray(
        (
            (
                node_cos * periapsis_cos - node_sin * periapsis_sin * inclination_cos,
                -node_cos * periapsis_sin - node_sin * periapsis_cos * inclination_cos,
                node_sin * inclination_sin,
            ),
            (
                node_sin * periapsis_cos + node_cos * periapsis_sin * inclination_cos,
                -node_sin * periapsis_sin + node_cos * periapsis_cos * inclination_cos,
                -node_cos * inclination_sin,
            ),
            (
                periapsis_sin * inclination_sin,
                periapsis_cos * inclination_sin,
                inclination_cos,
            ),
        ),
        dtype=np.float64,
    )


def radial_transverse_normal_basis(position_m: Vector, velocity_m_s: Vector) -> Matrix:
    """Return inertial columns for radial, transverse, and orbit-normal axes."""

    radial_norm = float(np.linalg.norm(position_m))
    angular_momentum = np.cross(position_m, velocity_m_s)
    normal_norm = float(np.linalg.norm(angular_momentum))
    if radial_norm == 0.0 or normal_norm == 0.0:
        raise ValueError("position and angular momentum must be non-zero")
    radial = position_m / radial_norm
    normal = angular_momentum / normal_norm
    transverse = np.cross(normal, radial)
    return np.column_stack((radial, transverse, normal))


def translate_relative_state(
    position_m: Vector,
    velocity_m_s: Vector,
    old_origin_position_m: Vector,
    old_origin_velocity_m_s: Vector,
    new_origin_position_m: Vector,
    new_origin_velocity_m_s: Vector,
) -> tuple[Vector, Vector]:
    """Change origins while preserving inertial position and velocity."""

    inertial_position = position_m + old_origin_position_m
    inertial_velocity = velocity_m_s + old_origin_velocity_m_s
    return (
        inertial_position - new_origin_position_m,
        inertial_velocity - new_origin_velocity_m_s,
    )
