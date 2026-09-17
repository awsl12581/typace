"""Canonical Earth and Moon force models."""

from typace.config.physics import (
    EARTH_EQUATORIAL_RADIUS_M,
    EARTH_GRAVITATIONAL_PARAMETER_M3_S2,
    EARTH_J2,
    EARTH_MEAN_RADIUS_M,
    EARTH_ROTATION_RATE_RAD_S,
    MOON_GRAVITATIONAL_PARAMETER_M3_S2,
    MOON_MEAN_RADIUS_M,
)
from typace.physics.forces import ForceModel


def force_model_for(primary_body_id: str) -> ForceModel:
    if primary_body_id == "earth":
        return ForceModel(
            "earth",
            EARTH_GRAVITATIONAL_PARAMETER_M3_S2,
            EARTH_MEAN_RADIUS_M,
            EARTH_J2,
            EARTH_EQUATORIAL_RADIUS_M,
            EARTH_ROTATION_RATE_RAD_S,
        )
    if primary_body_id == "moon":
        return ForceModel(
            "moon", MOON_GRAVITATIONAL_PARAMETER_M3_S2, MOON_MEAN_RADIUS_M
        )
    raise ValueError("unsupported primary body")
