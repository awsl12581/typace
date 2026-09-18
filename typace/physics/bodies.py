"""Force models derived from the bundled solar-system catalog."""

from functools import lru_cache

from typace.config.physics import (
    EARTH_EQUATORIAL_RADIUS_M,
    EARTH_GRAVITATIONAL_PARAMETER_M3_S2,
    EARTH_J2,
    EARTH_MEAN_RADIUS_M,
    EARTH_ROTATION_RATE_RAD_S,
    MOON_GRAVITATIONAL_PARAMETER_M3_S2,
    MOON_MEAN_RADIUS_M,
    UNIVERSAL_GRAVITATIONAL_CONSTANT_M3_KG_S2,
)
from typace.physics.forces import ForceModel
from typace.solar_system import load_solar_system

_PRECISE_MODELS = {
    "earth": ForceModel(
        "earth",
        EARTH_GRAVITATIONAL_PARAMETER_M3_S2,
        EARTH_MEAN_RADIUS_M,
        EARTH_J2,
        EARTH_EQUATORIAL_RADIUS_M,
        EARTH_ROTATION_RATE_RAD_S,
    ),
    "moon": ForceModel("moon", MOON_GRAVITATIONAL_PARAMETER_M3_S2, MOON_MEAN_RADIUS_M),
}


@lru_cache(maxsize=None)
def force_model_for(primary_body_id: str) -> ForceModel:
    precise = _PRECISE_MODELS.get(primary_body_id)
    if precise is not None:
        return precise
    try:
        body = load_solar_system().body(primary_body_id)
    except KeyError as error:
        raise ValueError("unsupported primary body") from error
    return ForceModel(
        primary_body_id,
        UNIVERSAL_GRAVITATIONAL_CONSTANT_M3_KG_S2 * body.mass_kg,
        body.radius_m,
    )
