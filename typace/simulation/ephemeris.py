"""Offline Astropy ephemeris adapter for Earth, Moon, and Sun."""

from dataclasses import dataclass
from typing import cast

from astropy import units
from astropy.coordinates import get_body_barycentric_posvel
from astropy.coordinates.representation import CartesianRepresentation
from astropy.time import Time
import numpy as np
from numpy.typing import NDArray

from typace.config.astronomy import configure_offline_astronomy

type Vector = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class BodyState:
    position_m: Vector
    velocity_m_s: Vector


@dataclass(frozen=True, slots=True)
class EarthMoonEphemeris:
    earth: BodyState
    moon: BodyState
    sun: BodyState

    def relative(self, target_id: str, origin_id: str) -> BodyState:
        target = self._body(target_id)
        origin = self._body(origin_id)
        return BodyState(
            target.position_m - origin.position_m,
            target.velocity_m_s - origin.velocity_m_s,
        )

    def _body(self, body_id: str) -> BodyState:
        if body_id == "earth":
            return self.earth
        if body_id == "moon":
            return self.moon
        if body_id == "sun":
            return self.sun
        raise KeyError(body_id)


def ephemeris_at(epoch: Time) -> EarthMoonEphemeris:
    configure_offline_astronomy()
    return EarthMoonEphemeris(
        _body_state("earth", epoch),
        _body_state("moon", epoch),
        _body_state("sun", epoch),
    )


def _body_state(body_id: str, epoch: Time) -> BodyState:
    position, velocity = get_body_barycentric_posvel(body_id, epoch)
    position_representation = cast(CartesianRepresentation, position)
    velocity_representation = cast(CartesianRepresentation, velocity)
    return BodyState(
        np.asarray(position_representation.xyz.to_value(units.m), dtype=np.float64),
        np.asarray(
            velocity_representation.xyz.to_value(units.m / units.s),
            dtype=np.float64,
        ),
    )
