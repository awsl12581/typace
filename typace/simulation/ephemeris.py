"""Offline solar-system ephemeris derived from the bundled body catalog."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import inf
from typing import cast

from astropy.time import Time, TimeDelta
import numpy as np
from numpy.typing import NDArray

from typace.celestial.model import CelestialSystem
from typace.celestial.orbital import body_kinematics
from typace.physics.soi import OriginState, change_primary, sphere_of_influence_radius_m
from typace.solar_system import load_solar_system

type Vector = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class BodyState:
    position_m: Vector
    velocity_m_s: Vector


@dataclass(frozen=True, slots=True)
class SolarSystemEphemeris:
    system: CelestialSystem
    states: Mapping[str, BodyState]

    def relative(self, target_id: str, origin_id: str) -> BodyState:
        target = self.body(target_id)
        origin = self.body(origin_id)
        return BodyState(
            target.position_m - origin.position_m,
            target.velocity_m_s - origin.velocity_m_s,
        )

    def body(self, body_id: str) -> BodyState:
        return self.states[body_id]

    def parent_id(self, body_id: str) -> str | None:
        parent = self.system.parent_of(self.system.body(body_id))
        return None if parent is None else parent.id

    def common_ancestor_id(self, first_id: str, second_id: str) -> str:
        first_ancestors = set(self._lineage(first_id))
        for body_id in self._lineage(second_id):
            if body_id in first_ancestors:
                return body_id
        raise ValueError("bodies do not share a catalog ancestor")

    def sphere_of_influence_radius_m(self, body_id: str) -> float:
        body = self.system.body(body_id)
        parent = self.system.parent_of(body)
        if parent is None:
            return inf
        return sphere_of_influence_radius_m(
            body.semimajor_axis_m,
            body.mass_kg,
            parent.mass_kg,
        )

    def translate_primary(
        self,
        position_m: Vector,
        velocity_m_s: Vector,
        old_primary_id: str,
        new_primary_id: str,
    ) -> tuple[Vector, Vector]:
        old = self.body(old_primary_id)
        new = self.body(new_primary_id)
        return change_primary(
            position_m,
            velocity_m_s,
            OriginState(old.position_m, old.velocity_m_s),
            OriginState(new.position_m, new.velocity_m_s),
        )

    def _lineage(self, body_id: str) -> tuple[str, ...]:
        lineage: list[str] = []
        current_id: str | None = body_id
        while current_id is not None:
            lineage.append(current_id)
            current_id = self.parent_id(current_id)
        return tuple(lineage)


def ephemeris_at(
    epoch: Time, system: CelestialSystem | None = None
) -> SolarSystemEphemeris:
    active_system = load_solar_system() if system is None else system
    epoch_delta = cast(TimeDelta, epoch - Time("J2000", scale="tt"))
    elapsed_seconds = cast(float, epoch_delta.to_value("sec"))
    states = {
        body_id: BodyState(state.position_m, state.velocity_m_s)
        for body_id, state in body_kinematics(active_system, elapsed_seconds).items()
    }
    return SolarSystemEphemeris(active_system, states)
