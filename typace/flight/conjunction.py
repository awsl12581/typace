"""Deterministic same-primary conjunction screening."""

from dataclasses import dataclass
from math import inf

import numpy as np

from typace.config.simulation import PHYSICS_STEP_SECONDS
from typace.physics.bodies import force_model_for
from typace.physics.propagation import PropagationContext, TranslationalState, propagate
from typace.satellites.state import SatelliteState


@dataclass(frozen=True, slots=True)
class ConjunctionCandidate:
    first_id: str
    second_id: str
    time_to_closest_approach_s: float
    predicted_distance_m: float
    required_distance_m: float

    @property
    def yielding_satellite_id(self) -> str:
        """Select one stable actor so paired autopilots never maneuver symmetrically."""

        return max(self.first_id, self.second_id)


def screen_conjunctions(
    satellites: tuple[SatelliteState, ...],
    horizon_s: float,
) -> tuple[ConjunctionCandidate, ...]:
    if horizon_s <= 0.0:
        raise ValueError("conjunction horizon must be positive")
    candidates: list[ConjunctionCandidate] = []
    for index, first in enumerate(satellites):
        for second in satellites[index + 1 :]:
            if first.primary_body_id != second.primary_body_id:
                continue
            candidate = _screen_pair(first, second, horizon_s)
            if candidate is not None:
                candidates.append(candidate)
    return tuple(sorted(candidates, key=lambda item: (item.first_id, item.second_id)))


def _screen_pair(
    first: SatelliteState, second: SatelliteState, horizon_s: float
) -> ConjunctionCandidate | None:
    relative_position = first.translation.position_m - second.translation.position_m
    relative_velocity = first.translation.velocity_m_s - second.translation.velocity_m_s
    speed_squared = float(np.dot(relative_velocity, relative_velocity))
    closest_time_s = 0.0
    if speed_squared > 0.0:
        closest_time_s = float(
            np.clip(
                -np.dot(relative_position, relative_velocity) / speed_squared,
                0.0,
                horizon_s,
            )
        )
    first_state = _propagate(first, closest_time_s)
    second_state = _propagate(second, closest_time_s)
    distance_m = float(np.linalg.norm(first_state.position_m - second_state.position_m))
    required_m = (
        first.definition.autonomy.avoidance_distance_m
        + second.definition.autonomy.avoidance_distance_m
    )
    if distance_m >= required_m:
        return None
    return ConjunctionCandidate(
        min(first.definition.id, second.definition.id),
        max(first.definition.id, second.definition.id),
        closest_time_s,
        distance_m,
        required_m,
    )


def _propagate(satellite: SatelliteState, duration_s: float) -> TranslationalState:
    return propagate(
        satellite.translation,
        PropagationContext(
            force_model_for(satellite.primary_body_id),
            satellite.definition.drag_area_m2,
            satellite.definition.drag_coefficient,
            np.zeros(3),
        ),
        duration_s,
        PHYSICS_STEP_SECONDS,
    )
