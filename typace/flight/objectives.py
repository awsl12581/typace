"""Fixed-priority selection of one satellite flight objective."""

from dataclasses import dataclass
from enum import IntEnum

from typace.satellites.commands import SatelliteCommand


class ObjectivePriority(IntEnum):
    COLLISION_AVOIDANCE = 1
    VEHICLE_SAFETY = 2
    ACTIVE_BURN = 3
    USER_COMMAND = 4
    ORBIT_MAINTENANCE = 5


@dataclass(frozen=True, slots=True)
class FlightObjective:
    id: str
    priority: ObjectivePriority
    command: SatelliteCommand


def select_objective(
    candidates: tuple[FlightObjective, ...],
) -> FlightObjective | None:
    if not candidates:
        return None
    return min(candidates, key=lambda objective: (objective.priority, objective.id))
