"""Immutable commands shared by UI, automation, and the simulation world."""

from dataclasses import dataclass
from enum import StrEnum

type Vector3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class MaintainOrbit:
    pass


@dataclass(frozen=True, slots=True)
class SetOrbitAltitude:
    altitude_m: float


@dataclass(frozen=True, slots=True)
class SetApsides:
    periapsis_altitude_m: float
    apoapsis_altitude_m: float


@dataclass(frozen=True, slots=True)
class SetInclination:
    inclination_rad: float


@dataclass(frozen=True, slots=True)
class TransferPrimary:
    primary_body_id: str


@dataclass(frozen=True, slots=True)
class ReturnStableOrbit:
    pass


@dataclass(frozen=True, slots=True)
class Deorbit:
    pass


@dataclass(frozen=True, slots=True)
class AvoidCollision:
    conjunction_id: str


@dataclass(frozen=True, slots=True)
class TakeManualControl:
    pass


@dataclass(frozen=True, slots=True)
class ReturnAutonomousControl:
    pass


@dataclass(frozen=True, slots=True)
class ManualActuation:
    main_throttle: float
    wheel_torque_n_m: Vector3
    rcs_torque_n_m: Vector3


type SatelliteCommand = (
    MaintainOrbit
    | SetOrbitAltitude
    | SetApsides
    | SetInclination
    | TransferPrimary
    | ReturnStableOrbit
    | Deorbit
    | AvoidCollision
    | TakeManualControl
    | ReturnAutonomousControl
    | ManualActuation
)


class CommandStatus(StrEnum):
    ACCEPTED = "accepted"
    UNKNOWN_SATELLITE = "unknown_satellite"
    NOT_ALLOWED = "not_allowed"
    INVALID = "invalid"
    POWER_SAFE = "power_safe"
    QUEUE_FULL = "queue_full"


@dataclass(frozen=True, slots=True)
class CommandResult:
    status: CommandStatus
    reason: str = ""

    @property
    def accepted(self) -> bool:
        return self.status is CommandStatus.ACCEPTED
