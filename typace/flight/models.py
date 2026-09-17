"""Immutable flight plan and planning failure contracts."""

from dataclasses import dataclass
from enum import StrEnum

type Vector3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class BurnStep:
    start_after_s: float
    duration_s: float
    throttle: float
    delta_v_m_s: Vector3
    purpose: str


@dataclass(frozen=True, slots=True)
class CoastStep:
    duration_s: float


type FlightPlanStep = BurnStep | CoastStep


@dataclass(frozen=True, slots=True)
class FlightPlan:
    objective_id: str
    steps: tuple[FlightPlanStep, ...]
    completion_condition: str
    estimated_propellant_kg: float
    target_primary_body_id: str


class PlanningFailureCode(StrEnum):
    POWER_SAFE = "power_safe"
    INVALID_TARGET = "invalid_target"
    INSUFFICIENT_PROPELLANT = "insufficient_propellant"
    MISSED_WINDOW = "missed_window"
    UNREACHABLE = "unreachable"
    NO_CORRECTION = "no_correction"


@dataclass(frozen=True, slots=True)
class PlanningFailure:
    code: PlanningFailureCode
    reason: str


type PlanningResult = FlightPlan | PlanningFailure
