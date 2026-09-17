"""Deterministic out-of-plane conjunction avoidance planning."""

from dataclasses import dataclass

import numpy as np

from typace.config.flight import AVOIDANCE_MANEUVER_LEAD_FRACTION
from typace.config.physics import STANDARD_GRAVITY_M_S2
from typace.satellites.definition import SatelliteDefinition
from typace.flight.models import (
    FlightPlan,
    PlanningFailure,
    PlanningFailureCode,
    PlanningResult,
)
from typace.flight.navigation import NavigationSolution
from typace.flight.planning.burns import corrected_burn


@dataclass(frozen=True, slots=True)
class ConjunctionRisk:
    id: str
    time_to_closest_approach_s: float
    predicted_distance_m: float
    required_distance_m: float


def plan_avoidance(
    navigation: NavigationSolution | None,
    definition: SatelliteDefinition,
    risk: ConjunctionRisk,
) -> PlanningResult:
    if navigation is None:
        return PlanningFailure(PlanningFailureCode.POWER_SAFE, "navigation is frozen")
    if risk.predicted_distance_m >= risk.required_distance_m:
        return PlanningFailure(
            PlanningFailureCode.NO_CORRECTION, "predicted separation is already safe"
        )
    if risk.time_to_closest_approach_s <= 0.0:
        return PlanningFailure(
            PlanningFailureCode.MISSED_WINDOW, "avoidance window has passed"
        )
    maneuver_time_s = risk.time_to_closest_approach_s * AVOIDANCE_MANEUVER_LEAD_FRACTION
    required_displacement_m = risk.required_distance_m - risk.predicted_distance_m
    delta_v_m_s = required_displacement_m / maneuver_time_s
    orbit_normal = np.cross(navigation.position_m, navigation.velocity_m_s)
    normal_length = float(np.linalg.norm(orbit_normal))
    if normal_length == 0.0:
        return PlanningFailure(
            PlanningFailureCode.UNREACHABLE, "orbit normal is undefined"
        )
    burn = corrected_burn(
        navigation,
        definition,
        orbit_normal / normal_length * delta_v_m_s,
        "avoid_collision",
    )
    if isinstance(burn, PlanningFailure):
        return burn
    mass_flow_kg_s = definition.propulsion.main_thrust_n / (
        definition.propulsion.main_specific_impulse_s * STANDARD_GRAVITY_M_S2
    )
    return FlightPlan(
        f"avoid_{risk.id}",
        (burn,),
        "predicted closest approach exceeds required distance",
        mass_flow_kg_s * burn.duration_s,
        navigation.primary_body_id,
    )
