"""Fixed Izzo zero-revolution patched-conic transfer planning."""

from dataclasses import dataclass
from math import sqrt

import numpy as np

from typace.config.physics import STANDARD_GRAVITY_M_S2
from typace.physics.bodies import force_model_for
from typace.physics.lambert import LambertFailure, solve_lambert
from typace.satellites.definition import SatelliteDefinition
from typace.flight.models import (
    BurnStep,
    CoastStep,
    FlightPlan,
    PlanningFailure,
    PlanningFailureCode,
    PlanningResult,
)
from typace.flight.navigation import NavigationSolution
from typace.flight.planning.burns import corrected_burn


@dataclass(frozen=True, slots=True)
class TransferTarget:
    primary_body_id: str
    departure_origin_position_earth_m: np.ndarray
    departure_origin_velocity_earth_m_s: np.ndarray
    target_body_position_earth_m: np.ndarray
    target_body_velocity_earth_m_s: np.ndarray
    insertion_position_m: np.ndarray
    flight_time_s: float
    departure_after_s: float = 0.0
    prograde: bool = True
    low_path: bool = True


def plan_transfer(
    navigation: NavigationSolution | None,
    definition: SatelliteDefinition,
    target: TransferTarget,
) -> PlanningResult:
    if navigation is None:
        return PlanningFailure(PlanningFailureCode.POWER_SAFE, "navigation is frozen")
    if target.departure_after_s < 0.0:
        return PlanningFailure(
            PlanningFailureCode.MISSED_WINDOW, "departure window passed"
        )
    target_is_supported = (
        target.primary_body_id in ("earth", "moon")
        and target.primary_body_id != navigation.primary_body_id
        and target.flight_time_s > 0.0
    )
    target_vectors = (
        target.departure_origin_position_earth_m,
        target.departure_origin_velocity_earth_m_s,
        target.target_body_position_earth_m,
        target.target_body_velocity_earth_m_s,
        target.insertion_position_m,
    )
    vectors_are_valid = all(
        vector.shape == (3,) and np.isfinite(vector).all() for vector in target_vectors
    )
    insertion_is_nonzero = float(np.linalg.norm(target.insertion_position_m)) > 0.0
    if not (target_is_supported and vectors_are_valid and insertion_is_nonzero):
        return PlanningFailure(
            PlanningFailureCode.INVALID_TARGET, "transfer target is invalid"
        )
    earth_model = force_model_for("earth")
    departure_position_earth_m = (
        target.departure_origin_position_earth_m + navigation.position_m
    )
    departure_velocity_earth_m_s = (
        target.departure_origin_velocity_earth_m_s + navigation.velocity_m_s
    )
    arrival_position_earth_m = (
        target.target_body_position_earth_m + target.insertion_position_m
    )
    try:
        lambert = solve_lambert(
            earth_model.gravitational_parameter_m3_s2,
            departure_position_earth_m,
            arrival_position_earth_m,
            target.flight_time_s,
            prograde=target.prograde,
            low_path=target.low_path,
        )
    except LambertFailure:
        return PlanningFailure(
            PlanningFailureCode.UNREACHABLE, "Lambert transfer is unreachable"
        )
    departure_delta_v = lambert.departure_velocity_m_s - departure_velocity_earth_m_s
    departure_burn = corrected_burn(
        navigation, definition, departure_delta_v, "transfer:departure"
    )
    if isinstance(departure_burn, PlanningFailure):
        return departure_burn

    capture_delta_v = _capture_delta_v_m_s(target, lambert.arrival_velocity_m_s)
    mass_flow_kg_s = definition.propulsion.main_thrust_n / (
        definition.propulsion.main_specific_impulse_s * STANDARD_GRAVITY_M_S2
    )
    departure_propellant_kg = mass_flow_kg_s * departure_burn.duration_s
    arrival_relative_velocity_m_s = (
        lambert.arrival_velocity_m_s - target.target_body_velocity_earth_m_s
    )
    arrival_navigation = NavigationSolution(
        navigation.satellite_id,
        target.primary_body_id,
        target.insertion_position_m,
        arrival_relative_velocity_m_s,
        navigation.mass_kg - departure_propellant_kg,
        navigation.main_propellant_kg - departure_propellant_kg,
        navigation.rcs_propellant_kg,
        navigation.orbit,
    )
    capture_burn = corrected_burn(
        arrival_navigation, definition, capture_delta_v, "transfer:capture"
    )
    if isinstance(capture_burn, PlanningFailure):
        return capture_burn
    departure_burn = BurnStep(
        target.departure_after_s,
        departure_burn.duration_s,
        departure_burn.throttle,
        departure_burn.delta_v_m_s,
        departure_burn.purpose,
    )
    capture_start_s = target.departure_after_s + target.flight_time_s
    capture_burn = BurnStep(
        capture_start_s,
        capture_burn.duration_s,
        capture_burn.throttle,
        capture_burn.delta_v_m_s,
        capture_burn.purpose,
    )
    propellant_kg = _estimated_propellant_kg(
        definition, departure_burn.duration_s + capture_burn.duration_s
    )
    usable_propellant_kg = (
        navigation.main_propellant_kg
        - definition.autonomy.minimum_main_propellant_reserve_kg
    )
    if propellant_kg > usable_propellant_kg:
        return PlanningFailure(
            PlanningFailureCode.INSUFFICIENT_PROPELLANT,
            "combined departure and capture burns exceed reserve",
        )
    return FlightPlan(
        f"transfer_to_{target.primary_body_id}",
        (
            departure_burn,
            CoastStep(target.flight_time_s),
            capture_burn,
        ),
        "SOI transition and capture orbit conditions reached",
        propellant_kg,
        target.primary_body_id,
    )


def _capture_delta_v_m_s(
    target: TransferTarget, arrival_velocity_m_s: np.ndarray
) -> np.ndarray:
    insertion_radius_m = float(np.linalg.norm(target.insertion_position_m))
    if insertion_radius_m == 0.0:
        raise ValueError("insertion position must be non-zero")
    radial = target.insertion_position_m / insertion_radius_m
    reference_normal = np.asarray((0.0, 0.0, 1.0))
    tangent = np.cross(reference_normal, radial)
    if float(np.linalg.norm(tangent)) == 0.0:
        reference_normal = np.asarray((0.0, 1.0, 0.0))
        tangent = np.cross(reference_normal, radial)
    tangent /= float(np.linalg.norm(tangent))
    target_mu_m3_s2 = force_model_for(
        target.primary_body_id
    ).gravitational_parameter_m3_s2
    circular_velocity_m_s = tangent * sqrt(target_mu_m3_s2 / insertion_radius_m)
    arrival_relative_velocity_m_s = (
        arrival_velocity_m_s - target.target_body_velocity_earth_m_s
    )
    return circular_velocity_m_s - arrival_relative_velocity_m_s


def _estimated_propellant_kg(
    definition: SatelliteDefinition, burn_duration_s: float
) -> float:
    mass_flow_kg_s = definition.propulsion.main_thrust_n / (
        definition.propulsion.main_specific_impulse_s * STANDARD_GRAVITY_M_S2
    )
    return mass_flow_kg_s * burn_duration_s
