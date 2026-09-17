"""Hohmann, plane-change, maintenance, and deorbit planning."""

from math import pi, sin, sqrt

import numpy as np

from typace.config.flight import (
    CONTROLLED_DEORBIT_PERIAPSIS_ALTITUDE_M,
    ORBIT_MAINTENANCE_EXIT_HYSTERESIS,
    PLANNING_ZERO_DELTA_V_M_S,
)
from typace.config.physics import STANDARD_GRAVITY_M_S2
from typace.physics.bodies import force_model_for
from typace.satellites.commands import (
    Deorbit,
    MaintainOrbit,
    SetApsides,
    SetInclination,
    SetOrbitAltitude,
)
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

type OrbitCommand = MaintainOrbit | SetOrbitAltitude | SetApsides | SetInclination | Deorbit


def plan_orbit_command(
    navigation: NavigationSolution | None,
    definition: SatelliteDefinition,
    command: OrbitCommand,
    *,
    maintenance_active: bool = False,
) -> PlanningResult:
    if navigation is None:
        return PlanningFailure(PlanningFailureCode.POWER_SAFE, "navigation is frozen")
    if isinstance(command, SetInclination):
        return _plan_inclination(navigation, definition, command.inclination_rad)
    target_apsides = _target_apsides_m(navigation, definition, command)
    if target_apsides is None:
        return PlanningFailure(
            PlanningFailureCode.INVALID_TARGET, "invalid orbit target"
        )
    target_periapsis_m, target_apoapsis_m = target_apsides
    if isinstance(command, MaintainOrbit):
        correction_needed = _maintenance_needed(
            navigation,
            definition,
            target_periapsis_m,
            target_apoapsis_m,
            maintenance_active,
        )
        if not correction_needed:
            return FlightPlan(
                "maintain_orbit",
                (),
                "inside maintenance hysteresis band",
                0.0,
                navigation.primary_body_id,
            )
    return _plan_apsides(
        navigation,
        definition,
        target_periapsis_m,
        target_apoapsis_m,
        "deorbit" if isinstance(command, Deorbit) else "set_apsides",
    )


def _target_apsides_m(
    navigation: NavigationSolution,
    definition: SatelliteDefinition,
    command: OrbitCommand,
) -> tuple[float, float] | None:
    radius_m = force_model_for(definition.primary_body_id).body_radius_m
    if isinstance(command, MaintainOrbit):
        return (
            radius_m + definition.autonomy.target_periapsis_altitude_m,
            radius_m + definition.autonomy.target_apoapsis_altitude_m,
        )
    if isinstance(command, SetOrbitAltitude):
        if command.altitude_m < 0.0:
            return None
        target_radius_m = radius_m + command.altitude_m
        return target_radius_m, target_radius_m
    if isinstance(command, SetApsides):
        target_is_valid = (
            0.0 <= command.periapsis_altitude_m <= command.apoapsis_altitude_m
        )
        if not target_is_valid:
            return None
        return (
            radius_m + command.periapsis_altitude_m,
            radius_m + command.apoapsis_altitude_m,
        )
    if isinstance(command, Deorbit):
        current_apoapsis_m = navigation.orbit.semi_major_axis_m * (
            1.0 + navigation.orbit.eccentricity
        )
        return radius_m + CONTROLLED_DEORBIT_PERIAPSIS_ALTITUDE_M, current_apoapsis_m
    return None


def _maintenance_needed(
    navigation: NavigationSolution,
    definition: SatelliteDefinition,
    target_periapsis_m: float,
    target_apoapsis_m: float,
    maintenance_active: bool,
) -> bool:
    current_periapsis_m = navigation.orbit.semi_major_axis_m * (
        1.0 - navigation.orbit.eccentricity
    )
    current_apoapsis_m = navigation.orbit.semi_major_axis_m * (
        1.0 + navigation.orbit.eccentricity
    )
    threshold_m = definition.autonomy.altitude_tolerance_m
    if maintenance_active:
        threshold_m *= ORBIT_MAINTENANCE_EXIT_HYSTERESIS
    periapsis_error_m = abs(current_periapsis_m - target_periapsis_m)
    apoapsis_error_m = abs(current_apoapsis_m - target_apoapsis_m)
    return max(periapsis_error_m, apoapsis_error_m) > threshold_m


def _plan_apsides(
    navigation: NavigationSolution,
    definition: SatelliteDefinition,
    target_periapsis_m: float,
    target_apoapsis_m: float,
    objective_id: str,
) -> PlanningResult:
    mu_m3_s2 = force_model_for(navigation.primary_body_id).gravitational_parameter_m3_s2
    current_radius_m = float(np.linalg.norm(navigation.position_m))
    if current_radius_m <= 0.0 or target_periapsis_m <= 0.0:
        return PlanningFailure(
            PlanningFailureCode.INVALID_TARGET, "orbit radius is invalid"
        )
    direction = navigation.velocity_m_s / float(np.linalg.norm(navigation.velocity_m_s))
    raises_apoapsis = target_apoapsis_m >= current_radius_m
    opposite_radius_m = target_apoapsis_m if raises_apoapsis else target_periapsis_m
    transfer_semi_major_axis_m = (current_radius_m + opposite_radius_m) / 2.0
    target_semi_major_axis_m = (target_periapsis_m + target_apoapsis_m) / 2.0
    current_speed_m_s = float(np.linalg.norm(navigation.velocity_m_s))
    transfer_start_speed_m_s = sqrt(
        mu_m3_s2 * (2.0 / current_radius_m - 1.0 / transfer_semi_major_axis_m)
    )
    transfer_arrival_speed_m_s = sqrt(
        mu_m3_s2 * (2.0 / opposite_radius_m - 1.0 / transfer_semi_major_axis_m)
    )
    target_arrival_speed_m_s = sqrt(
        mu_m3_s2 * (2.0 / opposite_radius_m - 1.0 / target_semi_major_axis_m)
    )
    first_delta_v_m_s = transfer_start_speed_m_s - current_speed_m_s
    second_delta_v_m_s = target_arrival_speed_m_s - transfer_arrival_speed_m_s
    transfer_time_s = pi * sqrt(transfer_semi_major_axis_m**3 / mu_m3_s2)
    first = _correct_optional_burn(
        navigation,
        definition,
        direction * first_delta_v_m_s,
        f"{objective_id}:depart",
    )
    if isinstance(first, PlanningFailure):
        return first
    first_propellant_kg = _burn_propellant_kg(
        definition, () if first is None else (first,)
    )
    arrival_navigation = NavigationSolution(
        navigation.satellite_id,
        navigation.primary_body_id,
        -navigation.position_m / current_radius_m * opposite_radius_m,
        -direction * transfer_arrival_speed_m_s,
        navigation.mass_kg - first_propellant_kg,
        navigation.main_propellant_kg - first_propellant_kg,
        navigation.rcs_propellant_kg,
        navigation.orbit,
    )
    second = _correct_optional_burn(
        arrival_navigation,
        definition,
        -direction * second_delta_v_m_s,
        f"{objective_id}:arrive",
    )
    if isinstance(second, PlanningFailure):
        return second
    burns = tuple(burn for burn in (first, second) if burn is not None)
    steps: list[BurnStep | CoastStep] = []
    if first is not None:
        steps.append(first)
    steps.append(CoastStep(transfer_time_s))
    if second is not None:
        steps.append(
            BurnStep(
                transfer_time_s,
                second.duration_s,
                second.throttle,
                second.delta_v_m_s,
                second.purpose,
            )
        )
    propellant_kg = _burn_propellant_kg(definition, burns)
    return FlightPlan(
        objective_id,
        tuple(steps),
        "target apsides reached within tolerance",
        propellant_kg,
        navigation.primary_body_id,
    )


def _plan_inclination(
    navigation: NavigationSolution,
    definition: SatelliteDefinition,
    target_inclination_rad: float,
) -> PlanningResult:
    if not 0.0 <= target_inclination_rad <= pi:
        return PlanningFailure(
            PlanningFailureCode.INVALID_TARGET, "inclination is invalid"
        )
    inclination_change_rad = target_inclination_rad - navigation.orbit.inclination_rad
    speed_m_s = float(np.linalg.norm(navigation.velocity_m_s))
    delta_v_m_s = 2.0 * speed_m_s * sin(abs(inclination_change_rad) / 2.0)
    normal = np.cross(navigation.position_m, navigation.velocity_m_s)
    normal /= float(np.linalg.norm(normal))
    if inclination_change_rad < 0.0:
        normal = -normal
    burn = corrected_burn(
        navigation, definition, normal * delta_v_m_s, "set_inclination"
    )
    if isinstance(burn, PlanningFailure):
        return burn
    return FlightPlan(
        "set_inclination",
        (burn,),
        "target inclination reached within tolerance",
        _burn_propellant_kg(definition, (burn,)),
        navigation.primary_body_id,
    )


def _burn_propellant_kg(
    definition: SatelliteDefinition, burns: tuple[BurnStep, ...]
) -> float:
    mass_flow_kg_s = definition.propulsion.main_thrust_n / (
        definition.propulsion.main_specific_impulse_s * STANDARD_GRAVITY_M_S2
    )
    return mass_flow_kg_s * sum(burn.duration_s for burn in burns)


def _correct_optional_burn(
    navigation: NavigationSolution,
    definition: SatelliteDefinition,
    delta_v_m_s: np.ndarray,
    purpose: str,
) -> BurnStep | PlanningFailure | None:
    if float(np.linalg.norm(delta_v_m_s)) <= PLANNING_ZERO_DELTA_V_M_S:
        return None
    return corrected_burn(navigation, definition, delta_v_m_s, purpose)
