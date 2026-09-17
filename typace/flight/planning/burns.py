"""Finite-burn correction through the shared force model and propagator."""

import numpy as np
from scipy.optimize import root_scalar

from typace.config.flight import (
    FINITE_BURN_MAX_ITERATIONS,
    FINITE_BURN_RESIDUAL_TOLERANCE_M_S,
    FINITE_BURN_TIME_TOLERANCE_SECONDS,
    PLANNING_PROPAGATION_STEP_SECONDS,
)
from typace.config.physics import STANDARD_GRAVITY_M_S2
from typace.physics.bodies import force_model_for
from typace.physics.propagation import PropagationContext, TranslationalState, propagate
from typace.satellites.definition import SatelliteDefinition
from typace.flight.models import (
    BurnStep,
    PlanningFailure,
    PlanningFailureCode,
)
from typace.flight.navigation import NavigationSolution


def corrected_burn(
    navigation: NavigationSolution,
    definition: SatelliteDefinition,
    delta_v_m_s: np.ndarray,
    purpose: str,
) -> BurnStep | PlanningFailure:
    delta_v = np.asarray(delta_v_m_s, dtype=np.float64)
    requested_delta_v_m_s = float(np.linalg.norm(delta_v))
    if requested_delta_v_m_s == 0.0:
        return PlanningFailure(PlanningFailureCode.NO_CORRECTION, "no burn is required")
    direction = delta_v / requested_delta_v_m_s
    propulsion = definition.propulsion
    mass_flow_kg_s = propulsion.main_thrust_n / (
        propulsion.main_specific_impulse_s * STANDARD_GRAVITY_M_S2
    )
    usable_propellant_kg = max(
        0.0,
        navigation.main_propellant_kg
        - definition.autonomy.minimum_main_propellant_reserve_kg,
    )
    final_mass_kg = navigation.mass_kg - usable_propellant_kg
    available_delta_v_m_s = (
        propulsion.main_specific_impulse_s
        * STANDARD_GRAVITY_M_S2
        * np.log(navigation.mass_kg / final_mass_kg)
    )
    if requested_delta_v_m_s > available_delta_v_m_s or mass_flow_kg_s == 0.0:
        return PlanningFailure(
            PlanningFailureCode.INSUFFICIENT_PROPELLANT,
            "main propellant reserve prevents this burn",
        )
    maximum_duration_s = usable_propellant_kg / mass_flow_kg_s
    initial = TranslationalState(
        navigation.position_m, navigation.velocity_m_s, navigation.mass_kg
    )
    model = force_model_for(navigation.primary_body_id)
    thrust_n = direction * propulsion.main_thrust_n

    def residual(duration_s: float) -> float:
        powered = propagate(
            initial,
            PropagationContext(
                model,
                definition.drag_area_m2,
                definition.drag_coefficient,
                thrust_n,
                mass_flow_kg_s,
            ),
            duration_s,
            PLANNING_PROPAGATION_STEP_SECONDS,
            allow_analytic=False,
        )
        coast = propagate(
            initial,
            PropagationContext(
                model,
                definition.drag_area_m2,
                definition.drag_coefficient,
                np.zeros(3),
            ),
            duration_s,
            PLANNING_PROPAGATION_STEP_SECONDS,
            allow_analytic=False,
        )
        achieved_delta_v = float(
            np.dot(powered.velocity_m_s - coast.velocity_m_s, direction)
        )
        return achieved_delta_v - requested_delta_v_m_s

    if residual(maximum_duration_s) < 0.0:
        return PlanningFailure(
            PlanningFailureCode.UNREACHABLE,
            "finite thrust cannot converge before the propellant reserve",
        )
    solution = root_scalar(
        residual,
        bracket=(0.0, maximum_duration_s),
        method="brentq",
        xtol=FINITE_BURN_TIME_TOLERANCE_SECONDS,
        maxiter=FINITE_BURN_MAX_ITERATIONS,
    )
    if not solution.converged:
        return PlanningFailure(
            PlanningFailureCode.UNREACHABLE,
            "finite burn correction did not converge",
        )
    duration_s = float(solution.root)
    if abs(residual(duration_s)) > FINITE_BURN_RESIDUAL_TOLERANCE_M_S:
        return PlanningFailure(
            PlanningFailureCode.UNREACHABLE,
            "finite burn correction exceeds the velocity tolerance",
        )
    return BurnStep(
        0.0,
        duration_s,
        1.0,
        (float(delta_v[0]), float(delta_v[1]), float(delta_v[2])),
        purpose,
    )
