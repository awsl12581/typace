"""The single fixed-step translation propagation path and event refinement."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from math import ceil

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root_scalar

from typace.config.simulation import EVENT_TIME_TOLERANCE_SECONDS
from typace.physics.elements import CartesianState, analytic_kepler_step
from typace.physics.forces import (
    ForceInputs,
    ForceModel,
    has_drag_acceleration,
    has_j2_acceleration,
    total_acceleration_m_s2,
)

type Vector = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class TranslationalState:
    position_m: Vector
    velocity_m_s: Vector
    mass_kg: float


@dataclass(frozen=True, slots=True)
class PropagationContext:
    force_model: ForceModel
    drag_area_m2: float
    drag_coefficient: float
    thrust_inertial_n: Vector
    mass_flow_kg_s: float = 0.0


def _derivative(
    state: TranslationalState, context: PropagationContext
) -> tuple[Vector, Vector, float]:
    inputs = ForceInputs(
        state.mass_kg,
        context.drag_area_m2,
        context.drag_coefficient,
        context.thrust_inertial_n,
    )
    return (
        state.velocity_m_s,
        total_acceleration_m_s2(
            context.force_model, state.position_m, state.velocity_m_s, inputs
        ),
        -context.mass_flow_kg_s,
    )


def _offset(
    state: TranslationalState,
    derivative: tuple[Vector, Vector, float],
    duration_s: float,
) -> TranslationalState:
    return TranslationalState(
        state.position_m + derivative[0] * duration_s,
        state.velocity_m_s + derivative[1] * duration_s,
        state.mass_kg + derivative[2] * duration_s,
    )


def rk4_step(
    state: TranslationalState, context: PropagationContext, duration_s: float
) -> TranslationalState:
    """Advance translation and total mass through one classical RK4 step."""

    if duration_s == 0.0:
        raise ValueError("step duration must be non-zero")
    if state.mass_kg <= 0.0:
        raise ValueError("mass must be positive")
    if duration_s < 0.0 and context.mass_flow_kg_s != 0.0:
        raise ValueError("cannot integrate mass flow backward")
    consumed_mass_kg = context.mass_flow_kg_s * duration_s
    if consumed_mass_kg >= state.mass_kg:
        raise ValueError("step would exhaust total mass")
    first = _derivative(state, context)
    second = _derivative(_offset(state, first, duration_s / 2.0), context)
    third = _derivative(_offset(state, second, duration_s / 2.0), context)
    fourth = _derivative(_offset(state, third, duration_s), context)
    sixth = duration_s / 6.0
    return TranslationalState(
        state.position_m
        + sixth * (first[0] + 2.0 * second[0] + 2.0 * third[0] + fourth[0]),
        state.velocity_m_s
        + sixth * (first[1] + 2.0 * second[1] + 2.0 * third[1] + fourth[1]),
        state.mass_kg
        + sixth * (first[2] + 2.0 * second[2] + 2.0 * third[2] + fourth[2]),
    )


def can_use_analytic_step(context: PropagationContext) -> bool:
    model = context.force_model
    has_central_gravity = model.gravitational_parameter_m3_s2 > 0.0
    has_thrust = bool(np.linalg.norm(context.thrust_inertial_n))
    has_drag = has_drag_acceleration(
        model, context.drag_area_m2, context.drag_coefficient
    )
    has_j2 = has_j2_acceleration(model)
    has_numerical_force = has_thrust or has_drag or has_j2
    has_mass_flow = context.mass_flow_kg_s != 0.0
    return has_central_gravity and not (has_numerical_force or has_mass_flow)


def can_use_analytic_state(
    state: TranslationalState, context: PropagationContext
) -> bool:
    mu_m3_s2 = context.force_model.gravitational_parameter_m3_s2
    radius_m = float(np.linalg.norm(state.position_m))
    angular_momentum = np.cross(state.position_m, state.velocity_m_s)
    has_orbit_plane = float(np.linalg.norm(angular_momentum)) > 0.0
    specific_energy = (
        float(np.dot(state.velocity_m_s, state.velocity_m_s)) / 2.0
        - mu_m3_s2 / radius_m
        if radius_m > 0.0
        else 0.0
    )
    return radius_m > 0.0 and has_orbit_plane and specific_energy < 0.0


def propagate(
    state: TranslationalState,
    context: PropagationContext,
    duration_s: float,
    maximum_step_s: float,
    *,
    allow_analytic: bool = True,
) -> TranslationalState:
    """Advance through the sole production path with stable fixed substeps."""

    if maximum_step_s <= 0.0:
        raise ValueError("maximum step must be positive")
    if duration_s == 0.0:
        return state
    use_analytic = (
        allow_analytic
        and can_use_analytic_step(context)
        and can_use_analytic_state(state, context)
    )
    if use_analytic:
        advanced = analytic_kepler_step(
            CartesianState(state.position_m, state.velocity_m_s),
            context.force_model.gravitational_parameter_m3_s2,
            duration_s,
        )
        return replace(
            state,
            position_m=advanced.position_m,
            velocity_m_s=advanced.velocity_m_s,
        )
    step_count = max(1, ceil(abs(duration_s) / maximum_step_s))
    step_duration_s = duration_s / step_count
    current = state
    for _ in range(step_count):
        current = rk4_step(current, context, step_duration_s)
    return current


def locate_event_time_s(
    duration_s: float,
    state_at: Callable[[float], TranslationalState],
    event_value: Callable[[TranslationalState], float],
) -> float:
    """Refine one sign-changing event inside a known bracket through SciPy."""

    if duration_s <= 0.0:
        raise ValueError("event bracket duration must be positive")
    start_value = event_value(state_at(0.0))
    end_value = event_value(state_at(duration_s))
    if start_value == 0.0:
        return 0.0
    if start_value * end_value > 0.0:
        raise ValueError("event does not cross zero inside the bracket")

    result = root_scalar(
        lambda elapsed_s: event_value(state_at(float(elapsed_s))),
        bracket=(0.0, duration_s),
        method="brentq",
        xtol=EVENT_TIME_TOLERANCE_SECONDS,
    )
    if not result.converged:
        raise ValueError("event time did not converge")
    return float(result.root)
