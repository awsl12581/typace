"""Thin orchestration of navigation, objectives, planning, and execution."""

from dataclasses import dataclass

import numpy as np

from typace.satellites.commands import (
    Deorbit,
    MaintainOrbit,
    SetApsides,
    SetInclination,
    SetOrbitAltitude,
    TransferPrimary,
    ReturnStableOrbit,
    AvoidCollision,
)
from typace.satellites.definition import SatelliteDefinition
from typace.satellites.state import ControlMode, SatelliteSnapshot
from typace.vehicle.actuators import ActuatorCommand
from typace.vehicle.state import VehicleState
from typace.flight.attitude_control import AttitudeControllerState
from typace.flight.execution import (
    ExecutionState,
    ExecutionStatus,
    execute_plan,
)
from typace.flight.models import (
    FlightPlan,
    PlanningFailure,
    PlanningFailureCode,
    PlanningResult,
)
from typace.flight.navigation import NavigationState, update_navigation
from typace.flight.objectives import FlightObjective
from typace.flight.planning.orbit import plan_orbit_command
from typace.flight.planning.avoidance import ConjunctionRisk, plan_avoidance
from typace.flight.planning.transfer import TransferTarget, plan_transfer
from typace.flight.safety import SafetyDecision, SafetyReason, SafetyTelemetry

type OrbitCommand = MaintainOrbit | SetOrbitAltitude | SetApsides | SetInclination | Deorbit


@dataclass(frozen=True, slots=True)
class AutopilotState:
    navigation: NavigationState
    objective: FlightObjective | None
    plan: FlightPlan | None
    execution: ExecutionState | None
    controller: AttitudeControllerState
    planning_failure: PlanningFailure | None = None

    @classmethod
    def idle(cls) -> "AutopilotState":
        return cls(
            NavigationState(),
            None,
            None,
            None,
            AttitudeControllerState(np.zeros(3)),
        )


@dataclass(frozen=True, slots=True)
class AutopilotOutput:
    state: AutopilotState
    decision: SafetyDecision


def autopilot_step(
    state: AutopilotState,
    snapshot: SatelliteSnapshot,
    vehicle: VehicleState,
    definition: SatelliteDefinition,
    telemetry: SafetyTelemetry,
    duration_s: float,
    *,
    objective: FlightObjective | None = None,
    planned_result: PlanningResult | None = None,
    transfer_target: TransferTarget | None = None,
    conjunction_risk: ConjunctionRisk | None = None,
    planning_pending: bool = False,
) -> AutopilotOutput:
    objective_changed = objective is not None and objective != state.objective
    navigation_primary_changed = (
        state.navigation.solution is not None
        and state.navigation.solution.primary_body_id != snapshot.primary_body_id
    )
    navigation_refresh_needed = (
        state.navigation.solution is None
        or vehicle.requires_replan
        or objective_changed
        or navigation_primary_changed
    )
    navigation_failure: PlanningFailure | None = None
    try:
        navigation = (
            update_navigation(state.navigation, snapshot)
            if navigation_refresh_needed
            else state.navigation
        )
    except (ArithmeticError, ValueError):
        navigation = state.navigation
        navigation_failure = PlanningFailure(
            PlanningFailureCode.INVALID_TARGET,
            "current state does not define a plannable orbit",
        )
    if snapshot.control_mode is not ControlMode.AUTONOMOUS:
        cancelled = AutopilotState(
            navigation,
            state.objective,
            None,
            None,
            state.controller,
        )
        return AutopilotOutput(cancelled, _idle_decision())

    selected_objective = objective if objective is not None else state.objective
    result = planned_result if planned_result is not None else navigation_failure
    has_no_planning_result = state.plan is None and state.planning_failure is None
    needs_plan = objective_changed or has_no_planning_result or vehicle.requires_replan
    should_plan_now = (
        needs_plan
        and result is None
        and selected_objective is not None
        and not planning_pending
    )
    if should_plan_now:
        assert selected_objective is not None
        try:
            result = plan_objective(
                navigation,
                definition,
                selected_objective,
                transfer_target,
                conjunction_risk,
            )
        except (ArithmeticError, ValueError):
            result = PlanningFailure(
                PlanningFailureCode.UNREACHABLE,
                "current state cannot produce a flight plan",
            )
    plan, failure = _resolve_plan(
        state.plan, state.planning_failure, result, needs_plan
    )
    execution = state.execution
    if plan is not None and (execution is None or execution.plan != plan):
        execution = ExecutionState(plan)
    execution_is_terminal = execution is not None and execution.status in (
        ExecutionStatus.COMPLETED,
        ExecutionStatus.ABORTED,
    )
    if not needs_plan and execution_is_terminal:
        return AutopilotOutput(state, _idle_decision())
    if execution is None:
        idle = AutopilotState(
            navigation,
            selected_objective,
            plan,
            None,
            state.controller,
            failure,
        )
        return AutopilotOutput(idle, _idle_decision())
    output = execute_plan(
        execution,
        state.controller,
        vehicle,
        definition,
        telemetry,
        duration_s,
    )
    updated = AutopilotState(
        navigation,
        selected_objective,
        plan,
        output.execution,
        output.controller,
        failure,
    )
    return AutopilotOutput(updated, output.decision)


def plan_objective(
    navigation: NavigationState,
    definition: SatelliteDefinition,
    objective: FlightObjective,
    transfer_target: TransferTarget | None,
    conjunction_risk: ConjunctionRisk | None,
) -> PlanningResult | None:
    command = objective.command
    if isinstance(
        command,
        MaintainOrbit | SetOrbitAltitude | SetApsides | SetInclination | Deorbit,
    ):
        return plan_orbit_command(navigation.solution, definition, command)
    if isinstance(command, TransferPrimary | ReturnStableOrbit):
        if transfer_target is None:
            return PlanningFailure(
                PlanningFailureCode.INVALID_TARGET,
                "transfer target is unavailable",
            )
        return plan_transfer(navigation.solution, definition, transfer_target)
    if isinstance(command, AvoidCollision):
        if conjunction_risk is None:
            return PlanningFailure(
                PlanningFailureCode.INVALID_TARGET,
                "conjunction risk is unavailable",
            )
        return plan_avoidance(navigation.solution, definition, conjunction_risk)
    return None


def _resolve_plan(
    existing: FlightPlan | None,
    existing_failure: PlanningFailure | None,
    result: PlanningResult | None,
    needs_plan: bool,
) -> tuple[FlightPlan | None, PlanningFailure | None]:
    if not needs_plan:
        return existing, existing_failure
    if isinstance(result, FlightPlan):
        return result, None
    if isinstance(result, PlanningFailure):
        return None, result
    return None, None


def _idle_decision() -> SafetyDecision:
    return SafetyDecision(
        ActuatorCommand(0.0, np.zeros(3), np.zeros(3)), SafetyReason.CLEAR
    )
