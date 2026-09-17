"""Thin orchestration of navigation, objectives, planning, and execution."""

from dataclasses import dataclass

import numpy as np

from typace.satellites.commands import (
    Deorbit,
    MaintainOrbit,
    SetApsides,
    SetInclination,
    SetOrbitAltitude,
)
from typace.satellites.definition import SatelliteDefinition
from typace.satellites.state import ControlMode, SatelliteSnapshot
from typace.vehicle.actuators import ActuatorCommand
from typace.vehicle.state import VehicleState
from typace.flight.attitude_control import AttitudeControllerState
from typace.flight.execution import (
    ExecutionState,
    execute_plan,
)
from typace.flight.models import FlightPlan, PlanningFailure, PlanningResult
from typace.flight.navigation import NavigationState, update_navigation
from typace.flight.objectives import FlightObjective
from typace.flight.planning.orbit import plan_orbit_command
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
) -> AutopilotOutput:
    navigation = update_navigation(state.navigation, snapshot)
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
    result = planned_result
    needs_plan = state.plan is None or vehicle.requires_replan
    if needs_plan and result is None and selected_objective is not None:
        result = _plan_objective(navigation, definition, selected_objective)
    plan, failure = _resolve_plan(state.plan, result, needs_plan)
    execution = state.execution
    if plan is not None and (execution is None or execution.plan != plan):
        execution = ExecutionState(plan)
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


def _plan_objective(
    navigation: NavigationState,
    definition: SatelliteDefinition,
    objective: FlightObjective,
) -> PlanningResult | None:
    command = objective.command
    if isinstance(
        command,
        MaintainOrbit | SetOrbitAltitude | SetApsides | SetInclination | Deorbit,
    ):
        return plan_orbit_command(navigation.solution, definition, command)
    return None


def _resolve_plan(
    existing: FlightPlan | None,
    result: PlanningResult | None,
    needs_plan: bool,
) -> tuple[FlightPlan | None, PlanningFailure | None]:
    if not needs_plan:
        return existing, None
    if isinstance(result, FlightPlan):
        return result, None
    if isinstance(result, PlanningFailure):
        return None, result
    return None, None


def _idle_decision() -> SafetyDecision:
    return SafetyDecision(
        ActuatorCommand(0.0, np.zeros(3), np.zeros(3)), SafetyReason.CLEAR
    )
