"""Flight-plan execution lifecycle and shared actuator requests."""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from typace.config.flight import (
    BURN_ALIGNMENT_TOLERANCE_RAD,
    WHEEL_UNLOAD_START_FRACTION,
)
from typace.satellites.definition import SatelliteDefinition
from typace.vehicle.actuators import (
    ActuatorCommand,
    momentum_unload_command,
)
from typace.vehicle.state import VehicleState
from typace.flight.attitude_control import (
    AttitudeControllerState,
    control_attitude,
)
from typace.flight.guidance import guidance_for_plan
from typace.flight.models import FlightPlan
from typace.flight.safety import (
    SafetyDecision,
    SafetyReason,
    SafetyTelemetry,
    filter_actuators,
)


class ExecutionStatus(StrEnum):
    READY = "ready"
    TURNING = "turning"
    COASTING = "coasting"
    BURNING = "burning"
    COMPLETED = "completed"
    ABORTED = "aborted"


@dataclass(frozen=True, slots=True)
class ExecutionState:
    plan: FlightPlan
    elapsed_s: float = 0.0
    status: ExecutionStatus = ExecutionStatus.READY
    abort_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutionOutput:
    execution: ExecutionState
    controller: AttitudeControllerState
    decision: SafetyDecision


def execute_plan(
    execution: ExecutionState,
    controller: AttitudeControllerState,
    vehicle: VehicleState,
    definition: SatelliteDefinition,
    telemetry: SafetyTelemetry,
    duration_s: float,
) -> ExecutionOutput:
    if duration_s <= 0.0:
        raise ValueError("execution duration must be positive")
    if execution.status in (ExecutionStatus.COMPLETED, ExecutionStatus.ABORTED):
        return ExecutionOutput(execution, controller, _idle_decision())
    guidance = guidance_for_plan(execution.plan, execution.elapsed_s, definition)
    if guidance is None:
        completed = ExecutionState(
            execution.plan, execution.elapsed_s, ExecutionStatus.COMPLETED
        )
        return ExecutionOutput(completed, controller, _idle_decision())

    control = control_attitude(
        controller,
        vehicle.attitude,
        guidance.target_quaternion_wxyz,
        np.asarray(definition.reaction_wheel_maximum_torque_n_m),
        duration_s,
    )
    aligned = control.attitude_error_rad <= BURN_ALIGNMENT_TOLERANCE_RAD
    throttle = guidance.throttle if aligned else 0.0
    request = ActuatorCommand(throttle, control.wheel_torque_n_m, np.zeros(3))
    if _wheel_unload_is_needed(vehicle, definition) and guidance.throttle == 0.0:
        request = momentum_unload_command(
            vehicle.attitude.wheel_momentum_n_m_s,
            np.asarray(definition.reaction_wheel_maximum_torque_n_m),
            definition.propulsion.rcs_maximum_torque_n_m,
            duration_s,
        )
    decision = filter_actuators(request, vehicle, definition, telemetry)
    status = _execution_status(guidance.throttle, aligned)
    next_execution = ExecutionState(
        execution.plan,
        execution.elapsed_s + duration_s,
        status,
    )
    if decision.reason is not SafetyReason.CLEAR:
        next_execution = ExecutionState(
            execution.plan,
            execution.elapsed_s,
            ExecutionStatus.ABORTED,
            decision.reason.value,
        )
    return ExecutionOutput(next_execution, control.state, decision)


def _execution_status(throttle: float, aligned: bool) -> ExecutionStatus:
    if throttle == 0.0:
        return ExecutionStatus.COASTING if aligned else ExecutionStatus.TURNING
    return ExecutionStatus.BURNING if aligned else ExecutionStatus.TURNING


def _wheel_unload_is_needed(
    vehicle: VehicleState, definition: SatelliteDefinition
) -> bool:
    maximum = np.asarray(definition.reaction_wheel_maximum_momentum_n_m_s)
    threshold = maximum * WHEEL_UNLOAD_START_FRACTION
    return bool(np.any(np.abs(vehicle.attitude.wheel_momentum_n_m_s) >= threshold))


def _idle_decision() -> SafetyDecision:
    return SafetyDecision(
        ActuatorCommand(0.0, np.zeros(3), np.zeros(3)), SafetyReason.CLEAR
    )
