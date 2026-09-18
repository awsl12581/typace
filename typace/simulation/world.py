"""Single-clock deterministic multi-satellite world."""

from concurrent.futures import Future
from dataclasses import dataclass, replace
from math import isfinite, pi, radians, sqrt
from typing import cast

from astropy.time import Time, TimeDelta
import numpy as np

from typace.config.simulation import (
    CONJUNCTION_PREDICTION_HORIZON_SECONDS,
    EPOCH_NORMALIZATION_STEP_SECONDS,
    MAX_PENDING_COMMANDS_PER_SATELLITE,
    MAX_PHYSICS_SUBSTEPS_PER_WORLD_STEP,
    PHYSICS_STEP_SECONDS,
    TIME_WARPS,
)
from typace.config.vehicle import QUATERNION_NORM_TOLERANCE
from typace.flight.autopilot import AutopilotState, autopilot_step, plan_objective
from typace.flight.conjunction import ConjunctionCandidate, screen_conjunctions
from typace.flight.execution import ExecutionStatus
from typace.flight.models import PlanningFailure, PlanningFailureCode, PlanningResult
from typace.flight.manual import manual_control
from typace.flight.navigation import NavigationState
from typace.flight.objectives import FlightObjective, ObjectivePriority
from typace.flight.planning.avoidance import ConjunctionRisk
from typace.flight.planning.transfer import TransferTarget
from typace.flight.safety import SafetyDecision, SafetyReason, SafetyTelemetry
from typace.physics.elements import CartesianState, ClassicalElements, elements_to_state
from typace.physics.bodies import force_model_for
from typace.physics.environment import OccludingBody
from typace.physics.propagation import PropagationContext, TranslationalState, propagate
from typace.satellites.commands import (
    AvoidCollision,
    CommandResult,
    CommandStatus,
    Deorbit,
    MaintainOrbit,
    ManualActuation,
    ReturnAutonomousControl,
    ReturnStableOrbit,
    SatelliteCommand,
    SetApsides,
    SetInclination,
    SetOrbitAltitude,
    TakeManualControl,
    TransferPrimary,
)
from typace.satellites.definition import (
    KeplerianDefinition,
    SatelliteCatalog,
    SatelliteDefinition,
    StateVectorDefinition,
)
from typace.satellites.state import (
    ControlMode,
    SatelliteSnapshot,
    SatelliteState,
)
from typace.simulation.ephemeris import SolarSystemEphemeris, ephemeris_at
from typace.simulation.events import (
    DestructionCause,
    DestructionEvent,
    locate_collision_time_s,
    locate_destruction_time_s,
)
from typace.tasks import ThreadTaskManager
from typace.vehicle.actuators import ActuatorCommand
from typace.vehicle.dynamics import PowerEnvironment, advance_vehicle
from typace.vehicle.power import PowerMode
from typace.vehicle.state import VehicleState


@dataclass(frozen=True, slots=True)
class WorldSnapshot:
    scenario_epoch_utc: str
    elapsed_seconds: float
    selected_time_warp: float
    effective_time_warp: float
    time_warp_limit_reason: str | None
    satellites: tuple[SatelliteSnapshot, ...]
    destruction_events: tuple[DestructionEvent, ...] = ()

    def satellite(self, satellite_id: str) -> SatelliteSnapshot:
        for satellite in self.satellites:
            if satellite.id == satellite_id:
                return satellite
        raise KeyError(satellite_id)


@dataclass(frozen=True, slots=True)
class _SatelliteStepInput:
    state: SatelliteState
    autopilot: AutopilotState
    previous_safety_reason: SafetyReason
    duration_s: float
    ephemeris: SolarSystemEphemeris
    conjunction: ConjunctionCandidate | None
    ready_plan: "_ReadyPlan | None"
    defer_planning: bool


@dataclass(frozen=True, slots=True)
class _SatelliteStepOutput:
    satellite_id: str
    controlled_update: tuple[SatelliteState, AutopilotState, SafetyDecision] | None
    planning_request: "_PlanningRequest | None" = None


@dataclass(frozen=True, slots=True)
class _PlanningRequest:
    satellite_id: str
    primary_body_id: str
    objective: FlightObjective
    navigation: NavigationState
    definition: SatelliteDefinition
    transfer_target: TransferTarget | None
    conjunction_risk: ConjunctionRisk | None


@dataclass(frozen=True, slots=True)
class _PendingPlan:
    primary_body_id: str
    objective: FlightObjective
    future: Future[PlanningResult]


@dataclass(frozen=True, slots=True)
class _ReadyPlan:
    objective: FlightObjective
    result: PlanningResult


class SimulationWorld:
    def __init__(
        self,
        scenario_epoch: Time,
        satellites: dict[str, SatelliteState],
    ) -> None:
        self._scenario_epoch = scenario_epoch.copy()
        self._elapsed_seconds = 0.0
        self._satellites = satellites
        self._selected_time_warp = 1.0
        self._effective_time_warp = 1.0
        self._time_warp_limit_reason: str | None = None
        self._destruction_events: list[DestructionEvent] = []
        self._autopilots = {
            satellite_id: AutopilotState.idle() for satellite_id in satellites
        }
        self._safety_reasons = {
            satellite_id: SafetyReason.CLEAR for satellite_id in satellites
        }
        self._conjunction_alerts = {satellite_id: () for satellite_id in satellites}
        self._pending_plans: dict[str, _PendingPlan] = {}

    @classmethod
    def from_catalog(
        cls, catalog: SatelliteCatalog, scenario_epoch: Time | None = None
    ) -> "SimulationWorld":
        epoch = catalog.scenario_epoch if scenario_epoch is None else scenario_epoch
        satellites = {
            definition.id: _initial_state(definition, epoch)
            for definition in sorted(catalog.satellites, key=lambda item: item.id)
        }
        return cls(epoch, satellites)

    @property
    def scenario_epoch(self) -> Time:
        return self._scenario_epoch.copy()

    def set_time_warp(self, multiplier: float) -> None:
        if multiplier not in TIME_WARPS:
            raise ValueError("time warp is not configured")
        self._selected_time_warp = multiplier

    def set_elapsed_seconds(self, elapsed_seconds: float) -> None:
        if not isfinite(elapsed_seconds):
            raise ValueError("elapsed time must be finite")
        self._elapsed_seconds = elapsed_seconds

    def submit(self, satellite_id: str, command: SatelliteCommand) -> CommandResult:
        state = self._satellites.get(satellite_id)
        if state is None:
            return CommandResult(CommandStatus.UNKNOWN_SATELLITE, "unknown satellite")
        rejection = _command_rejection(state, command)
        if rejection is not None:
            return rejection
        next_mode = state.control_mode
        pending_commands = state.pending_commands
        next_vehicle = state.vehicle
        if isinstance(command, TakeManualControl):
            next_mode = ControlMode.MANUAL
            pending_commands = ()
        elif isinstance(command, ReturnAutonomousControl):
            next_mode = ControlMode.AUTONOMOUS
            pending_commands = ()
            next_vehicle = replace(state.vehicle, requires_replan=True)
        elif isinstance(command, ManualActuation):
            pending_commands = tuple(
                pending
                for pending in pending_commands
                if not isinstance(pending, ManualActuation)
            ) + (command,)
        else:
            pending_commands = (*pending_commands, command)
        self._satellites[satellite_id] = replace(
            state,
            vehicle=next_vehicle,
            control_mode=next_mode,
            pending_commands=pending_commands,
        )
        return CommandResult(CommandStatus.ACCEPTED)

    def step(self, duration_s: float) -> WorldSnapshot:
        return self._step(duration_s, None)

    def step_parallel(
        self, duration_s: float, tasks: ThreadTaskManager
    ) -> WorldSnapshot:
        """Advance independent satellites on the shared compute pool."""
        return self._step(duration_s, tasks)

    def _step(
        self, duration_s: float, tasks: ThreadTaskManager | None
    ) -> WorldSnapshot:
        if duration_s < 0.0:
            raise ValueError("duration cannot be negative")
        if duration_s == 0.0:
            return self.snapshot()
        simulation_duration_s = self._limited_simulation_duration_s(duration_s)
        next_epoch = self._scenario_epoch + TimeDelta(
            self._elapsed_seconds + simulation_duration_s, format="sec"
        )
        ephemeris = ephemeris_at(next_epoch)
        frozen_states = tuple(
            self._satellites[satellite_id] for satellite_id in sorted(self._satellites)
        )
        conjunctions = screen_conjunctions(
            frozen_states, CONJUNCTION_PREDICTION_HORIZON_SECONDS
        )
        risks_by_satellite = _yielding_risks(conjunctions)
        conjunction_alerts = _conjunction_alerts(frozen_states, conjunctions)
        ready_plans = self._ready_plans() if tasks is not None else {}
        updates: dict[str, SatelliteState] = {}
        autopilot_updates: dict[str, AutopilotState] = {}
        safety_updates: dict[str, SafetyReason] = {}
        failure_events: list[DestructionEvent] = []
        step_inputs = tuple(
            _SatelliteStepInput(
                state,
                self._autopilots[state.definition.id],
                self._safety_reasons[state.definition.id],
                simulation_duration_s,
                ephemeris,
                risks_by_satellite.get(state.definition.id),
                ready_plans.get(state.definition.id),
                tasks is not None,
            )
            for state in frozen_states
        )
        step_outputs = (
            tuple(_advance_satellite(value) for value in step_inputs)
            if tasks is None
            else tasks.map_compute(_advance_satellite, step_inputs)
        )
        states_by_id = {state.definition.id: state for state in frozen_states}
        for step_output in step_outputs:
            satellite_id = step_output.satellite_id
            state = states_by_id[satellite_id]
            satellite_id = state.definition.id
            controlled_update = step_output.controlled_update
            update_is_invalid = controlled_update is None or not _state_is_valid(
                controlled_update[0]
            )
            if update_is_invalid:
                failure_events.append(
                    DestructionEvent(
                        f"numerical_failure-{state.definition.id}-{self._elapsed_seconds + simulation_duration_s:.6f}",
                        self._elapsed_seconds + simulation_duration_s,
                        (state.definition.id,),
                        DestructionCause.NUMERICAL_FAILURE,
                    )
                )
                continue
            assert controlled_update is not None
            update, autopilot, decision = controlled_update
            updates[satellite_id] = update
            autopilot_updates[satellite_id] = autopilot
            safety_updates[satellite_id] = decision.reason
        healthy_previous = tuple(
            state for state in frozen_states if state.definition.id in updates
        )
        events = _detect_destructions(
            healthy_previous,
            updates,
            self._elapsed_seconds,
            simulation_duration_s,
        )
        events = tuple(
            sorted(
                (*failure_events, *events),
                key=lambda event: (event.elapsed_seconds, event.event_id),
            )
        )
        destroyed_ids = {
            satellite_id for event in events for satellite_id in event.satellite_ids
        }
        self._destruction_events.extend(events)
        updates = {
            satellite_id: state
            for satellite_id, state in updates.items()
            if satellite_id not in destroyed_ids
        }
        self._satellites = updates
        self._autopilots = {
            satellite_id: autopilot_updates[satellite_id] for satellite_id in updates
        }
        self._safety_reasons = {
            satellite_id: safety_updates[satellite_id] for satellite_id in updates
        }
        self._conjunction_alerts = {
            satellite_id: conjunction_alerts[satellite_id] for satellite_id in updates
        }
        if tasks is not None:
            self._update_planning_tasks(step_outputs, updates, tasks)
        self._elapsed_seconds += simulation_duration_s
        return self.snapshot()

    def _ready_plans(self) -> dict[str, _ReadyPlan]:
        ready: dict[str, _ReadyPlan] = {}
        for satellite_id, pending in tuple(self._pending_plans.items()):
            if pending.future.cancelled():
                self._pending_plans.pop(satellite_id)
            elif pending.future.done():
                ready[satellite_id] = _ReadyPlan(
                    pending.objective, pending.future.result()
                )
        return ready

    def _update_planning_tasks(
        self,
        outputs: tuple[_SatelliteStepOutput, ...],
        updates: dict[str, SatelliteState],
        tasks: ThreadTaskManager,
    ) -> None:
        requests = {
            output.satellite_id: output.planning_request
            for output in outputs
            if output.satellite_id in updates
        }
        for satellite_id, pending in tuple(self._pending_plans.items()):
            request = requests.get(satellite_id)
            request_matches_pending = (
                request is not None
                and request.primary_body_id == pending.primary_body_id
                and request.objective == pending.objective
            )
            if not request_matches_pending:
                pending.future.cancel()
                self._pending_plans.pop(satellite_id)
        for satellite_id, request in requests.items():
            if request is None or satellite_id in self._pending_plans:
                continue
            self._pending_plans[satellite_id] = _PendingPlan(
                request.primary_body_id,
                request.objective,
                tasks.submit_planning(_run_planning_request, request),
            )

    def snapshot(self) -> WorldSnapshot:
        satellites = tuple(
            _snapshot(
                self._satellites[satellite_id],
                self._autopilots[satellite_id],
                self._safety_reasons[satellite_id],
                self._conjunction_alerts[satellite_id],
            )
            for satellite_id in sorted(self._satellites)
        )
        return WorldSnapshot(
            cast(str, self._scenario_epoch.to_value("isot")),
            self._elapsed_seconds,
            self._selected_time_warp,
            self._effective_time_warp,
            self._time_warp_limit_reason,
            satellites,
            tuple(self._destruction_events),
        )

    def _limited_simulation_duration_s(self, real_duration_s: float) -> float:
        requested_duration_s = real_duration_s * self._selected_time_warp
        maximum_duration_s = PHYSICS_STEP_SECONDS * MAX_PHYSICS_SUBSTEPS_PER_WORLD_STEP
        simulation_duration_s = min(requested_duration_s, maximum_duration_s)
        self._effective_time_warp = simulation_duration_s / real_duration_s
        self._time_warp_limit_reason = (
            "physics_step" if simulation_duration_s < requested_duration_s else None
        )
        return simulation_duration_s


def _initial_state(
    definition: SatelliteDefinition, scenario_epoch: Time
) -> SatelliteState:
    force_model = force_model_for(definition.primary_body_id)
    orbit = definition.initial_orbit
    cartesian = _cartesian_state(orbit, force_model.gravitational_parameter_m3_s2)
    vehicle = VehicleState.from_definition(definition)
    translation = TranslationalState(
        cartesian.position_m,
        cartesian.velocity_m_s,
        vehicle.resources.total_mass_kg,
    )
    epoch_delta = cast(TimeDelta, scenario_epoch - orbit.epoch)
    delta_seconds = cast(float, epoch_delta.to_value("sec"))
    if delta_seconds != 0.0:
        translation = propagate(
            translation,
            PropagationContext(
                force_model,
                definition.drag_area_m2,
                definition.drag_coefficient,
                np.zeros(3),
            ),
            delta_seconds,
            EPOCH_NORMALIZATION_STEP_SECONDS,
        )
    return SatelliteState(
        definition,
        definition.primary_body_id,
        translation,
        vehicle,
        ControlMode.AUTONOMOUS,
    )


def _cartesian_state(
    orbit: StateVectorDefinition | KeplerianDefinition, mu_m3_s2: float
) -> CartesianState:
    if isinstance(orbit, StateVectorDefinition):
        return CartesianState(
            np.asarray(orbit.position_m), np.asarray(orbit.velocity_m_s)
        )
    return elements_to_state(
        ClassicalElements(
            orbit.semi_major_axis_m,
            orbit.eccentricity,
            radians(orbit.inclination_deg),
            radians(orbit.ascending_node_deg),
            radians(orbit.periapsis_argument_deg),
            radians(orbit.mean_anomaly_deg),
        ),
        mu_m3_s2,
    )


def _advance_state(
    state: SatelliteState,
    ephemeris: SolarSystemEphemeris,
    command: ActuatorCommand,
    duration_s: float,
) -> SatelliteState:
    primary = state.primary_body_id
    sun = ephemeris.relative("sun", primary)
    power_environment = PowerEnvironment(
        sun.position_m,
        _occluding_bodies(ephemeris, primary),
    )
    result = advance_vehicle(
        state.definition,
        state.translation,
        state.vehicle,
        force_model_for(primary),
        command,
        power_environment,
        duration_s,
        PHYSICS_STEP_SECONDS,
    )
    control_mode = state.control_mode
    if result.vehicle.power_mode is PowerMode.POWER_SAFE:
        control_mode = ControlMode.POWER_SAFE
    elif control_mode is ControlMode.POWER_SAFE:
        control_mode = ControlMode.AUTONOMOUS
    pending_commands = (
        () if control_mode is ControlMode.POWER_SAFE else state.pending_commands
    )
    return replace(
        state,
        translation=result.translation,
        vehicle=result.vehicle,
        control_mode=control_mode,
        pending_commands=pending_commands,
    )


def _control_request(
    state: SatelliteState,
    autopilot: AutopilotState,
    previous_safety_reason: SafetyReason,
    duration_s: float,
    ephemeris: SolarSystemEphemeris,
    conjunction: ConjunctionCandidate | None,
    ready_plan: _ReadyPlan | None,
    defer_planning: bool,
) -> tuple[
    SatelliteState,
    AutopilotState,
    SafetyDecision,
    _PlanningRequest | None,
]:
    if conjunction is None and _autopilot_is_quiescent(state, autopilot):
        return (
            state,
            autopilot,
            SafetyDecision(
                ActuatorCommand(0.0, np.zeros(3), np.zeros(3)),
                SafetyReason.CLEAR,
            ),
            None,
        )
    snapshot = _snapshot(state, autopilot, previous_safety_reason, ())
    telemetry = SafetyTelemetry()
    if state.control_mode is ControlMode.MANUAL:
        command = _manual_command(state)
        decision = manual_control(command, state.vehicle, state.definition, telemetry)
        cancelled = autopilot_step(
            autopilot,
            snapshot,
            state.vehicle,
            state.definition,
            telemetry,
            duration_s,
        )
        return state, cancelled.state, decision, None

    objective, pending_commands = _next_objective(state, autopilot, conjunction)
    if objective is not autopilot.objective:
        autopilot = replace(
            autopilot,
            plan=None,
            execution=None,
            planning_failure=None,
        )
    needs_plan = (
        objective != autopilot.objective
        or (autopilot.plan is None and autopilot.planning_failure is None)
        or state.vehicle.requires_replan
    )
    planned_result = (
        ready_plan.result
        if ready_plan is not None and ready_plan.objective == objective
        else None
    )
    plan_is_ready = planned_result is not None
    output = autopilot_step(
        autopilot,
        snapshot,
        state.vehicle,
        state.definition,
        telemetry,
        duration_s,
        objective=objective,
        planned_result=planned_result,
        transfer_target=_transfer_target(state, objective, ephemeris),
        conjunction_risk=_conjunction_risk(conjunction),
        planning_pending=defer_planning and needs_plan and not plan_is_ready,
    )

    vehicle = state.vehicle
    planning_finished = (
        output.state.plan is not None or output.state.planning_failure is not None
    )
    if vehicle.requires_replan and planning_finished:
        vehicle = replace(vehicle, requires_replan=False)
    controlled_state = replace(
        state, vehicle=vehicle, pending_commands=pending_commands
    )
    planning_request = None
    planning_is_needed = (
        defer_planning
        and needs_plan
        and not plan_is_ready
        and output.state.plan is None
        and output.state.planning_failure is None
    )
    navigation = output.state.navigation
    if planning_is_needed and navigation.solution is not None:
        planning_request = _PlanningRequest(
            state.definition.id,
            state.primary_body_id,
            objective,
            navigation,
            state.definition,
            _transfer_target(state, objective, ephemeris),
            _conjunction_risk(conjunction),
        )
    return (
        controlled_state,
        output.state,
        output.decision,
        planning_request,
    )


def _advance_satellite(value: _SatelliteStepInput) -> _SatelliteStepOutput:
    satellite_id = value.state.definition.id
    try:
        controlled_state, autopilot, decision, planning_request = _control_request(
            value.state,
            value.autopilot,
            value.previous_safety_reason,
            value.duration_s,
            value.ephemeris,
            value.conjunction,
            value.ready_plan,
            value.defer_planning,
        )
        update = _advance_state(
            controlled_state,
            value.ephemeris,
            decision.command,
            value.duration_s,
        )
        update = _transition_primary(update, value.ephemeris)
        controlled_update = update, autopilot, decision
    except Exception:
        controlled_update = None
        planning_request = None
    return _SatelliteStepOutput(satellite_id, controlled_update, planning_request)


def _run_planning_request(request: _PlanningRequest) -> PlanningResult:
    try:
        result = plan_objective(
            request.navigation,
            request.definition,
            request.objective,
            request.transfer_target,
            request.conjunction_risk,
        )
    except (ArithmeticError, ValueError):
        result = None
    if result is None:
        return PlanningFailure(
            PlanningFailureCode.UNREACHABLE,
            "current state cannot produce a flight plan",
        )
    return result


def _autopilot_is_quiescent(state: SatelliteState, autopilot: AutopilotState) -> bool:
    execution = autopilot.execution
    return (
        state.control_mode is ControlMode.AUTONOMOUS
        and not state.pending_commands
        and not state.vehicle.requires_replan
        and execution is not None
        and execution.status is ExecutionStatus.COMPLETED
    )


def _occluding_bodies(
    ephemeris: SolarSystemEphemeris, primary_body_id: str
) -> tuple[OccludingBody, ...]:
    primary = ephemeris.system.body(primary_body_id)
    related = list(ephemeris.system.children_of(primary))
    parent = ephemeris.system.parent_of(primary)
    if parent is not None and parent.id != "sun":
        related.append(parent)
    return (
        OccludingBody(np.zeros(3), primary.radius_m),
        *(
            OccludingBody(
                ephemeris.relative(body.id, primary_body_id).position_m,
                body.radius_m,
            )
            for body in related
        ),
    )


def _transition_primary(
    state: SatelliteState, ephemeris: SolarSystemEphemeris
) -> SatelliteState:
    current_id = state.primary_body_id
    current_body = ephemeris.system.body(current_id)
    current_origin = ephemeris.body(current_id)
    system_position_m = current_origin.position_m + state.translation.position_m
    child_entries: list[tuple[float, str]] = []
    for child in ephemeris.system.children_of(current_body):
        child_distance_m = float(
            np.linalg.norm(system_position_m - ephemeris.body(child.id).position_m)
        )
        soi_radius_m = ephemeris.sphere_of_influence_radius_m(child.id)
        if child_distance_m <= soi_radius_m:
            child_entries.append((child_distance_m / soi_radius_m, child.id))
    next_primary_id = (
        min(child_entries)[1] if child_entries else _exited_primary(state, ephemeris)
    )
    if next_primary_id is None or next_primary_id == current_id:
        return state
    position_m, velocity_m_s = ephemeris.translate_primary(
        state.translation.position_m,
        state.translation.velocity_m_s,
        current_id,
        next_primary_id,
    )
    return replace(
        state,
        primary_body_id=next_primary_id,
        translation=replace(
            state.translation,
            position_m=position_m,
            velocity_m_s=velocity_m_s,
        ),
        vehicle=state.vehicle,
    )


def _exited_primary(
    state: SatelliteState, ephemeris: SolarSystemEphemeris
) -> str | None:
    parent_id = ephemeris.parent_id(state.primary_body_id)
    if parent_id is None:
        return None
    distance_m = float(np.linalg.norm(state.translation.position_m))
    if distance_m > ephemeris.sphere_of_influence_radius_m(state.primary_body_id):
        return parent_id
    return state.primary_body_id


def _transfer_target(
    state: SatelliteState,
    objective: FlightObjective,
    ephemeris: SolarSystemEphemeris,
) -> TransferTarget | None:
    command = objective.command
    target_id: str | None = None
    if isinstance(command, TransferPrimary):
        target_id = command.primary_body_id
    elif isinstance(command, ReturnStableOrbit):
        target_id = ephemeris.parent_id(state.primary_body_id)
    if target_id is None or target_id == state.primary_body_id:
        return None
    center_id = ephemeris.common_ancestor_id(state.primary_body_id, target_id)
    departure_origin = ephemeris.relative(state.primary_body_id, center_id)
    target_body = ephemeris.relative(target_id, center_id)
    departure_position_m = departure_origin.position_m + state.translation.position_m
    arrival_direction = _arrival_direction(
        target_body.position_m, departure_origin.position_m
    )
    target = ephemeris.system.body(target_id)
    insertion_radius_m = (
        target.radius_m + state.definition.autonomy.target_periapsis_altitude_m
    )
    insertion_position_m = arrival_direction * insertion_radius_m
    transfer_radius_m = (
        float(np.linalg.norm(departure_position_m))
        + float(np.linalg.norm(target_body.position_m + insertion_position_m))
    ) / 2.0
    center_mu_m3_s2 = force_model_for(center_id).gravitational_parameter_m3_s2
    flight_time_s = pi * sqrt(transfer_radius_m**3 / center_mu_m3_s2)
    return TransferTarget(
        target_id,
        center_id,
        departure_origin.position_m,
        departure_origin.velocity_m_s,
        target_body.position_m,
        target_body.velocity_m_s,
        insertion_position_m,
        flight_time_s,
    )


def _arrival_direction(
    target_position_m: np.ndarray, departure_position_m: np.ndarray
) -> np.ndarray:
    reference = target_position_m
    if not np.any(reference):
        reference = -departure_position_m
    if not np.any(reference):
        return np.asarray((1.0, 0.0, 0.0))
    return reference / float(np.linalg.norm(reference))


def _conjunction_id(conjunction: ConjunctionCandidate) -> str:
    return f"{conjunction.first_id}:{conjunction.second_id}"


def _conjunction_risk(
    conjunction: ConjunctionCandidate | None,
) -> ConjunctionRisk | None:
    if conjunction is None:
        return None
    return ConjunctionRisk(
        _conjunction_id(conjunction),
        conjunction.time_to_closest_approach_s,
        conjunction.predicted_distance_m,
        conjunction.required_distance_m,
    )


def _yielding_risks(
    conjunctions: tuple[ConjunctionCandidate, ...],
) -> dict[str, ConjunctionCandidate]:
    risks: dict[str, ConjunctionCandidate] = {}
    ordered = sorted(
        conjunctions,
        key=lambda item: (
            item.time_to_closest_approach_s,
            item.first_id,
            item.second_id,
        ),
    )
    for conjunction in ordered:
        risks.setdefault(conjunction.yielding_satellite_id, conjunction)
    return risks


def _conjunction_alerts(
    states: tuple[SatelliteState, ...],
    conjunctions: tuple[ConjunctionCandidate, ...],
) -> dict[str, tuple[str, ...]]:
    alerts: dict[str, list[str]] = {state.definition.id: [] for state in states}
    for conjunction in conjunctions:
        conjunction_id = _conjunction_id(conjunction)
        alerts[conjunction.first_id].append(conjunction_id)
        alerts[conjunction.second_id].append(conjunction_id)
    return {
        satellite_id: tuple(sorted(conjunction_ids))
        for satellite_id, conjunction_ids in alerts.items()
    }


def _manual_command(state: SatelliteState) -> ManualActuation:
    for command in reversed(state.pending_commands):
        if isinstance(command, ManualActuation):
            return command
    return ManualActuation(0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


def _next_objective(
    state: SatelliteState,
    autopilot: AutopilotState,
    conjunction: ConjunctionCandidate | None,
) -> tuple[FlightObjective, tuple[SatelliteCommand, ...]]:
    if conjunction is not None:
        conjunction_id = _conjunction_id(conjunction)
        return (
            FlightObjective(
                f"avoidance-{conjunction_id}",
                ObjectivePriority.COLLISION_AVOIDANCE,
                AvoidCollision(conjunction_id),
            ),
            state.pending_commands,
        )
    for index, command in enumerate(state.pending_commands):
        if isinstance(
            command,
            MaintainOrbit
            | SetOrbitAltitude
            | SetApsides
            | SetInclination
            | TransferPrimary
            | ReturnStableOrbit
            | Deorbit
            | AvoidCollision,
        ):
            objective = FlightObjective(
                f"user-{state.definition.id}-{type(command).__name__}",
                ObjectivePriority.USER_COMMAND,
                command,
            )
            pending = (
                *state.pending_commands[:index],
                *state.pending_commands[index + 1 :],
            )
            return objective, pending
    if autopilot.objective is not None:
        return autopilot.objective, state.pending_commands
    return (
        FlightObjective(
            f"maintenance-{state.definition.id}",
            ObjectivePriority.ORBIT_MAINTENANCE,
            MaintainOrbit(),
        ),
        state.pending_commands,
    )


def _command_rejection(
    state: SatelliteState, command: SatelliteCommand
) -> CommandResult | None:
    if state.control_mode is ControlMode.POWER_SAFE:
        return CommandResult(CommandStatus.POWER_SAFE, "satellite is in power safe")
    if _would_overflow_queue(state, command):
        return CommandResult(CommandStatus.QUEUE_FULL, "command queue is full")
    if not _command_is_valid(command):
        return CommandResult(CommandStatus.INVALID, "command values are invalid")
    if isinstance(command, ManualActuation):
        if state.control_mode is not ControlMode.MANUAL:
            return CommandResult(
                CommandStatus.NOT_ALLOWED, "manual control is inactive"
            )
    command_name = _allowed_command_name(command)
    command_is_disabled = (
        command_name is not None
        and command_name not in state.definition.autonomy.allowed_commands
    )
    if command_is_disabled:
        return CommandResult(CommandStatus.NOT_ALLOWED, "command is disabled")
    return None


def _snapshot(
    state: SatelliteState,
    autopilot: AutopilotState,
    safety_reason: SafetyReason,
    conjunction_alert_ids: tuple[str, ...],
) -> SatelliteSnapshot:
    translation = state.translation
    attitude = state.vehicle.attitude
    resources = state.vehicle.resources
    orbit = (
        None
        if autopilot.navigation.solution is None
        else autopilot.navigation.solution.orbit
    )
    force_model = force_model_for(state.primary_body_id)
    periapsis_altitude_m = (
        None
        if orbit is None
        else orbit.semi_major_axis_m * (1.0 - orbit.eccentricity)
        - force_model.body_radius_m
    )
    apoapsis_altitude_m = (
        None
        if orbit is None
        else orbit.semi_major_axis_m * (1.0 + orbit.eccentricity)
        - force_model.body_radius_m
    )
    orbital_period_s = (
        None
        if orbit is None
        else 2.0
        * pi
        * sqrt(orbit.semi_major_axis_m**3 / force_model.gravitational_parameter_m3_s2)
    )
    return SatelliteSnapshot(
        state.definition.id,
        state.definition.display_name,
        state.primary_body_id,
        _vector3(translation.position_m),
        _vector3(translation.velocity_m_s),
        translation.mass_kg,
        _quaternion(attitude.quaternion_wxyz),
        _vector3(attitude.angular_velocity_rad_s),
        _vector3(attitude.wheel_momentum_n_m_s),
        resources.main_propellant_kg,
        resources.rcs_propellant_kg,
        resources.battery_energy_j,
        state.control_mode,
        len(state.pending_commands),
        state.vehicle.requires_replan,
        None if autopilot.plan is None else autopilot.plan.objective_id,
        None if autopilot.execution is None else autopilot.execution.status.value,
        (
            None
            if autopilot.planning_failure is None
            else autopilot.planning_failure.reason
        ),
        safety_reason.value,
        conjunction_alert_ids,
        periapsis_altitude_m,
        apoapsis_altitude_m,
        None if orbit is None else np.degrees(orbit.inclination_rad),
        orbital_period_s,
    )


def _vector3(values: np.ndarray) -> tuple[float, float, float]:
    return float(values[0]), float(values[1]), float(values[2])


def _quaternion(values: np.ndarray) -> tuple[float, float, float, float]:
    return float(values[0]), float(values[1]), float(values[2]), float(values[3])


def _state_is_valid(state: SatelliteState) -> bool:
    translation = state.translation
    attitude = state.vehicle.attitude
    arrays = (
        translation.position_m,
        translation.velocity_m_s,
        attitude.quaternion_wxyz,
        attitude.angular_velocity_rad_s,
        attitude.wheel_momentum_n_m_s,
    )
    values_are_finite = bool(np.all(np.isfinite(np.concatenate(arrays))))
    mass_is_valid = isfinite(translation.mass_kg) and translation.mass_kg > 0.0
    quaternion_norm = float(
        np.sqrt(np.dot(attitude.quaternion_wxyz, attitude.quaternion_wxyz))
    )
    quaternion_is_valid = abs(quaternion_norm - 1.0) <= QUATERNION_NORM_TOLERANCE
    return values_are_finite and mass_is_valid and quaternion_is_valid


def _detect_destructions(
    previous: tuple[SatelliteState, ...],
    updates: dict[str, SatelliteState],
    elapsed_seconds: float,
    duration_s: float,
) -> tuple[DestructionEvent, ...]:
    events: list[DestructionEvent] = []
    previous_by_id = {state.definition.id: state for state in previous}
    for satellite_id, updated in sorted(updates.items()):
        before = previous_by_id[satellite_id]
        state_at = lambda time_s, first=before.translation, second=updated.translation: _linear_state(
            first, second, time_s / duration_s
        )
        destruction = locate_destruction_time_s(
            before,
            duration_s,
            state_at,
            lambda state: state.velocity_m_s,
        )
        if destruction is not None:
            event_time_s, cause = destruction
            events.append(
                DestructionEvent(
                    f"{cause.value}-{satellite_id}-{elapsed_seconds + event_time_s:.6f}",
                    elapsed_seconds + event_time_s,
                    (satellite_id,),
                    cause,
                )
            )
    destroyed = {
        satellite_id for event in events for satellite_id in event.satellite_ids
    }
    remaining = tuple(
        state for state in previous if state.definition.id not in destroyed
    )
    for first_index, second_index in _collision_candidates(remaining, updates):
        first_before = remaining[first_index]
        second_before = remaining[second_index]
        first_after = updates[first_before.definition.id]
        second_after = updates[second_before.definition.id]
        collision_time_s = _collision_time_s(
            first_before,
            first_after,
            second_before,
            second_after,
            duration_s,
        )
        if collision_time_s is None:
            continue
        first_id = min(first_before.definition.id, second_before.definition.id)
        second_id = max(first_before.definition.id, second_before.definition.id)
        events.append(
            DestructionEvent(
                f"collision-{first_id}-{second_id}-{elapsed_seconds + collision_time_s:.6f}",
                elapsed_seconds + collision_time_s,
                (first_id, second_id),
                DestructionCause.COLLISION,
            )
        )
    return tuple(
        sorted(events, key=lambda event: (event.elapsed_seconds, event.event_id))
    )


def _collision_candidates(
    previous: tuple[SatelliteState, ...], updates: dict[str, SatelliteState]
) -> tuple[tuple[int, int], ...]:
    if len(previous) < 2:
        return ()
    start = np.stack(tuple(state.translation.position_m for state in previous))
    end = np.stack(
        tuple(updates[state.definition.id].translation.position_m for state in previous)
    )
    relative_start = start[:, None, :] - start[None, :, :]
    relative_motion = (end - start)[:, None, :] - (end - start)[None, :, :]
    motion_squared = np.sum(relative_motion * relative_motion, axis=2)
    dot = np.sum(relative_start * relative_motion, axis=2)
    fractions = np.zeros_like(motion_squared)
    np.divide(-dot, motion_squared, out=fractions, where=motion_squared > 0.0)
    np.clip(fractions, 0.0, 1.0, out=fractions)
    closest = relative_start + relative_motion * fractions[:, :, None]
    distance_squared = np.sum(closest * closest, axis=2)
    radii = np.asarray(tuple(state.definition.collision_radius_m for state in previous))
    combined_radii_squared = (radii[:, None] + radii[None, :]) ** 2
    same_primary = np.equal.outer(
        tuple(state.primary_body_id for state in previous),
        tuple(state.primary_body_id for state in previous),
    )
    candidate_mask = np.triu(
        same_primary & (distance_squared <= combined_radii_squared), k=1
    )
    return tuple(
        (int(first), int(second)) for first, second in np.argwhere(candidate_mask)
    )


def _collision_time_s(
    first_before: SatelliteState,
    first_after: SatelliteState,
    second_before: SatelliteState,
    second_after: SatelliteState,
    duration_s: float,
) -> float | None:
    start_position = (
        first_before.translation.position_m - second_before.translation.position_m
    )
    end_position = (
        first_after.translation.position_m - second_after.translation.position_m
    )
    displacement = end_position - start_position
    displacement_squared = float(np.dot(displacement, displacement))
    closest_fraction = 0.0
    if displacement_squared > 0.0:
        closest_fraction = float(
            np.clip(
                -np.dot(start_position, displacement) / displacement_squared, 0.0, 1.0
            )
        )
    combined_radius_m = (
        first_before.definition.collision_radius_m
        + second_before.definition.collision_radius_m
    )
    closest_distance_m = float(
        np.linalg.norm(start_position + displacement * closest_fraction)
    )
    if closest_distance_m > combined_radius_m:
        return None
    closest_time_s = duration_s * closest_fraction
    if closest_time_s == 0.0:
        return 0.0

    def relative_state_at(time_s: float) -> TranslationalState:
        fraction = time_s / duration_s
        position = start_position + displacement * fraction
        velocity = (
            first_before.translation.velocity_m_s
            - second_before.translation.velocity_m_s
        )
        return TranslationalState(position, velocity, 1.0)

    return locate_collision_time_s(closest_time_s, relative_state_at, combined_radius_m)


def _linear_state(
    first: TranslationalState,
    second: TranslationalState,
    fraction: float,
) -> TranslationalState:
    return TranslationalState(
        first.position_m + (second.position_m - first.position_m) * fraction,
        first.velocity_m_s + (second.velocity_m_s - first.velocity_m_s) * fraction,
        first.mass_kg + (second.mass_kg - first.mass_kg) * fraction,
    )


def _would_overflow_queue(state: SatelliteState, command: SatelliteCommand) -> bool:
    replaces_control_state = isinstance(
        command,
        (TakeManualControl, ReturnAutonomousControl, ManualActuation),
    )
    return (
        not replaces_control_state
        and len(state.pending_commands) >= MAX_PENDING_COMMANDS_PER_SATELLITE
    )


def _command_is_valid(command: SatelliteCommand) -> bool:
    if isinstance(command, SetOrbitAltitude):
        return isfinite(command.altitude_m) and command.altitude_m >= 0.0
    if isinstance(command, SetApsides):
        values_are_finite = isfinite(command.periapsis_altitude_m) and isfinite(
            command.apoapsis_altitude_m
        )
        apsides_are_ordered = (
            0.0 <= command.periapsis_altitude_m <= command.apoapsis_altitude_m
        )
        return values_are_finite and apsides_are_ordered
    if isinstance(command, SetInclination):
        return (
            isfinite(command.inclination_rad)
            and 0.0 <= command.inclination_rad <= np.pi
        )
    if isinstance(command, TransferPrimary):
        return _primary_body_is_supported(command.primary_body_id)
    if isinstance(command, AvoidCollision):
        return bool(command.conjunction_id)
    if isinstance(command, ManualActuation):
        values = (
            command.main_throttle,
            *command.wheel_torque_n_m,
            *command.rcs_torque_n_m,
        )
        values_are_finite = all(isfinite(value) for value in values)
        throttle_is_valid = 0.0 <= command.main_throttle <= 1.0
        return values_are_finite and throttle_is_valid
    return True


def _primary_body_is_supported(primary_body_id: str) -> bool:
    try:
        force_model_for(primary_body_id)
    except ValueError:
        return False
    return True


def _allowed_command_name(command: SatelliteCommand) -> str | None:
    if isinstance(command, MaintainOrbit):
        return "maintain_orbit"
    if isinstance(command, SetOrbitAltitude | SetApsides):
        return "set_apsides"
    if isinstance(command, SetInclination):
        return "set_inclination"
    if isinstance(command, TransferPrimary):
        return "transfer_primary"
    if isinstance(command, ReturnStableOrbit):
        return "return_stable_orbit"
    if isinstance(command, Deorbit):
        return "deorbit"
    if isinstance(command, AvoidCollision):
        return "avoid_collision"
    return None
