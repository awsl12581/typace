"""Single-clock deterministic multi-satellite world."""

from dataclasses import dataclass, replace
from math import isfinite, radians
from typing import cast

from astropy.time import Time, TimeDelta
import numpy as np

from typace.config.physics import (
    EARTH_MEAN_RADIUS_M,
    MOON_MEAN_RADIUS_M,
)
from typace.config.simulation import (
    EPOCH_NORMALIZATION_STEP_SECONDS,
    MAX_PENDING_COMMANDS_PER_SATELLITE,
    MAX_PHYSICS_SUBSTEPS_PER_WORLD_STEP,
    PHYSICS_STEP_SECONDS,
    TIME_WARPS,
)
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
from typace.simulation.ephemeris import EarthMoonEphemeris, ephemeris_at
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

    def satellite(self, satellite_id: str) -> SatelliteSnapshot:
        for satellite in self.satellites:
            if satellite.id == satellite_id:
                return satellite
        raise KeyError(satellite_id)


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
        updates = {
            state.definition.id: _advance_state(state, ephemeris, simulation_duration_s)
            for state in frozen_states
        }
        self._satellites = updates
        self._elapsed_seconds += simulation_duration_s
        return self.snapshot()

    def snapshot(self) -> WorldSnapshot:
        satellites = tuple(
            _snapshot(self._satellites[satellite_id])
            for satellite_id in sorted(self._satellites)
        )
        return WorldSnapshot(
            cast(str, self._scenario_epoch.to_value("isot")),
            self._elapsed_seconds,
            self._selected_time_warp,
            self._effective_time_warp,
            self._time_warp_limit_reason,
            satellites,
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
    ephemeris: EarthMoonEphemeris,
    duration_s: float,
) -> SatelliteState:
    primary = state.primary_body_id
    sun = ephemeris.relative("sun", primary)
    other_body_id = "moon" if primary == "earth" else "earth"
    other = ephemeris.relative(other_body_id, primary)
    other_radius_m = (
        MOON_MEAN_RADIUS_M if other_body_id == "moon" else EARTH_MEAN_RADIUS_M
    )
    power_environment = PowerEnvironment(
        sun.position_m,
        (
            OccludingBody(np.zeros(3), force_model_for(primary).body_radius_m),
            OccludingBody(other.position_m, other_radius_m),
        ),
    )
    command = _actuator_command(state)
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


def _actuator_command(state: SatelliteState) -> ActuatorCommand:
    if state.control_mode is not ControlMode.MANUAL:
        return ActuatorCommand(0.0, np.zeros(3), np.zeros(3))
    for command in reversed(state.pending_commands):
        if isinstance(command, ManualActuation):
            return ActuatorCommand(
                command.main_throttle,
                np.asarray(command.wheel_torque_n_m),
                np.asarray(command.rcs_torque_n_m),
            )
    return ActuatorCommand(0.0, np.zeros(3), np.zeros(3))


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


def _snapshot(state: SatelliteState) -> SatelliteSnapshot:
    translation = state.translation
    attitude = state.vehicle.attitude
    resources = state.vehicle.resources
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
    )


def _vector3(values: np.ndarray) -> tuple[float, float, float]:
    return float(values[0]), float(values[1]), float(values[2])


def _quaternion(values: np.ndarray) -> tuple[float, float, float, float]:
    return float(values[0]), float(values[1]), float(values[2]), float(values[3])


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
        return command.primary_body_id in ("earth", "moon")
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
