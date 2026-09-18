"""Single vehicle step joining dynamics, actuators, fuel, and power."""

from dataclasses import dataclass, replace
from math import ceil

import numpy as np

from typace.config.physics import STANDARD_GRAVITY_M_S2
from typace.config.vehicle import MASS_CONSISTENCY_TOLERANCE_KG
from typace.physics.environment import OccludingBody
from typace.physics.forces import ForceModel
from typace.physics.propagation import (
    PropagationContext,
    TranslationalState,
    propagate,
)
from typace.satellites.definition import SatelliteDefinition
from typace.vehicle.actuators import (
    ActuatorCommand,
    ActuatorOutput,
    limit_rcs_torque,
    limit_reaction_wheel_torque,
)
from typace.vehicle.attitude import integrate_attitude, rotate_body_to_inertial
from typace.vehicle.power import PowerMode, power_step, solar_power_w
from typace.vehicle.resources import ResourceState
from typace.vehicle.state import VehicleState


@dataclass(frozen=True, slots=True)
class PowerEnvironment:
    sun_position_m: np.ndarray
    occluding_bodies: tuple[OccludingBody, ...]


@dataclass(frozen=True, slots=True)
class VehicleStepResult:
    translation: TranslationalState
    vehicle: VehicleState


def resolve_actuators(
    definition: SatelliteDefinition,
    state: VehicleState,
    command: ActuatorCommand,
    duration_s: float,
) -> ActuatorOutput:
    if duration_s <= 0.0:
        raise ValueError("duration must be positive")
    if not 0.0 <= command.main_throttle <= 1.0:
        raise ValueError("main throttle must be between zero and one")
    if state.power_mode is PowerMode.POWER_SAFE:
        return _idle_output()
    command_is_idle = (
        command.main_throttle == 0.0
        and not np.any(command.wheel_torque_n_m)
        and not np.any(command.rcs_torque_n_m)
    )
    if command_is_idle:
        return _idle_output()

    propulsion = definition.propulsion
    throttle = command.main_throttle
    if throttle > 0.0:
        throttle = max(throttle, propulsion.main_minimum_throttle)
    main_mass_flow = (
        propulsion.main_thrust_n
        * throttle
        / (propulsion.main_specific_impulse_s * STANDARD_GRAVITY_M_S2)
    )
    main_duration = _available_duration_s(
        state.resources.main_propellant_kg, main_mass_flow, duration_s
    )
    wheel_torque = limit_reaction_wheel_torque(
        command.wheel_torque_n_m,
        state.attitude.wheel_momentum_n_m_s,
        np.asarray(definition.reaction_wheel_maximum_torque_n_m),
        np.asarray(definition.reaction_wheel_maximum_momentum_n_m_s),
        duration_s,
    )
    rcs_torque = limit_rcs_torque(
        command.rcs_torque_n_m, propulsion.rcs_maximum_torque_n_m
    )
    rcs_fraction = _rcs_fraction(rcs_torque, propulsion.rcs_maximum_torque_n_m)
    rcs_mass_flow = (
        propulsion.rcs_thrust_n
        * rcs_fraction
        / (propulsion.rcs_specific_impulse_s * STANDARD_GRAVITY_M_S2)
    )
    rcs_duration = _available_duration_s(
        state.resources.rcs_propellant_kg, rcs_mass_flow, duration_s
    )
    return ActuatorOutput(
        throttle,
        wheel_torque,
        rcs_torque,
        main_mass_flow,
        rcs_mass_flow,
        main_duration,
        rcs_duration,
    )


def advance_vehicle(
    definition: SatelliteDefinition,
    translation: TranslationalState,
    vehicle: VehicleState,
    force_model: ForceModel,
    command: ActuatorCommand,
    power_environment: PowerEnvironment,
    duration_s: float,
    maximum_step_s: float,
) -> VehicleStepResult:
    """Advance all vehicle state through fixed substeps and one force path."""

    if duration_s < 0.0 or maximum_step_s <= 0.0:
        raise ValueError(
            "duration cannot be negative and maximum step must be positive"
        )
    mass_difference_kg = abs(translation.mass_kg - vehicle.resources.total_mass_kg)
    if mass_difference_kg > MASS_CONSISTENCY_TOLERANCE_KG:
        raise ValueError("translation mass does not match vehicle resources")
    if duration_s == 0.0:
        return VehicleStepResult(translation, vehicle)

    step_count = max(1, ceil(duration_s / maximum_step_s))
    step_duration_s = duration_s / step_count
    current_translation = translation
    current_vehicle = vehicle
    for _ in range(step_count):
        current_translation, current_vehicle = _advance_substep(
            definition,
            current_translation,
            current_vehicle,
            force_model,
            command,
            power_environment,
            step_duration_s,
            allow_analytic=step_duration_s >= maximum_step_s,
        )
    return VehicleStepResult(current_translation, current_vehicle)


def _advance_substep(
    definition: SatelliteDefinition,
    translation: TranslationalState,
    vehicle: VehicleState,
    force_model: ForceModel,
    command: ActuatorCommand,
    power_environment: PowerEnvironment,
    duration_s: float,
    *,
    allow_analytic: bool,
) -> tuple[TranslationalState, VehicleState]:
    output = resolve_actuators(definition, vehicle, command, duration_s)
    boundaries = sorted(
        {0.0, output.main_active_duration_s, output.rcs_active_duration_s, duration_s}
    )
    current_translation = translation
    current_attitude = vehicle.attitude
    current_resources = vehicle.resources
    for start_s, end_s in zip(boundaries, boundaries[1:]):
        segment_duration_s = end_s - start_s
        if segment_duration_s == 0.0:
            continue
        main_active = start_s < output.main_active_duration_s
        rcs_active = start_s < output.rcs_active_duration_s
        thrust_inertial_n = np.zeros(3)
        if main_active:
            body_axis = np.asarray(definition.propulsion.main_body_axis)
            body_axis_length = float(np.linalg.norm(body_axis))
            if body_axis_length == 0.0:
                raise ValueError("main body axis must be non-zero")
            thrust_body_n = (
                body_axis
                / body_axis_length
                * definition.propulsion.main_thrust_n
                * output.main_throttle
            )
            thrust_inertial_n = rotate_body_to_inertial(
                current_attitude.quaternion_wxyz, thrust_body_n
            )
        main_mass_flow = output.main_mass_flow_kg_s if main_active else 0.0
        rcs_mass_flow = output.rcs_mass_flow_kg_s if rcs_active else 0.0
        context = PropagationContext(
            force_model,
            definition.drag_area_m2,
            definition.drag_coefficient,
            thrust_inertial_n,
            main_mass_flow + rcs_mass_flow,
        )
        current_translation = propagate(
            current_translation,
            context,
            segment_duration_s,
            segment_duration_s,
            allow_analytic=allow_analytic,
        )
        rcs_torque = output.rcs_torque_n_m if rcs_active else np.zeros(3)
        attitude_is_stationary = (
            not np.any(current_attitude.angular_velocity_rad_s)
            and not np.any(output.wheel_torque_n_m)
            and not np.any(rcs_torque)
        )
        if not attitude_is_stationary:
            current_attitude = integrate_attitude(
                current_attitude,
                output.wheel_torque_n_m,
                rcs_torque,
                np.asarray(definition.inertia_diagonal_kg_m2),
                segment_duration_s,
            )
        current_resources = _consume_flows(
            current_resources,
            main_mass_flow,
            rcs_mass_flow,
            segment_duration_s,
        )

    panel_normal = rotate_body_to_inertial(
        current_attitude.quaternion_wxyz,
        np.asarray(definition.power.panel_body_normal),
    )
    generated_power_w = solar_power_w(
        current_translation.position_m,
        power_environment.sun_position_m,
        panel_normal,
        definition.power,
        power_environment.occluding_bodies,
    )
    actuator_enabled = _output_is_active(output)
    powered_resources, next_power_mode = power_step(
        current_resources,
        generated_power_w,
        definition.power,
        duration_s,
        vehicle.power_mode,
        actuator_enabled=actuator_enabled,
    )
    recovered = (
        vehicle.power_mode is PowerMode.POWER_SAFE
        and next_power_mode is PowerMode.NOMINAL
    )
    next_vehicle = VehicleState(
        current_attitude,
        powered_resources,
        next_power_mode,
        vehicle.requires_replan or recovered,
    )
    corrected_translation = replace(
        current_translation, mass_kg=powered_resources.total_mass_kg
    )
    return corrected_translation, next_vehicle


def _consume_flows(
    resources: ResourceState,
    main_mass_flow_kg_s: float,
    rcs_mass_flow_kg_s: float,
    duration_s: float,
) -> ResourceState:
    return ResourceState(
        resources.dry_mass_kg,
        max(0.0, resources.main_propellant_kg - main_mass_flow_kg_s * duration_s),
        max(0.0, resources.rcs_propellant_kg - rcs_mass_flow_kg_s * duration_s),
        resources.battery_energy_j,
    )


def _available_duration_s(
    propellant_kg: float, mass_flow_kg_s: float, requested_duration_s: float
) -> float:
    if mass_flow_kg_s == 0.0:
        return 0.0
    return min(requested_duration_s, propellant_kg / mass_flow_kg_s)


def _rcs_fraction(rcs_torque_n_m: np.ndarray, maximum_torque_n_m: float) -> float:
    if maximum_torque_n_m == 0.0:
        return 0.0
    return float(np.linalg.norm(rcs_torque_n_m)) / maximum_torque_n_m


def _idle_output() -> ActuatorOutput:
    return ActuatorOutput(0.0, np.zeros(3), np.zeros(3), 0.0, 0.0, 0.0, 0.0)


def _output_is_active(output: ActuatorOutput) -> bool:
    has_main_thrust = output.main_active_duration_s > 0.0
    has_rcs_torque = output.rcs_active_duration_s > 0.0
    has_wheel_torque = bool(np.linalg.norm(output.wheel_torque_n_m))
    return has_main_thrust or has_rcs_torque or has_wheel_torque
