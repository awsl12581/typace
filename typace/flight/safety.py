"""Shared safety filter for autonomous and manual actuator requests."""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from typace.satellites.definition import SatelliteDefinition
from typace.vehicle.actuators import ActuatorCommand
from typace.vehicle.power import PowerMode
from typace.vehicle.state import VehicleState


class SafetyReason(StrEnum):
    CLEAR = "clear"
    POWER_SAFE = "power_safe"
    DYNAMIC_PRESSURE = "dynamic_pressure"
    HEAT_FLUX = "heat_flux"
    MAIN_PROPELLANT_EMPTY = "main_propellant_empty"


@dataclass(frozen=True, slots=True)
class SafetyTelemetry:
    dynamic_pressure_pa: float = 0.0
    heat_flux_w_m2: float = 0.0
    collision_imminent: bool = False


@dataclass(frozen=True, slots=True)
class SafetyDecision:
    command: ActuatorCommand
    reason: SafetyReason


def filter_actuators(
    command: ActuatorCommand,
    vehicle: VehicleState,
    definition: SatelliteDefinition,
    telemetry: SafetyTelemetry,
) -> SafetyDecision:
    reason = _blocking_reason(command, vehicle, definition, telemetry)
    if reason is SafetyReason.CLEAR:
        return SafetyDecision(command, reason)
    return SafetyDecision(ActuatorCommand(0.0, np.zeros(3), np.zeros(3)), reason)


def _blocking_reason(
    command: ActuatorCommand,
    vehicle: VehicleState,
    definition: SatelliteDefinition,
    telemetry: SafetyTelemetry,
) -> SafetyReason:
    if vehicle.power_mode is PowerMode.POWER_SAFE:
        return SafetyReason.POWER_SAFE
    if telemetry.dynamic_pressure_pa >= definition.maximum_dynamic_pressure_pa:
        return SafetyReason.DYNAMIC_PRESSURE
    if telemetry.heat_flux_w_m2 >= definition.maximum_heat_flux_w_m2:
        return SafetyReason.HEAT_FLUX
    main_burn_without_fuel = (
        command.main_throttle > 0.0 and vehicle.resources.main_propellant_kg <= 0.0
    )
    if main_burn_without_fuel:
        return SafetyReason.MAIN_PROPELLANT_EMPTY
    return SafetyReason.CLEAR
