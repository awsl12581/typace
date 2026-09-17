"""Solar generation, eclipse, and POWER_SAFE thresholds."""

from enum import StrEnum

import numpy as np

from typace.config.physics import ASTRONOMICAL_UNIT_M, SOLAR_CONSTANT_W_M2
from typace.config.vehicle import (
    POWER_SAFE_ENTER_FRACTION,
    POWER_SAFE_RECOVER_FRACTION,
)
from typace.satellites.definition import PowerDefinition
from typace.physics.environment import OccludingBody, is_fully_eclipsed
from typace.vehicle.resources import ResourceState, update_battery


class PowerMode(StrEnum):
    NOMINAL = "nominal"
    POWER_SAFE = "power_safe"


def solar_power_w(
    satellite_position_m: np.ndarray,
    sun_position_m: np.ndarray,
    panel_normal_inertial: np.ndarray,
    power: PowerDefinition,
    occluding_bodies: tuple[OccludingBody, ...],
) -> float:
    sun_vector = sun_position_m - satellite_position_m
    distance_m = float(np.linalg.norm(sun_vector))
    if distance_m == 0.0:
        raise ValueError("sun and satellite positions must differ")
    direction = sun_vector / distance_m
    panel_normal_length = float(np.linalg.norm(panel_normal_inertial))
    if panel_normal_length == 0.0:
        raise ValueError("panel normal must be non-zero")
    panel_direction = panel_normal_inertial / panel_normal_length
    incidence = max(0.0, float(np.dot(panel_direction, direction)))
    eclipse_factor = (
        0.0
        if is_fully_eclipsed(satellite_position_m, sun_position_m, occluding_bodies)
        else 1.0
    )
    return float(
        SOLAR_CONSTANT_W_M2
        * (ASTRONOMICAL_UNIT_M / distance_m) ** 2
        * power.panel_area_m2
        * power.panel_efficiency
        * incidence
        * eclipse_factor
    )


def power_step(
    resources: ResourceState,
    generated_power_w: float,
    power: PowerDefinition,
    duration_s: float,
    current_mode: PowerMode,
    *,
    actuator_enabled: bool,
) -> tuple[ResourceState, PowerMode]:
    if generated_power_w < 0.0:
        raise ValueError("generated power cannot be negative")
    load_w = power.base_power_w
    if current_mode is PowerMode.NOMINAL:
        load_w += power.computer_power_w + power.sensor_power_w
        if actuator_enabled:
            load_w += power.actuator_power_w
    updated = update_battery(
        resources, generated_power_w - load_w, duration_s, power.battery_capacity_j
    )
    brownout = (
        updated.battery_energy_j <= power.battery_capacity_j * POWER_SAFE_ENTER_FRACTION
    )
    recovered = (
        updated.battery_energy_j
        >= power.battery_capacity_j * POWER_SAFE_RECOVER_FRACTION
    )
    if current_mode is PowerMode.POWER_SAFE:
        next_mode = PowerMode.NOMINAL if recovered else PowerMode.POWER_SAFE
    else:
        next_mode = PowerMode.POWER_SAFE if brownout else PowerMode.NOMINAL
    return updated, next_mode
