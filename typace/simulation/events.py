"""Event root localization and deterministic destruction records."""

from dataclasses import dataclass
from collections.abc import Callable
from enum import StrEnum

import numpy as np

from typace.physics.bodies import force_model_for
from typace.physics.environment import (
    atmosphere_relative_velocity_m_s,
    atmospheric_density_kg_m3,
    dynamic_pressure_pa,
    stagnation_heat_flux_w_m2,
)
from typace.physics.propagation import TranslationalState, locate_event_time_s
from typace.satellites.state import SatelliteState


class DestructionCause(StrEnum):
    COLLISION = "collision"
    SURFACE_IMPACT = "surface_impact"
    DYNAMIC_PRESSURE = "dynamic_pressure"
    HEAT_FLUX = "heat_flux"
    NUMERICAL_FAILURE = "numerical_failure"


@dataclass(frozen=True, slots=True)
class DestructionEvent:
    event_id: str
    elapsed_seconds: float
    satellite_ids: tuple[str, ...]
    cause: DestructionCause


def locate_collision_time_s(
    duration_s: float,
    state_at: Callable[[float], TranslationalState],
    combined_radius_m: float,
) -> float:
    return locate_event_time_s(
        duration_s,
        state_at,
        lambda state: float(np.linalg.norm(state.position_m)) - combined_radius_m,
    )


def locate_destruction_time_s(
    satellite: SatelliteState,
    duration_s: float,
    state_at: Callable[[float], TranslationalState],
    relative_velocity_at: Callable[[TranslationalState], np.ndarray],
) -> tuple[float, DestructionCause] | None:
    initial = state_at(0.0)
    final = state_at(duration_s)
    checks = (
        (
            DestructionCause.SURFACE_IMPACT,
            lambda state: _surface_value(satellite, state),
        ),
        (
            DestructionCause.DYNAMIC_PRESSURE,
            lambda state: _dynamic_pressure_value(
                satellite, state, relative_velocity_at
            ),
        ),
        (
            DestructionCause.HEAT_FLUX,
            lambda state: _heat_flux_value(satellite, state, relative_velocity_at),
        ),
    )
    initial_values = _destruction_values(satellite, initial, relative_velocity_at)
    final_values = _destruction_values(satellite, final, relative_velocity_at)
    candidates: list[tuple[float, DestructionCause]] = []
    for (cause, value), initial_value, final_value in zip(
        checks, initial_values, final_values
    ):
        if initial_value <= 0.0:
            return 0.0, cause
        if final_value > 0.0:
            continue
        candidates.append((locate_event_time_s(duration_s, state_at, value), cause))
    return min(candidates, default=None)


def _destruction_values(
    satellite: SatelliteState,
    state: TranslationalState,
    relative_velocity_at: Callable[[TranslationalState], np.ndarray],
) -> tuple[float, float, float]:
    model = force_model_for(satellite.primary_body_id)
    altitude_m = float(np.linalg.norm(state.position_m)) - model.body_radius_m
    if model.primary_body_id != "earth":
        return altitude_m, 1.0, 1.0
    relative = atmosphere_relative_velocity_m_s(
        state.position_m,
        relative_velocity_at(state),
        model.atmosphere_rotation_rate_rad_s,
    )
    density = atmospheric_density_kg_m3(altitude_m)
    dynamic_pressure_margin = (
        satellite.definition.maximum_dynamic_pressure_pa
        - dynamic_pressure_pa(density, relative)
    )
    heat_flux_margin = (
        satellite.definition.maximum_heat_flux_w_m2
        - stagnation_heat_flux_w_m2(
            density,
            satellite.definition.nose_radius_m,
            relative,
        )
    )
    return altitude_m, dynamic_pressure_margin, heat_flux_margin


def _surface_value(satellite: SatelliteState, state: TranslationalState) -> float:
    return (
        float(np.linalg.norm(state.position_m))
        - force_model_for(satellite.primary_body_id).body_radius_m
    )


def _dynamic_pressure_value(
    satellite: SatelliteState,
    state: TranslationalState,
    relative_velocity_at: Callable[[TranslationalState], np.ndarray],
) -> float:
    model = force_model_for(satellite.primary_body_id)
    if model.primary_body_id != "earth":
        return 1.0
    velocity = relative_velocity_at(state)
    relative = atmosphere_relative_velocity_m_s(
        state.position_m, velocity, model.atmosphere_rotation_rate_rad_s
    )
    density = atmospheric_density_kg_m3(
        float(np.linalg.norm(state.position_m)) - model.body_radius_m
    )
    return satellite.definition.maximum_dynamic_pressure_pa - dynamic_pressure_pa(
        density, relative
    )


def _heat_flux_value(
    satellite: SatelliteState,
    state: TranslationalState,
    relative_velocity_at: Callable[[TranslationalState], np.ndarray],
) -> float:
    model = force_model_for(satellite.primary_body_id)
    if model.primary_body_id != "earth":
        return 1.0
    velocity = relative_velocity_at(state)
    relative = atmosphere_relative_velocity_m_s(
        state.position_m, velocity, model.atmosphere_rotation_rate_rad_s
    )
    density = atmospheric_density_kg_m3(
        float(np.linalg.norm(state.position_m)) - model.body_radius_m
    )
    return satellite.definition.maximum_heat_flux_w_m2 - stagnation_heat_flux_w_m2(
        density, satellite.definition.nose_radius_m, relative
    )
