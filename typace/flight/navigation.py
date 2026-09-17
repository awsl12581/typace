"""Deterministic ideal navigation derived only from world snapshots."""

from dataclasses import dataclass

import numpy as np

from typace.physics.bodies import force_model_for
from typace.physics.elements import CartesianState, ClassicalElements, state_to_elements
from typace.satellites.state import ControlMode, SatelliteSnapshot


@dataclass(frozen=True, slots=True)
class NavigationSolution:
    satellite_id: str
    primary_body_id: str
    position_m: np.ndarray
    velocity_m_s: np.ndarray
    mass_kg: float
    main_propellant_kg: float
    rcs_propellant_kg: float
    orbit: ClassicalElements


@dataclass(frozen=True, slots=True)
class NavigationState:
    solution: NavigationSolution | None = None


def update_navigation(
    previous: NavigationState, snapshot: SatelliteSnapshot
) -> NavigationState:
    if snapshot.control_mode is ControlMode.POWER_SAFE:
        return previous
    model = force_model_for(snapshot.primary_body_id)
    position = np.asarray(snapshot.position_m)
    velocity = np.asarray(snapshot.velocity_m_s)
    orbit = state_to_elements(
        CartesianState(position, velocity), model.gravitational_parameter_m3_s2
    )
    return NavigationState(
        NavigationSolution(
            snapshot.id,
            snapshot.primary_body_id,
            position,
            velocity,
            snapshot.mass_kg,
            snapshot.main_propellant_kg,
            snapshot.rcs_propellant_kg,
            orbit,
        )
    )
