"""Aggregate vehicle state initialized from one catalog definition."""

from dataclasses import dataclass

import numpy as np

from typace.satellites.definition import SatelliteDefinition
from typace.vehicle.attitude import AttitudeState
from typace.vehicle.power import PowerMode
from typace.vehicle.resources import ResourceState


@dataclass(frozen=True, slots=True)
class VehicleState:
    attitude: AttitudeState
    resources: ResourceState
    power_mode: PowerMode
    requires_replan: bool = False

    @classmethod
    def from_definition(cls, definition: SatelliteDefinition) -> "VehicleState":
        return cls(
            AttitudeState(
                np.asarray((1.0, 0.0, 0.0, 0.0)),
                np.zeros(3),
                np.zeros(3),
            ),
            ResourceState(
                definition.dry_mass_kg,
                definition.main_propellant_mass_kg,
                definition.rcs_propellant_mass_kg,
                definition.power.initial_battery_energy_j,
            ),
            PowerMode.NOMINAL,
        )
