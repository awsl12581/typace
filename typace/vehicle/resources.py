"""Propellant and battery accounting."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResourceState:
    dry_mass_kg: float
    main_propellant_kg: float
    rcs_propellant_kg: float
    battery_energy_j: float

    @property
    def total_mass_kg(self) -> float:
        return self.dry_mass_kg + self.main_propellant_kg + self.rcs_propellant_kg


def update_battery(
    resources: ResourceState, net_power_w: float, duration_s: float, capacity_j: float
) -> ResourceState:
    if duration_s < 0.0 or capacity_j <= 0.0:
        raise ValueError("duration must be non-negative and capacity positive")
    return ResourceState(
        resources.dry_mass_kg,
        resources.main_propellant_kg,
        resources.rcs_propellant_kg,
        min(
            capacity_j, max(0.0, resources.battery_energy_j + net_power_w * duration_s)
        ),
    )
