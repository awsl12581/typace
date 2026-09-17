"""Immutable definitions loaded from satellite JSON catalogs."""

from dataclasses import dataclass
from enum import StrEnum

from astropy.time import Time

type Vector3 = tuple[float, float, float]


class ProvenanceKind(StrEnum):
    OBSERVED = "observed"
    DERIVED = "derived"
    ASSUMED = "assumed"


@dataclass(frozen=True, slots=True)
class SourceRecord:
    id: str
    kind: ProvenanceKind
    organization: str
    document_title: str
    published_date: str
    url: str | None


@dataclass(frozen=True, slots=True)
class StateVectorDefinition:
    epoch: Time
    frame: str
    position_m: Vector3
    velocity_m_s: Vector3


@dataclass(frozen=True, slots=True)
class KeplerianDefinition:
    epoch: Time
    frame: str
    semi_major_axis_m: float
    eccentricity: float
    inclination_deg: float
    ascending_node_deg: float
    periapsis_argument_deg: float
    mean_anomaly_deg: float


type InitialOrbit = StateVectorDefinition | KeplerianDefinition


@dataclass(frozen=True, slots=True)
class PropulsionDefinition:
    main_thrust_n: float
    main_specific_impulse_s: float
    main_minimum_throttle: float
    main_body_axis: Vector3
    rcs_thrust_n: float
    rcs_specific_impulse_s: float
    rcs_maximum_torque_n_m: float


@dataclass(frozen=True, slots=True)
class PowerDefinition:
    panel_area_m2: float
    panel_efficiency: float
    panel_body_normal: Vector3
    battery_capacity_j: float
    initial_battery_energy_j: float
    base_power_w: float
    computer_power_w: float
    sensor_power_w: float
    actuator_power_w: float


@dataclass(frozen=True, slots=True)
class AutonomyDefinition:
    target_periapsis_altitude_m: float
    target_apoapsis_altitude_m: float
    target_inclination_deg: float
    altitude_tolerance_m: float
    minimum_main_propellant_reserve_kg: float
    avoidance_distance_m: float
    allowed_commands: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SatelliteDefinition:
    id: str
    display_name: str
    operator: str
    primary_body_id: str
    initial_orbit: InitialOrbit
    dry_mass_kg: float
    main_propellant_mass_kg: float
    rcs_propellant_mass_kg: float
    inertia_diagonal_kg_m2: Vector3
    collision_radius_m: float
    drag_area_m2: float
    drag_coefficient: float
    nose_radius_m: float
    propulsion: PropulsionDefinition
    reaction_wheel_maximum_torque_n_m: Vector3
    reaction_wheel_maximum_momentum_n_m_s: Vector3
    power: PowerDefinition
    maximum_dynamic_pressure_pa: float
    maximum_heat_flux_w_m2: float
    autonomy: AutonomyDefinition
    sources: tuple[SourceRecord, ...]
    field_provenance: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class SatelliteCatalog:
    scenario_epoch: Time
    satellites: tuple[SatelliteDefinition, ...]

    def satellite(self, satellite_id: str) -> SatelliteDefinition:
        for satellite in self.satellites:
            if satellite.id == satellite_id:
                return satellite
        raise KeyError(satellite_id)
