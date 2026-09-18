"""Runtime satellite state and immutable snapshot types."""

from dataclasses import dataclass
from enum import StrEnum

from typace.physics.propagation import TranslationalState
from typace.satellites.commands import SatelliteCommand
from typace.satellites.definition import SatelliteDefinition
from typace.vehicle.state import VehicleState

type Vector3 = tuple[float, float, float]
type Quaternion = tuple[float, float, float, float]


class ControlMode(StrEnum):
    AUTONOMOUS = "autonomous"
    MANUAL = "manual"
    POWER_SAFE = "power_safe"


@dataclass(frozen=True, slots=True)
class SatelliteState:
    definition: SatelliteDefinition
    primary_body_id: str
    translation: TranslationalState
    vehicle: VehicleState
    control_mode: ControlMode
    pending_commands: tuple[SatelliteCommand, ...] = ()


@dataclass(frozen=True, slots=True)
class SatelliteSnapshot:
    id: str
    display_name: str
    primary_body_id: str
    position_m: Vector3
    velocity_m_s: Vector3
    mass_kg: float
    attitude_quaternion_wxyz: Quaternion
    angular_velocity_rad_s: Vector3
    wheel_momentum_n_m_s: Vector3
    main_propellant_kg: float
    rcs_propellant_kg: float
    battery_energy_j: float
    control_mode: ControlMode
    pending_command_count: int
    requires_replan: bool
    plan_objective_id: str | None = None
    execution_status: str | None = None
    planning_failure: str | None = None
    safety_reason: str | None = None
    conjunction_alert_ids: tuple[str, ...] = ()
    periapsis_altitude_m: float | None = None
    apoapsis_altitude_m: float | None = None
    inclination_deg: float | None = None
    orbital_period_s: float | None = None
