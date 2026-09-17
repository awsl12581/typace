"""Direct player input translated to the shared actuator request."""

import numpy as np

from typace.satellites.commands import ManualActuation
from typace.satellites.definition import SatelliteDefinition
from typace.vehicle.actuators import ActuatorCommand
from typace.vehicle.state import VehicleState
from typace.flight.safety import SafetyDecision, SafetyTelemetry, filter_actuators


def manual_control(
    command: ManualActuation,
    vehicle: VehicleState,
    definition: SatelliteDefinition,
    telemetry: SafetyTelemetry,
) -> SafetyDecision:
    request = ActuatorCommand(
        command.main_throttle,
        np.asarray(command.wheel_torque_n_m),
        np.asarray(command.rcs_torque_n_m),
    )
    return filter_actuators(request, vehicle, definition, telemetry)
