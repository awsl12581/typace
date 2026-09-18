from dataclasses import replace
import unittest

import numpy as np

from typace.config.physics import EARTH_MEAN_RADIUS_M
from typace.satellites import load_catalog
from typace.satellites.commands import ManualActuation, SetOrbitAltitude
from typace.satellites.state import ControlMode, SatelliteSnapshot
from typace.vehicle.actuators import ActuatorCommand
from typace.vehicle.attitude import AttitudeState
from typace.vehicle.power import PowerMode
from typace.vehicle.state import VehicleState
from typace.flight.attitude_control import AttitudeControllerState, control_attitude
from typace.flight.autopilot import AutopilotState, autopilot_step
from typace.flight.execution import ExecutionState, ExecutionStatus, execute_plan
from typace.flight.guidance import guidance_for_plan
from typace.flight.manual import manual_control
from typace.flight.models import BurnStep, FlightPlan
from typace.flight.objectives import FlightObjective, ObjectivePriority
from typace.flight.safety import SafetyReason, SafetyTelemetry


class ControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.definition = load_catalog().satellite("gps-biir-2")
        self.vehicle = VehicleState.from_definition(self.definition)
        self.plan = FlightPlan(
            "test-burn",
            (BurnStep(0.0, 10.0, 1.0, (0.0, 100.0, 0.0), "raise"),),
            "burn complete",
            1.0,
            "earth",
        )

    def _snapshot(self, mode: ControlMode) -> SatelliteSnapshot:
        quaternion = self.vehicle.attitude.quaternion_wxyz
        return SatelliteSnapshot(
            self.definition.id,
            self.definition.display_name,
            "earth",
            (EARTH_MEAN_RADIUS_M + 20_180_000.0, 0.0, 0.0),
            (0.0, 3_874.0, 0.0),
            self.vehicle.resources.total_mass_kg,
            (
                float(quaternion[0]),
                float(quaternion[1]),
                float(quaternion[2]),
                float(quaternion[3]),
            ),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            self.vehicle.resources.main_propellant_kg,
            self.vehicle.resources.rcs_propellant_kg,
            self.vehicle.resources.battery_energy_j,
            mode,
            0,
            False,
        )

    def test_guidance_turns_before_thrust_and_execution_burns_when_aligned(
        self,
    ) -> None:
        guidance = guidance_for_plan(self.plan, 0.0, self.definition)
        self.assertIsNotNone(guidance)
        assert guidance is not None
        controller = AttitudeControllerState(np.zeros(3))
        turning = execute_plan(
            ExecutionState(self.plan),
            controller,
            self.vehicle,
            self.definition,
            SafetyTelemetry(),
            0.1,
        )
        self.assertEqual(turning.execution.status, ExecutionStatus.TURNING)
        self.assertEqual(turning.decision.command.main_throttle, 0.0)

        aligned_vehicle = replace(
            self.vehicle,
            attitude=replace(
                self.vehicle.attitude,
                quaternion_wxyz=guidance.target_quaternion_wxyz,
            ),
        )
        burning = execute_plan(
            ExecutionState(self.plan),
            controller,
            aligned_vehicle,
            self.definition,
            SafetyTelemetry(),
            0.1,
        )
        self.assertEqual(burning.execution.status, ExecutionStatus.BURNING)
        self.assertEqual(burning.decision.command.main_throttle, 1.0)

    def test_attitude_controller_saturates_without_integral_windup(self) -> None:
        controller = AttitudeControllerState(np.zeros(3))
        target = np.asarray((0.0, 1.0, 0.0, 0.0))
        maximum = np.asarray((0.01, 0.01, 0.01))
        output = control_attitude(
            controller,
            self.vehicle.attitude,
            target,
            maximum,
            1.0,
        )
        self.assertTrue(np.all(np.abs(output.wheel_torque_n_m) <= maximum))
        np.testing.assert_allclose(
            output.state.integral_error_rad_s,
            controller.integral_error_rad_s,
        )

    def test_manual_and_autonomous_requests_share_safety_filter(self) -> None:
        unsafe = SafetyTelemetry(
            dynamic_pressure_pa=self.definition.maximum_dynamic_pressure_pa
        )
        manual = manual_control(
            ManualActuation(1.0, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            self.vehicle,
            self.definition,
            unsafe,
        )
        self.assertEqual(manual.reason, SafetyReason.DYNAMIC_PRESSURE)
        self.assertEqual(manual.command.main_throttle, 0.0)

        safe_vehicle = replace(self.vehicle, power_mode=PowerMode.POWER_SAFE)
        autonomous = execute_plan(
            ExecutionState(self.plan),
            AttitudeControllerState(np.zeros(3)),
            safe_vehicle,
            self.definition,
            SafetyTelemetry(),
            0.1,
        )
        self.assertEqual(autonomous.execution.status, ExecutionStatus.ABORTED)
        self.assertEqual(autonomous.decision.reason, SafetyReason.POWER_SAFE)

    def test_autopilot_never_overrides_manual_and_accepts_preplanned_flight(
        self,
    ) -> None:
        objective = FlightObjective(
            "raise",
            ObjectivePriority.USER_COMMAND,
            SetOrbitAltitude(20_300_000.0),
        )
        manual = autopilot_step(
            AutopilotState.idle(),
            self._snapshot(ControlMode.MANUAL),
            self.vehicle,
            self.definition,
            SafetyTelemetry(),
            0.1,
            objective=objective,
            planned_result=self.plan,
        )
        self.assertEqual(manual.decision.command.main_throttle, 0.0)
        np.testing.assert_allclose(manual.decision.command.wheel_torque_n_m, 0.0)
        np.testing.assert_allclose(manual.decision.command.rcs_torque_n_m, 0.0)
        autonomous = autopilot_step(
            AutopilotState.idle(),
            self._snapshot(ControlMode.AUTONOMOUS),
            self.vehicle,
            self.definition,
            SafetyTelemetry(),
            0.1,
            objective=objective,
            planned_result=self.plan,
        )
        self.assertIsNotNone(autonomous.state.execution)

    def test_primary_transition_refreshes_navigation_without_dropping_plan(
        self,
    ) -> None:
        objective = FlightObjective(
            "transfer",
            ObjectivePriority.USER_COMMAND,
            SetOrbitAltitude(20_300_000.0),
        )
        started = autopilot_step(
            AutopilotState.idle(),
            self._snapshot(ControlMode.AUTONOMOUS),
            self.vehicle,
            self.definition,
            SafetyTelemetry(),
            0.1,
            objective=objective,
            planned_result=self.plan,
        )
        lunar_snapshot = replace(
            self._snapshot(ControlMode.AUTONOMOUS),
            primary_body_id="moon",
            position_m=(1_837_400.0, 0.0, 0.0),
            velocity_m_s=(0.0, 1_600.0, 0.0),
        )

        transitioned = autopilot_step(
            started.state,
            lunar_snapshot,
            self.vehicle,
            self.definition,
            SafetyTelemetry(),
            0.1,
            objective=objective,
        )

        self.assertIsNotNone(transitioned.state.navigation.solution)
        assert transitioned.state.navigation.solution is not None
        self.assertEqual(
            transitioned.state.navigation.solution.primary_body_id,
            "moon",
        )
        self.assertIs(transitioned.state.plan, started.state.plan)
        self.assertIsNotNone(transitioned.state.execution)


if __name__ == "__main__":
    unittest.main()
