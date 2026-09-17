from dataclasses import replace
from math import cos, log, pi, sin
import unittest

import numpy as np

from typace.config.physics import EARTH_MEAN_RADIUS_M, STANDARD_GRAVITY_M_S2
from typace.config.vehicle import (
    POWER_SAFE_ENTER_FRACTION,
    POWER_SAFE_RECOVER_FRACTION,
)
from typace.satellites import load_catalog
from typace.physics.forces import ForceModel
from typace.physics.propagation import TranslationalState
from typace.vehicle.actuators import (
    ActuatorCommand,
    limit_rcs_torque,
    limit_reaction_wheel_torque,
    momentum_unload_command,
)
from typace.vehicle.attitude import (
    AttitudeState,
    integrate_attitude,
    rotate_body_to_inertial,
)
from typace.vehicle.dynamics import PowerEnvironment, advance_vehicle, resolve_actuators
from typace.vehicle.power import PowerMode, power_step, solar_power_w
from typace.vehicle.resources import ResourceState
from typace.vehicle.state import VehicleState
from typace.physics.environment import OccludingBody


class VehicleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.satellite = load_catalog().satellite("iss")

    def test_attitude_normalizes_and_wheel_saturation_blocks_outward_torque(
        self,
    ) -> None:
        attitude = AttitudeState(
            np.asarray((1.0, 0.0, 0.0, 0.0)),
            np.asarray((0.0, 0.0, 0.1)),
            np.asarray((20_000.0, 0.0, 0.0)),
        )
        updated = integrate_attitude(
            attitude,
            np.asarray((100.0, 0.0, 0.0)),
            np.zeros(3),
            np.asarray(self.satellite.inertia_diagonal_kg_m2),
            0.1,
        )
        self.assertAlmostEqual(float(np.linalg.norm(updated.quaternion_wxyz)), 1.0)
        limited = limit_reaction_wheel_torque(
            np.asarray((100.0, 1.0, 1.0)),
            attitude.wheel_momentum_n_m_s,
            np.asarray((4_000.0, 4_000.0, 4_000.0)),
            np.asarray((20_000.0, 20_000.0, 20_000.0)),
            0.1,
        )
        self.assertGreater(limited[0], 0.0)
        blocked = limit_reaction_wheel_torque(
            np.asarray((-100.0, 0.0, 0.0)),
            attitude.wheel_momentum_n_m_s,
            np.asarray((4_000.0, 4_000.0, 4_000.0)),
            np.asarray((20_000.0, 20_000.0, 20_000.0)),
            0.1,
        )
        self.assertEqual(blocked[0], 0.0)
        self.assertLessEqual(
            float(
                np.linalg.norm(limit_rcs_torque(np.ones(3), 0.5)),
            ),
            0.5,
        )

    def test_finite_burn_matches_rocket_equation_and_uses_body_axis(self) -> None:
        propulsion = replace(
            self.satellite.propulsion,
            main_thrust_n=1_000.0,
            main_specific_impulse_s=300.0,
            main_body_axis=(1.0, 0.0, 0.0),
        )
        definition = replace(
            self.satellite,
            dry_mass_kg=100.0,
            main_propellant_mass_kg=10.0,
            rcs_propellant_mass_kg=1.0,
            drag_area_m2=0.0,
            propulsion=propulsion,
        )
        vehicle = VehicleState.from_definition(definition)
        translation = TranslationalState(
            np.asarray((1.0, 0.0, 0.0)),
            np.zeros(3),
            vehicle.resources.total_mass_kg,
        )
        duration_s = 10.0
        result = advance_vehicle(
            definition,
            translation,
            vehicle,
            ForceModel("test", 0.0, 0.0),
            ActuatorCommand(1.0, np.zeros(3), np.zeros(3)),
            PowerEnvironment(np.asarray((150_000_000_000.0, 0.0, 0.0)), ()),
            duration_s,
            0.1,
        )
        expected_delta_v_m_s = (
            propulsion.main_specific_impulse_s
            * STANDARD_GRAVITY_M_S2
            * log(translation.mass_kg / result.translation.mass_kg)
        )
        self.assertAlmostEqual(
            result.translation.velocity_m_s[0], expected_delta_v_m_s, delta=1.0e-6
        )
        self.assertAlmostEqual(result.translation.velocity_m_s[1], 0.0)
        quarter_turn = np.asarray((cos(pi / 4.0), 0.0, 0.0, sin(pi / 4.0)))
        rotated_axis = rotate_body_to_inertial(
            quarter_turn, np.asarray((1.0, 0.0, 0.0))
        )
        self.assertAlmostEqual(
            float(np.dot(rotated_axis, np.asarray((1.0, 0.0, 0.0)))), 0.0, delta=1.0e-12
        )

    def test_main_exhaustion_leaves_rcs_available_and_power_safe_stops_actuators(
        self,
    ) -> None:
        definition = replace(
            self.satellite,
            main_propellant_mass_kg=0.01,
            rcs_propellant_mass_kg=1.0,
        )
        vehicle = VehicleState.from_definition(definition)
        command = ActuatorCommand(
            1.0,
            np.zeros(3),
            np.asarray((definition.propulsion.rcs_maximum_torque_n_m, 0.0, 0.0)),
        )
        output = resolve_actuators(definition, vehicle, command, 10.0)
        self.assertLess(output.main_active_duration_s, 10.0)
        self.assertGreater(output.rcs_active_duration_s, output.main_active_duration_s)
        safe_vehicle = replace(vehicle, power_mode=PowerMode.POWER_SAFE)
        safe_output = resolve_actuators(definition, safe_vehicle, command, 10.0)
        self.assertEqual(safe_output.main_active_duration_s, 0.0)
        self.assertEqual(safe_output.rcs_active_duration_s, 0.0)

        translation = TranslationalState(
            np.asarray((1.0, 0.0, 0.0)),
            np.zeros(3),
            vehicle.resources.total_mass_kg,
        )
        result = advance_vehicle(
            definition,
            translation,
            vehicle,
            ForceModel("test", 0.0, 0.0),
            command,
            PowerEnvironment(np.asarray((150_000_000_000.0, 0.0, 0.0)), ()),
            10.0,
            1.0,
        )
        self.assertEqual(result.vehicle.resources.main_propellant_kg, 0.0)
        self.assertLess(
            result.vehicle.resources.rcs_propellant_kg,
            vehicle.resources.rcs_propellant_kg,
        )
        self.assertAlmostEqual(
            result.translation.mass_kg, result.vehicle.resources.total_mass_kg
        )

    def test_rcs_unloads_reaction_wheel_momentum(self) -> None:
        momentum = np.asarray((5.0, 0.0, 0.0))
        command = momentum_unload_command(
            momentum,
            np.asarray(self.satellite.reaction_wheel_maximum_torque_n_m),
            self.satellite.propulsion.rcs_maximum_torque_n_m,
            2.0,
        )
        attitude = AttitudeState(
            np.asarray((1.0, 0.0, 0.0, 0.0)), np.zeros(3), momentum
        )
        updated = integrate_attitude(
            attitude,
            command.wheel_torque_n_m,
            command.rcs_torque_n_m,
            np.asarray(self.satellite.inertia_diagonal_kg_m2),
            2.0,
        )
        self.assertLess(
            float(np.linalg.norm(updated.wheel_momentum_n_m_s)),
            float(np.linalg.norm(momentum)),
        )
        self.assertAlmostEqual(
            float(np.linalg.norm(updated.angular_velocity_rad_s)), 0.0
        )

    def test_power_safe_recovery_requires_replan(self) -> None:
        vehicle = VehicleState.from_definition(self.satellite)
        recovered_energy_j = self.satellite.power.battery_capacity_j * (
            POWER_SAFE_RECOVER_FRACTION * 1.1
        )
        safe_vehicle = replace(
            vehicle,
            resources=replace(vehicle.resources, battery_energy_j=recovered_energy_j),
            power_mode=PowerMode.POWER_SAFE,
        )
        translation = TranslationalState(
            np.asarray((EARTH_MEAN_RADIUS_M + 400_000.0, 0.0, 0.0)),
            np.asarray((0.0, 7_600.0, 0.0)),
            safe_vehicle.resources.total_mass_kg,
        )
        result = advance_vehicle(
            self.satellite,
            translation,
            safe_vehicle,
            ForceModel("test", 0.0, EARTH_MEAN_RADIUS_M),
            ActuatorCommand(1.0, np.ones(3), np.ones(3)),
            PowerEnvironment(np.asarray((150_000_000_000.0, 0.0, 0.0)), ()),
            1.0,
            1.0,
        )
        self.assertEqual(result.vehicle.power_mode, PowerMode.NOMINAL)
        self.assertTrue(result.vehicle.requires_replan)
        self.assertEqual(
            result.vehicle.resources.main_propellant_kg,
            safe_vehicle.resources.main_propellant_kg,
        )

    def test_solar_power_and_power_safe_hysteresis(self) -> None:
        power = self.satellite.power
        satellite_position = np.asarray((EARTH_MEAN_RADIUS_M + 400_000.0, 0.0, 0.0))
        sun_position = np.asarray((-150_000_000_000.0, 0.0, 0.0))
        panel_normal = np.asarray((-1.0, 0.0, 0.0))
        generated = solar_power_w(
            satellite_position,
            sun_position,
            panel_normal,
            power,
            (),
        )
        eclipsed = solar_power_w(
            satellite_position,
            sun_position,
            panel_normal,
            power,
            (OccludingBody(np.zeros(3), EARTH_MEAN_RADIUS_M),),
        )
        self.assertGreater(generated, 0.0)
        self.assertEqual(eclipsed, 0.0)
        starting_fraction = POWER_SAFE_ENTER_FRACTION * 1.2
        low = ResourceState(1.0, 0.0, 0.0, power.battery_capacity_j * starting_fraction)
        nominal_load_w = (
            power.base_power_w
            + power.computer_power_w
            + power.sensor_power_w
            + power.actuator_power_w
        )
        drain_duration_s = (
            power.battery_capacity_j
            * (starting_fraction - POWER_SAFE_ENTER_FRACTION / 2.0)
            / nominal_load_w
        )
        depleted, mode = power_step(
            low,
            0.0,
            power,
            drain_duration_s,
            PowerMode.NOMINAL,
            actuator_enabled=True,
        )
        self.assertEqual(mode, PowerMode.POWER_SAFE)
        recovered, recovered_mode = power_step(
            depleted,
            nominal_load_w,
            power,
            power.battery_capacity_j
            * POWER_SAFE_RECOVER_FRACTION
            / max(nominal_load_w - power.base_power_w, 1.0),
            mode,
            actuator_enabled=True,
        )
        self.assertEqual(recovered_mode, PowerMode.NOMINAL)
        self.assertGreaterEqual(recovered.battery_energy_j, depleted.battery_energy_j)


if __name__ == "__main__":
    unittest.main()
