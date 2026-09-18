from dataclasses import replace
from math import sqrt
import unittest

import numpy as np

from typace.config.physics import (
    EARTH_GRAVITATIONAL_PARAMETER_M3_S2,
    MOON_GRAVITATIONAL_PARAMETER_M3_S2,
)
from typace.satellites import load_catalog
from typace.satellites.commands import (
    AvoidCollision,
    MaintainOrbit,
    Deorbit,
    SetApsides,
    SetInclination,
    SetOrbitAltitude,
)
from typace.satellites.state import ControlMode, SatelliteSnapshot
from typace.flight.models import FlightPlan, PlanningFailure, PlanningFailureCode
from typace.flight.navigation import NavigationState, update_navigation
from typace.flight.objectives import (
    FlightObjective,
    ObjectivePriority,
    select_objective,
)
from typace.flight.planning.avoidance import ConjunctionRisk, plan_avoidance
from typace.flight.planning.orbit import plan_orbit_command
from typace.flight.planning.transfer import TransferTarget, plan_transfer


class PlanningTests(unittest.TestCase):
    def setUp(self) -> None:
        catalog = load_catalog()
        source = catalog.satellite("gps-biir-2")
        self.definition = replace(
            source,
            dry_mass_kg=1_000.0,
            main_propellant_mass_kg=10_000.0,
            rcs_propellant_mass_kg=100.0,
            drag_area_m2=0.0,
            propulsion=replace(
                source.propulsion,
                main_thrust_n=1_000_000.0,
                main_specific_impulse_s=900.0,
            ),
            autonomy=replace(
                source.autonomy,
                minimum_main_propellant_reserve_kg=50.0,
            ),
        )

    def _navigation(self, primary_body_id: str = "earth"):
        if primary_body_id == "earth":
            radius_m = 7_000_000.0
            mu_m3_s2 = EARTH_GRAVITATIONAL_PARAMETER_M3_S2
        else:
            radius_m = 1_837_400.0
            mu_m3_s2 = MOON_GRAVITATIONAL_PARAMETER_M3_S2
        speed_m_s = sqrt(mu_m3_s2 / radius_m)
        snapshot = SatelliteSnapshot(
            "planner-test",
            "Planner Test",
            primary_body_id,
            (radius_m, 0.0, 0.0),
            (0.0, speed_m_s, 0.0),
            11_100.0,
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            10_000.0,
            100.0,
            1_000_000.0,
            ControlMode.AUTONOMOUS,
            0,
            False,
        )
        state = update_navigation(NavigationState(), snapshot)
        assert state.solution is not None
        return state.solution, snapshot

    def test_navigation_freezes_in_power_safe_and_rebuilds_after_recovery(self) -> None:
        navigation, snapshot = self._navigation()
        previous = NavigationState(navigation)
        frozen = update_navigation(
            previous, replace(snapshot, control_mode=ControlMode.POWER_SAFE)
        )
        rebuilt = update_navigation(
            frozen,
            replace(snapshot, position_m=(7_000_001.0, 0.0, 0.0)),
        )
        self.assertIs(frozen.solution, navigation)
        self.assertIsNot(rebuilt.solution, navigation)

    def test_objective_priority_is_fixed_and_stable(self) -> None:
        maintenance = FlightObjective(
            "maintenance", ObjectivePriority.ORBIT_MAINTENANCE, MaintainOrbit()
        )
        collision = FlightObjective(
            "collision", ObjectivePriority.COLLISION_AVOIDANCE, AvoidCollision("c-1")
        )
        self.assertEqual(
            select_objective((maintenance, collision)),
            collision,
        )

    def test_orbit_inclination_maintenance_and_resource_failure(self) -> None:
        navigation, _ = self._navigation()
        altitude_plan = plan_orbit_command(
            navigation,
            self.definition,
            SetOrbitAltitude(800_000.0),
        )
        inclination_plan = plan_orbit_command(
            navigation,
            self.definition,
            SetInclination(0.02),
        )
        apsides_plan = plan_orbit_command(
            navigation,
            self.definition,
            SetApsides(650_000.0, 900_000.0),
        )
        deorbit_plan = plan_orbit_command(
            navigation,
            self.definition,
            Deorbit(),
        )
        relaxed_definition = replace(
            self.definition,
            autonomy=replace(
                self.definition.autonomy,
                target_periapsis_altitude_m=600_000.0,
                target_apoapsis_altitude_m=700_000.0,
                altitude_tolerance_m=200_000.0,
            ),
        )
        maintenance = plan_orbit_command(
            navigation, relaxed_definition, MaintainOrbit()
        )
        unavailable = plan_orbit_command(
            navigation,
            replace(
                self.definition,
                autonomy=replace(
                    self.definition.autonomy,
                    minimum_main_propellant_reserve_kg=10_000.0,
                ),
            ),
            SetOrbitAltitude(800_000.0),
        )
        self.assertIsInstance(altitude_plan, FlightPlan)
        self.assertIsInstance(inclination_plan, FlightPlan)
        self.assertIsInstance(apsides_plan, FlightPlan)
        self.assertIsInstance(deorbit_plan, FlightPlan)
        self.assertIsInstance(maintenance, FlightPlan)
        assert isinstance(maintenance, FlightPlan)
        self.assertEqual(maintenance.steps, ())
        self.assertIsInstance(unavailable, PlanningFailure)
        assert isinstance(unavailable, PlanningFailure)
        self.assertEqual(unavailable.code, PlanningFailureCode.INSUFFICIENT_PROPELLANT)

    def test_bidirectional_transfer_and_avoidance_plans(self) -> None:
        earth_navigation, _ = self._navigation("earth")
        moon_definition = replace(self.definition, primary_body_id="moon")
        moon_navigation, _ = self._navigation("moon")
        moon_position_earth_m = np.asarray((384_400_000.0, 0.0, 0.0))
        moon_velocity_earth_m_s = np.asarray((0.0, 1_022.0, 0.0))
        outbound = plan_transfer(
            earth_navigation,
            self.definition,
            TransferTarget(
                "moon",
                "earth",
                np.zeros(3),
                np.zeros(3),
                moon_position_earth_m,
                moon_velocity_earth_m_s,
                np.asarray((0.0, 1_837_400.0, 0.0)),
                5.0 * 86_400.0,
            ),
        )
        inbound = plan_transfer(
            moon_navigation,
            moon_definition,
            TransferTarget(
                "earth",
                "earth",
                moon_position_earth_m,
                moon_velocity_earth_m_s,
                np.zeros(3),
                np.zeros(3),
                np.asarray((0.0, 7_000_000.0, 0.0)),
                5.0 * 86_400.0,
            ),
        )
        avoidance = plan_avoidance(
            earth_navigation,
            self.definition,
            ConjunctionRisk("risk-1", 1_000.0, 100.0, 1_000.0),
        )
        self.assertIsInstance(outbound, FlightPlan)
        self.assertIsInstance(inbound, FlightPlan)
        self.assertIsInstance(avoidance, FlightPlan)


if __name__ == "__main__":
    unittest.main()
