from dataclasses import replace
import unittest

from astropy.time import TimeDelta

from typace.config.simulation import (
    MAX_PHYSICS_SUBSTEPS_PER_WORLD_STEP,
    PHYSICS_STEP_SECONDS,
)
from typace.satellites import load_catalog
from typace.satellites.commands import (
    CommandStatus,
    ManualActuation,
    SetOrbitAltitude,
    TakeManualControl,
)
from typace.satellites.definition import KeplerianDefinition, SatelliteCatalog
from typace.satellites.state import ControlMode
from typace.simulation import SimulationWorld


class SimulationWorldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = load_catalog()

    def test_catalog_order_does_not_change_stable_batch_update(self) -> None:
        forward = SimulationWorld.from_catalog(self.catalog)
        reversed_catalog = SatelliteCatalog(
            self.catalog.scenario_epoch,
            tuple(reversed(self.catalog.satellites)),
        )
        reversed_world = SimulationWorld.from_catalog(reversed_catalog)

        forward_snapshot = forward.step(1.0)
        reversed_snapshot = reversed_world.step(1.0)

        self.assertEqual(forward_snapshot, reversed_snapshot)
        self.assertEqual(
            tuple(item.id for item in forward_snapshot.satellites),
            tuple(sorted(item.id for item in forward_snapshot.satellites)),
        )
        self.assertTrue(
            all(
                item.position_m != initial.position_m
                for item, initial in zip(
                    forward_snapshot.satellites,
                    SimulationWorld.from_catalog(self.catalog).snapshot().satellites,
                )
            )
        )

    def test_initial_epochs_are_normalized_deterministically(self) -> None:
        definition = self.catalog.satellite("gps-biir-2")
        orbit = definition.initial_orbit
        self.assertIsInstance(orbit, KeplerianDefinition)
        assert isinstance(orbit, KeplerianDefinition)
        earlier_orbit = replace(
            orbit,
            epoch=orbit.epoch - TimeDelta(60.0, format="sec"),
        )
        shifted_definition = replace(definition, initial_orbit=earlier_orbit)
        shifted_catalog = SatelliteCatalog(
            self.catalog.scenario_epoch,
            (shifted_definition,),
        )

        first = SimulationWorld.from_catalog(shifted_catalog).snapshot()
        second = SimulationWorld.from_catalog(shifted_catalog).snapshot()
        earlier_scenario = self.catalog.scenario_epoch - TimeDelta(60.0, format="sec")
        backward = SimulationWorld.from_catalog(
            SatelliteCatalog(self.catalog.scenario_epoch, (definition,)),
            earlier_scenario,
        ).snapshot()

        self.assertEqual(first, second)
        self.assertEqual(first.scenario_epoch_utc, second.scenario_epoch_utc)
        self.assertEqual(len(backward.satellites), 1)

    def test_submit_returns_named_results_and_manual_request_is_direct(self) -> None:
        world = SimulationWorld.from_catalog(self.catalog)
        self.assertEqual(
            world.submit("missing", SetOrbitAltitude(100_000.0)).status,
            CommandStatus.UNKNOWN_SATELLITE,
        )
        self.assertEqual(
            world.submit("iss", SetOrbitAltitude(-1.0)).status,
            CommandStatus.INVALID,
        )
        self.assertTrue(world.submit("iss", TakeManualControl()).accepted)
        first_request = ManualActuation(0.1, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        second_request = ManualActuation(0.2, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        self.assertTrue(world.submit("iss", first_request).accepted)
        self.assertTrue(world.submit("iss", second_request).accepted)
        before = world.snapshot().satellite("iss")
        self.assertEqual(before.control_mode, ControlMode.MANUAL)
        self.assertEqual(before.pending_command_count, 1)

        after = world.step(1.0).satellite("iss")

        self.assertLess(after.main_propellant_kg, before.main_propellant_kg)
        self.assertEqual(after.pending_command_count, 1)

    def test_time_warp_reports_effective_limit(self) -> None:
        world = SimulationWorld.from_catalog(self.catalog)
        world.set_time_warp(10_000.0)

        snapshot = world.step(1.0)

        expected_advance_s = PHYSICS_STEP_SECONDS * MAX_PHYSICS_SUBSTEPS_PER_WORLD_STEP
        self.assertEqual(snapshot.elapsed_seconds, expected_advance_s)
        self.assertEqual(snapshot.effective_time_warp, expected_advance_s)
        self.assertEqual(snapshot.time_warp_limit_reason, "physics_step")


if __name__ == "__main__":
    unittest.main()
