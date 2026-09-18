from dataclasses import replace
from time import perf_counter
import unittest

from astropy.time import TimeDelta
import numpy as np

from typace.config.simulation import (
    MAX_PHYSICS_SUBSTEPS_PER_WORLD_STEP,
    MAX_SATELLITES_WITH_PERFORMANCE_GUARANTEE,
    PHYSICS_STEP_SECONDS,
    UI_REFRESH_HZ,
    UPDATE_P95_BUDGET_SECONDS,
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
from typace.simulation.events import DestructionCause


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

    def test_non_finite_satellite_isolated_without_stopping_world(self) -> None:
        world = SimulationWorld.from_catalog(self.catalog)
        failed = world._satellites["iss"]
        world._satellites["iss"] = replace(
            failed,
            translation=replace(
                failed.translation,
                position_m=np.asarray((float("nan"), 0.0, 0.0)),
            ),
        )

        snapshot = world.step(1.0 / UI_REFRESH_HZ)

        self.assertEqual(len(snapshot.satellites), len(self.catalog.satellites) - 1)
        self.assertNotIn("iss", tuple(item.id for item in snapshot.satellites))
        self.assertEqual(
            snapshot.destruction_events[-1].cause,
            DestructionCause.NUMERICAL_FAILURE,
        )

    def test_sixty_four_satellite_update_p95_meets_budget(self) -> None:
        definitions = []
        catalog_size = len(self.catalog.satellites)
        satellite_count = MAX_SATELLITES_WITH_PERFORMANCE_GUARANTEE
        for index in range(satellite_count):
            source = self.catalog.satellites[index % catalog_size]
            orbit = source.initial_orbit
            self.assertIsInstance(orbit, KeplerianDefinition)
            assert isinstance(orbit, KeplerianDefinition)
            definitions.append(
                replace(
                    source,
                    id=f"performance-{index:02d}",
                    display_name=f"Performance {index:02d}",
                    initial_orbit=replace(
                        orbit,
                        mean_anomaly_deg=(
                            orbit.mean_anomaly_deg + index * 360.0 / satellite_count
                        )
                        % 360.0,
                    ),
                )
            )
        world = SimulationWorld.from_catalog(
            SatelliteCatalog(self.catalog.scenario_epoch, tuple(definitions))
        )
        frame_duration_s = 1.0 / UI_REFRESH_HZ
        for _ in range(10):
            world.step(frame_duration_s)

        durations_s = []
        for _ in range(100):
            started = perf_counter()
            world.step(frame_duration_s)
            durations_s.append(perf_counter() - started)

        p95_s = float(np.percentile(durations_s, 95))
        self.assertEqual(len(world.snapshot().satellites), satellite_count)
        self.assertLessEqual(p95_s, UPDATE_P95_BUDGET_SECONDS)


if __name__ == "__main__":
    unittest.main()
