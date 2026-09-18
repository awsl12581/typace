from concurrent.futures import Future
from dataclasses import replace
from time import perf_counter
import unittest
from unittest.mock import patch

from astropy.time import TimeDelta
import numpy as np

from typace.config.simulation import (
    MAX_PHYSICS_SUBSTEPS_PER_WORLD_STEP,
    MAX_SATELLITES_WITH_PERFORMANCE_GUARANTEE,
    PHYSICS_STEP_SECONDS,
    UI_REFRESH_HZ,
    UPDATE_P95_BUDGET_SECONDS,
)
from typace.flight.models import PlanningFailure, PlanningFailureCode, PlanningResult
from typace.satellites import load_catalog
from typace.satellites.commands import (
    CommandStatus,
    ManualActuation,
    ReturnAutonomousControl,
    ReturnStableOrbit,
    SetOrbitAltitude,
    TakeManualControl,
    TransferPrimary,
)
from typace.satellites.definition import (
    KeplerianDefinition,
    SatelliteCatalog,
    StateVectorDefinition,
)
from typace.satellites.state import ControlMode
from typace.simulation import SimulationWorld
from typace.simulation.ephemeris import ephemeris_at
from typace.simulation.events import DestructionCause
from typace.simulation.world import _transition_primary
from typace.tasks import ThreadTaskManager
from typace.vehicle.power import PowerMode


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

    def test_high_level_command_is_consumed_once_and_exposes_plan(self) -> None:
        world = SimulationWorld.from_catalog(self.catalog)
        before = world.snapshot().satellite("iss")

        self.assertTrue(world.submit("iss", SetOrbitAltitude(450_000.0)).accepted)
        planned = world.step(1.0 / UI_REFRESH_HZ).satellite("iss")
        continued = world.step(1.0 / UI_REFRESH_HZ).satellite("iss")

        self.assertEqual(planned.pending_command_count, 0)
        self.assertIsNotNone(planned.plan_objective_id)
        self.assertIsNotNone(planned.execution_status)
        self.assertEqual(continued.pending_command_count, 0)
        self.assertEqual(continued.plan_objective_id, planned.plan_objective_id)
        self.assertLessEqual(continued.main_propellant_kg, before.main_propellant_kg)

    def test_interplanetary_command_uses_catalog_transfer_center(self) -> None:
        source = self.catalog.satellite("gps-biir-2")
        definition = replace(
            source,
            id="interplanetary",
            dry_mass_kg=1_000.0,
            main_propellant_mass_kg=1.0e9,
            drag_area_m2=0.0,
            propulsion=replace(
                source.propulsion,
                main_thrust_n=1.0e12,
                main_specific_impulse_s=10_000.0,
            ),
            autonomy=replace(
                source.autonomy,
                minimum_main_propellant_reserve_kg=0.0,
            ),
        )
        world = SimulationWorld.from_catalog(
            SatelliteCatalog(self.catalog.scenario_epoch, (definition,))
        )

        self.assertTrue(
            world.submit("interplanetary", TransferPrimary("mars")).accepted
        )
        snapshot = world.step(1.0 / UI_REFRESH_HZ).satellite("interplanetary")

        self.assertEqual(snapshot.pending_command_count, 0)
        self.assertEqual(snapshot.plan_objective_id, "transfer_to_mars")
        self.assertIsNone(snapshot.planning_failure)

    def test_root_primary_return_reports_unavailable_parent(self) -> None:
        source = self.catalog.satellite("gps-biir-2")
        definition = replace(
            source,
            id="solar-orbiter",
            primary_body_id="sun",
            initial_orbit=StateVectorDefinition(
                self.catalog.scenario_epoch,
                "sun_j2000",
                (1_000_000_000.0, 0.0, 0.0),
                (0.0, 360_000.0, 0.0),
            ),
            drag_area_m2=0.0,
        )
        world = SimulationWorld.from_catalog(
            SatelliteCatalog(self.catalog.scenario_epoch, (definition,))
        )

        self.assertTrue(world.submit("solar-orbiter", ReturnStableOrbit()).accepted)
        snapshot = world.step(1.0 / UI_REFRESH_HZ).satellite("solar-orbiter")

        self.assertEqual(snapshot.pending_command_count, 0)
        self.assertIsNone(snapshot.plan_objective_id)
        self.assertEqual(snapshot.planning_failure, "transfer target is unavailable")

    def test_soi_transition_preserves_system_inertial_state(self) -> None:
        world = SimulationWorld.from_catalog(self.catalog)
        ephemeris = ephemeris_at(world.scenario_epoch)
        state = world._satellites["iss"]
        earth_soi_m = ephemeris.sphere_of_influence_radius_m("earth")
        escaping = replace(
            state,
            translation=replace(
                state.translation,
                position_m=np.asarray((earth_soi_m * 1.01, 0.0, 0.0)),
            ),
        )
        before_position = (
            ephemeris.body("earth").position_m + escaping.translation.position_m
        )
        before_velocity = (
            ephemeris.body("earth").velocity_m_s + escaping.translation.velocity_m_s
        )

        transitioned = _transition_primary(escaping, ephemeris)

        self.assertEqual(transitioned.primary_body_id, "sun")
        np.testing.assert_allclose(
            ephemeris.body("sun").position_m + transitioned.translation.position_m,
            before_position,
        )
        np.testing.assert_allclose(
            ephemeris.body("sun").velocity_m_s + transitioned.translation.velocity_m_s,
            before_velocity,
        )

    def test_conjunction_alerts_select_one_stable_yielding_satellite(self) -> None:
        source = self.catalog.satellite("gps-biir-2")

        def definition(satellite_id: str, offset_m: float, radial_speed_m_s: float):
            return replace(
                source,
                id=satellite_id,
                initial_orbit=StateVectorDefinition(
                    self.catalog.scenario_epoch,
                    "earth_j2000",
                    (7_000_000.0 + offset_m, 0.0, 0.0),
                    (radial_speed_m_s, 7_500.0, 0.0),
                ),
                dry_mass_kg=1_000.0,
                main_propellant_mass_kg=10_000.0,
                drag_area_m2=0.0,
                propulsion=replace(
                    source.propulsion,
                    main_thrust_n=1_000_000.0,
                    main_specific_impulse_s=900.0,
                ),
                autonomy=replace(
                    source.autonomy,
                    minimum_main_propellant_reserve_kg=0.0,
                    avoidance_distance_m=1_000_000.0,
                ),
            )

        world = SimulationWorld.from_catalog(
            SatelliteCatalog(
                self.catalog.scenario_epoch,
                (definition("a", 0.0, 0.0), definition("b", 20_000.0, -20.0)),
            )
        )

        snapshot = world.step(1.0 / UI_REFRESH_HZ)

        first = snapshot.satellite("a")
        second = snapshot.satellite("b")
        self.assertEqual(first.conjunction_alert_ids, ("a:b",))
        self.assertEqual(second.conjunction_alert_ids, ("a:b",))
        self.assertNotEqual(first.plan_objective_id, "avoid_a:b")
        self.assertEqual(second.plan_objective_id, "avoid_a:b")

        insufficient = definition("b", 20_000.0, -20.0)
        insufficient = replace(
            insufficient,
            autonomy=replace(
                insufficient.autonomy,
                minimum_main_propellant_reserve_kg=(
                    insufficient.main_propellant_mass_kg
                ),
            ),
        )
        constrained_world = SimulationWorld.from_catalog(
            SatelliteCatalog(
                self.catalog.scenario_epoch,
                (definition("a", 0.0, 0.0), insufficient),
            )
        )

        constrained = constrained_world.step(1.0 / UI_REFRESH_HZ).satellite("b")

        self.assertEqual(constrained.conjunction_alert_ids, ("a:b",))
        self.assertIn("reserve", constrained.planning_failure or "")

    def test_manual_actuation_uses_main_wheel_and_rcs_resources(self) -> None:
        world = SimulationWorld.from_catalog(self.catalog)
        self.assertTrue(world.submit("iss", TakeManualControl()).accepted)
        self.assertTrue(
            world.submit(
                "iss",
                ManualActuation(0.5, (0.001, 0.0, 0.0), (0.01, 0.0, 0.0)),
            ).accepted
        )
        before = world.snapshot().satellite("iss")

        after = world.step(1.0).satellite("iss")

        self.assertLess(after.main_propellant_kg, before.main_propellant_kg)
        self.assertLess(after.rcs_propellant_kg, before.rcs_propellant_kg)
        self.assertNotEqual(after.wheel_momentum_n_m_s, before.wheel_momentum_n_m_s)
        self.assertEqual(after.safety_reason, "clear")

    def test_returning_control_replans_from_current_state(self) -> None:
        world = SimulationWorld.from_catalog(self.catalog)
        self.assertTrue(world.submit("iss", TakeManualControl()).accepted)
        self.assertTrue(
            world.submit(
                "iss",
                ManualActuation(0.25, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            ).accepted
        )
        manual = world.step(1.0).satellite("iss")
        self.assertTrue(world.submit("iss", ReturnAutonomousControl()).accepted)

        autonomous = world.step(1.0 / UI_REFRESH_HZ).satellite("iss")

        self.assertEqual(autonomous.control_mode, ControlMode.AUTONOMOUS)
        self.assertFalse(autonomous.requires_replan)
        self.assertIsNotNone(autonomous.plan_objective_id)
        self.assertNotEqual(autonomous.position_m, manual.position_m)

    def test_insufficient_propellant_is_visible_as_planning_failure(self) -> None:
        world = SimulationWorld.from_catalog(self.catalog)
        state = world._satellites["iss"]
        resources = replace(
            state.vehicle.resources,
            main_propellant_kg=state.definition.autonomy.minimum_main_propellant_reserve_kg,
        )
        vehicle = replace(state.vehicle, resources=resources)
        translation = replace(state.translation, mass_kg=resources.total_mass_kg)
        world._satellites["iss"] = replace(
            state,
            vehicle=vehicle,
            translation=translation,
        )
        self.assertTrue(world.submit("iss", SetOrbitAltitude(800_000.0)).accepted)

        snapshot = world.step(1.0 / UI_REFRESH_HZ).satellite("iss")

        self.assertEqual(snapshot.pending_command_count, 0)
        self.assertIsNone(snapshot.plan_objective_id)
        self.assertIn("reserve", snapshot.planning_failure or "")

    def test_power_safe_recovery_requests_and_completes_replan(self) -> None:
        world = SimulationWorld.from_catalog(self.catalog)
        state = world._satellites["iss"]
        resources = replace(
            state.vehicle.resources,
            battery_energy_j=state.definition.power.battery_capacity_j,
        )
        world._satellites["iss"] = replace(
            state,
            control_mode=ControlMode.POWER_SAFE,
            vehicle=replace(
                state.vehicle,
                resources=resources,
                power_mode=PowerMode.POWER_SAFE,
            ),
        )

        recovered = world.step(1.0 / UI_REFRESH_HZ).satellite("iss")
        replanned = world.step(1.0 / UI_REFRESH_HZ).satellite("iss")

        self.assertEqual(recovered.control_mode, ControlMode.AUTONOMOUS)
        self.assertTrue(recovered.requires_replan)
        self.assertEqual(recovered.safety_reason, "clear")
        self.assertFalse(replanned.requires_replan)
        self.assertIsNotNone(replanned.plan_objective_id)

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

    def test_parallel_step_plans_each_objective_once_in_background(self) -> None:
        deferred: list[Future[PlanningResult]] = []

        def defer_planning(
            _function: object, _request: object
        ) -> Future[PlanningResult]:
            future: Future[PlanningResult] = Future()
            deferred.append(future)
            return future

        world = SimulationWorld.from_catalog(self.catalog)
        tasks = ThreadTaskManager(compute_workers=2)
        try:
            with patch.object(
                tasks,
                "submit_planning",
                side_effect=defer_planning,
            ):
                first = world.step_parallel(1.0 / UI_REFRESH_HZ, tasks)
                initial_futures = tuple(
                    pending.future for pending in world._pending_plans.values()
                )

                second = world.step_parallel(1.0 / UI_REFRESH_HZ, tasks)
                repeated_futures = tuple(
                    pending.future for pending in world._pending_plans.values()
                )

                self.assertTrue(
                    all(item.plan_objective_id is None for item in first.satellites)
                )
                self.assertEqual(initial_futures, repeated_futures)
                self.assertEqual(first.elapsed_seconds, 1.0 / UI_REFRESH_HZ)
                self.assertEqual(second.elapsed_seconds, 2.0 / UI_REFRESH_HZ)

                for future in initial_futures:
                    future.set_result(
                        PlanningFailure(
                            PlanningFailureCode.NO_CORRECTION,
                            "background plan",
                        )
                    )

                planned = world.step_parallel(1.0 / UI_REFRESH_HZ, tasks)
                world.step_parallel(1.0 / UI_REFRESH_HZ, tasks)

                self.assertTrue(
                    all(
                        item.planning_failure == "background plan"
                        for item in planned.satellites
                    )
                )
                self.assertFalse(world._pending_plans)
        finally:
            tasks.shutdown()

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
