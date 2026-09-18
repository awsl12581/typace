from dataclasses import replace
import unittest

import numpy as np

from typace.config.physics import EARTH_MEAN_RADIUS_M, MOON_MEAN_RADIUS_M
from typace.physics.propagation import TranslationalState
from typace.satellites import load_catalog
from typace.satellites.definition import SatelliteCatalog, StateVectorDefinition
from typace.satellites.state import ControlMode, SatelliteState
from typace.simulation.events import (
    DestructionCause,
    locate_collision_time_s,
    locate_destruction_time_s,
)
from typace.simulation import SimulationWorld
from typace.flight.conjunction import screen_conjunctions
from typace.vehicle.state import VehicleState


class LifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        catalog = load_catalog()
        definition = catalog.satellite("iss")
        self.satellite = SatelliteState(
            definition,
            "earth",
            TranslationalState(
                np.asarray((EARTH_MEAN_RADIUS_M + 100_000.0, 0.0, 0.0)),
                np.asarray((0.0, 7_800.0, 0.0)),
                definition.dry_mass_kg
                + definition.main_propellant_mass_kg
                + definition.rcs_propellant_mass_kg,
            ),
            VehicleState.from_definition(definition),
            ControlMode.AUTONOMOUS,
        )

    def test_collision_root_is_refined_and_conjunction_screen_is_stable(self) -> None:
        def state_at(elapsed_s: float) -> TranslationalState:
            return TranslationalState(
                np.asarray((20.0 - 20.0 * elapsed_s, 0.0, 0.0)),
                np.zeros(3),
                1.0,
            )

        event_time = locate_collision_time_s(1.0, state_at, 1.0)
        self.assertAlmostEqual(event_time, 0.95, places=6)
        second_state = replace(
            self.satellite,
            definition=replace(self.satellite.definition, id="iss-2"),
            translation=replace(
                self.satellite.translation,
                position_m=self.satellite.translation.position_m
                + np.asarray((100.0, 0.0, 0.0)),
            ),
        )
        candidates = screen_conjunctions((second_state, self.satellite), 10.0)
        self.assertEqual(
            candidates,
            tuple(sorted(candidates, key=lambda item: (item.first_id, item.second_id))),
        )
        if candidates:
            self.assertEqual(
                candidates[0].yielding_satellite_id,
                max(candidates[0].first_id, candidates[0].second_id),
            )

    def test_surface_destruction_is_reported_and_world_removes_satellite_atomically(
        self,
    ) -> None:
        catalog = load_catalog()
        lunar_definition = catalog.satellite("lro")
        lunar_satellite = SatelliteState(
            lunar_definition,
            "moon",
            TranslationalState(
                np.asarray((MOON_MEAN_RADIUS_M + 100.0, 0.0, 0.0)),
                np.asarray((-200.0, 0.0, 0.0)),
                lunar_definition.dry_mass_kg
                + lunar_definition.main_propellant_mass_kg
                + lunar_definition.rcs_propellant_mass_kg,
            ),
            VehicleState.from_definition(lunar_definition),
            ControlMode.AUTONOMOUS,
        )

        def state_at(elapsed_s: float) -> TranslationalState:
            return TranslationalState(
                np.asarray((MOON_MEAN_RADIUS_M + 100.0 - 200.0 * elapsed_s, 0.0, 0.0)),
                np.asarray((-200.0, 0.0, 0.0)),
                lunar_satellite.translation.mass_kg,
            )

        result = locate_destruction_time_s(
            lunar_satellite,
            1.0,
            state_at,
            lambda state: state.velocity_m_s,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result[1], DestructionCause.SURFACE_IMPACT)

        lunar = catalog.satellite("lro")
        impactor = replace(
            lunar,
            id="impactor",
            initial_orbit=StateVectorDefinition(
                catalog.scenario_epoch,
                "moon_j2000",
                (1_737_410.0, 0.0, 0.0),
                (-100.0, 0.0, 0.0),
            ),
        )
        world = SimulationWorld.from_catalog(
            SatelliteCatalog(catalog.scenario_epoch, (impactor,))
        )
        snapshot = world.step(0.2)
        self.assertEqual(snapshot.satellites, ())
        self.assertEqual(
            snapshot.destruction_events[0].cause,
            DestructionCause.SURFACE_IMPACT,
        )

    def test_dynamic_pressure_and_heat_flux_have_distinct_destruction_causes(
        self,
    ) -> None:
        state = TranslationalState(
            np.asarray((EARTH_MEAN_RADIUS_M + 100_000.0, 0.0, 0.0)),
            np.asarray((0.0, 8_000.0, 0.0)),
            self.satellite.translation.mass_kg,
        )

        dynamic_pressure = locate_destruction_time_s(
            replace(
                self.satellite,
                definition=replace(
                    self.satellite.definition,
                    maximum_dynamic_pressure_pa=1.0,
                    maximum_heat_flux_w_m2=1.0e30,
                ),
            ),
            1.0,
            lambda _: state,
            lambda current: current.velocity_m_s,
        )
        heat_flux = locate_destruction_time_s(
            replace(
                self.satellite,
                definition=replace(
                    self.satellite.definition,
                    maximum_dynamic_pressure_pa=1.0e30,
                    maximum_heat_flux_w_m2=1.0,
                ),
            ),
            1.0,
            lambda _: state,
            lambda current: current.velocity_m_s,
        )

        self.assertEqual(
            dynamic_pressure,
            (0.0, DestructionCause.DYNAMIC_PRESSURE),
        )
        self.assertEqual(heat_flux, (0.0, DestructionCause.HEAT_FLUX))


if __name__ == "__main__":
    unittest.main()
