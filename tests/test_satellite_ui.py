import unittest

import numpy as np
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Label, ListItem, Static

from typace.satellites import load_catalog
from typace.satellites.commands import SetOrbitAltitude, TakeManualControl
from typace.satellites.rendering import (
    SATELLITE_CLOSE_MARKER,
    SATELLITE_ALERT_MARKER,
    SATELLITE_THRUST_MARKER,
    SATELLITE_OUTLINE_MARKER,
    SATELLITE_POINT_MARKER,
    marker_glyph,
    project_satellite_orbits,
    project_satellites,
)
from typace.celestial.rendering import TOP_BASIS
from typace.solar_system import load_solar_system
from typace.satellites.widgets import SatellitePanel
from typace.simulation import SimulationWorld


class SatellitePanelHost(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.panel = SatellitePanel()
        self.commands: list[SatellitePanel.CommandSubmitted] = []

    def compose(self) -> ComposeResult:
        yield self.panel

    def on_satellite_panel_command_submitted(
        self, message: SatellitePanel.CommandSubmitted
    ) -> None:
        self.commands.append(message)


class SatelliteRenderingTests(unittest.TestCase):
    def test_marker_levels_are_point_outline_and_close_shape(self) -> None:
        self.assertEqual(marker_glyph(0.1), SATELLITE_POINT_MARKER)
        self.assertEqual(marker_glyph(1.0), SATELLITE_OUTLINE_MARKER)
        self.assertEqual(marker_glyph(2.0), SATELLITE_CLOSE_MARKER)

    def test_alert_and_thrust_markers_are_visible_states(self) -> None:
        snapshot = SimulationWorld.from_catalog(load_catalog()).snapshot()
        satellite = snapshot.satellites[0]
        from dataclasses import replace

        alerted = replace(satellite, conjunction_alert_ids=("a:b",))
        thrusting = replace(satellite, execution_status="burning")
        positions = {satellite.primary_body_id: np.zeros(3)}
        alert_marker = project_satellites(
            (alerted,),
            positions,
            np.zeros(3),
            1.0e-6,
            TOP_BASIS,
            80,
            40,
            selected_satellite_id=None,
        )[0]
        thrust_marker = project_satellites(
            (thrusting,),
            positions,
            np.zeros(3),
            1.0e-6,
            TOP_BASIS,
            80,
            40,
            selected_satellite_id=None,
        )[0]
        self.assertEqual(alert_marker.glyph, SATELLITE_ALERT_MARKER)
        self.assertEqual(thrust_marker.glyph, SATELLITE_THRUST_MARKER)

    def test_orbits_are_dense_and_use_distinct_satellite_colors(self) -> None:
        snapshot = SimulationWorld.from_catalog(load_catalog()).snapshot()
        system = load_solar_system()
        positions = {body.id: np.zeros(3) for body in system.bodies}
        masses = {body.id: body.mass_kg for body in system.bodies}
        orbits = project_satellite_orbits(
            snapshot.satellites,
            positions,
            masses,
            np.zeros(3),
            1.0e-6,
            TOP_BASIS,
            200,
            100,
            selected_satellite_id=None,
        )
        self.assertEqual(len(orbits), len(snapshot.satellites))
        self.assertTrue(all(orbit.sample_count >= 720 for orbit in orbits))
        self.assertEqual(len({orbit.color for orbit in orbits}), len(orbits))

    def test_zoomed_satellite_uses_close_marker_and_model_scale(self) -> None:
        snapshot = SimulationWorld.from_catalog(load_catalog()).snapshot()
        satellite = snapshot.satellites[0]
        marker = project_satellites(
            (satellite,),
            {satellite.primary_body_id: np.zeros(3)},
            np.asarray(satellite.position_m),
            2.0e-5,
            TOP_BASIS,
            80,
            40,
            selected_satellite_id=satellite.id,
        )[0]
        self.assertEqual(marker.glyph, SATELLITE_CLOSE_MARKER)
        self.assertGreaterEqual(marker.projected_radius_cells, 1.5)

    def test_satellite_projection_matches_vertical_cell_scale(self) -> None:
        snapshot = SimulationWorld.from_catalog(load_catalog()).snapshot()
        satellite = snapshot.satellites[0]
        positions = {satellite.primary_body_id: np.zeros(3)}
        common = dict(
            satellites=(satellite,),
            primary_positions_m=positions,
            center_m=np.zeros(3),
            scale=1.0e-6,
            basis=TOP_BASIS,
            columns=200,
            rows=200,
            selected_satellite_id=None,
        )
        normal = project_satellites(**common, vertical_scale=1.0)[0]
        stretched = project_satellites(**common, vertical_scale=2.0)[0]
        expected_delta = round(satellite.position_m[1] * 1.0e-6)
        self.assertEqual(stretched.row - 100, expected_delta * 2)
        self.assertEqual(normal.row - 100, expected_delta)


class SatellitePanelTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_list_telemetry_and_commands(self) -> None:
        app = SatellitePanelHost()
        snapshot = SimulationWorld.from_catalog(load_catalog()).snapshot()

        async with app.run_test(size=(60, 34)) as pilot:
            app.panel.set_snapshot(snapshot)
            await pilot.pause()
            self.assertEqual(len(app.panel.query(ListItem)), 6)
            telemetry = app.panel.query_one("#satellite-telemetry", Static)
            self.assertIn("Danuri / KPLO", str(telemetry.render()))

            app.panel.query_one("#satellite-altitude", Input).value = "450000"
            await pilot.click("#satellite-set-orbit")
            await pilot.pause()
            self.assertIsInstance(app.commands[-1].command, SetOrbitAltitude)
            await pilot.click("#satellite-take-control")
            await pilot.pause()
            self.assertIsInstance(app.commands[-1].command, TakeManualControl)

            app.panel.set_locale("en_US")
            self.assertEqual(
                str(app.panel.query_one("#satellite-altitude-label", Label).render()),
                "Altitude m",
            )
            self.assertEqual(
                str(app.panel.query_one("#satellite-set-orbit", Button).label),
                "Set orbit",
            )


if __name__ == "__main__":
    unittest.main()
