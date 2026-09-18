import unittest

from textual.app import App, ComposeResult
from textual.widgets import Input, ListItem, Static

from typace.satellites import load_catalog
from typace.satellites.commands import SetOrbitAltitude, TakeManualControl
from typace.satellites.rendering import (
    SATELLITE_CLOSE_MARKER,
    SATELLITE_ALERT_MARKER,
    SATELLITE_THRUST_MARKER,
    SATELLITE_OUTLINE_MARKER,
    SATELLITE_POINT_MARKER,
    marker_glyph,
)
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
        from typace.satellites.rendering import project_satellites
        from typace.celestial.rendering import TOP_BASIS
        import numpy as np

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


if __name__ == "__main__":
    unittest.main()
