from dataclasses import replace
from math import cos, radians, sin, sqrt
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
from typace.celestial.rendering import (
    BRAILLE_COLUMNS_PER_CELL,
    BRAILLE_ROWS_PER_CELL,
    BrailleRaster,
    Projector,
    TOP_BASIS,
)
from typace.celestial.model import Vec3
from typace.config.physics import EARTH_MEAN_RADIUS_M
from typace.physics.bodies import force_model_for
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
        radii = {body.id: body.radius_m for body in system.bodies}
        orbits = project_satellite_orbits(
            snapshot.satellites,
            positions,
            radii,
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
            4.0e-5,
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
        normal = project_satellites(
            (satellite,),
            positions,
            np.zeros(3),
            1.0e-6,
            TOP_BASIS,
            200,
            200,
            selected_satellite_id=None,
            vertical_scale=1.0,
        )[0]
        stretched = project_satellites(
            (satellite,),
            positions,
            np.zeros(3),
            1.0e-6,
            TOP_BASIS,
            200,
            200,
            selected_satellite_id=None,
            vertical_scale=2.0,
        )[0]
        expected_delta = round(satellite.position_m[1] * 1.0e-6 / BRAILLE_ROWS_PER_CELL)
        self.assertEqual(stretched.row - 100, expected_delta * 2)
        self.assertEqual(normal.row - 100, expected_delta)

    def test_satellite_and_orbit_pan_in_braille_cell_units(self) -> None:
        snapshot = SimulationWorld.from_catalog(load_catalog()).snapshot()
        satellite = snapshot.satellites[0]
        positions = {satellite.primary_body_id: np.zeros(3)}
        radii = {satellite.primary_body_id: 1.0}
        scale = 1.0e-6
        vertical_scale = 1.0
        center_shift = np.asarray(
            (
                10.0 * BRAILLE_COLUMNS_PER_CELL / scale,
                5.0 * BRAILLE_ROWS_PER_CELL / (scale * vertical_scale),
                0.0,
            )
        )

        marker = project_satellites(
            (satellite,),
            positions,
            np.zeros(3),
            scale,
            TOP_BASIS,
            1_000,
            1_000,
            selected_satellite_id=None,
            vertical_scale=vertical_scale,
        )[0]
        shifted_marker = project_satellites(
            (satellite,),
            positions,
            center_shift,
            scale,
            TOP_BASIS,
            1_000,
            1_000,
            selected_satellite_id=None,
            vertical_scale=vertical_scale,
        )[0]
        orbit = project_satellite_orbits(
            (satellite,),
            positions,
            radii,
            np.zeros(3),
            scale,
            TOP_BASIS,
            1_000,
            1_000,
            selected_satellite_id=None,
            vertical_scale=vertical_scale,
        )[0]
        shifted_orbit = project_satellite_orbits(
            (satellite,),
            positions,
            radii,
            center_shift,
            scale,
            TOP_BASIS,
            1_000,
            1_000,
            selected_satellite_id=None,
            vertical_scale=vertical_scale,
        )[0]
        point = Vec3(*satellite.position_m)
        raster = BrailleRaster(1_000, 1_000)
        planet_column, planet_row = Projector(
            raster,
            Vec3(),
            scale,
            TOP_BASIS,
            vertical_scale,
        ).project(point)
        shifted_planet_column, shifted_planet_row = Projector(
            raster,
            Vec3(*center_shift),
            scale,
            TOP_BASIS,
            vertical_scale,
        ).project(point)

        self.assertEqual(
            (shifted_marker.column - marker.column, shifted_marker.row - marker.row),
            (-10, 5),
        )
        self.assertEqual(
            (
                (shifted_planet_column // BRAILLE_COLUMNS_PER_CELL)
                - (planet_column // BRAILLE_COLUMNS_PER_CELL),
                (shifted_planet_row // BRAILLE_ROWS_PER_CELL)
                - (planet_row // BRAILLE_ROWS_PER_CELL),
            ),
            (-10, 5),
        )
        self.assertEqual(
            shifted_orbit.points,
            tuple((column - 10, row + 5) for column, row in orbit.points),
        )

    def test_planet_occludes_orbit_samples_behind_camera_view(self) -> None:
        snapshot = SimulationWorld.from_catalog(load_catalog()).snapshot()
        satellite = snapshot.satellites[0]
        radius_m = EARTH_MEAN_RADIUS_M * 2.0
        speed_m_s = sqrt(
            force_model_for("earth").gravitational_parameter_m3_s2 / radius_m
        )
        inclination_rad = radians(80.0)
        edge_on = replace(
            satellite,
            primary_body_id="earth",
            position_m=(radius_m, 0.0, 0.0),
            velocity_m_s=(
                0.0,
                speed_m_s * cos(inclination_rad),
                speed_m_s * sin(inclination_rad),
            ),
        )
        positions = {"earth": np.zeros(3)}
        common = (
            (edge_on,),
            positions,
        )
        unobscured = project_satellite_orbits(
            *common,
            {"earth": 0.0},
            np.zeros(3),
            1.0e-5,
            TOP_BASIS,
            1_000,
            1_000,
            selected_satellite_id=None,
        )[0]
        obscured = project_satellite_orbits(
            *common,
            {"earth": EARTH_MEAN_RADIUS_M},
            np.zeros(3),
            1.0e-5,
            TOP_BASIS,
            1_000,
            1_000,
            selected_satellite_id=None,
        )[0]

        self.assertGreater(len(obscured.points), 0)
        self.assertLess(len(obscured.points), len(unobscured.points))


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
