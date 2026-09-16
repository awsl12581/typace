import hashlib
import os
from math import sin
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from rich.color import Color
from rich.style import Style
from textual import events
from textual.geometry import Size
from textual.widgets import Input, Select, Switch

from app.__main__ import main
from typace.application import ControlPanel, SettingsScreen, TyPaceApp
from typace.celestial.model import Vec3, load_system
from typace.celestial.orbital import (
    DAY_SECONDS,
    body_positions,
    relative_position,
    solve_kepler,
    state_at,
)
from typace.celestial.view import CelestialSystemView
from typace.celestial.rendering import (
    OBLIQUE_BASIS,
    SELECTION_COLOR,
    BrailleRaster,
    Projector,
    TOP_BASIS,
    _disk,
    _rings,
    _selection_ring,
)
from typace.config import DEFAULT_FALLBACK_FONTS, DEFAULT_FONT
from typace.solar_system import load_solar_system


class SolarSystemModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.system = load_solar_system()

    def test_catalog_contains_planets_textures_and_rings(self) -> None:
        planets = tuple(
            body for body in self.system.bodies if body.body_type == "Planet"
        )

        self.assertEqual(len(self.system.bodies), 18)
        self.assertEqual(len(planets), 8)
        self.assertIsNotNone(self.system.body("earth").texture)
        self.assertEqual(len(self.system.body("saturn").rings), 4)

    def test_generic_loader_accepts_a_catalog_path(self) -> None:
        catalog = (
            Path(__file__).parents[1] / "typace" / "solar_system" / "data" / "sol.json"
        )

        self.assertEqual(load_system(catalog), self.system)

    def test_kepler_solution_satisfies_equation(self) -> None:
        mean = 2.1
        eccentricity = 0.72
        eccentric = solve_kepler(mean, eccentricity)

        self.assertAlmostEqual(eccentric - eccentricity * sin(eccentric), mean)

    def test_earth_returns_to_its_position_after_one_orbit(self) -> None:
        earth = self.system.body("earth")
        start = relative_position(earth, 0.0)
        end = relative_position(earth, earth.orbit_period_days * DAY_SECONDS)

        self.assertLess((end - start).norm(), 1.0)

    def test_moon_position_is_relative_to_earth(self) -> None:
        elapsed_seconds = 12.5 * DAY_SECONDS
        positions = body_positions(self.system, elapsed_seconds)
        expected = positions["earth"] + relative_position(
            self.system.body("moon"), elapsed_seconds
        )

        self.assertLess((positions["moon"] - expected).norm(), 1e-6)

    def test_state_binds_catalog_time_and_positions(self) -> None:
        elapsed_seconds = 3.0 * DAY_SECONDS
        state = state_at(self.system, elapsed_seconds)

        self.assertIs(state.system, self.system)
        self.assertEqual(state.elapsed_seconds, elapsed_seconds)
        self.assertEqual(state.positions, body_positions(self.system, elapsed_seconds))


class AppEntryPointTests(unittest.TestCase):
    def test_terminal_backend_starts_application(self) -> None:
        with patch.object(sys, "argv", ["app", "--backend", "terminal"]), patch(
            "app.__main__.run"
        ) as start:
            main()

        self.assertIsInstance(start.call_args.args[0], TyPaceApp)
        self.assertEqual(start.call_args.kwargs, {})

    def test_sdl_backend_uses_bundled_font(self) -> None:
        with patch.object(sys, "argv", ["app", "--backend", "sdl"]), patch(
            "app.__main__.run"
        ) as start:
            main()

        options = start.call_args.kwargs
        self.assertIsInstance(start.call_args.args[0], TyPaceApp)
        self.assertEqual(options["backend"], "sdl")
        self.assertEqual(options["window"].font, str(DEFAULT_FONT))
        self.assertEqual(options["window"].fallback_fonts, DEFAULT_FALLBACK_FONTS)
        self.assertTrue(all(Path(path).is_file() for path in DEFAULT_FALLBACK_FONTS))
        self.assertEqual(options["window"].title, "typace - Solar System")
        self.assertTrue(DEFAULT_FONT.is_file())


class SolarSystemInteractionTests(unittest.IsolatedAsyncioTestCase):
    async def test_render_and_controls(self) -> None:
        application = TyPaceApp()
        self.assertFalse(application.ALLOW_SELECT)

        async with application.run_test(size=(80, 36)) as pilot:
            view = application.query_one(CelestialSystemView)
            await pilot.pause()

            output = view.render().plain
            self.assertIn("地球", output)
            self.assertTrue(any("\u2801" <= char <= "\u28ff" for char in output))

            await pilot.mouse_down(view, offset=(2, 2))
            await pilot.hover(view, offset=(20, 2))
            await pilot.mouse_up(view, offset=(20, 2))
            self.assertEqual(application.screen.selections, {})

            initial_zoom = view.zoom
            columns = view.size.width
            rows = view.size.height - 1
            await pilot.press(".")
            self.assertEqual(view.warp, 10.0)
            await pilot.press("space")
            self.assertTrue(view.paused)
            paused_elapsed = view.elapsed_seconds
            selection_seconds = view.selection_seconds
            view.advance()
            self.assertEqual(view.elapsed_seconds, paused_elapsed)
            self.assertGreater(view.selection_seconds, selection_seconds)
            await pilot.press("tab")
            self.assertEqual(view.selected_body, view.system.body("mars"))
            await pilot.press("g")
            self.assertIsNone(view.selected_body)
            await pilot.press("+")
            self.assertEqual(view.zoom, initial_zoom * 1.5)
            scale, vertical_scale = view._projection_scale(columns, rows)
            await pilot.press("right", "up")
            self.assertAlmostEqual(view.pan_x * scale, columns * 2 * 0.1)
            self.assertAlmostEqual(view.pan_y * scale * vertical_scale, rows * 4 * 0.1)

            panels = tuple(application.query(ControlPanel))
            self.assertEqual(len(panels), 2)
            self.assertTrue(application.simulation_panel.has_class("selected"))
            self.assertEqual(application.simulation_panel.border_title, "模拟控制")
            await pilot.press("]")
            self.assertEqual(application.selected_panel_index, 1)
            self.assertTrue(application.camera_panel.has_class("selected"))
            await pilot.press("[")
            self.assertEqual(application.selected_panel_index, 0)

    async def test_settings_apply_simulation_data_and_language(self) -> None:
        application = TyPaceApp()

        async with application.run_test(size=(80, 36)) as pilot:
            view = application.query_one(CelestialSystemView)
            await pilot.press("escape")
            self.assertIsInstance(application.screen, SettingsScreen)

            application.screen.query_one("#locale-setting", Select).value = "en_US"
            application.screen.query_one("#days-setting", Input).value = "42.5"
            application.screen.query_one("#warp-setting", Select).value = 3
            application.screen.query_one("#paused-setting", Switch).value = True
            application.screen.query_one("#simulation-panel-setting", Switch).value = (
                False
            )
            await pilot.click("#settings-apply")
            await pilot.pause()

            self.assertNotIsInstance(application.screen, SettingsScreen)
            self.assertAlmostEqual(view.elapsed_seconds, 42.5 * DAY_SECONDS)
            self.assertEqual(view.warp_index, 3)
            self.assertTrue(view.paused)
            self.assertIn("Earth", view.render().plain)
            self.assertFalse(application.simulation_panel.display)
            self.assertTrue(application.camera_panel.display)
            self.assertEqual(application.selected_panel_index, 1)
            self.assertEqual(application.camera_panel.border_title, "Camera")

            await pilot.press("escape")
            self.assertIsInstance(application.screen, SettingsScreen)
            application.screen.query_one("#camera-panel-setting", Switch).value = False
            await pilot.click("#settings-apply")
            self.assertNotIsInstance(application.screen, SettingsScreen)
            self.assertIsNone(application.selected_panel_index)
            await pilot.press("[")
            self.assertIsNone(application.selected_panel_index)

            await pilot.press("escape")
            self.assertIsInstance(application.screen, SettingsScreen)
            await pilot.press("escape")
            self.assertNotIsInstance(application.screen, SettingsScreen)

    async def test_layout_fits_wide_narrow_and_portrait_terminals(self) -> None:
        application = TyPaceApp()

        async with application.run_test(size=(80, 36)) as pilot:
            view = application.query_one(CelestialSystemView)
            for columns, rows in ((40, 16), (140, 30), (40, 50), (20, 6)):
                with self.subTest(size=(columns, rows)):
                    await pilot.resize_terminal(columns, rows)
                    await pilot.pause()
                    view.render()
                    scene_lines = view.last_scene.text.plain.splitlines()

                    self.assertEqual(view.region.width, columns)
                    self.assertEqual(view.region.height, rows)
                    self.assertEqual(len(scene_lines), view.size.height - 1)
                    self.assertTrue(
                        all(len(line) == view.size.width for line in scene_lines)
                    )

    async def test_pixel_cell_aspect_keeps_sdl_scene_proportional(self) -> None:
        application = TyPaceApp()

        async with application.run_test(size=(80, 36)) as pilot:
            application.post_message(
                events.Resize(
                    Size(60, 40),
                    Size(60, 40),
                    pixel_size=Size(960, 960),
                )
            )
            await pilot.pause()
            view = application.query_one(CelestialSystemView)

            self.assertAlmostEqual(view.cell_pixel_aspect_ratio, 2.0 / 3.0)
            with patch(
                "typace.celestial.view.render_system", return_value=view.last_scene
            ) as render:
                view.render()
            self.assertEqual(render.call_args.args[-3], view.elapsed_seconds)
            self.assertAlmostEqual(render.call_args.args[-2], 4.0 / 3.0)


class CelestialRenderingTests(unittest.TestCase):
    def test_full_braille_cell_preserves_material_boundary(self) -> None:
        raster = BrailleRaster(1, 1)
        ocean = (10, 40, 90)
        land = (90, 140, 40)
        for y in range(raster.height):
            for x in range(raster.width):
                material = land if x == 1 else ocean
                raster.set(x, y, material, material=material)

        scene = raster.to_scene()

        self.assertEqual(scene.text.plain, "\u28b8")
        style = scene.text.spans[0].style
        assert isinstance(style, Style)
        self.assertEqual(style.color, Color.from_rgb(*land))
        self.assertEqual(style.bgcolor, Color.from_rgb(*ocean))

    def test_lighting_variation_does_not_create_material_boundary(self) -> None:
        raster = BrailleRaster(1, 1)
        material = (30, 80, 150)
        colors = ((10, 20, 30), (30, 40, 50))
        for y in range(raster.height):
            for x in range(raster.width):
                raster.set(x, y, colors[x], material=material)

        scene = raster.to_scene()

        self.assertEqual(scene.text.plain, "\u28ff")
        style = scene.text.spans[0].style
        assert isinstance(style, Style)
        self.assertEqual(style.color, Color.from_rgb(20, 30, 40))

    def test_selection_ring_is_colored_segmented_and_rotates(self) -> None:
        first = BrailleRaster(20, 10)
        second = BrailleRaster(20, 10)
        first_projector = Projector(first, Vec3(), 1.0, TOP_BASIS, 1.0)
        second_projector = Projector(second, Vec3(), 1.0, TOP_BASIS, 1.0)

        _selection_ring(first_projector, Vec3(), 8, 0.0)
        _selection_ring(second_projector, Vec3(), 8, 0.3)

        first_pixels = set(zip(*first.occupied.nonzero()))
        second_pixels = set(zip(*second.occupied.nonzero()))
        self.assertNotEqual(first_pixels, second_pixels)
        self.assertFalse(first.occupied[first.height // 2, first.width // 2])
        self.assertTrue(
            all(first.get(x, y) == SELECTION_COLOR for y, x in first_pixels)
        )

    def test_disk_work_is_clipped_to_the_visible_raster(self) -> None:
        earth = load_solar_system().body("earth")
        raster = BrailleRaster(10, 5)
        projector = Projector(raster, Vec3(), 1.0, TOP_BASIS, 1.0)

        _disk(projector, earth, Vec3(1_000, 0, 0), None, None, 0.0, 10)
        self.assertEqual(raster.occupied.sum(), 0)

        _disk(projector, earth, Vec3(9, 0, 0), None, None, 0.0, 10)
        self.assertGreater(raster.occupied.sum(), 0)
        self.assertLess(raster.occupied.sum(), 250)

    def test_ring_work_is_skipped_outside_the_raster(self) -> None:
        saturn = load_solar_system().body("saturn")
        raster = BrailleRaster(10, 5)
        projector = Projector(raster, Vec3(), 1e-7, TOP_BASIS, 1.0)

        with patch("typace.celestial.rendering._line") as line:
            _rings(projector, saturn, Vec3(1_000_000_000, 0, 0), False)
            line.assert_not_called()

            _rings(projector, saturn, Vec3(), False)
            self.assertGreater(line.call_count, 0)

    def test_numpy_disk_matches_reference_texture(self) -> None:
        earth = load_solar_system().body("earth")
        raster = BrailleRaster(20, 10)
        projector = Projector(raster, Vec3(), 1.0, OBLIQUE_BASIS, 1.0)

        _disk(
            projector,
            earth,
            Vec3(),
            None,
            Vec3(-100, 50, 20),
            12_345.0,
            10,
        )

        content = bytearray()
        for y in range(raster.height):
            for x in range(raster.width):
                color = raster.get(x, y)
                content.extend((0, 0, 0, 0) if color is None else (1, *color))
                content.append(1 if raster.owners[y, x] else 0)
        self.assertEqual(
            hashlib.sha256(content).hexdigest(),
            "6af4eb500841057ed023099c4bc8be438999fd139961b11102b78a1d366ad5b0",
        )


@unittest.skipUnless(
    os.environ.get("TYPACE_TEST_SDL") == "1",
    "set TYPACE_TEST_SDL=1 for a real SDL/OpenGL test",
)
class SolarSystemSDLTests(unittest.TestCase):
    def test_real_sdl_renderer_draws_solar_system(self) -> None:
        from pyte.screens import Screen

        from typace.ui import WindowOptions, run
        from typace.ui.backends.sdl import driver as driver_module
        from typace.ui.backends.sdl.renderer import ScreenRenderer

        sdl = driver_module.sdl
        frames: list[str] = []
        framebuffer_color_counts: list[int] = []
        original_create = sdl.SDL_CreateWindow
        original_draw = ScreenRenderer.draw

        def create(title: bytes, width: int, height: int, flags: int):
            return original_create(title, width, height, flags | sdl.SDL_WINDOW_HIDDEN)

        def draw(
            renderer: ScreenRenderer, screen: Screen, width: int, height: int
        ) -> None:
            original_draw(renderer, screen, width, height)
            self.assertTrue(
                any(face.get_char_index(ord("地")) for face in renderer.faces[1:])
            )
            frames.append("\n".join(screen.display))
            pixels = renderer.ctx.screen.read(
                viewport=(0, 0, width, height), components=3, alignment=1
            )
            framebuffer_color_counts.append(len(set(pixels)))

        class Smoke(TyPaceApp):
            def on_mount(self) -> None:
                self.set_timer(0.8, self.exit)

        application = Smoke()
        with patch.object(sdl, "SDL_CreateWindow", side_effect=create), patch.object(
            ScreenRenderer, "draw", new=draw
        ):
            run(
                application,
                backend="sdl",
                window=WindowOptions(
                    str(DEFAULT_FONT),
                    fallback_fonts=DEFAULT_FALLBACK_FONTS,
                    title="typace solar-system test",
                    width=960,
                    height=640,
                ),
            )

        self.assertEqual(application.return_code, 0)
        self.assertGreater(application.cell_pixel_aspect_ratio, 0.6)
        self.assertLess(application.cell_pixel_aspect_ratio, 0.7)
        self.assertGreater(len(frames), 1)
        self.assertTrue(
            any(any("\u2801" <= char <= "\u28ff" for char in frame) for frame in frames)
        )
        self.assertGreater(max(framebuffer_color_counts), 8)


if __name__ == "__main__":
    unittest.main()
