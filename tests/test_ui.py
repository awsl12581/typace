import ctypes
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

from textual.widgets import Button, Input, Label

from samples.basic import Demo
from typace.config.ui import DEFAULT_FONT
from typace.ui import WindowOptions, run
from typace.ui.backends.sdl.driver import SDL_EVENT_POLL_SECONDS


class WrapperTests(unittest.IsolatedAsyncioTestCase):
    def test_sdl_input_poll_interval_is_interactive(self) -> None:
        self.assertLessEqual(SDL_EVENT_POLL_SECONDS, 1.0 / 500.0)

    async def test_native_textual_widgets(self) -> None:
        app = Demo()
        async with app.run_test(size=(90, 30)) as pilot:
            await pilot.click(Input)
            await pilot.press("a", "b", "c", "left", "backspace")
            self.assertEqual(app.query_one(Input).value, "ac")
            await pilot.click("#apply")
            self.assertEqual(str(app.query_one("#status", Label).render()), "ac")

    async def test_sdl_requires_font_configuration(self) -> None:
        with self.assertRaises(ValueError):
            run(Demo(), backend="sdl")
        with self.assertRaises(ValueError):
            WindowOptions("font.ttf", width=0)

    async def test_terminal_runner_preserves_driver_and_return_value(self) -> None:
        app = Demo()
        original_driver = app.driver_class
        with patch.object(app, "run", return_value=42) as app_run:
            self.assertEqual(run(app), 42)
        app_run.assert_called_once_with()
        self.assertIs(app.driver_class, original_driver)

    @unittest.skipUnless(sys.platform == "win32", "new terminal is Windows-only")
    async def test_new_terminal_relaunches_current_entry_point(self) -> None:
        app = Demo()
        with patch.object(
            sys, "orig_argv", ["python", "-m", "samples.basic", "--new-terminal"]
        ), patch.dict(os.environ, {"TYPACE_TERMINAL_CHILD": ""}), patch(
            "typace.ui.runner.subprocess.Popen"
        ) as popen, patch.object(
            app, "run"
        ) as app_run:
            self.assertIsNone(run(app, terminal="new"))

        command = popen.call_args.args[0]
        options = popen.call_args.kwargs
        self.assertEqual(
            command,
            [sys.executable, "-m", "samples.basic", "--new-terminal"],
        )
        self.assertEqual(
            options["creationflags"], getattr(subprocess, "CREATE_NEW_CONSOLE")
        )
        self.assertEqual(options["env"]["TYPACE_TERMINAL_CHILD"], "1")
        app_run.assert_not_called()

    async def test_new_terminal_child_runs_in_place(self) -> None:
        app = Demo()
        with patch.dict(os.environ, {"TYPACE_TERMINAL_CHILD": "1"}), patch.object(
            app, "run", return_value=42
        ) as app_run:
            self.assertEqual(run(app, terminal="new"), 42)
            self.assertNotIn("TYPACE_TERMINAL_CHILD", os.environ)
        app_run.assert_called_once_with()

    async def test_sdl_rejects_terminal_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires backend='terminal'"):
            run(
                Demo(),
                backend="sdl",
                terminal="new",
                window=WindowOptions("font.ttf"),
            )

    async def test_sdl_runner_restores_driver_on_failure(self) -> None:
        from textual import screen as textual_screen

        from typace.ui.backends.sdl.driver import SDLDriver
        from typace.ui.runner import SDL_TEXTUAL_REFRESH_SECONDS

        app = Demo()
        original_driver = app.driver_class
        original_update_period = textual_screen.UPDATE_PERIOD

        def fail() -> None:
            self.assertTrue(issubclass(app.driver_class, SDLDriver))
            self.assertEqual(textual_screen.UPDATE_PERIOD, SDL_TEXTUAL_REFRESH_SECONDS)
            raise RuntimeError("startup failed")

        with patch.object(app, "run", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "startup failed"):
                run(app, backend="sdl", window=WindowOptions("font.ttf"))
        self.assertIs(app.driver_class, original_driver)
        self.assertEqual(textual_screen.UPDATE_PERIOD, original_update_period)

    async def test_pyte_color_names(self) -> None:
        from pyte.graphics import FG_ANSI, BG_ANSI, FG_AIXTERM, BG_AIXTERM
        from typace.ui.backends.sdl.renderer import color

        for table in (FG_ANSI, BG_ANSI, FG_AIXTERM, BG_AIXTERM):
            for name in table.values():
                self.assertEqual(len(color(name, True)), 3)
        self.assertEqual(color("1234ab", True), (18, 52, 171))


@unittest.skipUnless(
    os.environ.get("TYPACE_TEST_SDL") == "1",
    "set TYPACE_TEST_SDL=1 for a real SDL/OpenGL test",
)
class SDLTests(unittest.TestCase):
    def test_real_driver_text_input_resize_and_render(self) -> None:
        from typace.ui.backends.sdl import driver as driver_module
        from typace.ui.backends.sdl.renderer import ScreenRenderer

        sdl = driver_module.sdl
        frames: list[tuple[int, int]] = []
        text_frames: list[str] = []
        windows = []
        cell_size = [0, 0]
        resized_pixel_size = [0, 0]
        original_create = sdl.SDL_CreateWindow
        original_draw = ScreenRenderer.draw

        def create(title, width, height, flags):
            window = original_create(
                title, width, height, flags | sdl.SDL_WINDOW_HIDDEN
            )
            windows.append(window)
            return window

        def draw(renderer, screen, width, height):
            original_draw(renderer, screen, width, height)
            self.assertFalse(screen.dirty)
            frames.append((width, height))
            text_frames.append("\n".join(screen.display))
            cell_size[:] = [renderer.cell_width, renderer.cell_height]
            # Read real GL pixels, not just the CPU buffer or a mocked draw call.
            pixels = renderer.ctx.screen.read(
                viewport=(0, 0, width, height), components=3, alignment=1
            )
            if "typace" in text_frames[-1]:
                self.assertGreater(len(set(pixels)), 8)
                texture = renderer.texture.read(alignment=1)
                stride = width * 3
                expected = b"".join(
                    texture[row * stride : (row + 1) * stride]
                    for row in range(height - 1, -1, -1)
                )
                self.assertTrue(
                    pixels == expected, "GL output must preserve colors and orientation"
                )

        class Smoke(Demo):
            final_value = ""
            final_label = ""

            def on_mount(self) -> None:
                self.set_timer(0.2, self.inject_text)
                self.set_timer(0.5, self.click_button)
                self.set_timer(0.8, self.resize_window)
                self.set_timer(1.2, self.finish)

            def inject_text(self) -> None:
                self.query_one(Input).focus()
                event = sdl.SDL_Event()
                event.type = sdl.SDL_EVENT_KEY_DOWN
                event.key.key = sdl.SDLK_X
                sdl.SDL_PushEvent(ctypes.byref(event))
                event = sdl.SDL_Event()
                event.type = sdl.SDL_EVENT_TEXT_INPUT
                event.text.text = "x中文".encode()
                # Keep the UTF-8 pointer alive until SDL consumes the event.
                self.text_event = event
                sdl.SDL_PushEvent(ctypes.byref(event))

            def click_button(self) -> None:
                region = self.query_one(Button).region
                window_width, window_height = ctypes.c_int(), ctypes.c_int()
                pixel_width, pixel_height = ctypes.c_int(), ctypes.c_int()
                sdl.SDL_GetWindowSize(
                    windows[0], ctypes.byref(window_width), ctypes.byref(window_height)
                )
                sdl.SDL_GetWindowSizeInPixels(
                    windows[0], ctypes.byref(pixel_width), ctypes.byref(pixel_height)
                )
                x = (
                    (region.x + 2)
                    * cell_size[0]
                    * window_width.value
                    / pixel_width.value
                )
                y = (
                    (region.y + 1)
                    * cell_size[1]
                    * window_height.value
                    / pixel_height.value
                )
                for kind in (
                    sdl.SDL_EVENT_MOUSE_BUTTON_DOWN,
                    sdl.SDL_EVENT_MOUSE_BUTTON_UP,
                ):
                    event = sdl.SDL_Event()
                    event.type = kind
                    event.button.button = 1
                    event.button.x = x
                    event.button.y = y
                    sdl.SDL_PushEvent(ctypes.byref(event))

            def resize_window(self) -> None:
                sdl.SDL_SetWindowSize(windows[0], 820, 520)
                width, height = ctypes.c_int(), ctypes.c_int()
                sdl.SDL_GetWindowSizeInPixels(
                    windows[0], ctypes.byref(width), ctypes.byref(height)
                )
                resized_pixel_size[:] = [width.value, height.value]

            def finish(self) -> None:
                self.final_value = self.query_one(Input).value
                self.final_label = str(self.query_one("#status", Label).render())
                self.exit()

        app = Smoke()
        original_driver = app.driver_class
        window = WindowOptions(
            str(DEFAULT_FONT),
            width=960,
            height=640,
        )
        with patch.object(sdl, "SDL_CreateWindow", side_effect=create), patch.object(
            ScreenRenderer, "draw", new=draw
        ):
            run(app, backend="sdl", window=window)
        self.assertIs(app.driver_class, original_driver)
        self.assertEqual(app.return_code, 0)
        self.assertGreater(len(frames), 1)
        self.assertIn(tuple(resized_pixel_size), frames)
        self.assertTrue(any("x中文" in frame for frame in text_frames))
        self.assertEqual(app.final_value, "x中文")
        self.assertEqual(app.final_label, "x中文")


if __name__ == "__main__":
    unittest.main()
