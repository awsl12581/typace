"""Textual Driver: ANSI -> pyte -> FreeType / OpenGL; SDL -> Textual events."""

import asyncio
import ctypes
import os
from queue import Empty, SimpleQueue

import pyte
from textual import events
from textual.driver import Driver
from textual.geometry import Size
from textual.keys import _character_to_key

from ...config import WindowOptions

os.environ.setdefault("SDL_CHECK_VERSION", "0")
os.environ.setdefault("SDL_DOC_GENERATOR", "0")
os.environ["SDL_MAIN_HANDLED"] = "1"

import sdl3 as sdl

from .renderer import ScreenRenderer


class SDLDriver(Driver):
    """Own the SDL window and adapt Textual output and input."""

    options: WindowOptions

    def start_application_mode(self) -> None:
        options = self.options
        self.pending: SimpleQueue[str] = SimpleQueue()
        self.window = None
        self.gl = None
        self.renderer: ScreenRenderer | None = None
        self.handle: asyncio.Handle | None = None
        self.enabled = True
        self.last_mouse = (0, 0)
        self.dirty = True
        sdl.SDL_SetMainReady()
        if not sdl.SDL_InitSubSystem(sdl.SDL_INIT_VIDEO):
            raise RuntimeError(sdl.SDL_GetError())
        try:
            for attribute, value in (
                (sdl.SDL_GL_CONTEXT_MAJOR_VERSION, 3),
                (sdl.SDL_GL_CONTEXT_MINOR_VERSION, 3),
                (sdl.SDL_GL_CONTEXT_PROFILE_MASK, sdl.SDL_GL_CONTEXT_PROFILE_CORE),
                (sdl.SDL_GL_DOUBLEBUFFER, 1),
            ):
                if not sdl.SDL_GL_SetAttribute(attribute, value):
                    raise RuntimeError(sdl.SDL_GetError())
            self.window = sdl.SDL_CreateWindow(
                options.title.encode(),
                options.width,
                options.height,
                sdl.SDL_WINDOW_OPENGL
                | sdl.SDL_WINDOW_RESIZABLE
                | sdl.SDL_WINDOW_HIGH_PIXEL_DENSITY,
            )
            if not self.window:
                raise RuntimeError(sdl.SDL_GetError())
            self.gl = sdl.SDL_GL_CreateContext(self.window)
            if not self.gl or not sdl.SDL_GL_MakeCurrent(self.window, self.gl):
                raise RuntimeError(sdl.SDL_GetError())
            sdl.SDL_GL_SetSwapInterval(0)
            self.renderer = ScreenRenderer(options)
            self.screen = pyte.Screen(1, 1)
            self.stream = pyte.Stream(self.screen)
            self.pixel_size = (0, 0)
            self._resize()
            if not sdl.SDL_StartTextInput(self.window):
                raise RuntimeError(sdl.SDL_GetError())
            self.handle = self._loop.call_soon(self._tick)
        except Exception:
            self.stop_application_mode()
            raise

    def write(self, data: str) -> None:
        # Textual may write on another thread; pyte and GL stay on the UI thread.
        self.pending.put(data)

    def _resize(self) -> None:
        assert self.renderer is not None
        width, height = ctypes.c_int(), ctypes.c_int()
        sdl.SDL_GetWindowSizeInPixels(
            self.window, ctypes.byref(width), ctypes.byref(height)
        )
        pixels = (width.value, height.value)
        if min(pixels) <= 0 or pixels == self.pixel_size:
            return
        self.pixel_size = pixels
        size = Size(
            max(1, pixels[0] // self.renderer.cell_width),
            max(1, pixels[1] // self.renderer.cell_height),
        )
        self.screen.resize(lines=size.height, columns=size.width)
        self.send_message(events.Resize(size, size, pixel_size=Size(*pixels)))
        self.dirty = True

    def _handle_key(self, event: sdl.SDL_KeyboardEvent) -> None:
        names = {
            sdl.SDLK_RETURN: "enter",
            sdl.SDLK_KP_ENTER: "enter",
            sdl.SDLK_ESCAPE: "escape",
            sdl.SDLK_TAB: "tab",
            sdl.SDLK_BACKSPACE: "backspace",
            sdl.SDLK_DELETE: "delete",
            sdl.SDLK_UP: "up",
            sdl.SDLK_DOWN: "down",
            sdl.SDLK_LEFT: "left",
            sdl.SDLK_RIGHT: "right",
            sdl.SDLK_HOME: "home",
            sdl.SDLK_END: "end",
            sdl.SDLK_PAGEUP: "pageup",
            sdl.SDLK_PAGEDOWN: "pagedown",
            **{getattr(sdl, f"SDLK_F{i}"): f"f{i}" for i in range(1, 13)},
        }
        modifiers = []
        for mask, name in (
            (sdl.SDL_KMOD_CTRL, "ctrl"),
            (sdl.SDL_KMOD_ALT, "alt"),
            (sdl.SDL_KMOD_SHIFT, "shift"),
            (sdl.SDL_KMOD_GUI, "super"),
        ):
            if event.mod & mask:
                modifiers.append(name)
        key = names.get(event.key)
        if key is None:
            # Printable text comes only from SDL_TEXTINPUT to prevent duplicates.
            if not event.mod & (
                sdl.SDL_KMOD_CTRL | sdl.SDL_KMOD_ALT | sdl.SDL_KMOD_GUI
            ):
                return
            raw = sdl.SDL_GetKeyName(event.key)
            key = raw.decode().lower() if raw else ""
            if len(key) != 1:
                return
        if len(key) == 1:
            key = _character_to_key(key)
        self.process_message(events.Key("+".join([*modifiers, key]), None))

    def _handle_mouse(self, event: sdl.SDL_Event) -> None:
        if not self._mouse:
            return
        assert self.renderer is not None
        mods = sdl.SDL_GetModState()
        button = 0
        if event.type == sdl.SDL_EVENT_MOUSE_MOTION:
            px, py = event.motion.x, event.motion.y
            button = next(
                (i for i in (1, 2, 3) if event.motion.state & (1 << (i - 1))), 0
            )
            cls = events.MouseMove
        elif event.type == sdl.SDL_EVENT_MOUSE_WHEEL:
            px, py = event.wheel.mouse_x, event.wheel.mouse_y
            delta = event.wheel.y
            if event.wheel.direction == sdl.SDL_MOUSEWHEEL_FLIPPED:
                delta = -delta
            if not delta:
                return
            cls = events.MouseScrollUp if delta > 0 else events.MouseScrollDown
        else:
            px, py = event.button.x, event.button.y
            button = event.button.button
            cls = (
                events.MouseDown
                if event.type == sdl.SDL_EVENT_MOUSE_BUTTON_DOWN
                else events.MouseUp
            )
        width, height = ctypes.c_int(), ctypes.c_int()
        sdl.SDL_GetWindowSize(self.window, ctypes.byref(width), ctypes.byref(height))
        x = int(
            px * self.pixel_size[0] / max(1, width.value) / self.renderer.cell_width
        )
        y = int(
            py * self.pixel_size[1] / max(1, height.value) / self.renderer.cell_height
        )
        old_x, old_y = self.last_mouse
        self.last_mouse = x, y
        self.process_message(
            cls(
                None,
                x,
                y,
                x - old_x,
                y - old_y,
                button,
                bool(mods & sdl.SDL_KMOD_SHIFT),
                bool(mods & sdl.SDL_KMOD_ALT),
                bool(mods & sdl.SDL_KMOD_CTRL),
            )
        )

    def _tick(self) -> None:
        if not self.enabled:
            return
        try:
            self._resize()
            event = sdl.SDL_Event()
            while sdl.SDL_PollEvent(ctypes.byref(event)):
                if event.type in (
                    sdl.SDL_EVENT_QUIT,
                    sdl.SDL_EVENT_WINDOW_CLOSE_REQUESTED,
                ):
                    self._app.exit()
                    return
                if event.type == sdl.SDL_EVENT_KEY_DOWN:
                    self._handle_key(event.key)
                elif event.type == sdl.SDL_EVENT_TEXT_INPUT:
                    raw = event.text.text
                    for char in raw.decode("utf-8") if raw else "":
                        self.process_message(events.Key(_character_to_key(char), char))
                elif event.type in (
                    sdl.SDL_EVENT_MOUSE_MOTION,
                    sdl.SDL_EVENT_MOUSE_BUTTON_DOWN,
                    sdl.SDL_EVENT_MOUSE_BUTTON_UP,
                    sdl.SDL_EVENT_MOUSE_WHEEL,
                ):
                    self._handle_mouse(event)
                elif event.type == sdl.SDL_EVENT_WINDOW_EXPOSED:
                    self.dirty = True
            while True:
                try:
                    self.stream.feed(self.pending.get_nowait())
                    self.dirty = True
                except Empty:
                    break
            if self.dirty:
                assert self.renderer is not None
                self.renderer.draw(self.screen, *self.pixel_size)
                sdl.SDL_GL_SwapWindow(self.window)
                self.dirty = False
        except Exception as error:
            self._app._handle_exception(error)
            return
        self.handle = self._loop.call_later(1 / 60, self._tick)

    def disable_input(self) -> None:
        self.enabled = False
        if self.handle is not None:
            self.handle.cancel()
            self.handle = None

    def stop_application_mode(self) -> None:
        self.disable_input()
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self.gl:
            sdl.SDL_GL_DestroyContext(self.gl)
            self.gl = None
        if self.window:
            sdl.SDL_StopTextInput(self.window)
            sdl.SDL_DestroyWindow(self.window)
            self.window = None
        sdl.SDL_QuitSubSystem(sdl.SDL_INIT_VIDEO)


def create_driver(options: WindowOptions) -> type[Driver]:
    """Textual constructs drivers itself, so bind options to a private subclass."""

    class ConfiguredSDLDriver(SDLDriver):
        pass

    ConfiguredSDLDriver.options = options
    return ConfiguredSDLDriver
