"""Public entry point: select a display backend, then run the Textual app."""

import os
import subprocess
import sys
from typing import Literal, TypeVar

from textual import screen as textual_screen
from textual.app import App

from .config import WindowOptions

Result = TypeVar("Result")
_TERMINAL_CHILD = "TYPACE_TERMINAL_CHILD"
SDL_TEXTUAL_REFRESH_SECONDS = 1.0 / 1_000.0


def _open_new_terminal() -> None:
    if os.name != "nt":
        raise RuntimeError("terminal='new' is currently supported only on Windows")
    arguments = sys.orig_argv[1:]
    if not arguments:
        raise RuntimeError("terminal='new' requires a script or module entry point")
    environment = os.environ.copy()
    environment[_TERMINAL_CHILD] = "1"
    subprocess.Popen(
        [sys.executable, *arguments],
        env=environment,
        creationflags=subprocess.CREATE_NEW_CONSOLE,
    )


def run(
    app: App[Result],
    *,
    backend: Literal["terminal", "sdl"] = "terminal",
    terminal: Literal["current", "new"] = "current",
    window: WindowOptions | None = None,
) -> Result | None:
    """Run synchronously and return the application's exit value.

    The terminal backend uses the app's existing Textual driver in this terminal
    or relaunches the current entry point in a new terminal. The SDL backend
    requires window options and temporarily installs the SDL driver.
    """
    if terminal not in ("current", "new"):
        raise ValueError(f"unknown terminal mode: {terminal}")
    if backend == "terminal":
        if window is not None:
            raise ValueError("window options require backend='sdl'")
        if terminal == "new" and os.environ.pop(_TERMINAL_CHILD, None) != "1":
            _open_new_terminal()
            return None
        return app.run()
    if backend != "sdl":
        raise ValueError(f"unknown backend: {backend}")
    if terminal != "current":
        raise ValueError("terminal mode requires backend='terminal'")
    if window is None:
        raise ValueError("backend='sdl' requires WindowOptions(font=...)")

    from .backends.sdl.driver import create_driver

    original_driver = app.driver_class
    original_update_period = textual_screen.UPDATE_PERIOD
    app.driver_class = create_driver(window)
    setattr(textual_screen, "UPDATE_PERIOD", SDL_TEXTUAL_REFRESH_SECONDS)
    try:
        return app.run()
    finally:
        setattr(textual_screen, "UPDATE_PERIOD", original_update_period)
        app.driver_class = original_driver
