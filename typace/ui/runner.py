"""Public entry point: select a display backend, then run the Textual app."""

from typing import Literal, TypeVar

from textual.app import App

from .config import WindowOptions

Result = TypeVar("Result")


def run(
    app: App[Result],
    *,
    backend: Literal["terminal", "sdl"] = "terminal",
    window: WindowOptions | None = None,
) -> Result | None:
    """Run synchronously and return the application's exit value.

    The terminal backend uses the app's existing Textual driver. The SDL backend
    requires window options and temporarily installs the SDL driver. Widgets,
    CSS and application lifecycle remain Textual's responsibility.
    """
    if backend == "terminal":
        if window is not None:
            raise ValueError("window options require backend='sdl'")
        return app.run()
    if backend != "sdl":
        raise ValueError(f"unknown backend: {backend}")
    if window is None:
        raise ValueError("backend='sdl' requires WindowOptions(font=...)")

    from .backends.sdl.driver import create_driver

    original_driver = app.driver_class
    app.driver_class = create_driver(window)
    try:
        return app.run()
    finally:
        app.driver_class = original_driver
