"""Animated Earth rendered by a regular Textual widget."""

import argparse
from functools import lru_cache
from math import asin, atan2, cos, degrees, radians, sin, sqrt
import os
from pathlib import Path

from rich.color import Color
from rich.style import Style
from rich.text import Text
from textual.app import App, ComposeResult
from textual.widget import Widget
from textual.widgets import Footer, Label

from typace.ui import WindowOptions, run

RGB = tuple[int, int, int]

OCEAN: RGB = (24, 82, 132)
LAND: RGB = (72, 122, 66)
DESERT: RGB = (183, 145, 85)
ICE: RGB = (220, 235, 244)
ATMOSPHERE: RGB = (69, 143, 205)

# (latitude, longitude, latitude radius, longitude radius)
CONTINENTS = (
    (47.0, -108.0, 27.0, 48.0),  # North America
    (18.0, -92.0, 17.0, 20.0),
    (-18.0, -60.0, 36.0, 19.0),  # South America
    (7.0, 20.0, 38.0, 25.0),  # Africa
    (49.0, 55.0, 26.0, 72.0),  # Europe and Asia
    (25.0, 105.0, 24.0, 42.0),
    (-25.0, 134.0, 15.0, 23.0),  # Australia
    (72.0, -41.0, 13.0, 14.0),  # Greenland
)


def _inside_ellipse(
    latitude: float,
    longitude: float,
    ellipse: tuple[float, float, float, float],
) -> bool:
    center_lat, center_lon, lat_radius, lon_radius = ellipse
    delta_lon = (longitude - center_lon + 180.0) % 360.0 - 180.0
    return ((latitude - center_lat) / lat_radius) ** 2 + (
        delta_lon / lon_radius
    ) ** 2 <= 1.0


def _surface_color(latitude: float, longitude: float) -> RGB:
    if latitude < -68.0 or latitude > 78.0:
        return ICE
    if not any(
        _inside_ellipse(latitude, longitude, continent) for continent in CONTINENTS
    ):
        return OCEAN
    if 12.0 < latitude < 34.0 and -18.0 < longitude < 58.0:
        return DESERT
    if -32.0 < latitude < -18.0 and 118.0 < longitude < 145.0:
        return DESERT
    return LAND


def _shade(color: RGB, light: float, rim: float) -> RGB:
    factor = 0.12 + 0.88 * max(0.0, light)
    atmosphere = max(0.0, (0.42 - rim) / 0.42) * 0.7
    red, green, blue = (
        min(255, round(channel * factor * (1.0 - atmosphere) + glow * atmosphere))
        for channel, glow in zip(color, ATMOSPHERE, strict=True)
    )
    return red, green, blue


def earth_pixel(x: float, y: float, rotation: float) -> RGB | None:
    """Return an Earth surface color for normalized screen coordinates."""
    radius_squared = x * x + y * y
    if radius_squared > 1.0:
        return None

    z = sqrt(max(0.0, 1.0 - radius_squared))
    tilt = radians(23.4)
    body_y = y * cos(tilt) + z * sin(tilt)
    body_z = z * cos(tilt) - y * sin(tilt)
    latitude = degrees(asin(max(-1.0, min(1.0, body_y))))
    longitude = degrees(atan2(x, body_z)) + rotation
    surface = _surface_color(latitude, longitude)

    # A fixed light makes the rotating texture cross the terminator.
    light = x * -0.55 + y * -0.25 + z * 0.80
    return _shade(surface, light, z)


BRAILLE_BITS = ((1, 8), (2, 16), (4, 32), (64, 128))


@lru_cache(maxsize=1024)
def _dot_style(color: RGB) -> Style:
    return Style(color=Color.from_rgb(*color))


class Earth(Widget):
    """A self-refreshing half-block Earth widget."""

    can_focus = True

    def __init__(self, *, rotation: float = -30.0) -> None:
        super().__init__()
        self.rotation = rotation
        self.paused = False

    def on_mount(self) -> None:
        self.set_interval(1 / 15, self.advance)

    def advance(self) -> None:
        if not self.paused:
            self.rotation = (self.rotation + 1.5) % 360.0
            self.refresh()

    def render(self) -> Text:
        width = max(1, self.size.width)
        pixel_width = width * 2
        pixel_height = max(4, self.size.height * 4)
        radius = max(1.0, min(pixel_width, pixel_height) * 0.46)
        center_x = (pixel_width - 1) / 2
        center_y = (pixel_height - 1) / 2
        result = Text(no_wrap=True)

        for row in range(self.size.height):
            for column in range(width):
                bits = 0
                colors: list[RGB] = []
                for dot_y, bit_row in enumerate(BRAILLE_BITS):
                    for dot_x, bit in enumerate(bit_row):
                        pixel_x = column * 2 + dot_x
                        pixel_y = row * 4 + dot_y
                        x = (pixel_x - center_x) / radius
                        y = (center_y - pixel_y) / radius
                        color = earth_pixel(x, y, self.rotation)
                        if color is not None:
                            bits |= bit
                            colors.append(color)
                if colors:
                    count = len(colors)
                    color = (
                        sum(rgb[0] for rgb in colors) // count,
                        sum(rgb[1] for rgb in colors) // count,
                        sum(rgb[2] for rgb in colors) // count,
                    )
                    result.append(chr(0x2800 + bits), _dot_style(color))
                else:
                    result.append(" ")
            if row + 1 < self.size.height:
                result.append("\n")
        return result


class PlanetDemo(App[None]):
    CSS = """
    Screen { background: #03060c; align: center middle; }
    #title { width: 100%; text-align: center; color: #9dc8ff; }
    Earth { width: 64; height: 28; }
    Footer { background: #07101b; }
    """
    BINDINGS = [
        ("space", "toggle_rotation", "Pause / resume"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Label("EARTH  /  23.4 deg axial tilt", id="title")
        yield Earth()
        yield Footer()

    def action_toggle_rotation(self) -> None:
        earth = self.query_one(Earth)
        earth.paused = not earth.paused


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("terminal", "sdl"), default="sdl")
    parser.add_argument(
        "--font",
        default=str(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/consola.ttf"),
    )
    args = parser.parse_args()
    assets = Path(__file__).resolve().parents[1] / "assets/fonts"
    system_fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    fallback_fonts = tuple(
        str(path)
        for path in (assets / "seguisym.ttf", system_fonts / "msyh.ttc")
        if path.exists()
    )
    if args.backend == "terminal":
        run(PlanetDemo())
    else:
        run(
            PlanetDemo(),
            backend="sdl",
            window=WindowOptions(
                font=args.font,
                fallback_fonts=fallback_fonts,
                title="typace - Earth",
            ),
        )


if __name__ == "__main__":
    main()
