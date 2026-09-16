"""Bundled Sol catalog."""

from pathlib import Path

from typace.celestial.model import CelestialSystem, load_system

_CATALOG_PATH = Path(__file__).with_name("data") / "sol.json"


def load_solar_system() -> CelestialSystem:
    """Load the bundled Sol catalog."""
    return load_system(_CATALOG_PATH)
