"""Deterministic offline configuration for Astropy."""

from astropy.coordinates import solar_system_ephemeris
from astropy.utils import iers


def configure_offline_astronomy() -> None:
    """Disable all Astropy data downloads and degraded IERS results."""

    iers.conf.auto_download = False
    iers.conf.iers_degraded_accuracy = "error"
    solar_system_ephemeris.set("builtin")
