"""Loaded satellite definitions and catalog boundaries."""

from typace.satellites.catalog import CatalogError, load_catalog
from typace.satellites.definition import SatelliteCatalog, SatelliteDefinition

__all__ = ("CatalogError", "SatelliteCatalog", "SatelliteDefinition", "load_catalog")
