"""Snapshot-only satellite marker selection and projection."""

from dataclasses import dataclass

import numpy as np

from typace.celestial.rendering import Basis
from typace.celestial.model import Vec3
from typace.satellites.state import SatelliteSnapshot

SATELLITE_POINT_MARKER = "·"
SATELLITE_OUTLINE_MARKER = "⊙"
SATELLITE_CLOSE_MARKER = "◇"
SATELLITE_POINT_MAX_PROJECTED_RADIUS = 0.35
SATELLITE_OUTLINE_MAX_PROJECTED_RADIUS = 1.5


@dataclass(frozen=True, slots=True)
class SatelliteMarker:
    satellite_id: str
    column: int
    row: int
    glyph: str
    selected: bool


def marker_glyph(projected_radius_cells: float) -> str:
    if projected_radius_cells <= SATELLITE_POINT_MAX_PROJECTED_RADIUS:
        return SATELLITE_POINT_MARKER
    if projected_radius_cells <= SATELLITE_OUTLINE_MAX_PROJECTED_RADIUS:
        return SATELLITE_OUTLINE_MARKER
    return SATELLITE_CLOSE_MARKER


def project_satellites(
    satellites: tuple[SatelliteSnapshot, ...],
    primary_positions_m: dict[str, np.ndarray],
    center_m: np.ndarray,
    scale: float,
    basis: Basis,
    columns: int,
    rows: int,
    *,
    selected_satellite_id: str | None,
) -> tuple[SatelliteMarker, ...]:
    markers: list[SatelliteMarker] = []
    for satellite in satellites:
        primary_position = primary_positions_m.get(satellite.primary_body_id)
        if primary_position is None:
            continue
        inertial_position = primary_position + np.asarray(satellite.position_m)
        offset = inertial_position - center_m
        column = int(
            round(columns / 2.0 + np.dot(offset, _basis_vector(basis.x)) * scale)
        )
        row = int(round(rows / 2.0 - np.dot(offset, _basis_vector(basis.y)) * scale))
        if not (0 <= column < columns and 0 <= row < rows):
            continue
        projected_radius = satellite.mass_kg ** (1.0 / 3.0) * scale
        markers.append(
            SatelliteMarker(
                satellite.id,
                column,
                row,
                marker_glyph(projected_radius),
                satellite.id == selected_satellite_id,
            )
        )
    return tuple(markers)


def _basis_vector(vector: Vec3) -> np.ndarray:
    return np.asarray((vector.x, vector.y, vector.z))
