"""Snapshot-only satellite marker selection and projection."""

from dataclasses import dataclass

import numpy as np
from typace.physics.bodies import force_model_for
from typace.physics.elements import CartesianState, state_to_elements
from typace.physics.frames import perifocal_to_inertial_matrix

from typace.celestial.rendering import (
    Basis,
    BRAILLE_COLUMNS_PER_CELL,
    BRAILLE_ROWS_PER_CELL,
)
from typace.celestial.model import Vec3
from typace.satellites.state import SatelliteSnapshot

SATELLITE_POINT_MARKER = "·"
SATELLITE_OUTLINE_MARKER = "⊙"
SATELLITE_CLOSE_MARKER = "◇"
SATELLITE_ALERT_MARKER = "!"
SATELLITE_THRUST_MARKER = "✦"
SATELLITE_POINT_MAX_PROJECTED_RADIUS = 0.35
SATELLITE_OUTLINE_MAX_PROJECTED_RADIUS = 1.5
# At least two samples per projected cell keep a curved orbit visually solid;
# the cap bounds the 30 Hz rendering cost at extreme zoom levels.
ORBIT_MINIMUM_SAMPLE_COUNT = 720
ORBIT_MAXIMUM_SAMPLE_COUNT = 4_096
ORBIT_SAMPLES_PER_PROJECTED_CELL = 2.0
# Apparent horizontal radius: keeps the close model visible at maximum zoom
# without affecting any physical size or collision calculation.
SATELLITE_DISPLAY_RADIUS_M = 200_000.0
SATELLITE_ORBIT_COLORS = (
    "#63d7ff",
    "#ffbf69",
    "#c7a5ff",
    "#72e6a8",
    "#ff7f9e",
    "#f4e06d",
)


@dataclass(frozen=True, slots=True)
class SatelliteMarker:
    satellite_id: str
    column: int
    row: int
    glyph: str
    selected: bool
    color: str = "#7ef5d2"
    projected_radius_cells: float = 0.0


@dataclass(frozen=True, slots=True)
class SatelliteOrbit:
    satellite_id: str
    points: tuple[tuple[int, int], ...]
    selected: bool
    color: str = "#397080"
    sample_count: int = 0


def satellite_color(satellite_id: str, palette_index: int | None = None) -> str:
    """Return a stable accent color without depending on hash randomization."""
    index = sum(
        (position + 1) * ord(character)
        for position, character in enumerate(satellite_id)
    )
    if palette_index is not None:
        index = palette_index
    return SATELLITE_ORBIT_COLORS[index % len(SATELLITE_ORBIT_COLORS)]


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
    vertical_scale: float = 1.0,
) -> tuple[SatelliteMarker, ...]:
    markers: list[SatelliteMarker] = []
    horizontal_scale = scale / BRAILLE_COLUMNS_PER_CELL
    vertical_cell_scale = scale * vertical_scale / BRAILLE_ROWS_PER_CELL
    for palette_index, satellite in enumerate(satellites):
        primary_position = primary_positions_m.get(satellite.primary_body_id)
        if primary_position is None:
            continue
        inertial_position = primary_position + np.asarray(satellite.position_m)
        offset = inertial_position - center_m
        column = int(
            round(
                columns / 2.0
                + np.dot(offset, _basis_vector(basis.x)) * horizontal_scale
            )
        )
        row = int(
            round(
                rows / 2.0
                - np.dot(offset, _basis_vector(basis.y)) * vertical_cell_scale
            )
        )
        if not (0 <= column < columns and 0 <= row < rows):
            continue
        projected_radius = max(
            satellite.mass_kg ** (1.0 / 3.0) * horizontal_scale,
            SATELLITE_DISPLAY_RADIUS_M * horizontal_scale,
        )
        color = satellite_color(satellite.id, palette_index)
        markers.append(
            SatelliteMarker(
                satellite.id,
                column,
                row,
                (
                    SATELLITE_ALERT_MARKER
                    if satellite.conjunction_alert_ids
                    else (
                        SATELLITE_THRUST_MARKER
                        if satellite.execution_status == "burning"
                        else marker_glyph(projected_radius)
                    )
                ),
                satellite.id == selected_satellite_id,
                color,
                projected_radius,
            )
        )
    return tuple(markers)


def project_satellite_orbits(
    satellites: tuple[SatelliteSnapshot, ...],
    primary_positions_m: dict[str, np.ndarray],
    body_radii_m: dict[str, float],
    center_m: np.ndarray,
    scale: float,
    basis: Basis,
    columns: int,
    rows: int,
    *,
    selected_satellite_id: str | None,
    vertical_scale: float = 1.0,
) -> tuple[SatelliteOrbit, ...]:
    """Project each satellite's osculating elliptic orbit into screen cells."""
    orbits: list[SatelliteOrbit] = []
    horizontal_scale = scale / BRAILLE_COLUMNS_PER_CELL
    vertical_cell_scale = scale * vertical_scale / BRAILLE_ROWS_PER_CELL
    for palette_index, satellite in enumerate(satellites):
        primary_position = primary_positions_m.get(satellite.primary_body_id)
        if primary_position is None:
            continue
        mu = force_model_for(satellite.primary_body_id).gravitational_parameter_m3_s2
        try:
            elements = state_to_elements(
                CartesianState(
                    np.asarray(satellite.position_m, dtype=np.float64),
                    np.asarray(satellite.velocity_m_s, dtype=np.float64),
                ),
                mu,
            )
        except (ValueError, FloatingPointError):
            continue
        projected_circumference = (
            2.0 * np.pi * elements.semi_major_axis_m * horizontal_scale
        )
        sample_count = max(
            ORBIT_MINIMUM_SAMPLE_COUNT,
            min(
                ORBIT_MAXIMUM_SAMPLE_COUNT,
                int(projected_circumference * ORBIT_SAMPLES_PER_PROJECTED_CELL),
            ),
        )
        eccentric_anomalies = np.linspace(
            0.0, 2.0 * np.pi, sample_count, endpoint=False
        )
        perifocal_positions = np.stack(
            (
                elements.semi_major_axis_m
                * (np.cos(eccentric_anomalies) - elements.eccentricity),
                elements.semi_major_axis_m
                * np.sqrt(1.0 - elements.eccentricity**2)
                * np.sin(eccentric_anomalies),
                np.zeros(sample_count),
            )
        )
        rotation = perifocal_to_inertial_matrix(
            elements.ascending_node_rad,
            elements.inclination_rad,
            elements.periapsis_argument_rad,
        )
        positions = (rotation @ perifocal_positions).T + primary_position
        visible = _visible_orbit_samples(
            positions,
            primary_positions_m,
            body_radii_m,
            basis,
        )
        offsets = positions - center_m
        projected_columns = np.rint(
            columns / 2.0 + offsets @ _basis_vector(basis.x) * horizontal_scale
        ).astype(np.int64)
        projected_rows = np.rint(
            rows / 2.0 - offsets @ _basis_vector(basis.y) * vertical_cell_scale
        ).astype(np.int64)
        points = tuple(
            dict.fromkeys(
                (int(column), int(row))
                for column, row, is_visible in zip(
                    projected_columns,
                    projected_rows,
                    visible,
                    strict=True,
                )
                if is_visible and 0 <= column < columns and 0 <= row < rows
            )
        )
        if points:
            orbits.append(
                SatelliteOrbit(
                    satellite.id,
                    points,
                    satellite.id == selected_satellite_id,
                    satellite_color(satellite.id, palette_index),
                    sample_count,
                )
            )
    return tuple(orbits)


def _basis_vector(vector: Vec3) -> np.ndarray:
    return np.asarray((vector.x, vector.y, vector.z))


def _visible_orbit_samples(
    positions_m: np.ndarray,
    body_positions_m: dict[str, np.ndarray],
    body_radii_m: dict[str, float],
    basis: Basis,
) -> np.ndarray:
    visible = np.ones(len(positions_m), dtype=np.bool_)
    camera_axis = _basis_vector(basis.depth)
    for body_id, body_position in body_positions_m.items():
        radius_m = body_radii_m.get(body_id)
        if radius_m is None or radius_m <= 0.0:
            continue
        relative = positions_m - body_position
        depth_m = relative @ camera_axis
        projected = relative - depth_m[:, np.newaxis] * camera_axis
        behind_body = depth_m < 0.0
        inside_disk = np.einsum("ij,ij->i", projected, projected) < radius_m**2
        visible &= ~(behind_body & inside_disk)
    return visible
