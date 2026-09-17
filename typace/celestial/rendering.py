"""Braille rendering for the solar-system Textual widget."""

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, cos, pi, radians, sin

import numpy as np
from numpy.typing import NDArray
from rich.color import Color
from rich.style import Style
from rich.text import Text

from typace.celestial.model import (
    CelestialBody,
    CelestialSystem,
    RGB,
    TextureEllipse,
    Vec3,
)
from typace.celestial.orbital import (
    CelestialState,
    elements_from_body,
    position_at_true_anomaly,
    true_anomaly,
)

BRAILLE_BITS = ((1, 8), (2, 16), (4, 32), (64, 128))
BRAILLE_BIT_WEIGHTS = np.asarray(BRAILLE_BITS, dtype=np.uint16)
TAU = 2.0 * pi
ORBIT_SAMPLES = 256
BODY_ORBIT_COLOR: RGB = (66, 76, 92)
SELECTED_ORBIT_COLOR: RGB = (116, 178, 214)
SELECTION_COLOR: RGB = (238, 184, 96)
BACKGROUND: RGB = (3, 6, 12)
NIGHT_LIGHT = 0.18
TERMINATOR_WIDTH = 0.15
TEXTURE_MIN_RADIUS_PX = 8
RING_OUTLINE_LIMIT = 12
# UI selection marker: four broken arcs complete one clockwise turn in 2.4 seconds.
SELECTION_RING_MARGIN_PX = 2
SELECTION_SEGMENT_COUNT = 4
SELECTION_ARC_FRACTION = 0.32
SELECTION_ARC_STEPS = 8
SELECTION_ROTATION_SECONDS = 2.4
# At the maximum 200 px disk radius this moves a surface point by at most
# 0.04 px, while allowing many low-warp frames to share one texture raster.
DISK_DIRECTION_QUANTUM = 0.0002

type BoolArray = NDArray[np.bool_]
type ColorArray = NDArray[np.uint8]
type FloatArray = NDArray[np.float64]
type MaterialArray = NDArray[np.uint32]


@dataclass(frozen=True, slots=True)
class Basis:
    x: Vec3
    y: Vec3
    depth: Vec3


TOP_BASIS = Basis(Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1))
SIDE_BASIS = Basis(Vec3(1, 0, 0), Vec3(0, 0, 1), Vec3(0, -1, 0))
OBLIQUE_BASIS = Basis(
    Vec3(0.894427, -0.447214, 0),
    Vec3(0.182574, 0.365148, 0.912871),
    Vec3(-0.408248, -0.816497, 0.408248),
)


@dataclass(frozen=True, slots=True)
class Scene:
    text: Text
    body_cells: dict[tuple[int, int], str]


@dataclass(frozen=True, slots=True)
class DiskRasterKey:
    radius: int
    vertical_radius: int
    bounds: tuple[int, int, int, int]
    basis: Basis
    body_x: tuple[int, int, int]
    body_y: tuple[int, int, int]
    axis: tuple[int, int, int]
    light: tuple[int, int, int]


@dataclass(slots=True)
class DiskRasterEntry:
    key: DiskRasterKey
    valid: BoolArray
    colors: ColorArray
    materials: ColorArray


type DiskRasterCache = dict[str, DiskRasterEntry]


@lru_cache(maxsize=4096)
def _style(foreground: RGB, background: RGB = BACKGROUND) -> Style:
    return Style(
        color=Color.from_rgb(*foreground),
        bgcolor=Color.from_rgb(*background),
    )


def _material_key(color: RGB) -> int:
    return 1 + (color[0] << 16) + (color[1] << 8) + color[2]


def _material_keys(colors: ColorArray) -> MaterialArray:
    values = colors.astype(np.uint32)
    return (1 + (values[..., 0] << 16) + (values[..., 1] << 8) + values[..., 2]).astype(
        np.uint32
    )


class BrailleRaster:
    """A render-local 2x4 sub-pixel buffer, not a UI or layout abstraction."""

    def __init__(self, columns: int, rows: int) -> None:
        self.columns = max(1, columns)
        self.rows = max(1, rows)
        self.width = self.columns * 2
        self.height = self.rows * 4
        self.colors: ColorArray = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.occupied: BoolArray = np.zeros((self.height, self.width), dtype=np.bool_)
        self.materials: MaterialArray = np.zeros(
            (self.height, self.width), dtype=np.uint32
        )
        self.owners: NDArray[np.object_] = np.full(
            (self.height, self.width), None, dtype=object
        )

    def set(
        self,
        x: int,
        y: int,
        color: RGB,
        owner: str | None = None,
        material: RGB | None = None,
    ) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            self.colors[y, x] = color
            self.occupied[y, x] = True
            self.materials[y, x] = _material_key(material or color)
            if owner is not None:
                self.owners[y, x] = owner

    def paint(
        self,
        x: int,
        y: int,
        mask: BoolArray,
        colors: ColorArray,
        materials: ColorArray,
        owner: str,
    ) -> None:
        height, width = mask.shape
        color_area = self.colors[y : y + height, x : x + width]
        occupied_area = self.occupied[y : y + height, x : x + width]
        material_area = self.materials[y : y + height, x : x + width]
        owner_area = self.owners[y : y + height, x : x + width]
        color_area[mask] = colors[mask]
        occupied_area[mask] = True
        material_area[mask] = _material_keys(materials)[mask]
        owner_area[mask] = owner

    def get(self, x: int, y: int) -> RGB | None:
        if 0 <= x < self.width and 0 <= y < self.height and self.occupied[y, x]:
            color = self.colors[y, x]
            return int(color[0]), int(color[1]), int(color[2])
        return None

    def to_scene(self) -> Scene:
        result = Text(no_wrap=True)
        body_cells: dict[tuple[int, int], str] = {}
        occupied_cells = self.occupied.reshape(self.rows, 4, self.columns, 2).transpose(
            0, 2, 1, 3
        )
        color_cells = self.colors.reshape(self.rows, 4, self.columns, 2, 3).transpose(
            0, 2, 1, 3, 4
        )
        owner_cells = self.owners.reshape(self.rows, 4, self.columns, 2).transpose(
            0, 2, 1, 3
        )
        counts = occupied_cells.sum(axis=(2, 3), dtype=np.uint8)
        bits = np.sum(
            occupied_cells * BRAILLE_BIT_WEIGHTS,
            axis=(2, 3),
            dtype=np.uint16,
        )
        color_sums = np.sum(
            color_cells * occupied_cells[..., None],
            axis=(2, 3),
            dtype=np.uint32,
        )
        mean_colors = np.zeros((self.rows, self.columns, 3), dtype=np.uint8)
        populated = counts > 0
        mean_colors[populated] = np.rint(
            color_sums[populated] / counts[populated, None]
        ).astype(np.uint8)

        for row in range(self.rows):
            run = ""
            run_style: Style | None = None
            for column in range(self.columns):
                bit_value = int(bits[row, column])
                color = mean_colors[row, column]
                style = (
                    _style((int(color[0]), int(color[1]), int(color[2])))
                    if bit_value
                    else _style(BACKGROUND)
                )
                character = chr(0x2800 + bit_value) if bit_value else " "
                if run_style is not None and style != run_style:
                    result.append(run, run_style)
                    run = ""
                run += character
                run_style = style

                owners = [
                    owner
                    for owner in owner_cells[row, column].flat
                    if isinstance(owner, str)
                ]
                if owners:
                    first_owner = owners[0]
                    body_cells[(column, row)] = (
                        first_owner
                        if all(owner == first_owner for owner in owners[1:])
                        else Counter(owners).most_common(1)[0][0]
                    )
            if run_style is not None:
                result.append(run, run_style)
            if row + 1 < self.rows:
                result.append("\n")
        return Scene(result, body_cells)


@dataclass(slots=True)
class Projector:
    raster: BrailleRaster
    center: Vec3
    scale: float
    basis: Basis
    vertical_scale: float

    def project_float(self, point: Vec3) -> tuple[float, float]:
        relative = point - self.center
        x = relative.dot(self.basis.x) * self.scale + self.raster.width / 2.0
        y = (
            self.raster.height / 2.0
            - relative.dot(self.basis.y) * self.scale * self.vertical_scale
        )
        return x, y

    def project(self, point: Vec3) -> tuple[int, int]:
        x, y = self.project_float(point)
        return round(x), round(y)


def _clip_line(
    x0: float, y0: float, x1: float, y1: float, width: int, height: int
) -> tuple[float, float, float, float] | None:
    dx, dy = x1 - x0, y1 - y0
    start, end = 0.0, 1.0
    for p, q in (
        (-dx, x0),
        (dx, width - 1 - x0),
        (-dy, y0),
        (dy, height - 1 - y0),
    ):
        if p == 0:
            if q < 0:
                return None
            continue
        ratio = q / p
        if p < 0:
            start = max(start, ratio)
        else:
            end = min(end, ratio)
        if start > end:
            return None
    return x0 + start * dx, y0 + start * dy, x0 + end * dx, y0 + end * dy


def _line(
    raster: BrailleRaster,
    start: tuple[float, float],
    end: tuple[float, float],
    color: RGB,
    on_pixels: int = 1,
    off_pixels: int = 0,
) -> None:
    clipped = _clip_line(*start, *end, raster.width, raster.height)
    if clipped is None:
        return
    x0, y0, x1, y1 = clipped
    steps = max(1, round(max(abs(x1 - x0), abs(y1 - y0))))
    period = max(1, on_pixels + off_pixels)
    for step in range(steps + 1):
        if step % period >= on_pixels:
            continue
        fraction = step / steps
        raster.set(
            round(x0 + (x1 - x0) * fraction), round(y0 + (y1 - y0) * fraction), color
        )


def _orbit(
    projector: Projector,
    body: CelestialBody,
    center: Vec3,
    color: RGB,
) -> None:
    elements = elements_from_body(body)
    previous: tuple[float, float] | None = None
    for sample in range(ORBIT_SAMPLES + 1):
        eccentric_anomaly = TAU * sample / ORBIT_SAMPLES
        anomaly = true_anomaly(eccentric_anomaly, body.eccentricity)
        point = center + position_at_true_anomaly(elements, anomaly)
        projected = projector.project_float(point)
        if previous is not None:
            _line(projector.raster, previous, projected, color, 1, 4)
        previous = projected


def _axis(body: CelestialBody) -> Vec3:
    tilt = radians(body.axial_tilt_deg)
    return Vec3(sin(tilt), 0.0, cos(tilt)).unit()


def _equatorial_basis(body: CelestialBody) -> tuple[Vec3, Vec3]:
    axis = _axis(body)
    reference = Vec3(1, 0, 0) if abs(axis.x) < 0.999 else Vec3(0, 1, 0)
    first = (reference - axis * reference.dot(axis)).unit()
    return first, axis.cross(first).unit()


def _body_basis(
    body: CelestialBody,
    rotation_seconds: float,
    position: Vec3,
    parent_position: Vec3 | None,
) -> tuple[Vec3, Vec3, Vec3]:
    axis = _axis(body)
    first, second = _equatorial_basis(body)
    if body.tidally_locked and parent_position is not None:
        direction = (parent_position - position).unit()
        projected = direction - axis * direction.dot(axis)
        body_x = projected.unit() if projected.norm() else first
    else:
        period_seconds = body.rotation_period_hours * 3600.0
        phase = TAU * rotation_seconds / period_seconds if period_seconds else 0.0
        body_x = first * cos(phase) + second * sin(phase)
    return body_x, axis.cross(body_x).unit(), axis


def _rounded_colors(values: FloatArray) -> ColorArray:
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def _quantized_direction(vector: Vec3) -> tuple[int, int, int]:
    return (
        round(vector.x / DISK_DIRECTION_QUANTUM),
        round(vector.y / DISK_DIRECTION_QUANTUM),
        round(vector.z / DISK_DIRECTION_QUANTUM),
    )


def _ellipse_mask(
    latitude: FloatArray,
    longitude: FloatArray,
    ellipse: TextureEllipse,
    valid: BoolArray,
) -> BoolArray:
    longitude_delta = (longitude - ellipse.longitude_deg + 180.0) % 360.0 - 180.0
    latitude_delta = latitude - ellipse.latitude_deg
    distance = (latitude_delta / ellipse.latitude_radius_deg) ** 2 + (
        longitude_delta / ellipse.longitude_radius_deg
    ) ** 2
    return valid & (distance < 1.0)


def _polygon_mask(
    latitude: FloatArray,
    longitude: FloatArray,
    vertices: tuple[tuple[float, float], ...],
    candidates: BoolArray,
) -> BoolArray:
    candidate_latitude = latitude[candidates]
    candidate_longitude = longitude[candidates]
    candidate_inside = np.zeros(candidate_latitude.shape, dtype=np.bool_)
    previous_latitude, previous_longitude = vertices[-1]
    previous_longitude_array = (
        previous_longitude - candidate_longitude + 180.0
    ) % 360.0 - 180.0
    for current_latitude, raw_longitude in vertices:
        if current_latitude == previous_latitude:
            previous_latitude = current_latitude
            previous_longitude_array = (
                raw_longitude - candidate_longitude + 180.0
            ) % 360.0 - 180.0
            continue
        current_longitude = (
            raw_longitude - candidate_longitude + 180.0
        ) % 360.0 - 180.0
        crosses = (current_latitude > candidate_latitude) != (
            previous_latitude > candidate_latitude
        )
        longitude_at_latitude = previous_longitude_array + (
            (current_longitude - previous_longitude_array)
            * (candidate_latitude - previous_latitude)
            / (current_latitude - previous_latitude)
        )
        candidate_inside ^= crosses & (longitude_at_latitude > 0.0)
        previous_latitude = current_latitude
        previous_longitude_array = current_longitude
    inside = np.zeros(latitude.shape, dtype=np.bool_)
    inside[candidates] = candidate_inside
    return inside


def _texture_colors(
    body: CelestialBody,
    latitude: FloatArray,
    longitude: FloatArray,
    radius_fraction: FloatArray,
    dx: FloatArray,
    dy: FloatArray,
    valid: BoolArray,
) -> tuple[ColorArray, ColorArray]:
    texture = body.texture
    assert texture is not None
    shape = (*latitude.shape, 3)
    if texture.star is not None:
        star = texture.star
        materials = np.empty(shape, dtype=np.uint8)
        materials[:] = star.surface
        radial = np.clip(radius_fraction, 0.0, 1.0)
        surface_amount = np.minimum(1.0, radial * 1.7)[..., None]
        colors = _rounded_colors(
            np.asarray(star.core, dtype=np.float64) * (1.0 - surface_amount)
            + np.asarray(star.surface, dtype=np.float64) * surface_amount
        )
        limb_amount = np.maximum(0.0, (radial - 0.6) / 0.4)[..., None]
        colors = _rounded_colors(
            colors.astype(np.float64) * (1.0 - limb_amount)
            + np.asarray(star.limb, dtype=np.float64) * limb_amount
        )
        noise = (np.sin(dx * 12.9898 + dy * 78.233 + star.seed) * 0.5 + 0.5)[..., None]
        darkened = _rounded_colors(colors.astype(np.float64) * (1.0 - star.granulation))
        colors = _rounded_colors(
            darkened.astype(np.float64) * (1.0 - noise)
            + colors.astype(np.float64) * noise
        )
    else:
        colors = np.empty(shape, dtype=np.uint8)
        colors[:] = texture.base
        materials = colors.copy()
        unassigned = valid.copy()
        for band in texture.bands:
            band_mask = (
                unassigned
                & (latitude >= band.latitude_min_deg)
                & (latitude < band.latitude_max_deg)
            )
            colors[band_mask] = band.color
            materials[band_mask] = band.color
            unassigned[band_mask] = False

    for region in texture.regions:
        region_latitudes = tuple(vertex[0] for vertex in region.vertices)
        region_longitudes = tuple(vertex[1] for vertex in region.vertices)
        candidates = (
            valid
            & (latitude >= min(region_latitudes))
            & (latitude <= max(region_latitudes))
        )
        if max(region_longitudes) - min(region_longitudes) < 360.0:
            longitude_origin = region_longitudes[0]
            region_longitude_deltas = tuple(
                (value - longitude_origin + 180.0) % 360.0 - 180.0
                for value in region_longitudes
            )
            longitude_delta = (longitude - longitude_origin + 180.0) % 360.0 - 180.0
            candidates &= longitude_delta >= min(region_longitude_deltas)
            candidates &= longitude_delta <= max(region_longitude_deltas)
        if np.any(candidates):
            region_mask = _polygon_mask(
                latitude, longitude, region.vertices, candidates
            )
            region_color = texture.biomes[region.kind]
            colors[region_mask] = region_color
            materials[region_mask] = region_color
    for ellipse in (*texture.continents, *texture.spots):
        if ellipse.color is not None:
            ellipse_mask = _ellipse_mask(latitude, longitude, ellipse, valid)
            colors[ellipse_mask] = ellipse.color
            materials[ellipse_mask] = ellipse.color
    for crater in texture.craters:
        crater_mask = _ellipse_mask(latitude, longitude, crater, valid)
        if crater.color is not None:
            colors[crater_mask] = crater.color
            materials[crater_mask] = crater.color
    if texture.limb_tint is not None:
        limb_mask = valid & (radius_fraction > 0.92)
        colors[limb_mask] = texture.limb_tint
        materials[limb_mask] = texture.limb_tint
    return colors, materials


def _pixel_radius(body: CelestialBody, scale: float, limit: int) -> int:
    true_radius = round(body.radius_m * scale)
    if true_radius >= 4:
        return min(limit, true_radius)
    if body.body_type == "Star":
        return 6
    if body.radius_m >= 20_000_000:
        return 4
    if body.radius_m >= 3_000_000:
        return 2
    return 1


def _disk(
    projector: Projector,
    body: CelestialBody,
    position: Vec3,
    parent_position: Vec3 | None,
    light_position: Vec3 | None,
    rotation_seconds: float,
    radius: int,
    disk_raster_cache: DiskRasterCache | None = None,
) -> None:
    center_x, center_y = projector.project(position)
    raster = projector.raster
    vertical_radius = max(1, round(radius * projector.vertical_scale))
    dx_start = max(-radius, -center_x)
    dx_stop = min(radius, raster.width - 1 - center_x)
    dy_start = max(-vertical_radius, -center_y)
    dy_stop = min(vertical_radius, raster.height - 1 - center_y)
    if dx_start > dx_stop or dy_start > dy_stop:
        return
    body_x, body_y, axis = _body_basis(
        body, rotation_seconds, position, parent_position
    )
    light_direction = (
        (light_position - position).unit() if light_position is not None else Vec3()
    )
    textured = body.texture is not None and radius >= TEXTURE_MIN_RADIUS_PX
    cache_key = (
        DiskRasterKey(
            radius,
            vertical_radius,
            (dx_start, dx_stop, dy_start, dy_stop),
            projector.basis,
            _quantized_direction(body_x),
            _quantized_direction(body_y),
            _quantized_direction(axis),
            _quantized_direction(light_direction),
        )
        if textured and disk_raster_cache is not None
        else None
    )
    cached = (
        disk_raster_cache.get(body.id)
        if cache_key is not None and disk_raster_cache is not None
        else None
    )
    if cached is not None and cached.key == cache_key:
        raster.paint(
            center_x + dx_start,
            center_y + dy_start,
            cached.valid,
            cached.colors,
            cached.materials,
            body.id,
        )
        return

    dx = np.arange(dx_start, dx_stop + 1, dtype=np.float64)[None, :]
    dy = np.arange(dy_start, dy_stop + 1, dtype=np.float64)[:, None]
    nx, ny = np.broadcast_arrays(dx / radius, -dy / vertical_radius)
    radius_fraction = nx * nx + ny * ny
    valid = radius_fraction <= 1.0
    nz = np.sqrt(np.maximum(0.0, 1.0 - radius_fraction))
    surface_x = (
        projector.basis.x.x * nx
        + projector.basis.y.x * ny
        + projector.basis.depth.x * nz
    )
    surface_y = (
        projector.basis.x.y * nx
        + projector.basis.y.y * ny
        + projector.basis.depth.y * nz
    )
    surface_z = (
        projector.basis.x.z * nx
        + projector.basis.y.z * ny
        + projector.basis.depth.z * nz
    )
    surface_norm = np.sqrt(
        surface_x * surface_x + surface_y * surface_y + surface_z * surface_z
    )
    surface_x /= surface_norm
    surface_y /= surface_norm
    surface_z /= surface_norm

    if textured:
        axis_projection = surface_x * axis.x + surface_y * axis.y + surface_z * axis.z
        latitude = np.degrees(np.arcsin(np.clip(axis_projection, -1.0, 1.0)))
        longitude = np.degrees(
            np.arctan2(
                surface_x * body_y.x + surface_y * body_y.y + surface_z * body_y.z,
                surface_x * body_x.x + surface_y * body_x.y + surface_z * body_x.z,
            )
        )
        colors, materials = _texture_colors(
            body, latitude, longitude, radius_fraction, dx, dy, valid
        )
    else:
        colors = np.empty((*valid.shape, 3), dtype=np.uint8)
        colors[:] = body.color
        materials = colors.copy()

    if body.body_type != "Star" and light_direction.norm():
        cosine = (
            surface_x * light_direction.x
            + surface_y * light_direction.y
            + surface_z * light_direction.z
        )
        blend = np.clip(
            (cosine + TERMINATOR_WIDTH) / (2.0 * TERMINATOR_WIDTH), 0.0, 1.0
        )
        smooth = blend * blend * (3.0 - 2.0 * blend)
        illumination = NIGHT_LIGHT + (1.0 - NIGHT_LIGHT) * smooth
        colors = _rounded_colors(colors.astype(np.float64) * illumination[..., None])
    if cache_key is not None and disk_raster_cache is not None:
        disk_raster_cache[body.id] = DiskRasterEntry(
            cache_key, valid, colors, materials
        )
    raster.paint(
        center_x + dx_start,
        center_y + dy_start,
        valid,
        colors,
        materials,
        body.id,
    )


def _rings(
    projector: Projector,
    body: CelestialBody,
    position: Vec3,
    front: bool,
) -> None:
    if not body.rings:
        return
    center_x, center_y = projector.project(position)
    outer_radius = max(band.outer_radius_m for band in body.rings)
    horizontal_radius = ceil(outer_radius * projector.scale)
    vertical_radius = ceil(horizontal_radius * projector.vertical_scale)
    horizontal_overlap = (
        center_x + horizontal_radius >= 0
        and center_x - horizontal_radius < projector.raster.width
    )
    vertical_overlap = (
        center_y + vertical_radius >= 0
        and center_y - vertical_radius < projector.raster.height
    )
    if not horizontal_overlap or not vertical_overlap:
        return
    first, second = _equatorial_basis(body)
    for band in body.rings:
        projected_width = max(
            1, round((band.outer_radius_m - band.inner_radius_m) * projector.scale)
        )
        outlines = min(RING_OUTLINE_LIMIT, projected_width)
        for outline in range(outlines):
            fraction = (outline + 0.5) / outlines
            radius = band.inner_radius_m + fraction * (
                band.outer_radius_m - band.inner_radius_m
            )
            previous: tuple[float, float] | None = None
            for sample in range(129):
                angle = TAU * sample / 128
                offset = first * (radius * cos(angle)) + second * (radius * sin(angle))
                is_front = offset.dot(projector.basis.depth) >= 0
                point = projector.project_float(position + offset)
                if previous is not None and is_front == front:
                    _line(projector.raster, previous, point, band.color)
                previous = point


def _selection_ring(
    projector: Projector,
    position: Vec3,
    body_radius: int,
    animation_seconds: float,
) -> None:
    center_x, center_y = projector.project(position)
    radius = body_radius + SELECTION_RING_MARGIN_PX
    vertical_radius = max(1, round(radius * projector.vertical_scale))
    phase = TAU * animation_seconds / SELECTION_ROTATION_SECONDS
    segment_spacing = TAU / SELECTION_SEGMENT_COUNT
    arc_angle = segment_spacing * SELECTION_ARC_FRACTION
    for segment in range(SELECTION_SEGMENT_COUNT):
        start_angle = phase + segment * segment_spacing
        previous: tuple[float, float] | None = None
        for step in range(SELECTION_ARC_STEPS + 1):
            angle = start_angle + arc_angle * step / SELECTION_ARC_STEPS
            point = (
                center_x + radius * cos(angle),
                center_y + vertical_radius * sin(angle),
            )
            if previous is not None:
                _line(projector.raster, previous, point, SELECTION_COLOR)
            previous = point


def render_system(
    state: CelestialState,
    columns: int,
    rows: int,
    center: Vec3,
    scale: float,
    basis: Basis,
    selected_id: str | None,
    rotation_seconds: float,
    vertical_scale: float,
    selection_seconds: float,
    disk_raster_cache: DiskRasterCache | None = None,
) -> Scene:
    system = state.system
    positions = state.positions
    raster = BrailleRaster(columns, rows)
    projector = Projector(raster, center, scale, basis, vertical_scale)
    orbit_bodies = system.bodies
    if selected_id is not None:
        selected = system.body(selected_id)
        orbit_bodies = (selected, *system.children_of(selected))
    for body in orbit_bodies:
        if body.semimajor_axis_m <= 0:
            continue
        parent = system.parent_of(body)
        orbit_center = positions[parent.id] if parent is not None else Vec3()
        orbit_color = (
            SELECTED_ORBIT_COLOR if body.id == selected_id else BODY_ORBIT_COLOR
        )
        _orbit(projector, body, orbit_center, orbit_color)

    depth_sorted = sorted(
        system.bodies,
        key=lambda body: (positions[body.id] - center).dot(basis.depth),
    )
    light_positions = tuple(
        positions[body.id] for body in system.bodies if body.body_type == "Star"
    )
    radius_limit = max(raster.width, raster.height)
    for body in depth_sorted:
        position = positions[body.id]
        parent = system.parent_of(body)
        parent_position = positions[parent.id] if parent is not None else None
        # ponytail: use the nearest star until a system needs additive multi-star light.
        light_position = (
            min(light_positions, key=lambda light: (light - position).norm())
            if light_positions
            else None
        )
        radius = _pixel_radius(body, scale, radius_limit)
        _rings(projector, body, position, False)
        _disk(
            projector,
            body,
            position,
            parent_position,
            light_position,
            rotation_seconds,
            radius,
            disk_raster_cache,
        )
        _rings(projector, body, position, True)
        if body.id == selected_id:
            _selection_ring(projector, position, radius, selection_seconds)
    return raster.to_scene()
