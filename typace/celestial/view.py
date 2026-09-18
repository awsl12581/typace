"""Textual celestial-system widget."""

from rich.text import Text
import numpy as np
from textual import events
from textual.message import Message
from collections import deque
from textual.widget import Widget

from typace.celestial.bindings import VIEW_BINDINGS
from typace.celestial.model import (
    CelestialBody,
    CelestialSystem,
    Vec3,
)
from typace.celestial.orbital import CelestialState, state_at
from typace.celestial.rendering import (
    Basis,
    DiskRasterCache,
    OBLIQUE_BASIS,
    Scene,
    SIDE_BASIS,
    TOP_BASIS,
    TextureAtlasCache,
    render_system,
)
from typace.i18n import DEFAULT_LOCALE, Locale
from typace.config.simulation import TIME_WARPS
from typace.satellites.rendering import (
    SatelliteMarker,
    SatelliteOrbit,
    project_satellite_orbits,
    project_satellites,
)
from typace.simulation.world import WorldSnapshot

# 30 Hz keeps motion visually continuous while leaving enough frame budget for
# terminal output and the measured 5-7 ms scene render cost.
FRAME_INTERVAL_SECONDS = 1.0 / 30.0
VIEW_MODES: tuple[tuple[str, Basis], ...] = (
    ("view.top", TOP_BASIS),
    ("view.oblique", OBLIQUE_BASIS),
    ("view.side", SIDE_BASIS),
)
MIN_FOCUS_RADIUS_M = 1_000_000.0
BODY_RADIUS_MARGIN = 8.0
CHILD_ORBIT_MARGIN = 1.2
RING_MARGIN = 1.3
# Camera actions move 10% of the viewport and change scale by 25% per press.
PAN_STEP_FRACTION = 0.1
ZOOM_STEP = 1.25
MIN_ZOOM = 0.05
MAX_ZOOM = 200.0
# A conventional terminal cell is half as wide as it is tall, making each
# point in a 2-by-4 Braille cell physically square.
DEFAULT_TERMINAL_CELL_PIXEL_ASPECT_RATIO = 0.5
BRAILLE_COLUMNS_PER_CELL = 2
BRAILLE_ROWS_PER_CELL = 4
SATELLITE_MODEL = ("  ╭─╮  ", "▤▤│◆│▤▤", "  ╰─╯  ")


class CelestialSystemView(Widget):
    """Focusable celestial-system map rendered by a Textual widget."""

    can_focus = True
    BINDINGS = VIEW_BINDINGS

    def __init__(
        self, system: CelestialSystem, *, selected_body_id: str | None = None
    ) -> None:
        super().__init__()
        self.system = system
        self.elapsed_seconds = 0.0
        self.selection_seconds = 0.0
        self.warp_index = TIME_WARPS.index(1.0)
        self.paused = False
        self.selected_index = (
            None
            if selected_body_id is None
            else self.system.bodies.index(self.system.body(selected_body_id))
        )
        self.view_index = 1
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.cell_pixel_aspect_ratio = DEFAULT_TERMINAL_CELL_PIXEL_ASPECT_RATIO
        self._ui_locale: Locale = DEFAULT_LOCALE
        self.last_scene = Scene(Text(), {})
        self.disk_raster_cache: DiskRasterCache = {}
        self.texture_atlas_cache: TextureAtlasCache = {}
        self.world_snapshot: WorldSnapshot | None = None
        self.selected_satellite_id: str | None = None
        self.focused_satellite_id: str | None = None
        self.satellite_markers: tuple[SatelliteMarker, ...] = ()
        self.satellite_orbits: tuple[SatelliteOrbit, ...] = ()
        self._world_clock_bound = False
        self._satellite_trails: dict[str, deque[tuple[int, int]]] = {}
        self._seen_destruction_event_ids: set[str] = set()

    class SatelliteSelected(Message):
        def __init__(self, satellite_id: str) -> None:
            super().__init__()
            self.satellite_id = satellite_id

    @property
    def selected_body(self) -> CelestialBody | None:
        if self.selected_index is None:
            return None
        return self.system.bodies[self.selected_index]

    @property
    def warp(self) -> float:
        return TIME_WARPS[self.warp_index]

    def on_mount(self) -> None:
        self.focus()
        self.set_interval(FRAME_INTERVAL_SECONDS, self.advance)

    def advance(self) -> None:
        self.selection_seconds += FRAME_INTERVAL_SECONDS
        if not self._world_clock_bound and not self.paused:
            self.elapsed_seconds += FRAME_INTERVAL_SECONDS * self.warp
        self.refresh()

    def _focus_radius(self) -> float:
        selected = self.selected_body
        if selected is None:
            return max(
                body.semimajor_axis_m * (1.0 + body.eccentricity)
                for body in self.system.bodies
                if body.parent_id is None
            )
        radius = max(MIN_FOCUS_RADIUS_M, selected.radius_m * BODY_RADIUS_MARGIN)
        for child in self.system.children_of(selected):
            radius = max(
                radius,
                child.semimajor_axis_m
                * (1.0 + child.eccentricity)
                * CHILD_ORBIT_MARGIN,
            )
        for ring in selected.rings:
            radius = max(radius, ring.outer_radius_m * RING_MARGIN)
        return radius

    def _reset_camera(self) -> None:
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.focused_satellite_id = None
        self.refresh()

    def set_cell_pixel_aspect_ratio(self, aspect_ratio: float) -> None:
        """Keep Braille geometry proportional for the current character cell."""
        self.cell_pixel_aspect_ratio = aspect_ratio
        self.refresh()

    def set_locale(self, locale: Locale) -> None:
        """Change the language used by the view status line."""
        self._ui_locale = locale
        self.refresh()

    def set_world_snapshot(
        self, snapshot: WorldSnapshot, selected_satellite_id: str | None
    ) -> None:
        self.world_snapshot = snapshot
        self.selected_satellite_id = selected_satellite_id
        self._world_clock_bound = True
        self.elapsed_seconds = snapshot.elapsed_seconds
        if snapshot.selected_time_warp in TIME_WARPS:
            self.warp_index = TIME_WARPS.index(snapshot.selected_time_warp)
        self.refresh()

    def set_texture_atlases(self, atlases: TextureAtlasCache) -> None:
        """Install background-prepared immutable texture atlases."""
        self.texture_atlas_cache = atlases
        self.disk_raster_cache.clear()
        self.refresh()

    def set_simulation(
        self, *, elapsed_seconds: float, warp_index: int, paused: bool
    ) -> None:
        """Apply editable simulation state from the settings screen."""
        self.elapsed_seconds = elapsed_seconds
        self.warp_index = max(0, min(len(TIME_WARPS) - 1, warp_index))
        self.paused = paused
        self.refresh()

    def _projection_scale(self, columns: int, rows: int) -> tuple[float, float]:
        vertical_scale = (
            self.cell_pixel_aspect_ratio
            * BRAILLE_ROWS_PER_CELL
            / BRAILLE_COLUMNS_PER_CELL
        )
        horizontal_extent = columns * BRAILLE_COLUMNS_PER_CELL
        vertical_extent = rows * BRAILLE_ROWS_PER_CELL / vertical_scale
        scale = (
            min(horizontal_extent, vertical_extent)
            / (2.0 * self._focus_radius())
            * self.zoom
        )
        return scale, vertical_scale

    def render(self) -> Text:
        columns = max(1, self.size.width)
        rows = max(1, self.size.height)
        snapshot = state_at(self.system, self.elapsed_seconds)
        selected = self.selected_body
        center = snapshot.positions[selected.id] if selected is not None else Vec3()
        if self.focused_satellite_id is not None and self.world_snapshot is not None:
            try:
                focused_satellite = self.world_snapshot.satellite(
                    self.focused_satellite_id
                )
                primary_position = snapshot.positions[focused_satellite.primary_body_id]
                relative_position = Vec3(*focused_satellite.position_m)
                center = primary_position + relative_position
            except KeyError:
                self.focused_satellite_id = None
        basis = VIEW_MODES[self.view_index][1]
        center += basis.x * self.pan_x + basis.y * self.pan_y
        scale, vertical_scale = self._projection_scale(columns, rows)
        self.last_scene = render_system(
            snapshot,
            columns,
            rows,
            center,
            scale,
            basis,
            selected.id if selected is not None else None,
            self.elapsed_seconds,
            vertical_scale,
            self.selection_seconds,
            disk_raster_cache=self.disk_raster_cache,
            texture_atlas_cache=self.texture_atlas_cache,
        )
        return self._render_satellites(snapshot, center, scale, basis, vertical_scale)

    def _render_satellites(
        self,
        celestial_snapshot: CelestialState,
        center: Vec3,
        scale: float,
        basis: Basis,
        vertical_scale: float,
    ) -> Text:
        if self.world_snapshot is None:
            self.satellite_markers = ()
            return self.last_scene.text
        primary_positions = {
            body_id: np.asarray((position.x, position.y, position.z))
            for body_id, position in celestial_snapshot.positions.items()
        }
        primary_masses = {body.id: body.mass_kg for body in self.system.bodies}
        center_vector = np.asarray((center.x, center.y, center.z))
        self.satellite_orbits = project_satellite_orbits(
            self.world_snapshot.satellites,
            primary_positions,
            primary_masses,
            center_vector,
            scale,
            basis,
            self.size.width,
            self.size.height,
            selected_satellite_id=self.selected_satellite_id,
            vertical_scale=vertical_scale,
        )
        self.satellite_markers = project_satellites(
            self.world_snapshot.satellites,
            primary_positions,
            center_vector,
            scale,
            basis,
            self.size.width,
            self.size.height,
            selected_satellite_id=self.selected_satellite_id,
            vertical_scale=vertical_scale,
        )
        for marker in self.satellite_markers:
            trail = self._satellite_trails.setdefault(
                marker.satellite_id, deque(maxlen=24)
            )
            trail.append((marker.column, marker.row))
        lines = [list(line) for line in self.last_scene.text.plain.splitlines()]
        for trail in self._satellite_trails.values():
            for column, row in trail:
                if 0 <= row < len(lines) and 0 <= column < len(lines[row]):
                    if lines[row][column] == " ":
                        lines[row][column] = "·"
        for orbit in self.satellite_orbits:
            for column, row in orbit.points:
                if lines[row][column] in (" ", "·"):
                    lines[row][column] = ":"
        for marker in self.satellite_markers:
            lines[marker.row][marker.column] = marker.glyph
        for marker in self.satellite_markers:
            if marker.projected_radius_cells < 1.5:
                continue
            for row_offset, model_line in enumerate(SATELLITE_MODEL, -1):
                target_row = marker.row + row_offset
                if not 0 <= target_row < len(lines):
                    continue
                for column_offset, glyph in enumerate(model_line, -3):
                    target_column = marker.column + column_offset
                    if 0 <= target_column < len(lines[target_row]) and glyph != " ":
                        lines[target_row][target_column] = glyph
        if self.world_snapshot is not None:
            for event in self.world_snapshot.destruction_events:
                if event.event_id not in self._seen_destruction_event_ids:
                    self._seen_destruction_event_ids.add(event.event_id)
                    trail = self._satellite_trails.get(event.satellite_ids[0], ())
                    if trail:
                        column, row = trail[-1]
                        if 0 <= row < len(lines) and 0 <= column < len(lines[row]):
                            lines[row][column] = "✹"
        overlaid = self.last_scene.text.copy()
        overlaid.plain = "\n".join("".join(line) for line in lines)
        for orbit in self.satellite_orbits:
            color = "#ffcf66" if orbit.selected else orbit.color
            for column, row in orbit.points:
                offset = row * (self.size.width + 1) + column
                overlaid.stylize(color, offset, offset + 1)
        for marker in self.satellite_markers:
            offset = marker.row * (self.size.width + 1) + marker.column
            color = "bold #ffcf66" if marker.selected else f"bold {marker.color}"
            overlaid.stylize(color, offset, offset + 1)
            if marker.projected_radius_cells >= 1.5:
                for row_offset in range(-1, 2):
                    target_row = marker.row + row_offset
                    if not 0 <= target_row < self.size.height:
                        continue
                    start_column = max(0, marker.column - 3)
                    end_column = min(self.size.width, marker.column + 4)
                    model_offset = target_row * (self.size.width + 1)
                    overlaid.stylize(
                        color,
                        model_offset + start_column,
                        model_offset + end_column,
                    )
        return overlaid

    def action_toggle_pause(self) -> None:
        self.paused = not self.paused
        self.refresh()

    def action_warp_down(self) -> None:
        self.warp_index = max(0, self.warp_index - 1)
        self.refresh()

    def action_warp_up(self) -> None:
        self.warp_index = min(len(TIME_WARPS) - 1, self.warp_index + 1)
        self.refresh()

    def action_focus_next(self) -> None:
        if self.selected_index is None:
            self.selected_index = 0
        else:
            next_index = self.selected_index + 1
            self.selected_index = (
                next_index if next_index < len(self.system.bodies) else None
            )
        self._reset_camera()

    def action_focus_previous(self) -> None:
        if self.selected_index is None:
            self.selected_index = len(self.system.bodies) - 1
        elif self.selected_index == 0:
            self.selected_index = None
        else:
            self.selected_index -= 1
        self._reset_camera()

    def action_system_view(self) -> None:
        self.selected_index = None
        self._reset_camera()

    def focus_satellite(self, satellite_id: str) -> None:
        """Center subsequent zoom and pan actions on a selected satellite."""
        self._reset_camera()
        self.focused_satellite_id = satellite_id
        self.selected_satellite_id = satellite_id
        self.refresh()

    def action_zoom_in(self) -> None:
        self.zoom = min(MAX_ZOOM, self.zoom * ZOOM_STEP)
        self.refresh()

    def action_zoom_out(self) -> None:
        self.zoom = max(MIN_ZOOM, self.zoom / ZOOM_STEP)
        self.refresh()

    def action_cycle_view(self) -> None:
        self.view_index = (self.view_index + 1) % len(VIEW_MODES)
        self.refresh()

    def _pan(self, horizontal: float, vertical: float) -> None:
        columns = max(1, self.size.width)
        rows = max(1, self.size.height)
        scale, vertical_scale = self._projection_scale(columns, rows)
        horizontal_distance = (
            columns * BRAILLE_COLUMNS_PER_CELL * PAN_STEP_FRACTION / scale
        )
        vertical_distance = (
            rows * BRAILLE_ROWS_PER_CELL * PAN_STEP_FRACTION / (scale * vertical_scale)
        )
        self.pan_x += horizontal * horizontal_distance
        self.pan_y += vertical * vertical_distance
        self.refresh()

    def action_pan_left(self) -> None:
        self._pan(-1.0, 0.0)

    def action_pan_right(self) -> None:
        self._pan(1.0, 0.0)

    def action_pan_up(self) -> None:
        self._pan(0.0, 1.0)

    def action_pan_down(self) -> None:
        self._pan(0.0, -1.0)

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        event.stop()
        self.action_zoom_in()

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        event.stop()
        self.action_zoom_out()

    def on_mouse_down(self, event: events.MouseDown) -> None:
        offset = event.get_content_offset(self)
        if offset is None:
            return
        for marker in self.satellite_markers:
            if (marker.column, marker.row) == (offset.x, offset.y):
                self.selected_satellite_id = marker.satellite_id
                self.focus_satellite(marker.satellite_id)
                self.post_message(self.SatelliteSelected(marker.satellite_id))
                return
        body_id = self.last_scene.body_cells.get((offset.x, offset.y))
        if body_id is None:
            return
        self.selected_index = next(
            index for index, body in enumerate(self.system.bodies) if body.id == body_id
        )
        self.focused_satellite_id = None
        self._reset_camera()
