"""The typace Textual application."""

from dataclasses import dataclass
from concurrent.futures import Future
from math import isfinite
from pathlib import Path
from typing import Literal

from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.validation import Number
from textual.widgets import Button, Input, Label, Select, Static, Switch

from typace.celestial.orbital import DAY_SECONDS
from typace.celestial.view import (
    DEFAULT_TERMINAL_CELL_PIXEL_ASPECT_RATIO,
    FRAME_INTERVAL_SECONDS,
    TIME_WARPS,
    VIEW_MODES,
    CelestialSystemView,
)
from typace.i18n import (
    DEFAULT_LOCALE,
    LOCALE_OPTIONS,
    Locale,
    body_name,
    parse_locale,
    translate,
)
from typace.keybindings import APP_BINDINGS, SETTINGS_BINDINGS
from typace.solar_system import load_solar_system
from typace.satellites import load_catalog
from typace.satellites.widgets import SatellitePanel
from typace.simulation import SimulationWorld, WorldSnapshot
from typace.tasks import ThreadTaskManager
from typace.celestial.rendering import TextureAtlasCache, prepare_texture_atlases

CONTROL_PANEL_REFRESH_SECONDS = 0.25
FLOATING_PANEL_WIDTH = 36
FLOATING_PANEL_MARGIN = 2
FLOATING_PANEL_STACK_OFFSET = 4
STATUS_REFRESH_SECONDS = 0.25

type PanelKind = Literal["simulation", "camera"]


class StatusBar(Static):
    """Compact application and simulation status shown above the scene."""

    def __init__(self, view: CelestialSystemView, locale: Locale) -> None:
        super().__init__(id="status-bar")
        self.view = view
        self._ui_locale: Locale = locale

    def on_mount(self) -> None:
        self.set_interval(STATUS_REFRESH_SECONDS, self.refresh)

    def set_locale(self, locale: Locale) -> None:
        self._ui_locale = locale
        self.refresh()

    def render(self) -> Text:
        selected = self.view.selected_body
        focus_name = (
            body_name(self._ui_locale, selected.id, selected.name)
            if selected is not None
            else translate(self._ui_locale, "system.sol")
        )
        state = (
            translate(self._ui_locale, "state.paused")
            if self.view.paused
            else f"warp {self.view.warp:g}x"
        )
        elapsed_days = self.view.elapsed_seconds / DAY_SECONDS
        left = f" typace  -  {self.view.system.name}  -  focus: {focus_name}"
        right = (
            f"J2000 {elapsed_days:+,.2f} d   {state}   "
            f"[Esc] {translate(self._ui_locale, 'app.settings')} "
        )
        result = Text(left, style="bold #68d5ff", no_wrap=True)
        result.append(" " * max(2, self.size.width - result.cell_len - len(right)))
        result.append(right, style="#9fb0bf")
        result.truncate(self.size.width, overflow="crop", pad=True)
        return result


def _hint_text(locale: Locale) -> str:
    return "  ·  ".join(
        translate(locale, key)
        for key in (
            "shortcut.view",
            "shortcut.zoom",
            "shortcut.pan",
            "shortcut.pause",
            "shortcut.settings",
        )
    )


class ControlPanel(Static):
    """One floating simulation or camera control window."""

    def __init__(
        self, view: CelestialSystemView, locale: Locale, kind: PanelKind
    ) -> None:
        super().__init__(id=f"{kind}-panel", classes="floating-panel")
        self.view = view
        self._ui_locale: Locale = locale
        self.kind = kind
        self.selected = False
        self.position_in_visible = 0
        self.visible_count = 0
        self._update_frame()

    def on_mount(self) -> None:
        self.set_interval(CONTROL_PANEL_REFRESH_SECONDS, self.refresh)

    def set_locale(self, locale: Locale) -> None:
        self._ui_locale = locale
        self._update_frame()
        self.refresh()

    def set_selected(self, selected: bool, position: int, total: int) -> None:
        self.selected = selected
        self.position_in_visible = position
        self.visible_count = total
        self.set_class(selected, "selected")
        self._update_frame()

    def _update_frame(self) -> None:
        title_key = "panel.simulation" if self.kind == "simulation" else "panel.camera"
        self.border_title = translate(self._ui_locale, title_key)
        self.border_subtitle = (
            translate(
                self._ui_locale,
                "panel.navigation",
                current=self.position_in_visible,
                total=self.visible_count,
            )
            if self.selected and self.visible_count
            else ""
        )

    def render(self) -> Text:
        selected = self.view.selected_body
        focus_name = (
            body_name(self._ui_locale, selected.id, selected.name)
            if selected is not None
            else translate(self._ui_locale, "system.sol")
        )
        if self.kind == "simulation":
            state_key = "state.paused" if self.view.paused else "state.running"
            lines = (
                f"{translate(self._ui_locale, 'panel.focus')}:  {focus_name}",
                f"{translate(self._ui_locale, 'panel.time')}:  {self.view.elapsed_seconds / DAY_SECONDS:+,.2f} d",
                f"{translate(self._ui_locale, 'panel.speed')}:  {self.view.warp:g}x",
                f"{translate(self._ui_locale, 'panel.state')}:  {translate(self._ui_locale, state_key)}",
            )
        else:
            view_name = translate(self._ui_locale, VIEW_MODES[self.view.view_index][0])
            lines = (
                f"{translate(self._ui_locale, 'panel.view')}:  {view_name}",
                f"{translate(self._ui_locale, 'panel.zoom')}:  {self.view.zoom:.2f}x",
                f"{translate(self._ui_locale, 'panel.pan')}:  {self.view.pan_x:.2g}, {self.view.pan_y:.2g}",
            )
        return Text("\n".join(lines), style="#c8d7e8")


@dataclass(frozen=True, slots=True)
class SettingsResult:
    locale: Locale
    elapsed_seconds: float
    warp_index: int
    paused: bool
    simulation_panel_visible: bool
    camera_panel_visible: bool


class SettingsScreen(Screen[SettingsResult | None]):
    """Edit application and simulation settings without duplicating state."""

    BINDINGS = SETTINGS_BINDINGS
    CSS = """
    SettingsScreen {
        align: center middle;
        background: #02050a 85%;
    }
    #settings-dialog {
        width: 52;
        max-width: 95%;
        height: auto;
        padding: 1 2;
        border: solid #36718f;
        background: #08131f;
    }
    #settings-title {
        width: 100%;
        margin-bottom: 1;
        text-style: bold;
        color: #68d5ff;
    }
    .settings-label { margin-top: 1; color: #a9bfd3; }
    #paused-setting { margin: 0 0 1 0; }
    #settings-error { height: 1; color: #ff7777; }
    #settings-actions { height: 3; align-horizontal: right; }
    #settings-actions Button { width: 14; margin-left: 1; }
    """

    def __init__(
        self,
        view: CelestialSystemView,
        locale: Locale,
        *,
        simulation_panel_visible: bool,
        camera_panel_visible: bool,
    ) -> None:
        super().__init__()
        self.view = view
        self._ui_locale: Locale = locale
        self.simulation_panel_visible = simulation_panel_visible
        self.camera_panel_visible = camera_panel_visible

    def compose(self) -> ComposeResult:
        with Vertical(id="settings-dialog"):
            yield Label(
                translate(self._ui_locale, "settings.title"), id="settings-title"
            )
            yield Label(
                translate(self._ui_locale, "settings.language"),
                classes="settings-label",
            )
            yield Select(
                LOCALE_OPTIONS,
                value=self._ui_locale,
                allow_blank=False,
                id="locale-setting",
            )
            yield Label(
                translate(self._ui_locale, "settings.elapsed_days"),
                classes="settings-label",
            )
            yield Input(
                f"{self.view.elapsed_seconds / DAY_SECONDS:.6f}",
                type="number",
                validators=Number(),
                id="days-setting",
            )
            yield Label(
                translate(self._ui_locale, "settings.warp"), classes="settings-label"
            )
            yield Select(
                tuple((f"{warp:g}x", index) for index, warp in enumerate(TIME_WARPS)),
                value=self.view.warp_index,
                allow_blank=False,
                id="warp-setting",
            )
            yield Label(
                translate(self._ui_locale, "settings.paused"), classes="settings-label"
            )
            yield Switch(self.view.paused, id="paused-setting")
            yield Label(
                translate(self._ui_locale, "settings.panels"), classes="settings-label"
            )
            with Horizontal(classes="panel-visibility-setting"):
                yield Label(translate(self._ui_locale, "settings.simulation_panel"))
                yield Switch(
                    self.simulation_panel_visible, id="simulation-panel-setting"
                )
            with Horizontal(classes="panel-visibility-setting"):
                yield Label(translate(self._ui_locale, "settings.camera_panel"))
                yield Switch(self.camera_panel_visible, id="camera-panel-setting")
            yield Label("", id="settings-error")
            with Horizontal(id="settings-actions"):
                yield Button(
                    translate(self._ui_locale, "settings.cancel"), id="settings-cancel"
                )
                yield Button(
                    translate(self._ui_locale, "settings.apply"),
                    id="settings-apply",
                    variant="primary",
                )

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "settings-cancel":
            self.action_cancel()
            return
        if event.button.id != "settings-apply":
            return
        days_input = self.query_one("#days-setting", Input)
        try:
            days = float(days_input.value)
        except ValueError:
            days = float("nan")
        if not isfinite(days):
            self.query_one("#settings-error", Label).update(
                translate(self._ui_locale, "settings.invalid_days")
            )
            return
        locale = parse_locale(self.query_one("#locale-setting", Select).value)
        warp_index = self.query_one("#warp-setting", Select).value
        if locale is None or not isinstance(warp_index, int):
            return
        self.dismiss(
            SettingsResult(
                locale=locale,
                elapsed_seconds=days * DAY_SECONDS,
                warp_index=warp_index,
                paused=self.query_one("#paused-setting", Switch).value,
                simulation_panel_visible=self.query_one(
                    "#simulation-panel-setting", Switch
                ).value,
                camera_panel_visible=self.query_one(
                    "#camera-panel-setting", Switch
                ).value,
            )
        )


class TyPaceApp(App[None]):
    """Explore celestial systems in a terminal or SDL window."""

    ALLOW_SELECT = False
    CSS = """
    Screen { background: #020407; }
    #status-bar {
        dock: top;
        height: 1;
        background: #05090d;
    }
    #workspace {
        width: 100%;
        height: 1fr;
        border: solid #37cdf5;
        background: #03060c;
        layers: scene panels selected-panel;
    }
    #hint-strip {
        dock: bottom;
        width: 100%;
        height: 1;
        padding: 0 1;
        color: #6f7d88;
        background: #05090d;
    }
    CelestialSystemView {
        layer: scene;
        width: 100%;
        height: 100%;
    }
    .floating-panel {
        layer: panels;
        position: absolute;
        width: 36;
        max-width: 96%;
        padding: 0 1;
        border: solid #37cdf5;
        background: #05090d 96%;
        color: #c8d7e8;
    }
    #simulation-panel { height: 7; }
    #camera-panel { height: 6; }
    SatellitePanel {
        layer: selected-panel;
        position: absolute;
        width: 42;
        height: auto;
        max-height: 96%;
        padding: 0 1;
        border: solid #37cdf5;
        background: #05090d 96%;
        color: #c8d7e8;
    }
    .floating-panel.selected {
        layer: selected-panel;
        border: solid #eeb860;
        background: #101722 97%;
    }
    .panel-visibility-setting { height: 3; }
    .panel-visibility-setting Label { width: 1fr; content-align: left middle; }
    .panel-visibility-setting Switch { width: auto; }
    """
    BINDINGS = APP_BINDINGS
    cell_pixel_aspect_ratio = DEFAULT_TERMINAL_CELL_PIXEL_ASPECT_RATIO

    def __init__(self, satellite_catalogs: tuple[Path, ...] = ()) -> None:
        super().__init__()
        self._ui_locale: Locale = DEFAULT_LOCALE
        self.celestial_view = CelestialSystemView(
            load_solar_system(), selected_body_id="earth"
        )
        self.simulation_panel = ControlPanel(
            self.celestial_view, self._ui_locale, "simulation"
        )
        self.camera_panel = ControlPanel(self.celestial_view, self._ui_locale, "camera")
        self.control_panels = (self.simulation_panel, self.camera_panel)
        self.status_bar = StatusBar(self.celestial_view, self._ui_locale)
        self.hint_strip = Label(_hint_text(self._ui_locale), id="hint-strip")
        self.selected_panel_index: int | None = 0
        self.tasks = ThreadTaskManager()
        self._world_step_future: Future[WorldSnapshot] | None = None
        self._pending_real_seconds = 0.0
        self.world = SimulationWorld.from_catalog(load_catalog(satellite_catalogs))
        self.satellite_panel = SatellitePanel(self._ui_locale)
        snapshot = self.world.snapshot()
        self.satellite_panel.set_snapshot(snapshot)
        self.selected_satellite_id = snapshot.satellites[0].id
        self.celestial_view.set_world_snapshot(snapshot, self.selected_satellite_id)

    def compose(self) -> ComposeResult:
        yield self.status_bar
        yield Container(
            self.celestial_view,
            *self.control_panels,
            self.satellite_panel,
            id="workspace",
        )
        yield self.hint_strip

    def on_mount(self) -> None:
        self._apply_cell_pixel_aspect_ratio()
        self._position_panels(self.size.width)
        self._update_panel_selection()
        self.set_interval(FRAME_INTERVAL_SECONDS, self._advance_world)
        self.set_focus(self.celestial_view)
        self.call_after_refresh(lambda: self.set_focus(self.celestial_view))
        atlas_future = self.tasks.submit_serial(
            prepare_texture_atlases, self.celestial_view.system
        )
        atlas_future.add_done_callback(self._on_atlases_ready)

    def _advance_world(self) -> None:
        if self.celestial_view.paused:
            return
        self._pending_real_seconds += FRAME_INTERVAL_SECONDS
        if self._world_step_future is not None:
            return
        duration_s = self._pending_real_seconds
        self._pending_real_seconds = 0.0
        self._world_step_future = self.tasks.submit_serial(
            self._step_world, duration_s, self.celestial_view.warp
        )
        self._world_step_future.add_done_callback(self._on_world_step_ready)

    def _step_world(self, duration_s: float, time_warp: float) -> WorldSnapshot:
        self.world.set_time_warp(time_warp)
        return self.world.step_parallel(duration_s, self.tasks)

    def _on_world_step_ready(self, future: Future[WorldSnapshot]) -> None:
        if self._loop is not None and self._loop.is_closed():
            return
        self.call_from_thread(self._apply_world_step, future)

    def _apply_world_step(self, future: Future[WorldSnapshot]) -> None:
        if future is not self._world_step_future:
            return
        self._world_step_future = None
        snapshot = future.result()
        self.satellite_panel.set_snapshot(snapshot)
        self.celestial_view.set_world_snapshot(snapshot, self.selected_satellite_id)
        if self._pending_real_seconds > 0.0 and not self.celestial_view.paused:
            self._advance_world()

    def _on_atlases_ready(self, future: Future[TextureAtlasCache]) -> None:
        if self._loop is not None and self._loop.is_closed():
            return
        self.call_from_thread(self.celestial_view.set_texture_atlases, future.result())

    def on_unmount(self) -> None:
        # Do not block Textual's UI loop while a long first physics step drains.
        self.tasks.shutdown(wait=False)

    def on_celestial_system_view_satellite_selected(
        self, message: CelestialSystemView.SatelliteSelected
    ) -> None:
        self.selected_satellite_id = message.satellite_id
        self.satellite_panel.set_snapshot(self.world.snapshot())

    def on_satellite_panel_selected(self, message: SatellitePanel.Selected) -> None:
        self.selected_satellite_id = message.selection.satellite_id
        self.celestial_view.focus_satellite(self.selected_satellite_id)
        self.celestial_view.set_world_snapshot(
            self.world.snapshot(), self.selected_satellite_id
        )

    def on_satellite_panel_command_submitted(
        self, message: SatellitePanel.CommandSubmitted
    ) -> None:
        result = self.world.submit(message.satellite_id, message.command)
        self.satellite_panel.set_feedback(
            translate(self._ui_locale, "satellite.accepted")
            if result.accepted
            else result.reason
        )
        snapshot = self.world.snapshot()
        self.satellite_panel.set_snapshot(snapshot)
        self.celestial_view.set_world_snapshot(snapshot, self.selected_satellite_id)

    def on_resize(self, event: events.Resize) -> None:
        if event.pixel_size is None:
            self.cell_pixel_aspect_ratio = DEFAULT_TERMINAL_CELL_PIXEL_ASPECT_RATIO
        else:
            cell_width_px = event.pixel_size.width / event.size.width
            cell_height_px = event.pixel_size.height / event.size.height
            self.cell_pixel_aspect_ratio = cell_width_px / cell_height_px
        self._apply_cell_pixel_aspect_ratio()
        self._position_panels(event.size.width)

    def _apply_cell_pixel_aspect_ratio(self) -> None:
        self.celestial_view.set_cell_pixel_aspect_ratio(self.cell_pixel_aspect_ratio)

    def _position_panels(self, columns: int) -> None:
        left = min(FLOATING_PANEL_MARGIN, max(0, columns - 1))
        right = max(left, columns - FLOATING_PANEL_WIDTH - FLOATING_PANEL_MARGIN)
        self.simulation_panel.offset = (left, FLOATING_PANEL_MARGIN)
        camera_y = (
            FLOATING_PANEL_MARGIN
            if right > left
            else FLOATING_PANEL_MARGIN + FLOATING_PANEL_STACK_OFFSET
        )
        self.camera_panel.offset = (right, camera_y)
        self.satellite_panel.offset = (
            max(0, columns - 42 - FLOATING_PANEL_MARGIN),
            FLOATING_PANEL_MARGIN,
        )

    def _visible_panel_indices(self) -> tuple[int, ...]:
        return tuple(
            index for index, panel in enumerate(self.control_panels) if panel.display
        )

    def _select_panel(self, offset: int) -> None:
        visible = self._visible_panel_indices()
        if not visible:
            self.selected_panel_index = None
        elif self.selected_panel_index not in visible:
            self.selected_panel_index = visible[0]
        else:
            position = visible.index(self.selected_panel_index)
            self.selected_panel_index = visible[(position + offset) % len(visible)]
        self._update_panel_selection()

    def _update_panel_selection(self) -> None:
        visible = self._visible_panel_indices()
        for index, panel in enumerate(self.control_panels):
            position = visible.index(index) + 1 if index in visible else 0
            panel.set_selected(
                index == self.selected_panel_index, position, len(visible)
            )

    def action_previous_panel(self) -> None:
        self._select_panel(-1)

    def action_next_panel(self) -> None:
        self._select_panel(1)

    def action_open_settings(self) -> None:
        self.push_screen(
            SettingsScreen(
                self.celestial_view,
                self._ui_locale,
                simulation_panel_visible=self.simulation_panel.display,
                camera_panel_visible=self.camera_panel.display,
            ),
            self._apply_settings,
        )

    def _apply_settings(self, result: SettingsResult | None) -> None:
        if result is None:
            self.celestial_view.focus()
            return
        if self._world_step_future is not None:
            self._world_step_future.result()
            self._world_step_future = None
        self._pending_real_seconds = 0.0
        self._ui_locale = result.locale
        self.status_bar.set_locale(result.locale)
        self.hint_strip.update(_hint_text(result.locale))
        self.celestial_view.set_locale(result.locale)
        self.satellite_panel.set_locale(result.locale)
        self.celestial_view.set_simulation(
            elapsed_seconds=result.elapsed_seconds,
            warp_index=result.warp_index,
            paused=result.paused,
        )
        self.world.set_elapsed_seconds(result.elapsed_seconds)
        self.world.set_time_warp(TIME_WARPS[result.warp_index])
        snapshot = self.world.snapshot()
        self.satellite_panel.set_snapshot(snapshot)
        self.celestial_view.set_world_snapshot(snapshot, self.selected_satellite_id)
        self.simulation_panel.display = result.simulation_panel_visible
        self.camera_panel.display = result.camera_panel_visible
        for panel in self.control_panels:
            panel.set_locale(result.locale)
        self._select_panel(0)
        self.celestial_view.focus()
