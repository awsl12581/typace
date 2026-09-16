"""The typace Textual application."""

from dataclasses import dataclass
from math import isfinite
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

CONTROL_PANEL_REFRESH_SECONDS = 0.25
FLOATING_PANEL_WIDTH = 36
FLOATING_PANEL_MARGIN = 2
FLOATING_PANEL_STACK_OFFSET = 4

type PanelKind = Literal["simulation", "camera"]


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
                "",
                translate(self._ui_locale, "shortcut.pause"),
                translate(self._ui_locale, "shortcut.warp"),
                translate(self._ui_locale, "shortcut.focus"),
                translate(self._ui_locale, "shortcut.system"),
                translate(self._ui_locale, "shortcut.settings"),
            )
        else:
            view_name = translate(self._ui_locale, VIEW_MODES[self.view.view_index][0])
            lines = (
                f"{translate(self._ui_locale, 'panel.view')}:  {view_name}",
                f"{translate(self._ui_locale, 'panel.zoom')}:  {self.view.zoom:.2f}x",
                f"{translate(self._ui_locale, 'panel.pan')}:  {self.view.pan_x:.2g}, {self.view.pan_y:.2g}",
                "",
                translate(self._ui_locale, "shortcut.view"),
                translate(self._ui_locale, "shortcut.zoom"),
                translate(self._ui_locale, "shortcut.pan"),
                translate(self._ui_locale, "shortcut.mouse"),
                translate(self._ui_locale, "shortcut.settings"),
                translate(self._ui_locale, "shortcut.quit"),
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
    Screen { background: #03060c; }
    #workspace {
        width: 100%;
        height: 100%;
        layers: scene panels selected-panel;
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
        height: 14;
        padding: 1 2;
        border: solid #36718f;
        background: #07101b 94%;
        color: #c8d7e8;
    }
    .floating-panel.selected {
        layer: selected-panel;
        border: double #ff4090;
        background: #101523 97%;
    }
    .panel-visibility-setting { height: 3; }
    .panel-visibility-setting Label { width: 1fr; content-align: left middle; }
    .panel-visibility-setting Switch { width: auto; }
    """
    BINDINGS = APP_BINDINGS
    cell_pixel_aspect_ratio = DEFAULT_TERMINAL_CELL_PIXEL_ASPECT_RATIO

    def __init__(self) -> None:
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
        self.selected_panel_index: int | None = 0

    def compose(self) -> ComposeResult:
        yield Container(self.celestial_view, *self.control_panels, id="workspace")

    def on_mount(self) -> None:
        self._apply_cell_pixel_aspect_ratio()
        self._position_panels(self.size.width)
        self._update_panel_selection()

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
        self._ui_locale = result.locale
        self.celestial_view.set_locale(result.locale)
        self.celestial_view.set_simulation(
            elapsed_seconds=result.elapsed_seconds,
            warp_index=result.warp_index,
            paused=result.paused,
        )
        self.simulation_panel.display = result.simulation_panel_visible
        self.camera_panel.display = result.camera_panel_visible
        for panel in self.control_panels:
            panel.set_locale(result.locale)
        self._select_panel(0)
        self.celestial_view.focus()
