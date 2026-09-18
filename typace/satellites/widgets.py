"""Native Textual satellite list, telemetry, and control panel."""

from dataclasses import dataclass
from math import isfinite, radians

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Input, Label, ListItem, ListView, Static

from typace.privacy import sanitize_text
from typace.i18n import DEFAULT_LOCALE, Locale, translate
from typace.satellites.commands import (
    Deorbit,
    ManualActuation,
    ReturnAutonomousControl,
    ReturnStableOrbit,
    SatelliteCommand,
    SetInclination,
    SetApsides,
    SetOrbitAltitude,
    TransferPrimary,
    TakeManualControl,
)
from typace.satellites.state import SatelliteSnapshot
from typace.simulation.world import WorldSnapshot


@dataclass(frozen=True, slots=True)
class SatelliteSelection:
    satellite_id: str


class SatellitePanel(Widget):
    """Select a satellite and emit commands without owning simulation state."""

    DEFAULT_CSS = """
    SatellitePanel { width: 42; height: 100%; background: #071019; border-left: solid #37cdf5; }
    #satellite-list { height: 7; border-bottom: solid #264b5c; }
    #satellite-telemetry { height: 12; padding: 0 1; color: #c8d7e8; overflow-y: hidden; }
    .satellite-input-row { height: 3; }
    .satellite-input-row Label { width: 10; content-align: left middle; }
    .satellite-input-row Input { width: 1fr; }
    .satellite-actions { height: 3; }
    .satellite-actions Button { width: 1fr; min-width: 8; }
    #satellite-feedback { height: 2; padding: 0 1; color: #ffbd66; }
    """

    class Selected(Message):
        def __init__(self, selection: SatelliteSelection) -> None:
            super().__init__()
            self.selection = selection

    class CommandSubmitted(Message):
        def __init__(self, satellite_id: str, command: SatelliteCommand) -> None:
            super().__init__()
            self.satellite_id = satellite_id
            self.command = command

    def __init__(self, locale: Locale = DEFAULT_LOCALE) -> None:
        super().__init__(id="satellite-panel", classes="floating-panel")
        self._ui_locale: Locale = locale
        self._snapshot: WorldSnapshot | None = None
        self._selected_satellite_id: str | None = None
        self._listed_satellite_ids: tuple[str, ...] = ()
        self.border_title = translate(locale, "satellite.title")

    def compose(self) -> ComposeResult:
        with Vertical():
            yield ListView(id="satellite-list")
            yield Static("", id="satellite-telemetry")
            with Horizontal(classes="satellite-input-row"):
                yield Label(
                    translate(self._ui_locale, "satellite.altitude"),
                    id="satellite-altitude-label",
                )
                yield Input(type="number", id="satellite-altitude")
            with Horizontal(classes="satellite-input-row"):
                yield Label(
                    translate(self._ui_locale, "satellite.inclination"),
                    id="satellite-inclination-label",
                )
                yield Input(type="number", id="satellite-inclination")
            with Horizontal(classes="satellite-input-row"):
                yield Label(
                    translate(self._ui_locale, "satellite.apsides"),
                    id="satellite-apsides-label",
                )
                yield Input(type="number", id="satellite-periapsis")
                yield Input(type="number", id="satellite-apoapsis")
            with Horizontal(classes="satellite-input-row"):
                yield Label(
                    translate(self._ui_locale, "satellite.target"),
                    id="satellite-target-label",
                )
                yield Input(id="satellite-target-body")
            with Horizontal(classes="satellite-actions"):
                yield Button(
                    translate(self._ui_locale, "satellite.set_orbit"),
                    id="satellite-set-orbit",
                )
                yield Button(
                    translate(self._ui_locale, "satellite.take_control"),
                    id="satellite-take-control",
                )
                yield Button(
                    translate(self._ui_locale, "satellite.set_apsides"),
                    id="satellite-set-apsides",
                )
                yield Button(
                    translate(self._ui_locale, "satellite.transfer"),
                    id="satellite-transfer",
                )
            with Horizontal(classes="satellite-actions"):
                yield Button(
                    translate(self._ui_locale, "satellite.return_stable"),
                    id="satellite-return-stable",
                )
            with Horizontal(classes="satellite-actions"):
                yield Button(
                    translate(self._ui_locale, "satellite.return_auto"),
                    id="satellite-return-auto",
                )
                yield Button(
                    translate(self._ui_locale, "satellite.deorbit"),
                    id="satellite-deorbit",
                    variant="warning",
                )
            with Horizontal(classes="satellite-actions"):
                yield Button(
                    translate(self._ui_locale, "satellite.ignite"),
                    id="satellite-ignite",
                    variant="primary",
                )
                yield Button(
                    translate(self._ui_locale, "satellite.shutdown"),
                    id="satellite-shutdown",
                )
            with Horizontal(classes="satellite-input-row"):
                yield Label(
                    translate(self._ui_locale, "satellite.throttle"),
                    id="satellite-throttle-label",
                )
                yield Input(type="number", id="satellite-throttle")
                yield Input(placeholder="wheel x,y,z", id="satellite-wheel")
                yield Input(placeholder="RCS x,y,z", id="satellite-rcs")
            with Horizontal(classes="satellite-actions"):
                yield Button(
                    translate(self._ui_locale, "satellite.manual_apply"),
                    id="satellite-manual-apply",
                )
            yield Static("", id="satellite-feedback")

    def set_snapshot(self, snapshot: WorldSnapshot) -> None:
        self._snapshot = snapshot
        available_ids = tuple(item.id for item in snapshot.satellites)
        if self._selected_satellite_id not in available_ids:
            self._selected_satellite_id = available_ids[0] if available_ids else None
        if self.is_mounted:
            if available_ids != self._listed_satellite_ids:
                self._refresh_list()
            self._refresh_telemetry()

    def set_locale(self, locale: Locale) -> None:
        """Update all visible panel labels without rebuilding simulation state."""
        self._ui_locale = locale
        self.border_title = translate(locale, "satellite.title")
        if not self.is_mounted:
            return
        labels = {
            "#satellite-altitude-label": "satellite.altitude",
            "#satellite-inclination-label": "satellite.inclination",
            "#satellite-apsides-label": "satellite.apsides",
            "#satellite-target-label": "satellite.target",
            "#satellite-throttle-label": "satellite.throttle",
        }
        buttons = {
            "#satellite-set-orbit": "satellite.set_orbit",
            "#satellite-take-control": "satellite.take_control",
            "#satellite-set-apsides": "satellite.set_apsides",
            "#satellite-transfer": "satellite.transfer",
            "#satellite-return-stable": "satellite.return_stable",
            "#satellite-return-auto": "satellite.return_auto",
            "#satellite-deorbit": "satellite.deorbit",
            "#satellite-ignite": "satellite.ignite",
            "#satellite-shutdown": "satellite.shutdown",
            "#satellite-manual-apply": "satellite.manual_apply",
        }
        for selector, key in labels.items():
            self.query_one(selector, Label).update(translate(locale, key))
        for selector, key in buttons.items():
            self.query_one(selector, Button).label = translate(locale, key)
        self._refresh_telemetry()

    def set_feedback(self, text: str) -> None:
        if not self.is_mounted:
            return
        try:
            self.query_one("#satellite-feedback", Static).update(sanitize_text(text))
        except NoMatches:
            return

    def on_mount(self) -> None:
        self._refresh_list()
        self._refresh_telemetry()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.item.id is None:
            return
        self._selected_satellite_id = event.item.id.removeprefix("satellite-")
        self._refresh_telemetry()
        self.post_message(
            self.Selected(SatelliteSelection(self._selected_satellite_id))
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if self._selected_satellite_id is None:
            self.set_feedback(translate(self._ui_locale, "satellite.no_active"))
            return
        command = self._command_for_button(event.button.id)
        if command is not None:
            self.post_message(
                self.CommandSubmitted(self._selected_satellite_id, command)
            )

    def _command_for_button(self, button_id: str | None) -> SatelliteCommand | None:
        if button_id == "satellite-set-orbit":
            altitude = self._number_input("#satellite-altitude")
            inclination_deg = self._number_input("#satellite-inclination")
            if altitude is not None:
                return SetOrbitAltitude(altitude)
            if inclination_deg is not None:
                return SetInclination(radians(inclination_deg))
            self.set_feedback(translate(self._ui_locale, "satellite.enter_orbit"))
            return None
        if button_id == "satellite-take-control":
            return TakeManualControl()
        if button_id == "satellite-set-apsides":
            periapsis = self._number_input("#satellite-periapsis")
            apoapsis = self._number_input("#satellite-apoapsis")
            if periapsis is None or apoapsis is None:
                self.set_feedback(translate(self._ui_locale, "satellite.enter_apsides"))
                return None
            return SetApsides(periapsis, apoapsis)
        if button_id == "satellite-transfer":
            target = self.query_one("#satellite-target-body", Input).value.strip()
            if target:
                return TransferPrimary(target)
            self.set_feedback(translate(self._ui_locale, "satellite.enter_target"))
            return None
        if button_id == "satellite-return-stable":
            return ReturnStableOrbit()
        if button_id == "satellite-return-auto":
            return ReturnAutonomousControl()
        if button_id == "satellite-deorbit":
            return Deorbit()
        if button_id == "satellite-ignite":
            return ManualActuation(1.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        if button_id == "satellite-shutdown":
            return ManualActuation(0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        if button_id == "satellite-manual-apply":
            throttle = self._number_input("#satellite-throttle") or 0.0
            wheel = self._vector_input("#satellite-wheel") or (0.0, 0.0, 0.0)
            rcs = self._vector_input("#satellite-rcs") or (0.0, 0.0, 0.0)
            return ManualActuation(throttle, wheel, rcs)
        return None

    def _number_input(self, selector: str) -> float | None:
        value = self.query_one(selector, Input).value.strip()
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    def _vector_input(self, selector: str) -> tuple[float, float, float] | None:
        values = self.query_one(selector, Input).value.strip().split(",")
        if len(values) != 3:
            return None
        try:
            vector = tuple(float(value.strip()) for value in values)
        except ValueError:
            return None
        typed_vector = (vector[0], vector[1], vector[2])
        return typed_vector if all(isfinite(value) for value in typed_vector) else None

    def _refresh_list(self) -> None:
        view = self.query_one("#satellite-list", ListView)
        view.clear()
        if self._snapshot is None:
            return
        self._listed_satellite_ids = tuple(
            satellite.id for satellite in self._snapshot.satellites
        )
        for satellite in self._snapshot.satellites:
            view.append(
                ListItem(
                    Label(sanitize_text(satellite.display_name)),
                    id=f"satellite-{satellite.id}",
                )
            )

    def _refresh_telemetry(self) -> None:
        if not self.is_mounted:
            return
        try:
            telemetry = self.query_one("#satellite-telemetry", Static)
        except NoMatches:
            return
        satellite = self._selected_satellite()
        if satellite is None:
            telemetry.update(translate(self._ui_locale, "satellite.no_active"))
            return
        snapshot = self._snapshot
        assert snapshot is not None
        recent_event = next(
            (
                event
                for event in reversed(snapshot.destruction_events)
                if satellite.id in event.satellite_ids
            ),
            None,
        )
        event_text = "-" if recent_event is None else recent_event.cause.value
        mode_text = translate(
            self._ui_locale, f"satellite.mode.{satellite.control_mode.value}"
        )
        telemetry.update(
            "\n".join(
                (
                    sanitize_text(satellite.display_name),
                    f"{translate(self._ui_locale, 'satellite.primary')}  {satellite.primary_body_id}",
                    f"{translate(self._ui_locale, 'satellite.mode')}     {mode_text}",
                    f"{translate(self._ui_locale, 'satellite.mass')}     {satellite.mass_kg:,.1f} kg",
                    f"{translate(self._ui_locale, 'satellite.main')}     {satellite.main_propellant_kg:,.1f} kg",
                    f"{translate(self._ui_locale, 'satellite.rcs')}      {satellite.rcs_propellant_kg:,.1f} kg",
                    f"{translate(self._ui_locale, 'satellite.battery')}  {satellite.battery_energy_j / 3_600_000.0:,.2f} kWh",
                    f"{translate(self._ui_locale, 'satellite.queue')}    {satellite.pending_command_count}",
                    f"{translate(self._ui_locale, 'satellite.orbit')}    p {satellite.periapsis_altitude_m or 0:,.0f} / a {satellite.apoapsis_altitude_m or 0:,.0f} m",
                    f"{translate(self._ui_locale, 'satellite.incl')}    {satellite.inclination_deg or 0:,.2f} deg",
                    f"{translate(self._ui_locale, 'satellite.plan')}     {satellite.plan_objective_id or '-'} / {satellite.execution_status or '-'}",
                    f"{translate(self._ui_locale, 'satellite.alerts')}   {', '.join(satellite.conjunction_alert_ids) or '-'}",
                    f"{translate(self._ui_locale, 'satellite.safety')}   {satellite.safety_reason or '-'}",
                    f"{translate(self._ui_locale, 'satellite.failure')}  {satellite.planning_failure or '-'}",
                    f"{translate(self._ui_locale, 'satellite.event')}    {event_text}",
                )
            )
        )

    def _selected_satellite(self) -> SatelliteSnapshot | None:
        if self._snapshot is None or self._selected_satellite_id is None:
            return None
        try:
            return self._snapshot.satellite(self._selected_satellite_id)
        except KeyError:
            return None
