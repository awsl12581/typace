"""Native Textual satellite list, telemetry, and control panel."""

from dataclasses import dataclass
from math import radians

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Input, Label, ListItem, ListView, Static

from typace.privacy import sanitize_text
from typace.satellites.commands import (
    Deorbit,
    ManualActuation,
    ReturnAutonomousControl,
    SatelliteCommand,
    SetInclination,
    SetOrbitAltitude,
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
    SatellitePanel { width: 38; height: 100%; background: #071019; border-left: solid #37cdf5; }
    #satellite-list { height: 9; border-bottom: solid #264b5c; }
    #satellite-telemetry { height: 10; padding: 0 1; color: #c8d7e8; }
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

    def __init__(self) -> None:
        super().__init__(id="satellite-panel")
        self._snapshot: WorldSnapshot | None = None
        self._selected_satellite_id: str | None = None

    def compose(self) -> ComposeResult:
        with Vertical():
            yield ListView(id="satellite-list")
            yield Static("", id="satellite-telemetry")
            with Horizontal(classes="satellite-input-row"):
                yield Label("Altitude m")
                yield Input(type="number", id="satellite-altitude")
            with Horizontal(classes="satellite-input-row"):
                yield Label("Inclination")
                yield Input(type="number", id="satellite-inclination")
            with Horizontal(classes="satellite-actions"):
                yield Button("Set orbit", id="satellite-set-orbit")
                yield Button("Take control", id="satellite-take-control")
            with Horizontal(classes="satellite-actions"):
                yield Button("Return auto", id="satellite-return-auto")
                yield Button("Deorbit", id="satellite-deorbit", variant="warning")
            with Horizontal(classes="satellite-actions"):
                yield Button("Ignite", id="satellite-ignite", variant="primary")
                yield Button("Shutdown", id="satellite-shutdown")
            yield Static("", id="satellite-feedback")

    def set_snapshot(self, snapshot: WorldSnapshot) -> None:
        self._snapshot = snapshot
        available_ids = tuple(item.id for item in snapshot.satellites)
        if self._selected_satellite_id not in available_ids:
            self._selected_satellite_id = available_ids[0] if available_ids else None
        if self.is_mounted:
            self._refresh_list()
            self._refresh_telemetry()

    def set_feedback(self, text: str) -> None:
        self.query_one("#satellite-feedback", Static).update(sanitize_text(text))

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
            self.set_feedback("No active satellite")
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
            self.set_feedback("Enter altitude or inclination")
            return None
        if button_id == "satellite-take-control":
            return TakeManualControl()
        if button_id == "satellite-return-auto":
            return ReturnAutonomousControl()
        if button_id == "satellite-deorbit":
            return Deorbit()
        if button_id == "satellite-ignite":
            return ManualActuation(1.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        if button_id == "satellite-shutdown":
            return ManualActuation(0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        return None

    def _number_input(self, selector: str) -> float | None:
        value = self.query_one(selector, Input).value.strip()
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    def _refresh_list(self) -> None:
        view = self.query_one("#satellite-list", ListView)
        view.clear()
        if self._snapshot is None:
            return
        for satellite in self._snapshot.satellites:
            view.append(
                ListItem(
                    Label(sanitize_text(satellite.display_name)),
                    id=f"satellite-{satellite.id}",
                )
            )

    def _refresh_telemetry(self) -> None:
        telemetry = self.query_one("#satellite-telemetry", Static)
        satellite = self._selected_satellite()
        if satellite is None:
            telemetry.update("No active satellite")
            return
        telemetry.update(
            "\n".join(
                (
                    sanitize_text(satellite.display_name),
                    f"Primary  {satellite.primary_body_id}",
                    f"Mode     {satellite.control_mode.value}",
                    f"Mass     {satellite.mass_kg:,.1f} kg",
                    f"Main     {satellite.main_propellant_kg:,.1f} kg",
                    f"RCS      {satellite.rcs_propellant_kg:,.1f} kg",
                    f"Battery  {satellite.battery_energy_j / 3_600_000.0:,.2f} kWh",
                    f"Queue    {satellite.pending_command_count}",
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
