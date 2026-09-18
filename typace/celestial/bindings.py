"""Keyboard bindings for celestial-system views."""

from textual.binding import Binding, BindingType

from typace.i18n import DEFAULT_LOCALE, translate

VIEW_BINDINGS: list[BindingType] = [
    Binding(
        "space", "toggle_pause", translate(DEFAULT_LOCALE, "shortcut.pause"), show=False
    ),
    Binding(
        "comma", "warp_down", translate(DEFAULT_LOCALE, "shortcut.warp"), show=False
    ),
    Binding(
        "full_stop", "warp_up", translate(DEFAULT_LOCALE, "shortcut.warp"), show=False
    ),
    Binding(
        "tab", "focus_next", translate(DEFAULT_LOCALE, "shortcut.focus"), show=False
    ),
    Binding(
        "shift+tab",
        "focus_previous",
        translate(DEFAULT_LOCALE, "shortcut.focus"),
        show=False,
    ),
    Binding("v", "cycle_view", translate(DEFAULT_LOCALE, "shortcut.view"), show=False),
    Binding("+", "zoom_in", translate(DEFAULT_LOCALE, "shortcut.zoom"), show=False),
    Binding("-", "zoom_out", translate(DEFAULT_LOCALE, "shortcut.zoom"), show=False),
    Binding(
        "g", "system_view", translate(DEFAULT_LOCALE, "shortcut.system"), show=False
    ),
    Binding("left", "pan_left", translate(DEFAULT_LOCALE, "shortcut.pan"), show=False),
    Binding(
        "right", "pan_right", translate(DEFAULT_LOCALE, "shortcut.pan"), show=False
    ),
    Binding("up", "pan_up", translate(DEFAULT_LOCALE, "shortcut.pan"), show=False),
    Binding("down", "pan_down", translate(DEFAULT_LOCALE, "shortcut.pan"), show=False),
]
