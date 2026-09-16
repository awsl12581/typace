"""Application-wide keyboard bindings."""

from textual.binding import Binding, BindingType

from typace.i18n import DEFAULT_LOCALE, translate

APP_BINDINGS: list[BindingType] = [
    Binding("ctrl+q", "quit", translate(DEFAULT_LOCALE, "app.quit"), show=False),
    Binding(
        "left_square_bracket",
        "previous_panel",
        translate(DEFAULT_LOCALE, "app.previous_panel"),
        show=False,
    ),
    Binding(
        "right_square_bracket",
        "next_panel",
        translate(DEFAULT_LOCALE, "app.next_panel"),
        show=False,
    ),
    Binding(
        "escape",
        "open_settings",
        translate(DEFAULT_LOCALE, "app.settings"),
        show=False,
    ),
]

SETTINGS_BINDINGS: list[BindingType] = [
    Binding(
        "escape",
        "cancel",
        translate(DEFAULT_LOCALE, "settings.cancel"),
        show=False,
    )
]
