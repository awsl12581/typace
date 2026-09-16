"""Run the same Textual UI in a terminal or an SDL window."""

import argparse

from textual.app import App, ComposeResult
from textual.widgets import Button, Footer, Input, Label

from typace.config import DEFAULT_FONT
from typace.ui import WindowOptions, run


class Demo(App[None]):
    CSS = """
    Screen { align: center middle; }
    Input, Button, Label { width: 60; margin: 1 2; }
    """
    BINDINGS = [("ctrl+q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Label("typace — 同一份 Textual UI")
        yield Input(placeholder="输入文本 / Enter text", id="text")
        yield Button("Apply", id="apply", variant="primary")
        yield Label("Ready", id="status")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.query_one("#status", Label).update(self.query_one(Input).value or "Ready")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("terminal", "sdl"), default="terminal")
    parser.add_argument("--new-terminal", action="store_true")
    parser.add_argument(
        "--font",
        default=str(DEFAULT_FONT),
    )
    args = parser.parse_args()
    if args.new_terminal and args.backend != "terminal":
        parser.error("--new-terminal requires --backend terminal")
    if args.backend == "terminal":
        run(Demo(), terminal="new" if args.new_terminal else "current")
    else:
        run(
            Demo(),
            backend="sdl",
            window=WindowOptions(
                font=args.font,
            ),
        )


if __name__ == "__main__":
    main()
