"""Start the application in a terminal or SDL window."""

import argparse

from typace.config import DEFAULT_FALLBACK_FONTS, DEFAULT_FONT
from typace.application import TyPaceApp
from typace.ui import WindowOptions, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("terminal", "sdl"), default="terminal")
    args = parser.parse_args()

    application = TyPaceApp()
    if args.backend == "terminal":
        run(application)
        return

    run(
        application,
        backend="sdl",
        window=WindowOptions(
            font=str(DEFAULT_FONT),
            fallback_fonts=DEFAULT_FALLBACK_FONTS,
            title="typace - Solar System",
        ),
    )


if __name__ == "__main__":
    main()
