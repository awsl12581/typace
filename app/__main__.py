"""Start the application in a terminal or SDL window."""

import argparse
from pathlib import Path

from typace.application import TyPaceApp
from typace.config.ui import DEFAULT_FALLBACK_FONTS, DEFAULT_FONT
from typace.privacy import sanitize_text
from typace.ui import WindowOptions, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("terminal", "sdl"), default="terminal")
    parser.add_argument(
        "--satellite-catalog",
        action="append",
        type=Path,
        default=[],
        help="local satellite JSON file or directory; repeat to add catalogs",
    )
    args = parser.parse_args()

    try:
        application = TyPaceApp(tuple(args.satellite_catalog))
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
    except Exception as error:
        raise SystemExit(sanitize_text(str(error))) from None


if __name__ == "__main__":
    main()
