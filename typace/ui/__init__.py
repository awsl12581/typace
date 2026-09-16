"""Run an ordinary Textual application in a terminal or an SDL window."""

from .config import WindowOptions
from .runner import run

__all__ = ["WindowOptions", "run"]
