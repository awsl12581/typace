"""Options for the SDL window; independent of backend implementation."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WindowOptions:
    font: str
    fallback_fonts: tuple[str, ...] = ()
    title: str = "typace"
    width: int = 1000
    height: int = 600
    font_size: int = 18

    def __post_init__(self) -> None:
        if min(self.width, self.height, self.font_size) <= 0:
            raise ValueError("window dimensions and font_size must be positive")
