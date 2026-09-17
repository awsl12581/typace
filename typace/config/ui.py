"""Window and font configuration for the Textual backends."""

from pathlib import Path

DEFAULT_FONT = Path(__file__).resolve().parents[2] / "assets/fonts/seguisym.ttf"

_SYSTEM_CJK_FONT_CANDIDATES = (
    Path("/System/Library/Fonts/STHeiti Medium.ttc"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
)
DEFAULT_FALLBACK_FONTS = tuple(
    str(path) for path in _SYSTEM_CJK_FONT_CANDIDATES if path.is_file()
)
