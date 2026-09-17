"""Sanitize user-visible diagnostics at one output boundary."""

from pathlib import Path
import re
from urllib.parse import urlsplit, urlunsplit

_EMAIL_PATTERN = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
_POSIX_ABSOLUTE_PATH_PATTERN = re.compile(r"(?<![\w.])/(?:[^\s/:]+/)+[^\s:]*")
_WINDOWS_ABSOLUTE_PATH_PATTERN = re.compile(r"(?i)(?<![\w])(?:[a-z]:\\|\\\\)[^\s:]+")
_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)\b(token|secret|password|api[_-]?key)=([^\s;]+)"
)


def safe_catalog_location(alias: str, relative_path: Path) -> str:
    """Return a stable catalog label without exposing its filesystem root."""

    normalized = relative_path.as_posix().lstrip("/")
    return f"{alias}:{normalized or '<root>'}"


def sanitize_url(value: str) -> str:
    """Remove user information, query parameters, and fragments from a URL."""

    parts = urlsplit(value)
    hostname = parts.hostname or ""
    return urlunsplit((parts.scheme, hostname, parts.path, "", ""))


def sanitize_text(value: str) -> str:
    """Remove machine-specific paths and common sensitive fields."""

    sanitized = _EMAIL_PATTERN.sub("<redacted-email>", value)
    sanitized = _WINDOWS_ABSOLUTE_PATH_PATTERN.sub("<redacted-path>", sanitized)
    sanitized = _POSIX_ABSOLUTE_PATH_PATTERN.sub("<redacted-path>", sanitized)
    return _SECRET_ASSIGNMENT_PATTERN.sub(r"\1=<redacted>", sanitized)
