"""Inline suppression comment parsing (`# type-narrow: ignore[CODE]`)."""

from __future__ import annotations

import re
from dataclasses import dataclass

IGNORE_PATTERN = re.compile(
    r"#\s*type-narrow:\s*ignore(?:\s*\[\s*([A-Za-z0-9_,\s]+)\s*\])?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LineIgnore:
    """Suppression directive found on a single source line."""

    ignore_all: bool
    codes: frozenset[str]

    def suppresses(self, code: str) -> bool:
        """Return True if this directive suppresses the given rule code."""
        return self.ignore_all or code.upper() in self.codes


def parse_file_ignores(source: str) -> dict[int, LineIgnore]:
    """Parse all inline `# type-narrow: ignore` directives in a source string.

    Returns a mapping from 1-based line number to the parsed directive.
    """
    ignores: dict[int, LineIgnore] = {}
    for lineno, line in enumerate(source.splitlines(), start=1):
        directive = _parse_line_ignore(line)
        if directive is not None:
            ignores[lineno] = directive
    return ignores


def is_ignored(ignores: dict[int, LineIgnore], line: int, code: str) -> bool:
    """Return True if a diagnostic at the given 1-based line and rule code is suppressed."""
    directive = ignores.get(line)
    if directive is not None and directive.suppresses(code):
        return True
    # Also support a comment placed directly on the line above a def/parameter.
    prev_directive = ignores.get(line - 1)
    return prev_directive is not None and prev_directive.suppresses(code)


def _parse_line_ignore(line: str) -> LineIgnore | None:
    """Extract a LineIgnore from a single line of Python source code."""
    match = IGNORE_PATTERN.search(line)
    if match is None:
        return None
    raw_codes = match.group(1)
    if raw_codes is None:
        return LineIgnore(ignore_all=True, codes=frozenset())
    codes = frozenset(part.strip().upper() for part in raw_codes.split(",") if part.strip())
    if not codes:
        return LineIgnore(ignore_all=True, codes=frozenset())
    return LineIgnore(ignore_all=False, codes=codes)
