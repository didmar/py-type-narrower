"""Optional thin flake8 plugin exposing Tier 1 single-file narrowing rules."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from py_type_narrower import __version__
from py_type_narrower.config import NarrowerConfig
from py_type_narrower.indexer import index_files
from py_type_narrower.rules import run_rules

if TYPE_CHECKING:
    import ast
    from collections.abc import Iterator, Sequence


class TypeNarrowerChecker:
    """Flake8 plugin running Tier 1 local type-narrowing rules (NT101, NT102, NT103)."""

    name = "py-type-narrower"
    version = __version__

    def __init__(
        self,
        tree: ast.AST,  # noqa: ARG002
        filename: str = "<stdin>",
        lines: Sequence[str] | None = None,
    ) -> None:
        """Initialize the flake8 checker with the file path and optional lines."""
        self.filename = filename
        self.lines = lines

    def run(self) -> Iterator[tuple[int, int, str, type[TypeNarrowerChecker]]]:
        """Yield flake8 `(line, col, message, checker_type)` tuples for Tier 1 findings."""
        path = Path(self.filename).resolve() if self.filename != "<stdin>" else Path("<stdin>.py")
        overrides: dict[Path, str] | None = None
        if self.lines is not None:
            overrides = {path: "".join(self.lines)}

        config = NarrowerConfig(max_tier=1, include_test_files=True)
        index = index_files([path], project_root=path.parent, overrides=overrides)
        for cand in run_rules(index, config):
            yield (cand.line, cand.col, f"{cand.code.value} {cand.message}", type(self))
