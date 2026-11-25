from pathlib import Path

from py_type_narrower.config import NarrowerConfig
from py_type_narrower.indexer import index_files
from py_type_narrower.models import NarrowingCandidate, RuleCode, TargetKind, Tier
from py_type_narrower.rewriter import compute_unified_diffs, rewrite_files, rewrite_module_source
from py_type_narrower.rules import run_rules


def test_rewriter_inserts_literal_import_after_docstring_and_future(tmp_path: Path) -> None:
    source = '''"""Module docstring."""

from __future__ import annotations

def choose(mode: str) -> int:
    if mode == "a":
        return 1
    elif mode == "b":
        return 2
    else:
        raise ValueError(mode)
'''
    file_path = tmp_path / "mod.py"
    file_path.write_text(source, encoding="utf-8")
    index = index_files([file_path], project_root=tmp_path)
    cands = run_rules(index, NarrowerConfig(max_tier=1))

    updated = rewrite_module_source(source, cands)
    expected = '''"""Module docstring."""

from __future__ import annotations
from typing import Literal

def choose(mode: Literal["a", "b"]) -> Literal[1, 2]:
    if mode == "a":
        return 1
    elif mode == "b":
        return 2
    else:
        raise ValueError(mode)
'''
    assert updated == expected

    # Idempotency: re-running analysis on the rewritten source finds zero candidates.
    index2 = index_files([file_path], project_root=tmp_path, overrides={file_path: updated})
    assert run_rules(index2, NarrowerConfig(max_tier=1)) == []


def test_rewriter_merges_existing_typing_import(tmp_path: Path) -> None:
    source = """from typing import Any

def _status(ok: bool) -> str:
    if ok:
        return "ok"
    return "err"
"""
    file_path = tmp_path / "mod.py"
    file_path.write_text(source, encoding="utf-8")
    index = index_files([file_path], project_root=tmp_path)
    cands = run_rules(index, NarrowerConfig(max_tier=1))

    updated = rewrite_module_source(source, cands)
    assert "from typing import Any, Literal" in updated
    assert '-> Literal["err", "ok"]:' in updated


def test_rewriter_drops_unused_none_default_and_diffs(tmp_path: Path) -> None:
    source = """def _run(limit: int | None = None) -> int:
    return limit or 0

def main() -> None:
    _run(5)
    _run(10)
"""
    file_path = tmp_path / "mod.py"
    file_path.write_text(source, encoding="utf-8")
    index = index_files([file_path], project_root=tmp_path)
    cands = run_rules(index, NarrowerConfig(max_tier=2))

    updated_map = rewrite_files(cands, index.files)
    assert file_path in updated_map
    assert "def _run(limit: int) -> int:" in updated_map[file_path]

    diffs = compute_unified_diffs(index.files, updated_map, tmp_path)
    assert file_path in diffs
    assert "-def _run(limit: int | None = None) -> int:" in diffs[file_path]
    assert "+def _run(limit: int) -> int:" in diffs[file_path]


def test_rewriter_and_rules_preserve_valid_default_param_ordering(tmp_path: Path) -> None:
    source = """from dataclasses import dataclass

def _helper(a: int = 1, b: str | None = None, *, c: str | None = None) -> str:
    return (b or "") + (c or str(a))

@dataclass
class _Cfg:
    a: int = 1
    b: str | None = None

def main() -> None:
    _helper(1, "x", c="y")
    _helper(2, "z", c="w")
    _Cfg(1, "x")
    _Cfg(2, "y")
"""
    file_path = tmp_path / "mod.py"
    file_path.write_text(source, encoding="utf-8")
    index = index_files([file_path], project_root=tmp_path)
    cands = run_rules(index, NarrowerConfig(max_tier=2))

    # Positional `b` after defaulted `a` is skipped, while kw-only `c` can still drop `= None`.
    names = {(c.code, c.target_name) for c in cands}
    assert (RuleCode.NT202, "b") not in names
    assert (RuleCode.NT203, "b") not in names
    assert (RuleCode.NT202, "c") in names

    updated = rewrite_module_source(source, cands)
    assert "b: str | None = None, *, c: str) -> str:" in updated

    # Even if a direct caller forces drop_none_default=True on `b`, rewriter never raises CSTValidationError.
    forced = NarrowingCandidate(
        code=RuleCode.NT202,
        tier=Tier.WHOLE_REPO,
        file=file_path,
        line=3,
        col=24,
        symbol="mod._helper",
        target_kind=TargetKind.PARAMETER,
        target_name="b",
        current_type="str | None",
        suggested_type="str",
        needs_literal_import=False,
        message="forced",
        drop_none_default=True,
    )
    safe_out = rewrite_module_source(source, [forced])
    assert "b: str = None" in safe_out
