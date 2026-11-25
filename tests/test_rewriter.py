from pathlib import Path

from py_type_narrower.config import NarrowerConfig
from py_type_narrower.indexer import index_files
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
