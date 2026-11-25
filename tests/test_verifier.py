import json
from pathlib import Path

import pytest

from py_type_narrower.config import NarrowerConfig
from py_type_narrower.indexer import index_files
from py_type_narrower.models import CheckerDiagnostic, RuleCode, VerificationStatus
from py_type_narrower.rules import run_rules
from py_type_narrower.verifier import (
    CustomCommandAdapter,
    TyAdapter,
    _parse_mypy_output,
    _parse_pyright_json,
    _parse_ty_output,
    resolve_verifier,
    verify_candidates,
)


class _FakeAdapter:
    """Deterministic fake type checker that flags a specific forbidden annotation."""

    name = "fake"

    def __init__(self, forbidden_substring: str) -> None:
        self._forbidden = forbidden_substring

    def is_available(self) -> bool:
        return True

    def check(self, workdir: Path, python_exe: Path | None = None) -> set[CheckerDiagnostic]:
        _ = python_exe
        diags: set[CheckerDiagnostic] = set()
        for py_file in workdir.rglob("*.py"):
            text = py_file.read_text(encoding="utf-8")
            if self._forbidden in text:
                diags.add(
                    CheckerDiagnostic(
                        file=str(py_file.relative_to(workdir)),
                        line=1,
                        col=1,
                        severity="error",
                        rule="invalid-argument-type",
                        message=f"Forbidden edit detected: {self._forbidden}",
                    )
                )
        return diags


def test_parse_ty_mypy_pyright_outputs(tmp_path: Path) -> None:
    ty_out = "pkg/a.py:12:5: error[invalid-argument-type] Argument of type `str` is not assignable"
    ty_diags = _parse_ty_output(ty_out, tmp_path)
    assert len(ty_diags) == 1
    diag = next(iter(ty_diags))
    assert (diag.file, diag.line, diag.col, diag.rule) == (
        "pkg/a.py",
        12,
        5,
        "invalid-argument-type",
    )

    mypy_out = "pkg/a.py:8:3: error: Incompatible types [arg-type]"
    mypy_diags = _parse_mypy_output(mypy_out, tmp_path)
    assert len(mypy_diags) == 1
    mdiag = next(iter(mypy_diags))
    assert (mdiag.file, mdiag.line, mdiag.col, mdiag.rule) == ("pkg/a.py", 8, 3, "arg-type")

    pyright_payload = json.dumps(
        {
            "generalDiagnostics": [
                {
                    "file": str(tmp_path / "pkg/a.py"),
                    "severity": "error",
                    "rule": "reportArgumentType",
                    "message": "Type str cannot be assigned",
                    "range": {"start": {"line": 4, "character": 9}},
                }
            ]
        }
    )
    pr_diags = _parse_pyright_json(pyright_payload, tmp_path)
    assert len(pr_diags) == 1
    pdiag = next(iter(pr_diags))
    assert (pdiag.file, pdiag.line, pdiag.col, pdiag.rule) == (
        "pkg/a.py",
        5,
        10,
        "reportArgumentType",
    )


def test_bisect_verify_isolates_bad_candidate(tmp_path: Path) -> None:
    mod = tmp_path / "mod.py"
    mod.write_text(
        """
def safe_fn(flag: bool) -> str | None:
    if flag:
        return "a"
    return "b"

def guarded_fn(mode: str) -> int:
    if mode == "fast":
        return 1
    elif mode == "slow":
        return 2
    else:
        raise ValueError(mode)
""",
        encoding="utf-8",
    )
    index = index_files([mod], project_root=tmp_path)
    cands = run_rules(index, NarrowerConfig(max_tier=1))
    assert len(cands) == 3

    adapter = _FakeAdapter('Literal["fast", "slow"]')
    verified, rejected = verify_candidates(cands, index.files, tmp_path, adapter)

    assert {c.code for c in verified} == {RuleCode.NT102, RuleCode.NT103}
    assert all(c.verification == VerificationStatus.VERIFIED for c in verified)

    assert len(rejected) == 1
    assert rejected[0].code == RuleCode.NT101
    assert rejected[0].verification == VerificationStatus.REJECTED
    assert rejected[0].blocking_diagnostics


def test_real_ty_soundness_verification(tmp_path: Path) -> None:
    adapter = TyAdapter()
    if not adapter.is_available():
        pytest.skip("ty is not installed in PATH")

    mod = tmp_path / "app.py"
    original = """def safe_return(flag: bool) -> str | None:
    if flag:
        return "yes"
    return "no"

def guarded(mode: str) -> int:
    if mode == "a":
        return 1
    elif mode == "b":
        return 2
    else:
        raise ValueError(mode)

def dynamic_caller(raw: str) -> int:
    # Passing an arbitrary str breaks narrowing `mode` to Literal["a", "b"]!
    return guarded(raw)
"""
    mod.write_text(original, encoding="utf-8")

    index = index_files([mod], project_root=tmp_path)
    cands = run_rules(index, NarrowerConfig(max_tier=1))
    assert {c.code for c in cands} == {RuleCode.NT101, RuleCode.NT102, RuleCode.NT103}

    verified, rejected = verify_candidates(cands, index.files, tmp_path, adapter)

    # `safe_return` (NT102) and `guarded`'s return type (NT103) are both sound!
    assert {c.code for c in verified} == {RuleCode.NT102, RuleCode.NT103}
    # `guarded`'s param narrowing (NT101) breaks `dynamic_caller(raw: str)` and MUST be rejected by ty!
    assert [c.code for c in rejected] == [RuleCode.NT101]
    assert rejected[0].blocking_diagnostics
    # Real working file must remain completely untouched.
    assert mod.read_text(encoding="utf-8") == original


def test_resolve_verifier_errors_and_custom(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown verifier"):
        resolve_verifier("nonexistent", tmp_path)

    with pytest.raises(ValueError, match="requires `--verify-command`"):
        resolve_verifier("custom", tmp_path)

    custom = resolve_verifier("custom", tmp_path, custom_command="true")
    assert isinstance(custom, CustomCommandAdapter)
    assert custom.check(tmp_path) == set()
