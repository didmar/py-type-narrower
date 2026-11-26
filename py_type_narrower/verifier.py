"""Configurable type-checker soundness verifier with shadow workspace and batch bisection."""

from __future__ import annotations

import json
import logging
import re
import shlex
import shutil
import subprocess
import tempfile
import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable

from py_type_narrower.models import (
    CheckerDiagnostic,
    NarrowingCandidate,
    VerificationStatus,
)
from py_type_narrower.rewriter import rewrite_files

logger = logging.getLogger(__name__)

TY_CONCISE_RE = re.compile(
    r"^(?P<file>.+?):(?P<line>\d+):(?P<col>\d+):\s+(?P<sev>error|warning|info)"
    r"(?:\[(?P<rule>[^\]]+)\])?\s+(?P<msg>.+)$"
)
MYPY_LINE_RE = re.compile(
    r"^(?P<file>.+?):(?P<line>\d+):(?:(?P<col>\d+):)?\s+(?P<sev>error|warning|note):\s+"
    r"(?P<msg>.+?)(?:\s+\[(?P<rule>[^\]]+)\])?$"
)


class TypeCheckerAdapter(Protocol):
    """Protocol implemented by every supported type-checker backend."""

    name: str

    def is_available(self) -> bool:
        """Return True if the type checker executable is available."""
        ...

    def check(self, workdir: Path, python_exe: Path | None = None) -> set[CheckerDiagnostic]:
        """Run the type checker against workdir and return normalized diagnostics."""
        ...


@dataclass(frozen=True)
class _BisectContext:
    """Shared verification state threaded through recursive bisection."""

    sources: dict[Path, str]
    project_root: Path
    adapter: TypeCheckerAdapter
    python_exe: Path | None
    baseline_counts: Counter[tuple[str, str, str]]


def verify_candidates(
    candidates: list[NarrowingCandidate],
    sources: dict[Path, str],
    project_root: Path,
    adapter: TypeCheckerAdapter,
) -> tuple[list[NarrowingCandidate], list[NarrowingCandidate]]:
    """Verify narrowing candidates against a type-checker baseline using batch bisection.

    Returns `(verified_candidates, rejected_candidates)` and mutates each candidate's
    `verification` status and `blocking_diagnostics`.
    """
    if not candidates:
        return ([], [])

    python_exe = _detect_project_python(project_root)
    source_paths = tuple(sources.keys())
    baseline = _run_check_in_shadow(
        project_root, {}, adapter, python_exe, source_paths=source_paths
    )
    ctx = _BisectContext(
        sources=sources,
        project_root=project_root,
        adapter=adapter,
        python_exe=python_exe,
        baseline_counts=Counter(
            (d.file, d.rule, d.message) for d in baseline if d.severity == "error"
        ),
    )

    verified: list[NarrowingCandidate] = []
    rejected: list[NarrowingCandidate] = []
    _bisect_verify(candidates, ctx, verified, rejected)
    return (verified, rejected)


def _bisect_verify(
    candidates: list[NarrowingCandidate],
    ctx: _BisectContext,
    verified: list[NarrowingCandidate],
    rejected: list[NarrowingCandidate],
) -> None:
    """Verify a batch of candidates; bisect recursively on any newly introduced errors."""
    patched = rewrite_files(candidates, ctx.sources)
    diags = _run_check_in_shadow(
        ctx.project_root,
        patched,
        ctx.adapter,
        ctx.python_exe,
        source_paths=ctx.sources.keys(),
    )
    new_errors = _find_new_errors(diags, ctx.baseline_counts)

    if not new_errors:
        for cand in candidates:
            cand.verification = VerificationStatus.VERIFIED
            verified.append(cand)
        return

    if len(candidates) == 1:
        cand = candidates[0]
        cand.verification = VerificationStatus.REJECTED
        cand.blocking_diagnostics = sorted({d.format_short() for d in new_errors})[:5]
        rejected.append(cand)
        return

    mid = len(candidates) // 2
    _bisect_verify(candidates[:mid], ctx, verified, rejected)
    _bisect_verify(candidates[mid:], ctx, verified, rejected)


def _find_new_errors(
    diags: set[CheckerDiagnostic],
    baseline_counts: Counter[tuple[str, str, str]],
) -> list[CheckerDiagnostic]:
    """Return error diagnostics exceeding baseline counts (tolerant of import line shifts)."""
    remaining = Counter(baseline_counts)
    new_errors: list[CheckerDiagnostic] = []
    for diag in sorted(diags, key=lambda d: (d.file, d.line, d.col, d.rule, d.message)):
        if diag.severity != "error":
            continue
        key = (diag.file, diag.rule, diag.message)
        if remaining[key] > 0:
            remaining[key] -= 1
        else:
            new_errors.append(diag)
    return new_errors


def _run_check_in_shadow(
    project_root: Path,
    patched_files: dict[Path, str],
    adapter: TypeCheckerAdapter,
    python_exe: Path | None,
    *,
    source_paths: Iterable[Path] = (),
) -> set[CheckerDiagnostic]:
    """Materialize a lightweight symlinked shadow workspace and invoke the type checker."""
    root = project_root.resolve()
    with tempfile.TemporaryDirectory(prefix="py_type_narrower_") as tmp:
        shadow_root = Path(tmp)
        _populate_shadow_tree(root, shadow_root, patched_files, source_paths)
        return adapter.check(shadow_root, python_exe=python_exe)


def _populate_shadow_tree(
    source_root: Path,
    shadow_root: Path,
    patched_files: dict[Path, str],
    source_paths: Iterable[Path] = (),
) -> None:
    """Symlink unmodified entries and materialize parent dirs for all analyzed & patched files."""
    rel_patched: dict[Path, str] = {}
    dirs_to_materialize: set[Path] = {Path()}
    for abs_path in (*source_paths, *patched_files.keys()):
        try:
            rel = abs_path.resolve().relative_to(source_root)
        except ValueError:
            continue
        curr = rel.parent
        while curr != Path():
            dirs_to_materialize.add(curr)
            curr = curr.parent
    for abs_path, content in patched_files.items():
        try:
            rel = abs_path.resolve().relative_to(source_root)
        except ValueError:
            continue
        rel_patched[rel] = content

    _mirror_directory(source_root, shadow_root, Path(), dirs_to_materialize, rel_patched)


def _mirror_directory(
    source_root: Path,
    shadow_root: Path,
    rel_dir: Path,
    dirs_to_materialize: set[Path],
    rel_patched: dict[Path, str],
) -> None:
    """Recursively symlink unchanged entries and materialize directories containing patched files."""
    src_dir = source_root / rel_dir
    dst_dir = shadow_root / rel_dir
    dst_dir.mkdir(parents=True, exist_ok=True)

    for entry in src_dir.iterdir():
        rel_entry = rel_dir / entry.name
        dst_entry = dst_dir / entry.name
        if rel_entry in rel_patched:
            dst_entry.write_text(rel_patched[rel_entry], encoding="utf-8")
        elif entry.is_dir() and not entry.is_symlink() and rel_entry in dirs_to_materialize:
            _mirror_directory(source_root, shadow_root, rel_entry, dirs_to_materialize, rel_patched)
        else:
            dst_entry.symlink_to(entry)


def _detect_project_python(project_root: Path) -> Path | None:
    """Locate `.venv/bin/python` under project_root without resolving the venv symlink."""
    root = project_root.resolve()
    for rel in (".venv/bin/python", "venv/bin/python"):
        candidate = root / rel
        if candidate.exists():
            return candidate
    return None


class TyAdapter:
    """Soundness verifier backend powered by Astral's `ty`."""

    name = "ty"

    def is_available(self) -> bool:
        """Return True if `ty` is on PATH."""
        return shutil.which("ty") is not None

    def check(self, workdir: Path, python_exe: Path | None = None) -> set[CheckerDiagnostic]:
        """Run `ty check` with concise output and parse diagnostics."""
        cmd = [
            "ty",
            "check",
            "--project",
            str(workdir),
            "--output-format",
            "concise",
            "--color",
            "never",
            "--no-progress",
        ]
        if python_exe is not None:
            cmd.extend(["--python", str(python_exe)])
        proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, check=False)  # noqa: S603
        return _parse_ty_output(proc.stdout + "\n" + proc.stderr, workdir)


class MypyAdapter:
    """Soundness verifier backend powered by `mypy`."""

    name = "mypy"

    def is_available(self) -> bool:
        """Return True if `mypy` is on PATH."""
        return shutil.which("mypy") is not None

    def check(self, workdir: Path, python_exe: Path | None = None) -> set[CheckerDiagnostic]:
        """Run `mypy` and parse diagnostics."""
        cmd = [
            "mypy",
            "--show-column-numbers",
            "--no-error-summary",
            "--no-color-output",
            "--show-error-codes",
        ]
        if python_exe is not None:
            cmd.extend(["--python-executable", str(python_exe)])
        cmd.append(".")
        proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, check=False)  # noqa: S603
        return _parse_mypy_output(proc.stdout + "\n" + proc.stderr, workdir)


class PyrightAdapter:
    """Soundness verifier backend powered by `pyright` or `basedpyright`."""

    def __init__(self, executable: str = "pyright") -> None:
        """Initialize the adapter with the pyright or basedpyright binary name."""
        self.name = executable

    def is_available(self) -> bool:
        """Return True if the pyright executable is on PATH."""
        return shutil.which(self.name) is not None

    def check(self, workdir: Path, python_exe: Path | None = None) -> set[CheckerDiagnostic]:
        """Run pyright/basedpyright with `--outputjson` and parse diagnostics."""
        cmd = [self.name, "--outputjson"]
        if python_exe is not None:
            cmd.extend(["--pythonpath", str(python_exe)])
        proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, check=False)  # noqa: S603
        return _parse_pyright_json(proc.stdout, workdir)


class CustomCommandAdapter:
    """Soundness verifier backend running an arbitrary user-configured command."""

    def __init__(self, command: str) -> None:
        """Initialize the adapter by parsing the custom command string."""
        self.name = f"custom({command})"
        self._argv = shlex.split(command)

    def is_available(self) -> bool:
        """Return True if the custom command has an executable token."""
        return bool(self._argv) and shutil.which(self._argv[0]) is not None

    def check(self, workdir: Path, python_exe: Path | None = None) -> set[CheckerDiagnostic]:  # noqa: ARG002
        """Execute the custom command in workdir and treat non-empty output lines on failure as errors."""
        proc = subprocess.run(self._argv, cwd=workdir, capture_output=True, text=True, check=False)  # noqa: S603
        if proc.returncode == 0:
            return set()
        combined = proc.stdout + "\n" + proc.stderr
        parsed = _parse_ty_output(combined, workdir) | _parse_mypy_output(combined, workdir)
        if parsed:
            return parsed
        lines = [line.strip() for line in combined.splitlines() if line.strip()]
        return {
            CheckerDiagnostic(file="<custom>", line=0, col=0, severity="error", message=line)
            for line in lines
        }


def resolve_verifier(
    verifier_name: str,
    project_root: Path,
    custom_command: str | None = None,
) -> TypeCheckerAdapter:
    """Resolve a TypeCheckerAdapter by name ('auto', 'ty', 'mypy', 'pyright', 'basedpyright', 'custom')."""
    normalized = verifier_name.strip().lower()
    if normalized == "custom":
        if not custom_command:
            msg = "`--verifier custom` requires `--verify-command` or `verify-command` in pyproject.toml"
            raise ValueError(msg)
        return CustomCommandAdapter(custom_command)

    if normalized == "auto":
        return _auto_detect_adapter(project_root, custom_command)

    adapters: dict[str, TypeCheckerAdapter] = {
        "ty": TyAdapter(),
        "mypy": MypyAdapter(),
        "pyright": PyrightAdapter("pyright"),
        "basedpyright": PyrightAdapter("basedpyright"),
    }
    adapter = adapters.get(normalized)
    if adapter is None:
        supported = ", ".join(["auto", *adapters.keys(), "custom"])
        msg = f"Unknown verifier `{verifier_name}`. Supported: {supported}"
        raise ValueError(msg)
    if not adapter.is_available():
        msg = f"Requested type checker `{adapter.name}` was not found in PATH"
        raise RuntimeError(msg)
    return adapter


def _auto_detect_adapter(
    project_root: Path,
    custom_command: str | None,
) -> TypeCheckerAdapter:
    """Auto-detect the best available type checker from pyproject.toml or PATH."""
    if custom_command:
        return CustomCommandAdapter(custom_command)

    preferred_order = _detect_configured_checkers(project_root)
    fallback_order = ["ty", "pyright", "basedpyright", "mypy"]
    seen: set[str] = set()
    ordered_names: list[str] = []
    for name in [*preferred_order, *fallback_order]:
        if name not in seen:
            seen.add(name)
            ordered_names.append(name)

    registry: dict[str, TypeCheckerAdapter] = {
        "ty": TyAdapter(),
        "pyright": PyrightAdapter("pyright"),
        "basedpyright": PyrightAdapter("basedpyright"),
        "mypy": MypyAdapter(),
    }
    for name in ordered_names:
        adapter = registry[name]
        if adapter.is_available():
            logger.debug("Auto-detected soundness verifier: %s", adapter.name)
            return adapter

    msg = (
        "No supported type checker found in PATH (tried: ty, pyright, basedpyright, mypy). "
        "Install one or pass `--verifier custom --verify-command '...'`."
    )
    raise RuntimeError(msg)


def _detect_configured_checkers(project_root: Path) -> list[str]:
    """Inspect pyproject.toml `[tool.*]` sections to see which type checkers a project configures."""
    pyproject = project_root / "pyproject.toml"
    if not pyproject.is_file():
        return []
    try:
        with pyproject.open("rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return []

    tool = data.get("tool", {})
    if not isinstance(tool, dict):
        return []
    return [name for name in ("ty", "basedpyright", "pyright", "mypy") if name in tool]


def _parse_ty_output(output: str, workdir: Path) -> set[CheckerDiagnostic]:
    """Parse `ty check --output-format concise` lines."""
    diags: set[CheckerDiagnostic] = set()
    for line in output.splitlines():
        match = TY_CONCISE_RE.match(line.strip())
        if match is None:
            continue
        diags.add(
            CheckerDiagnostic(
                file=_normalize_diag_path(match.group("file"), workdir),
                line=int(match.group("line")),
                col=int(match.group("col")),
                severity=match.group("sev"),
                rule=match.group("rule") or "",
                message=match.group("msg").strip(),
            )
        )
    return diags


def _parse_mypy_output(output: str, workdir: Path) -> set[CheckerDiagnostic]:
    """Parse standard mypy diagnostic output lines."""
    diags: set[CheckerDiagnostic] = set()
    for line in output.splitlines():
        match = MYPY_LINE_RE.match(line.strip())
        if match is None:
            continue
        diags.add(
            CheckerDiagnostic(
                file=_normalize_diag_path(match.group("file"), workdir),
                line=int(match.group("line")),
                col=int(match.group("col") or 0),
                severity=match.group("sev"),
                rule=match.group("rule") or "",
                message=match.group("msg").strip(),
            )
        )
    return diags


def _parse_pyright_json(raw_json: str, workdir: Path) -> set[CheckerDiagnostic]:
    """Parse JSON emitted by `pyright --outputjson` or `basedpyright --outputjson`."""
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError:
        return set()
    diags: set[CheckerDiagnostic] = set()
    for item in payload.get("generalDiagnostics", []):
        if not isinstance(item, dict):
            continue
        start = item.get("range", {}).get("start", {})
        diags.add(
            CheckerDiagnostic(
                file=_normalize_diag_path(str(item.get("file", "")), workdir),
                line=int(start.get("line", 0)) + 1,
                col=int(start.get("character", 0)) + 1,
                severity=str(item.get("severity", "error")),
                rule=str(item.get("rule", "")),
                message=str(item.get("message", "")).strip(),
            )
        )
    return diags


def _normalize_diag_path(raw_path: str, workdir: Path) -> str:
    """Normalize a diagnostic file path relative to workdir."""
    p = Path(raw_path)
    if p.is_absolute():
        try:
            return str(p.resolve().relative_to(workdir.resolve()))
        except ValueError:
            return raw_path
    return raw_path.removeprefix("./")
