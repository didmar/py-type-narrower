"""High-level analysis engine with multi-pass fixpoint propagation and soundness verification."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from py_type_narrower.indexer import discover_python_files, index_files
from py_type_narrower.models import (
    AnalysisResult,
    NarrowingCandidate,
    Tier,
    VerificationStatus,
)
from py_type_narrower.rewriter import compute_unified_diffs, rewrite_files
from py_type_narrower.rules import run_rules
from py_type_narrower.verifier import TypeCheckerAdapter, resolve_verifier, verify_candidates

if TYPE_CHECKING:
    from pathlib import Path

    from py_type_narrower.config import NarrowerConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunOptions:
    """Execution flags controlling fixing, diffing, and custom verifier injection."""

    apply_fixes: bool = False
    unsafe_fixes: bool = False
    compute_diffs: bool = False
    adapter: TypeCheckerAdapter | None = None


def analyze_paths(
    paths: list[Path],
    project_root: Path,
    config: NarrowerConfig,
    options: RunOptions | None = None,
) -> AnalysisResult:
    """Run single- or multi-pass narrowing analysis, optional verification, and optional fixing."""
    opts = options or RunOptions()
    target_files = discover_python_files(paths, config.exclude)
    if not target_files:
        return AnalysisResult()

    indexed_files = _resolve_index_files(target_files, project_root, config)
    target_set = set(target_files)
    resolved_adapter = _get_verifier_adapter(config, project_root, opts.adapter)
    original_sources: dict[Path, str] = {}
    working_sources: dict[Path, str] = {}
    all_candidates: list[NarrowingCandidate] = []
    all_rejected: list[NarrowingCandidate] = []
    all_applied: list[NarrowingCandidate] = []
    passes_run = 0

    for pass_num in range(1, max(1, config.max_passes) + 1):
        passes_run = pass_num
        index = index_files(indexed_files, project_root, overrides=working_sources or None)
        if pass_num == 1:
            original_sources = dict(index.files)
            working_sources = dict(index.files)

        pass_candidates = run_rules(index, config, target_files=target_set)
        if not pass_candidates:
            break

        verified_or_kept, pass_rejected = _verify_pass(
            pass_candidates, working_sources, project_root, resolved_adapter
        )
        all_candidates.extend(verified_or_kept)
        all_rejected.extend(pass_rejected)

        selected_for_rewrite = _select_fix_candidates(
            verified_or_kept,
            apply_all=opts.unsafe_fixes or (not opts.apply_fixes and config.max_passes > 1),
        )
        if not selected_for_rewrite:
            break

        changed = rewrite_files(selected_for_rewrite, working_sources)
        if not changed:
            break

        working_sources.update(changed)
        if opts.apply_fixes:
            all_applied.extend(selected_for_rewrite)
        if pass_num == config.max_passes:
            break

    if opts.apply_fixes and all_applied:
        _write_updated_files(original_sources, working_sources)

    diffs = (
        compute_unified_diffs(original_sources, working_sources, project_root)
        if (opts.compute_diffs or opts.apply_fixes)
        else {}
    )
    return AnalysisResult(
        candidates=all_candidates,
        rejected=all_rejected,
        applied=all_applied,
        diffs=diffs,
        passes_run=passes_run,
        verifier_used=resolved_adapter.name if resolved_adapter is not None else None,
    )


def _resolve_index_files(
    target_files: list[Path],
    project_root: Path,
    config: NarrowerConfig,
) -> list[Path]:
    """Return files to index: full project_root when Tier 2 rules are active, else target_files."""
    has_tier2 = config.max_tier >= Tier.WHOLE_REPO and any(
        rule.tier == Tier.WHOLE_REPO for rule in config.active_rules()
    )
    if not has_tier2 or not config.index_project_root:
        return target_files
    project_files = discover_python_files([project_root], config.exclude)
    return sorted({*project_files, *target_files})


def _get_verifier_adapter(
    config: NarrowerConfig,
    project_root: Path,
    explicit_adapter: TypeCheckerAdapter | None,
) -> TypeCheckerAdapter | None:
    """Resolve the TypeCheckerAdapter if verification is enabled."""
    if explicit_adapter is not None:
        return explicit_adapter
    if not config.verify:
        return None
    return resolve_verifier(config.verifier, project_root, config.verify_command)


def _verify_pass(
    candidates: list[NarrowingCandidate],
    sources: dict[Path, str],
    project_root: Path,
    adapter: TypeCheckerAdapter | None,
) -> tuple[list[NarrowingCandidate], list[NarrowingCandidate]]:
    """Run soundness verification on fixable candidates when an adapter is configured."""
    if adapter is None:
        return (candidates, [])
    fixable = [c for c in candidates if c.is_fixable]
    advisory = [c for c in candidates if not c.is_fixable]
    verified, rejected = verify_candidates(fixable, sources, project_root, adapter)
    return ([*verified, *advisory], rejected)


def _select_fix_candidates(
    candidates: list[NarrowingCandidate],
    *,
    apply_all: bool,
) -> list[NarrowingCandidate]:
    """Choose which candidates are eligible for automatic rewriting."""
    if apply_all:
        return [
            c for c in candidates if c.is_fixable and c.verification != VerificationStatus.REJECTED
        ]
    return [
        c
        for c in candidates
        if c.is_fixable and c.tier == Tier.LOCAL and c.verification != VerificationStatus.REJECTED
    ]


def _write_updated_files(
    original_sources: dict[Path, str],
    working_sources: dict[Path, str],
) -> None:
    """Persist modified files back to disk."""
    for file_path, new_content in working_sources.items():
        if original_sources.get(file_path) != new_content:
            file_path.write_text(new_content, encoding="utf-8")
            logger.info("Updated %s", file_path)
