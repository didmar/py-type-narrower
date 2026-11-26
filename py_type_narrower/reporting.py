"""Output formatters for text, JSON, SARIF 2.1.0, and unified diffs."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from py_type_narrower import __version__
from py_type_narrower.models import AnalysisResult, NarrowingCandidate, VerificationStatus

if TYPE_CHECKING:
    from pathlib import Path


def format_report(
    result: AnalysisResult,
    project_root: Path,
    *,
    output_format: str = "text",
    verbose: bool = False,
    show_rejected: bool = False,
) -> str:
    """Render an AnalysisResult in the requested output format ('text', 'json', 'sarif')."""
    if output_format == "json":
        return format_json(result, project_root)
    if output_format == "sarif":
        return format_sarif(result, project_root)
    return format_text(result, project_root, verbose=verbose, show_rejected=show_rejected)


def format_text(
    result: AnalysisResult,
    project_root: Path,
    *,
    verbose: bool = False,
    show_rejected: bool = False,
) -> str:
    """Render concise Ruff-style diagnostics with tier, score, and optional evidence."""
    lines: list[str] = []
    root = project_root.resolve()

    for cand in result.candidates:
        lines.extend(_format_candidate_text(cand, root, verbose=verbose))

    if show_rejected and result.rejected:
        lines.append("")
        lines.append("Rejected by soundness verifier:")
        for cand in result.rejected:
            lines.extend(_format_candidate_text(cand, root, verbose=True))

    summary = _format_summary_line(result)
    if lines:
        lines.append("")
    lines.append(summary)
    return "\n".join(lines)


def format_json(result: AnalysisResult, project_root: Path) -> str:
    """Render AnalysisResult as structured JSON."""
    root = project_root.resolve()
    payload = {
        "version": __version__,
        "passes_run": result.passes_run,
        "verifier": result.verifier_used,
        "candidates": [_candidate_to_dict(c, root) for c in result.candidates],
        "rejected": [_candidate_to_dict(c, root) for c in result.rejected],
        "applied_count": len(result.applied),
    }
    return json.dumps(payload, indent=2)


def format_sarif(result: AnalysisResult, project_root: Path) -> str:
    """Render AnalysisResult as a SARIF 2.1.0 JSON report."""
    root = project_root.resolve()
    sarif_results = [_candidate_to_sarif(c, root) for c in result.candidates]
    sarif_doc = {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "py-type-narrower",
                        "version": __version__,
                    }
                },
                "results": sarif_results,
            }
        ],
    }
    return json.dumps(sarif_doc, indent=2)


def format_diffs(result: AnalysisResult) -> str:
    """Concatenate all unified diffs in an AnalysisResult."""
    return "".join(result.diffs.values())


def _format_candidate_text(
    cand: NarrowingCandidate,
    root: Path,
    *,
    verbose: bool,
) -> list[str]:
    """Render a single NarrowingCandidate as one or more terminal lines."""
    rel_file = _relative_display(cand.file, root)
    verify_tag = _verification_badge(cand.verification)
    advisory_tag = "" if cand.is_fixable else " advisory"
    header = (
        f"{rel_file}:{cand.line}:{cand.col}: {cand.code.value} "
        f"[T{cand.tier.value} score={cand.score}{verify_tag}{advisory_tag}] {cand.message}"
    )
    lines = [header]
    if verbose:
        lines.append(f"  suggest: {cand.current_type} -> {cand.suggested_type}")
        lines.extend(f"  evidence ({ev.kind}): {ev.detail}" for ev in cand.evidence)
        lines.extend(f"  blocked by: {diag}" for diag in cand.blocking_diagnostics)
    return lines


def _verification_badge(status: VerificationStatus) -> str:
    """Return a short status tag for text output."""
    if status == VerificationStatus.VERIFIED:
        return " verified"
    if status == VerificationStatus.REJECTED:
        return " rejected"
    return ""


def _format_summary_line(result: AnalysisResult) -> str:
    """Render the final one-line run summary."""
    parts = [f"Found {len(result.candidates)} narrowing candidate(s)"]
    if result.applied:
        parts.append(f"applied {len(result.applied)}")
    if result.rejected:
        parts.append(f"rejected {len(result.rejected)} by {result.verifier_used or 'verifier'}")
    if result.passes_run > 1:
        parts.append(f"across {result.passes_run} passes")
    return ", ".join(parts) + "."


def _candidate_to_dict(cand: NarrowingCandidate, root: Path) -> dict[str, object]:
    """Convert a NarrowingCandidate to a JSON-serializable dictionary."""
    return {
        "code": cand.code.value,
        "tier": cand.tier.value,
        "file": _relative_display(cand.file, root),
        "line": cand.line,
        "col": cand.col,
        "symbol": cand.symbol,
        "target_kind": cand.target_kind.value,
        "target_name": cand.target_name,
        "current_type": cand.current_type,
        "suggested_type": cand.suggested_type,
        "score": cand.score,
        "is_public": cand.is_public,
        "is_single_constant": cand.is_single_constant,
        "is_fixable": cand.is_fixable,
        "verification": cand.verification.value,
        "message": cand.message,
        "evidence": [{"kind": e.kind, "detail": e.detail} for e in cand.evidence],
        "blocking_diagnostics": cand.blocking_diagnostics,
    }


def _candidate_to_sarif(cand: NarrowingCandidate, root: Path) -> dict[str, object]:
    """Convert a NarrowingCandidate to a SARIF result object."""
    level = "warning" if (cand.tier == 1 and cand.is_fixable) else "note"
    return {
        "ruleId": cand.code.value,
        "level": level,
        "message": {"text": cand.message},
        "properties": {
            "fixable": cand.is_fixable,
            "score": cand.score,
            "tier": cand.tier.value,
        },
        "locations": [
            {
                "physicalLocation": {
                    "artifactLocation": {"uri": _relative_display(cand.file, root)},
                    "region": {
                        "startLine": cand.line,
                        "startColumn": cand.col + 1,
                    },
                }
            }
        ],
    }


def _relative_display(path: Path, root: Path) -> str:
    """Display path relative to project root when possible."""
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path)
