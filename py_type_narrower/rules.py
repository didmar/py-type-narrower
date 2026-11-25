"""Narrowing rule implementations (Tier 1 local rules and Tier 2 whole-repo rules)."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from py_type_narrower.body_analysis import filter_literals_by_kind
from py_type_narrower.cst_utils import ExtractedLiteral, format_literal_type
from py_type_narrower.ignores import is_ignored
from py_type_narrower.models import (
    EvidenceItem,
    NarrowingCandidate,
    RuleCode,
    TargetKind,
    Tier,
)
from py_type_narrower.scorer import (
    CallSiteScoreContext,
    OptionalParamScoreContext,
    score_tier1_guarded_param,
    score_tier1_literal_return,
    score_tier1_unused_optional_return,
    score_tier2_callsite_literal,
    score_tier2_unused_optional_param,
)

if TYPE_CHECKING:
    from py_type_narrower.config import NarrowerConfig
    from py_type_narrower.indexer import FunctionInfo, ParamCallStats, ParamInfo, RepoIndex


def run_rules(index: RepoIndex, config: NarrowerConfig) -> list[NarrowingCandidate]:
    """Evaluate all enabled narrowing rules against a RepoIndex."""
    active = config.active_rules()
    candidates: list[NarrowingCandidate] = []

    for func in sorted(index.functions.values(), key=lambda f: (str(f.file), f.line, f.col)):
        if func.is_test_file and not config.include_test_files:
            continue
        _evaluate_function_rules(func, index, config, active, candidates)

    return _filter_ignored_and_scored(candidates, index, config)


def _evaluate_function_rules(
    func: FunctionInfo,
    index: RepoIndex,
    config: NarrowerConfig,
    active: set[RuleCode],
    out: list[NarrowingCandidate],
) -> None:
    """Run Tier 1 and Tier 2 rules on a single FunctionInfo."""
    if config.max_tier >= Tier.LOCAL:
        _run_tier1_rules(func, config, active, out)
    if config.max_tier >= Tier.WHOLE_REPO:
        _run_tier2_rules(func, index, config, active, out)


def _run_tier1_rules(
    func: FunctionInfo,
    config: NarrowerConfig,
    active: set[RuleCode],
    out: list[NarrowingCandidate],
) -> None:
    """Evaluate local high-confidence rules (NT101, NT102, NT103)."""
    if RuleCode.NT101 in active:
        out.extend(_rule_nt101_guarded_params(func, config))
    if RuleCode.NT102 in active:
        cand = _rule_nt102_unused_optional_return(func)
        if cand is not None:
            out.append(cand)
    if RuleCode.NT103 in active:
        cand = _rule_nt103_literal_return(func, config)
        if cand is not None:
            out.append(cand)


def _run_tier2_rules(
    func: FunctionInfo,
    index: RepoIndex,
    config: NarrowerConfig,
    active: set[RuleCode],
    out: list[NarrowingCandidate],
) -> None:
    """Evaluate whole-repo call-site rules (NT201, NT202)."""
    if _skip_param_narrowing_for_hierarchy(func):
        return
    # Track params already claimed by NT101 so we don't emit a duplicate weaker NT201.
    nt101_params = {
        c.target_name for c in out if c.symbol == func.qualname and c.code == RuleCode.NT101
    }
    for param in _non_receiver_params(func):
        stats = index.call_stats.get((func.qualname, param.name))
        if stats is None or not stats.observations or stats.has_dynamic_star_call:
            continue
        if RuleCode.NT201 in active and param.name not in nt101_params:
            cand = _rule_nt201_callsite_literal(func, param, stats, config)
            if cand is not None:
                out.append(cand)
        if RuleCode.NT202 in active:
            cand = _rule_nt202_unused_optional_param(func, param, stats, config)
            if cand is not None:
                out.append(cand)


def _rule_nt101_guarded_params(
    func: FunctionInfo,
    config: NarrowerConfig,
) -> list[NarrowingCandidate]:
    """NT101: Parameter typed str/int is exhaustively guarded to specific literals in body."""
    if _skip_param_narrowing_for_hierarchy(func):
        return []

    results: list[NarrowingCandidate] = []
    for param in _non_receiver_params(func):
        if param.primitive_kind is None:
            continue
        guard = func.body_facts.param_guards.get(param.name)
        if guard is None or not guard.strict_literals:
            continue
        matching = filter_literals_by_kind(guard.strict_literals, param.primitive_kind)
        if not matching or len(matching) > config.max_literals:
            continue
        if param.has_default and param.default_literal is not None:
            if param.default_literal.kind == param.primitive_kind:
                matching = matching | {param.default_literal}
            else:
                continue

        suggested = format_literal_type(matching)
        evidence = [
            EvidenceItem(kind="body_guard", detail=reason, file=func.file, line=func.line)
            for reason in guard.strict_evidence
        ]
        results.append(
            NarrowingCandidate(
                code=RuleCode.NT101,
                tier=Tier.LOCAL,
                file=func.file,
                line=param.line,
                col=param.col,
                symbol=func.qualname,
                target_kind=TargetKind.PARAMETER,
                target_name=param.name,
                current_type=param.annotation_str or param.primitive_kind,
                suggested_type=suggested,
                needs_literal_import=True,
                message=(
                    f"Parameter `{param.name}` in `{func.name}` is guarded to "
                    f"{suggested} in the function body"
                ),
                evidence=evidence,
                score=score_tier1_guarded_param(is_private=func.is_private),
                is_public=not func.is_private,
                is_single_constant=len(matching) == 1,
            )
        )
    return results


def _rule_nt102_unused_optional_return(func: FunctionInfo) -> NarrowingCandidate | None:
    """NT102: Return type `T | None` never returns `None` along any path."""
    if func.return_optional_inner is None or func.return_annotation_str is None:
        return None
    if func.is_abstract or func.in_protocol_or_abc:
        return None
    ret = func.body_facts.returns
    if ret.is_stub or ret.is_generator or not ret.has_returns:
        return None
    if ret.returns_none or ret.can_fallthrough or not ret.all_returns_non_none:
        return None

    return NarrowingCandidate(
        code=RuleCode.NT102,
        tier=Tier.LOCAL,
        file=func.file,
        line=func.return_line,
        col=func.return_col,
        symbol=func.qualname,
        target_kind=TargetKind.RETURN,
        target_name=None,
        current_type=func.return_annotation_str,
        suggested_type=func.return_optional_inner,
        needs_literal_import=False,
        message=(
            f"Return type of `{func.name}` is `{func.return_annotation_str}`, "
            f"but all return paths are non-None; narrow to `{func.return_optional_inner}`"
        ),
        evidence=[
            EvidenceItem(
                kind="return_analysis",
                detail="All return statements return non-None values and control flow cannot fall through",
                file=func.file,
                line=func.line,
            )
        ],
        score=score_tier1_unused_optional_return(is_private=func.is_private),
        is_public=not func.is_private,
    )


def _rule_nt103_literal_return(
    func: FunctionInfo,
    config: NarrowerConfig,
) -> NarrowingCandidate | None:
    """NT103: Return type `str`/`int` only returns constant literals."""
    ret = func.body_facts.returns
    if (
        func.return_primitive_kind is None
        or func.return_annotation_str is None
        or func.is_abstract
        or func.in_protocol_or_abc
    ):
        return None
    if (
        ret.is_stub
        or ret.is_generator
        or not ret.has_returns
        or ret.can_fallthrough
        or ret.has_non_literal_return
        or not ret.returned_literals
    ):
        return None

    matching = filter_literals_by_kind(ret.returned_literals, func.return_primitive_kind)
    if (
        len(matching) != len(ret.returned_literals)
        or not matching
        or len(matching) > config.max_literals
    ):
        return None

    suggested = format_literal_type(matching)
    return NarrowingCandidate(
        code=RuleCode.NT103,
        tier=Tier.LOCAL,
        file=func.file,
        line=func.return_line,
        col=func.return_col,
        symbol=func.qualname,
        target_kind=TargetKind.RETURN,
        target_name=None,
        current_type=func.return_annotation_str,
        suggested_type=suggested,
        needs_literal_import=True,
        message=(
            f"Return type of `{func.name}` is `{func.return_annotation_str}`, "
            f"but it only returns {suggested}"
        ),
        evidence=[
            EvidenceItem(
                kind="return_literals",
                detail=f"Every return statement yields a constant in {suggested}",
                file=func.file,
                line=func.line,
            )
        ],
        score=score_tier1_literal_return(
            is_private=func.is_private, closed_world=config.closed_world
        ),
        is_public=not func.is_private,
        is_single_constant=len(matching) == 1,
    )


def _rule_nt201_callsite_literal(
    func: FunctionInfo,
    param: ParamInfo,
    stats: ParamCallStats,
    config: NarrowerConfig,
) -> NarrowingCandidate | None:
    """NT201: Call-site `str`/`int` parameter only receives literals from a small set."""
    if param.primitive_kind is None or param.annotation_str is None:
        return None

    collected_counts: Counter[ExtractedLiteral] = Counter()
    for obs in stats.observations:
        if obs.is_unknown or obs.is_none or not obs.literals:
            return None
        for lit in obs.literals:
            if lit.kind != param.primitive_kind:
                return None
            collected_counts[lit] += 1

    # Account for default value if callers omit the parameter.
    if stats.omitted_count > 0:
        if param.default_literal is None or param.default_literal.kind != param.primitive_kind:
            return None
        collected_counts[param.default_literal] += stats.omitted_count

    unique_lits = set(collected_counts.keys())
    if not unique_lits or len(unique_lits) > config.max_literals:
        return None

    total_calls = len(stats.observations) + stats.omitted_count
    only_test = all(obs.is_test_caller for obs in stats.observations)
    guard = func.body_facts.param_guards.get(param.name)
    has_body_corr = bool(guard is not None and (guard.soft_literals & unique_lits))

    score, notes = score_tier2_callsite_literal(
        CallSiteScoreContext(
            is_private=func.is_private,
            closed_world=config.closed_world,
            call_count=total_calls,
            unique_value_count=len(unique_lits),
            min_call_sites=config.min_call_sites,
            only_test_callers=only_test,
            has_body_corroboration=has_body_corr,
            param_name=param.name,
        )
    )

    suggested = format_literal_type(unique_lits)
    is_single = len(unique_lits) == 1
    single_hint = " (single constant across all callers)" if is_single else ""
    evidence = _build_callsite_literal_evidence(collected_counts, total_calls, notes)

    return NarrowingCandidate(
        code=RuleCode.NT201,
        tier=Tier.WHOLE_REPO,
        file=func.file,
        line=param.line,
        col=param.col,
        symbol=func.qualname,
        target_kind=TargetKind.PARAMETER,
        target_name=param.name,
        current_type=param.annotation_str,
        suggested_type=suggested,
        needs_literal_import=True,
        message=(
            f"Parameter `{param.name}` in `{func.name}` only receives "
            f"{suggested} across {total_calls} call(s){single_hint}"
        ),
        evidence=evidence,
        score=score,
        is_public=not func.is_private,
        is_single_constant=is_single,
    )


def _rule_nt202_unused_optional_param(
    func: FunctionInfo,
    param: ParamInfo,
    stats: ParamCallStats,
    config: NarrowerConfig,
) -> NarrowingCandidate | None:
    """NT202: Parameter typed `T | None` never receives `None` across callers."""
    if param.optional_inner is None or param.annotation_str is None:
        return None
    if param.default_is_none and stats.omitted_count > 0:
        return None
    if any(obs.is_none or not obs.is_non_none for obs in stats.observations):
        return None

    call_count = len(stats.observations)
    only_test = all(obs.is_test_caller for obs in stats.observations)
    score, notes = score_tier2_unused_optional_param(
        OptionalParamScoreContext(
            is_private=func.is_private,
            closed_world=config.closed_world,
            call_count=call_count,
            min_call_sites=config.min_call_sites,
            only_test_callers=only_test,
            has_none_default=param.default_is_none,
        )
    )

    evidence = [
        EvidenceItem(
            kind="call_sites",
            detail=f"Observed {call_count} call site(s), all passing provably non-None values",
        ),
        *(EvidenceItem(kind="score_factor", detail=note) for note in notes),
    ]
    return NarrowingCandidate(
        code=RuleCode.NT202,
        tier=Tier.WHOLE_REPO,
        file=func.file,
        line=param.line,
        col=param.col,
        symbol=func.qualname,
        target_kind=TargetKind.PARAMETER,
        target_name=param.name,
        current_type=param.annotation_str,
        suggested_type=param.optional_inner,
        needs_literal_import=False,
        message=(
            f"Parameter `{param.name}` in `{func.name}` is `{param.annotation_str}`, "
            f"but None is never passed across {call_count} call(s); "
            f"narrow to `{param.optional_inner}`"
        ),
        evidence=evidence,
        score=score,
        is_public=not func.is_private,
        drop_none_default=param.default_is_none,
    )


def _skip_param_narrowing_for_hierarchy(func: FunctionInfo) -> bool:
    """Skip parameter narrowing on overrides, ABCs, Protocols, and abstract methods."""
    return func.is_override or func.is_abstract or func.in_protocol_or_abc


def _non_receiver_params(func: FunctionInfo) -> list[ParamInfo]:
    """Return parameters excluding `self`/`cls` on bound methods."""
    if func.is_method and not func.is_staticmethod and func.parameters:
        return func.parameters[1:]
    return func.parameters


def _build_callsite_literal_evidence(
    counts: Counter[ExtractedLiteral],
    total_calls: int,
    notes: list[str],
) -> list[EvidenceItem]:
    """Format human-readable value distribution and score factor evidence."""
    dist_parts = [f"{lit.value!r} ({count}x)" for lit, count in counts.most_common()]
    items = [
        EvidenceItem(
            kind="call_sites",
            detail=f"Observed {total_calls} call(s): {', '.join(dist_parts)}",
        )
    ]
    items.extend(EvidenceItem(kind="score_factor", detail=note) for note in notes)
    return items


def _filter_ignored_and_scored(
    candidates: list[NarrowingCandidate],
    index: RepoIndex,
    config: NarrowerConfig,
) -> list[NarrowingCandidate]:
    """Filter out suppressed findings and candidates below min_confidence."""
    kept: list[NarrowingCandidate] = []
    for cand in candidates:
        file_ignores = index.ignores.get(cand.file, {})
        if is_ignored(file_ignores, cand.line, cand.code.value):
            continue
        if cand.score < config.min_confidence:
            continue
        kept.append(cand)
    return kept
