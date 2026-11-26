"""Narrowing rule implementations (Tier 1 local rules and Tier 2 whole-repo rules)."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import libcst as cst

from py_type_narrower.body_analysis import filter_literals_by_kind
from py_type_narrower.cst_utils import (
    ExtractedLiteral,
    extract_optional_container_return,
    extract_union_members,
    extract_variadic_tuple_element,
    format_literal_type,
    format_union_type,
    matches_union_member,
)
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
    ConstantBoolScoreContext,
    OptionalParamScoreContext,
    score_tier1_always_initialized_attr,
    score_tier1_container_non_none,
    score_tier1_fixed_tuple_return,
    score_tier1_guarded_param,
    score_tier1_literal_return,
    score_tier1_unused_optional_return,
    score_tier1_unused_union_return,
    score_tier2_callsite_literal,
    score_tier2_constant_bool_param,
    score_tier2_unused_optional_param,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from py_type_narrower.body_analysis import ReturnSummary
    from py_type_narrower.config import NarrowerConfig
    from py_type_narrower.indexer import (
        ClassInfo,
        CodebaseIndex,
        FieldInfo,
        FunctionInfo,
        ParamCallStats,
        ParamInfo,
    )

    Tier1FunctionRule = Callable[[FunctionInfo, NarrowerConfig], list[NarrowingCandidate]]
    Tier2ParamRule = Callable[
        [FunctionInfo, ParamInfo, ParamCallStats, NarrowerConfig],
        NarrowingCandidate | None,
    ]

# Python data-model protocol methods whose signatures are fixed by the runtime/typeshed.
# User-defined API dunders like __init__, __call__, and __getitem__ are intentionally excluded.
DATA_MODEL_DUNDER_METHODS: frozenset[str] = frozenset(
    {
        "__repr__",
        "__str__",
        "__bytes__",
        "__format__",
        "__exit__",
        "__aexit__",
        "__bool__",
        "__len__",
        "__length_hint__",
        "__hash__",
        "__index__",
        "__int__",
        "__float__",
        "__complex__",
        "__eq__",
        "__ne__",
        "__lt__",
        "__le__",
        "__gt__",
        "__ge__",
        "__contains__",
        "__dir__",
        "__getattr__",
        "__getattribute__",
        "__setattr__",
        "__delattr__",
        "__sizeof__",
        "__fspath__",
    }
)

# PEP 562 module-level protocol hooks with fixed runtime signatures.
MODULE_DUNDER_FUNCTIONS: frozenset[str] = frozenset({"__getattr__", "__dir__"})


def run_rules(
    index: CodebaseIndex,
    config: NarrowerConfig,
    *,
    target_files: set[Path] | None = None,
) -> list[NarrowingCandidate]:
    """Evaluate all enabled narrowing rules against a CodebaseIndex.

    When `target_files` is provided, only definitions inside those files are reported,
    while the full index still supplies cross-file call sites and class hierarchies.
    """
    active = config.active_rules()
    resolved_targets = {p.resolve() for p in target_files} if target_files is not None else None
    candidates: list[NarrowingCandidate] = []

    for func in sorted(index.functions.values(), key=lambda f: (str(f.file), f.line, f.col)):
        if resolved_targets is not None and func.file.resolve() not in resolved_targets:
            continue
        if func.is_test_file and not config.include_test_files:
            continue
        _evaluate_function_rules(func, index, config, active, candidates)

    for cls in sorted(index.classes.values(), key=lambda c: (str(c.file), c.line, c.col)):
        if resolved_targets is not None and cls.file.resolve() not in resolved_targets:
            continue
        if cls.is_test_file and not config.include_test_files:
            continue
        _evaluate_class_rules(cls, index, config, active, candidates)

    return _filter_ignored_and_scored(candidates, index, config)


def _evaluate_function_rules(
    func: FunctionInfo,
    index: CodebaseIndex,
    config: NarrowerConfig,
    active: set[RuleCode],
    out: list[NarrowingCandidate],
) -> None:
    """Run Tier 1 and Tier 2 rules on a single FunctionInfo."""
    if config.max_tier >= Tier.LOCAL:
        _run_tier1_function_rules(func, config, active, out)
    if config.max_tier >= Tier.WHOLE_REPO:
        _run_tier2_function_rules(func, index, config, active, out)


def _evaluate_class_rules(
    cls: ClassInfo,
    index: CodebaseIndex,
    config: NarrowerConfig,
    active: set[RuleCode],
    out: list[NarrowingCandidate],
) -> None:
    """Run Tier 1 (NT104) and Tier 2 (NT203) rules on a single ClassInfo."""
    if config.max_tier >= Tier.LOCAL and RuleCode.NT104 in active:
        out.extend(_rule_nt104_always_initialized_attrs(cls))
    if config.max_tier >= Tier.WHOLE_REPO and RuleCode.NT203 in active:
        out.extend(_rule_nt203_structured_fields(cls, index, config))


def _run_tier1_function_rules(
    func: FunctionInfo,
    config: NarrowerConfig,
    active: set[RuleCode],
    out: list[NarrowingCandidate],
) -> None:
    """Evaluate local high-confidence function rules in priority order (NT101, NT105, NT102, NT103)."""
    rules: tuple[tuple[RuleCode, Tier1FunctionRule], ...] = (
        (RuleCode.NT101, _rule_nt101_guarded_params),
        (RuleCode.NT105, _rule_nt105_unused_union_return),
        (RuleCode.NT102, _rule_nt102_unused_optional_return),
        (RuleCode.NT103, _rule_nt103_literal_return),
        (RuleCode.NT106, _rule_nt106_fixed_tuple_return),
        (RuleCode.NT107, _rule_nt107_container_non_none_return),
    )
    seen_targets: set[tuple[TargetKind, str | None]] = set()
    for code, rule_fn in rules:
        if code not in active:
            continue
        for cand in rule_fn(func, config):
            target = (cand.target_kind, cand.target_name)
            if target not in seen_targets:
                seen_targets.add(target)
                out.append(cand)


def _run_tier2_function_rules(
    func: FunctionInfo,
    index: CodebaseIndex,
    config: NarrowerConfig,
    active: set[RuleCode],
    out: list[NarrowingCandidate],
) -> None:
    """Evaluate whole-repo call-site function rules (NT201, NT204, NT202, NT205)."""
    if _skip_param_narrowing_for_hierarchy(func):
        return
    if func.in_external_subclass and not func.is_private and func.name != "__init__":
        return
    nt101_params = {
        c.target_name for c in out if c.symbol == func.qualname and c.code == RuleCode.NT101
    }
    for param in _non_receiver_params(func):
        if param.name in nt101_params:
            continue
        stats = index.call_stats.get((func.qualname, param.name))
        if stats is None or not stats.observations or stats.has_dynamic_star_call:
            continue
        cand = _evaluate_param_tier2(func, param, stats, config, active)
        if cand is not None:
            out.append(cand)


def _evaluate_param_tier2(
    func: FunctionInfo,
    param: ParamInfo,
    stats: ParamCallStats,
    config: NarrowerConfig,
    active: set[RuleCode],
) -> NarrowingCandidate | None:
    """Evaluate Tier 2 parameter rules in priority order (NT201, NT204, NT202, NT205)."""
    rules: tuple[tuple[RuleCode, Tier2ParamRule], ...] = (
        (RuleCode.NT201, _rule_nt201_callsite_literal),
        (RuleCode.NT204, _rule_nt204_unused_union_param),
        (RuleCode.NT202, _rule_nt202_unused_optional_param),
        (RuleCode.NT205, _rule_nt205_constant_bool_param),
    )
    for code, rule_fn in rules:
        if code not in active:
            continue
        cand = rule_fn(func, param, stats, config)
        if cand is not None:
            return cand
    return None


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


def _rule_nt102_unused_optional_return(
    func: FunctionInfo,
    config: NarrowerConfig,
) -> list[NarrowingCandidate]:
    """NT102: Return type `T | None` never returns `None` along any path."""
    if (
        func.return_optional_inner is None
        or func.return_annotation_str is None
        or _skip_return_narrowing_for_protocol(func, config)
    ):
        return []
    ret = func.body_facts.returns
    if ret.is_stub or ret.is_generator or not ret.has_returns:
        return []
    if ret.returns_none or ret.can_fallthrough or not ret.all_returns_non_none:
        return []

    return [
        NarrowingCandidate(
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
    ]


def _rule_nt103_literal_return(
    func: FunctionInfo,
    config: NarrowerConfig,
) -> list[NarrowingCandidate]:
    """NT103: Return type `str`/`int` only returns constant literals."""
    ret = func.body_facts.returns
    if (
        func.return_primitive_kind is None
        or func.return_annotation_str is None
        or func.is_override
        or _skip_return_narrowing_for_protocol(func, config)
    ):
        return []
    if (
        ret.is_stub
        or ret.is_generator
        or not ret.has_returns
        or ret.can_fallthrough
        or ret.has_non_literal_return
        or not ret.returned_literals
    ):
        return []

    matching = filter_literals_by_kind(ret.returned_literals, func.return_primitive_kind)
    if (
        len(matching) != len(ret.returned_literals)
        or not matching
        or len(matching) > config.max_literals
        or _has_prose_or_overlong_literal(matching)
    ):
        return []
    if len(matching) == 1 and func.in_external_subclass and not func.is_private:
        return []

    suggested = format_literal_type(matching)
    return [
        NarrowingCandidate(
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
    ]


def _rule_nt104_always_initialized_attrs(cls: ClassInfo) -> list[NarrowingCandidate]:
    """NT104: Class attribute `T | None` is always initialized non-None and never set to None."""
    results: list[NarrowingCandidate] = []
    for fact in cls.attr_facts:
        loc_note = "in `__init__`" if fact.declared_in_init else "on class body and `__init__`"
        results.append(
            NarrowingCandidate(
                code=RuleCode.NT104,
                tier=Tier.LOCAL,
                file=cls.file,
                line=fact.line,
                col=fact.col,
                symbol=cls.qualname,
                target_kind=TargetKind.ATTRIBUTE,
                target_name=fact.name,
                current_type=fact.annotation_str,
                suggested_type=fact.optional_inner,
                needs_literal_import=False,
                message=(
                    f"Attribute `{cls.name}.{fact.name}` is `{fact.annotation_str}`, "
                    f"but is always initialized non-None in `__init__` and never assigned None; "
                    f"narrow to `{fact.optional_inner}`"
                ),
                evidence=[
                    EvidenceItem(
                        kind="class_attr_analysis",
                        detail=f"Always assigned non-None {loc_note} and never mutated to None",
                        file=cls.file,
                        line=fact.line,
                    )
                ],
                score=score_tier1_always_initialized_attr(is_private=cls.is_private),
                is_public=not cls.is_private,
                drop_none_default=fact.has_none_default,
            )
        )
    return results


def _rule_nt105_unused_union_return(
    func: FunctionInfo,
    config: NarrowerConfig,
) -> list[NarrowingCandidate]:
    """NT105: Return union `A | B | C` has unused non-None branch(es)."""
    members = func.return_union_members
    ret = func.body_facts.returns
    if (
        members is None
        or func.return_annotation_str is None
        or _skip_return_narrowing_for_protocol(func, config)
        or ret.is_stub
        or ret.is_generator
        or not ret.has_returns
        or ret.has_unknown_return_type
    ):
        return []

    observed = set(ret.returned_types)
    if ret.can_fallthrough:
        observed.add("None")

    used, unused = _partition_union_members(members, observed)
    if not used or not unused or unused == ["None"] or used == ["None"]:
        return []

    suggested = format_union_type(used)
    return [
        NarrowingCandidate(
            code=RuleCode.NT105,
            tier=Tier.LOCAL,
            file=func.file,
            line=func.return_line,
            col=func.return_col,
            symbol=func.qualname,
            target_kind=TargetKind.RETURN,
            target_name=None,
            current_type=func.return_annotation_str,
            suggested_type=suggested,
            needs_literal_import=False,
            message=(
                f"Return type of `{func.name}` is `{func.return_annotation_str}`, "
                f"but branch(es) {', '.join(unused)} are never returned; narrow to `{suggested}`"
            ),
            evidence=[
                EvidenceItem(
                    kind="return_types",
                    detail=f"Observed return types: {', '.join(sorted(observed))}",
                    file=func.file,
                    line=func.line,
                )
            ],
            score=score_tier1_unused_union_return(is_private=func.is_private),
            is_public=not func.is_private,
        )
    ]


_MAX_FIXED_TUPLE_LEN = 6
_ANY_OR_OBJECT_TYPES: frozenset[str] = frozenset({"Any", "typing.Any", "object", "builtins.object"})


def _rule_nt106_fixed_tuple_return(
    func: FunctionInfo,
    config: NarrowerConfig,
) -> list[NarrowingCandidate]:
    """NT106: Variadic `tuple[T, ...]` return always yields a fixed-length tuple literal."""
    if func.return_annotation_str is None or _skip_return_narrowing_for_protocol(func, config):
        return []
    parsed = extract_variadic_tuple_element(func.return_annotation_str)
    if parsed is None:
        return []
    outer_name, elem_type_str = parsed
    ret = func.body_facts.returns
    if (
        ret.is_stub
        or ret.is_generator
        or not ret.has_returns
        or ret.can_fallthrough
        or not ret.all_returns_fixed_tuple
        or ret.fixed_tuple_len is None
        or not (1 <= ret.fixed_tuple_len <= _MAX_FIXED_TUPLE_LEN)
    ):
        return []

    slot_types = _resolve_fixed_tuple_slot_types(
        elem_type_str,
        ret.fixed_tuple_len,
        ret.fixed_tuple_slot_types,
        ret.fixed_tuple_slot_unknown,
    )
    suggested = f"{outer_name}[{', '.join(slot_types)}]"
    return [
        NarrowingCandidate(
            code=RuleCode.NT106,
            tier=Tier.LOCAL,
            file=func.file,
            line=func.return_line,
            col=func.return_col,
            symbol=func.qualname,
            target_kind=TargetKind.RETURN,
            target_name=None,
            current_type=func.return_annotation_str,
            suggested_type=suggested,
            needs_literal_import=False,
            message=(
                f"Return type of `{func.name}` is `{func.return_annotation_str}`, "
                f"but every return yields a {ret.fixed_tuple_len}-tuple; narrow to `{suggested}`"
            ),
            evidence=[
                EvidenceItem(
                    kind="fixed_tuple_return",
                    detail=f"All return paths yield a {ret.fixed_tuple_len}-element tuple literal",
                    file=func.file,
                    line=func.line,
                )
            ],
            score=score_tier1_fixed_tuple_return(is_private=func.is_private),
            is_public=not func.is_private,
        )
    ]


def _resolve_fixed_tuple_slot_types(
    elem_type_str: str,
    tuple_len: int,
    slot_types: list[set[str]],
    slot_unknown: list[bool],
) -> list[str]:
    """Compute per-slot types for a fixed-length tuple narrowed from `tuple[elem_type_str, ...]`."""
    union_members = _parse_union_members(elem_type_str)
    is_broad = elem_type_str.strip() in _ANY_OR_OBJECT_TYPES
    resolved: list[str] = []
    for idx in range(tuple_len):
        obs = slot_types[idx]
        unknown = slot_unknown[idx]
        if union_members is not None and not unknown and obs:
            used, _unused = _partition_union_members(union_members, obs)
            resolved.append(format_union_type(used) if used else elem_type_str)
        elif is_broad and not unknown and obs:
            resolved.append(format_union_type(sorted(obs)))
        else:
            resolved.append(elem_type_str)
    return resolved


def _parse_union_members(type_str: str) -> tuple[str, ...] | None:
    """Parse a type string as a top-level union if possible."""
    try:
        expr = cst.parse_expression(type_str)
    except cst.ParserSyntaxError:
        return None
    return extract_union_members(cst.Annotation(annotation=expr))


def _rule_nt107_container_non_none_return(
    func: FunctionInfo,
    config: NarrowerConfig,
) -> list[NarrowingCandidate]:
    """NT107: Container/generator return `list[T | None]` / `Iterator[T | None]` never yields None."""
    if func.return_annotation_str is None or _skip_return_narrowing_for_protocol(func, config):
        return []
    spec = extract_optional_container_return(func.return_annotation_str)
    if spec is None:
        return []
    ret = func.body_facts.returns
    if ret.is_stub:
        return []

    detail = _match_container_non_none_detail(spec.kind, ret)
    if detail is None:
        return []

    return [
        NarrowingCandidate(
            code=RuleCode.NT107,
            tier=Tier.LOCAL,
            file=func.file,
            line=func.return_line,
            col=func.return_col,
            symbol=func.qualname,
            target_kind=TargetKind.RETURN,
            target_name=None,
            current_type=func.return_annotation_str,
            suggested_type=spec.narrowed_annotation,
            needs_literal_import=False,
            message=(
                f"Return type of `{func.name}` is `{func.return_annotation_str}`, "
                f"but {detail}; narrow to `{spec.narrowed_annotation}`"
            ),
            evidence=[
                EvidenceItem(
                    kind="container_non_none",
                    detail=detail,
                    file=func.file,
                    line=func.line,
                )
            ],
            score=score_tier1_container_non_none(is_private=func.is_private),
            is_public=not func.is_private,
        )
    ]


def _match_container_non_none_detail(kind: str, ret: ReturnSummary) -> str | None:
    """Return an evidence detail string if ReturnSummary proves no None element/yield."""
    if kind in {"generator", "iterable"} and ret.is_generator:
        return (
            "every `yield` expression is provably non-None"
            if (ret.has_yields and ret.all_yields_non_none)
            else None
        )
    if ret.is_generator or not ret.has_returns or ret.can_fallthrough:
        return None
    if (
        kind in {"seq", "iterable"}
        and ret.all_returns_non_none_seq
        and ret.has_non_empty_seq_element
    ):
        return "every returned element is provably non-None"
    if kind == "dict" and ret.all_returns_non_none_dict and ret.has_non_empty_dict_entry:
        return "every returned dictionary value is provably non-None"
    return None


def _rule_nt201_callsite_literal(
    func: FunctionInfo,
    param: ParamInfo,
    stats: ParamCallStats,
    config: NarrowerConfig,
) -> NarrowingCandidate | None:
    """NT201: Call-site `str`/`int` parameter only receives literals from a small set."""
    if param.primitive_kind is None or param.annotation_str is None:
        return None

    collected_counts = _collect_literal_counts(
        stats=stats,
        expected_kind=param.primitive_kind,
        has_default=param.has_default,
        default_literal=param.default_literal,
    )
    if collected_counts is None:
        return None

    unique_lits = set(collected_counts.keys())
    if (
        not unique_lits
        or len(unique_lits) > config.max_literals
        or _has_prose_or_overlong_literal(unique_lits)
    ):
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
    if _cannot_drop_param_none_default(func, param):
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


def _rule_nt203_structured_fields(
    cls: ClassInfo,
    index: CodebaseIndex,
    config: NarrowerConfig,
) -> list[NarrowingCandidate]:
    """NT203: Dataclass / NamedTuple field narrowing across constructor & assignment sites."""
    if not (cls.is_dataclass or cls.is_namedtuple) or cls.is_protocol_or_abc or not cls.fields:
        return []
    if _class_has_subclasses(cls, index):
        return []

    results: list[NarrowingCandidate] = []
    for fld in cls.fields:
        stats = index.field_stats.get((cls.qualname, fld.name))
        if stats is None or not stats.observations or stats.has_dynamic_star_call:
            continue
        lit_cand = _check_field_literal(cls, fld, stats, config)
        if lit_cand is not None:
            results.append(lit_cand)
            continue
        opt_cand = _check_field_optional(cls, fld, stats, config)
        if opt_cand is not None:
            results.append(opt_cand)
    return results


def _check_field_literal(
    cls: ClassInfo,
    fld: FieldInfo,
    stats: ParamCallStats,
    config: NarrowerConfig,
) -> NarrowingCandidate | None:
    """Check if a dataclass/NamedTuple str/int field only receives literals."""
    if fld.primitive_kind is None:
        return None
    counts = _collect_literal_counts(
        stats=stats,
        expected_kind=fld.primitive_kind,
        has_default=fld.has_default,
        default_literal=fld.default_literal,
    )
    if counts is None:
        return None
    unique_lits = set(counts.keys())
    if (
        not unique_lits
        or len(unique_lits) > config.max_literals
        or _has_prose_or_overlong_literal(unique_lits)
    ):
        return None

    total_calls = len(stats.observations) + stats.omitted_count
    only_test = all(obs.is_test_caller for obs in stats.observations)
    score, notes = score_tier2_callsite_literal(
        CallSiteScoreContext(
            is_private=cls.is_private,
            closed_world=config.closed_world,
            call_count=total_calls,
            unique_value_count=len(unique_lits),
            min_call_sites=config.min_call_sites,
            only_test_callers=only_test,
            has_body_corroboration=False,
            param_name=fld.name,
        )
    )
    suggested = format_literal_type(unique_lits)
    is_single = len(unique_lits) == 1
    single_hint = " (single constant across all instantiations)" if is_single else ""
    return NarrowingCandidate(
        code=RuleCode.NT203,
        tier=Tier.WHOLE_REPO,
        file=cls.file,
        line=fld.line,
        col=fld.col,
        symbol=cls.qualname,
        target_kind=TargetKind.ATTRIBUTE,
        target_name=fld.name,
        current_type=fld.annotation_str,
        suggested_type=suggested,
        needs_literal_import=True,
        message=(
            f"Field `{cls.name}.{fld.name}` only receives "
            f"{suggested} across {total_calls} instantiation(s){single_hint}"
        ),
        evidence=_build_callsite_literal_evidence(counts, total_calls, notes),
        score=score,
        is_public=not cls.is_private,
        is_single_constant=is_single,
    )


def _check_field_optional(
    cls: ClassInfo,
    fld: FieldInfo,
    stats: ParamCallStats,
    config: NarrowerConfig,
) -> NarrowingCandidate | None:
    """Check if a dataclass/NamedTuple `T | None` field never receives None."""
    if fld.optional_inner is None:
        return None
    if fld.default_is_none and stats.omitted_count > 0:
        return None
    if _cannot_drop_field_none_default(cls, fld):
        return None
    if any(obs.is_none or not obs.is_non_none for obs in stats.observations):
        return None

    call_count = len(stats.observations)
    only_test = all(obs.is_test_caller for obs in stats.observations)
    score, notes = score_tier2_unused_optional_param(
        OptionalParamScoreContext(
            is_private=cls.is_private,
            closed_world=config.closed_world,
            call_count=call_count,
            min_call_sites=config.min_call_sites,
            only_test_callers=only_test,
            has_none_default=fld.default_is_none,
        )
    )
    return NarrowingCandidate(
        code=RuleCode.NT203,
        tier=Tier.WHOLE_REPO,
        file=cls.file,
        line=fld.line,
        col=fld.col,
        symbol=cls.qualname,
        target_kind=TargetKind.ATTRIBUTE,
        target_name=fld.name,
        current_type=fld.annotation_str,
        suggested_type=fld.optional_inner,
        needs_literal_import=False,
        message=(
            f"Field `{cls.name}.{fld.name}` is `{fld.annotation_str}`, "
            f"but None is never passed across {call_count} instantiation(s); "
            f"narrow to `{fld.optional_inner}`"
        ),
        evidence=[
            EvidenceItem(
                kind="call_sites",
                detail=f"Observed {call_count} instantiation(s), all passing non-None values",
            ),
            *(EvidenceItem(kind="score_factor", detail=note) for note in notes),
        ],
        score=score,
        is_public=not cls.is_private,
        drop_none_default=fld.default_is_none,
    )


def _rule_nt204_unused_union_param(
    func: FunctionInfo,
    param: ParamInfo,
    stats: ParamCallStats,
    config: NarrowerConfig,
) -> NarrowingCandidate | None:
    """NT204: Parameter union `A | B | C` has unused non-None branch(es) across callers."""
    members = param.union_members
    if members is None or param.annotation_str is None:
        return None

    observed_types: set[str] = set()
    for obs in stats.observations:
        if obs.observed_type is None or obs.observed_type in {
            "Any",
            "typing.Any",
            "object",
            "builtins.object",
        }:
            return None
        observed_types.add(obs.observed_type)

    if stats.omitted_count > 0 or (param.has_default and not param.default_is_none):
        if param.default_type is None:
            return None
        observed_types.add(param.default_type)

    used, unused = _partition_union_members(members, observed_types)
    if not used or not unused or unused == ["None"] or used == ["None"]:
        return None

    call_count = len(stats.observations) + stats.omitted_count
    only_test = all(obs.is_test_caller for obs in stats.observations)
    drops_none = "None" in unused and param.default_is_none
    if drops_none and _cannot_drop_param_none_default(func, param):
        return None
    score, notes = score_tier2_unused_optional_param(
        OptionalParamScoreContext(
            is_private=func.is_private,
            closed_world=config.closed_world,
            call_count=call_count,
            min_call_sites=config.min_call_sites,
            only_test_callers=only_test,
            has_none_default=drops_none,
        )
    )
    suggested = format_union_type(used)
    return NarrowingCandidate(
        code=RuleCode.NT204,
        tier=Tier.WHOLE_REPO,
        file=func.file,
        line=param.line,
        col=param.col,
        symbol=func.qualname,
        target_kind=TargetKind.PARAMETER,
        target_name=param.name,
        current_type=param.annotation_str,
        suggested_type=suggested,
        needs_literal_import=False,
        message=(
            f"Parameter `{param.name}` in `{func.name}` is `{param.annotation_str}`, "
            f"but branch(es) {', '.join(unused)} are never passed across {call_count} call(s); "
            f"narrow to `{suggested}`"
        ),
        evidence=[
            EvidenceItem(
                kind="call_sites",
                detail=f"Observed {call_count} call(s) with types: {', '.join(sorted(observed_types))}",
            ),
            *(EvidenceItem(kind="score_factor", detail=note) for note in notes),
        ],
        score=score,
        is_public=not func.is_private,
        drop_none_default=drops_none,
    )


def _rule_nt205_constant_bool_param(
    func: FunctionInfo,
    param: ParamInfo,
    stats: ParamCallStats,
    config: NarrowerConfig,
) -> NarrowingCandidate | None:
    """NT205: Boolean parameter always receives a single constant (True or False)."""
    if not param.is_bool or param.annotation_str is None:
        return None

    counts = _collect_literal_counts(
        stats=stats,
        expected_kind="bool",
        has_default=param.has_default,
        default_literal=param.default_literal,
    )
    if counts is None or len(counts) != 1:
        return None

    only_lit = next(iter(counts.keys()))
    total_calls = len(stats.observations) + stats.omitted_count
    only_test = all(obs.is_test_caller for obs in stats.observations)
    score, notes = score_tier2_constant_bool_param(
        ConstantBoolScoreContext(
            is_private=func.is_private,
            closed_world=config.closed_world,
            call_count=total_calls,
            min_call_sites=config.min_call_sites,
            only_test_callers=only_test,
            has_default=param.has_default,
        )
    )
    suggested = format_literal_type([only_lit])
    return NarrowingCandidate(
        code=RuleCode.NT205,
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
            f"Boolean parameter `{param.name}` in `{func.name}` always receives "
            f"{only_lit.value!r} across {total_calls} call(s); narrow to {suggested}"
        ),
        evidence=_build_callsite_literal_evidence(counts, total_calls, notes),
        score=score,
        is_public=not func.is_private,
        is_single_constant=True,
    )


def _collect_literal_counts(
    *,
    stats: ParamCallStats,
    expected_kind: str,
    has_default: bool,
    default_literal: ExtractedLiteral | None,
) -> Counter[ExtractedLiteral] | None:
    """Aggregate literal counts across observations and parameter/field defaults."""
    counts: Counter[ExtractedLiteral] = Counter()
    for obs in stats.observations:
        if obs.is_unknown or obs.is_none or not obs.literals:
            return None
        for lit in obs.literals:
            if lit.kind != expected_kind:
                return None
            counts[lit] += 1

    if has_default:
        if default_literal is None or default_literal.kind != expected_kind:
            return None
        if stats.omitted_count > 0:
            counts[default_literal] += stats.omitted_count
        else:
            counts.setdefault(default_literal, 0)
    return counts


def _partition_union_members(
    members: tuple[str, ...],
    observed_types: set[str],
) -> tuple[list[str], list[str]]:
    """Split union members into `(used, unused)` based on observed type names.

    Returns `(list(members), [])` (no narrowing) if any observed type cannot be
    matched to at least one declared member, which indicates an unresolved type
    alias, protocol, or subclass relationship.
    """
    expanded_observed = _expand_observed_union_types(observed_types)
    union_has_bool = any(m.strip() in {"bool", "builtins.bool"} for m in members)
    union_has_int = any(m.strip() in {"int", "builtins.int"} for m in members)

    for obs in expanded_observed:
        if not any(
            matches_union_member(
                obs,
                member,
                union_has_bool=union_has_bool,
                union_has_int=union_has_int,
            )
            for member in members
        ):
            return (list(members), [])

    used: list[str] = []
    unused: list[str] = []
    for member in members:
        if any(
            matches_union_member(
                obs,
                member,
                union_has_bool=union_has_bool,
                union_has_int=union_has_int,
            )
            for obs in expanded_observed
        ):
            used.append(member)
        else:
            unused.append(member)
    return (used, unused)


def _expand_observed_union_types(observed_types: set[str]) -> set[str]:
    """Flatten any union type strings in `observed_types` into their constituent members."""
    expanded: set[str] = set()
    for obs in observed_types:
        try:
            ann = cst.Annotation(annotation=cst.parse_expression(obs))
        except cst.ParserSyntaxError:
            expanded.add(obs)
            continue
        sub_members = extract_union_members(ann)
        if sub_members is not None:
            expanded.update(sub_members)
        else:
            expanded.add(obs)
    return expanded


_MAX_LITERAL_STR_LEN = 40
_PROSE_WORD_THRESHOLD = 3


def _has_prose_or_overlong_literal(literals: set[ExtractedLiteral]) -> bool:
    """Return True if any string literal looks like UI prose or exceeds the token length cap."""
    return any(
        lit.kind == "str"
        and isinstance(lit.value, str)
        and (
            len(lit.value) > _MAX_LITERAL_STR_LEN
            or len(lit.value.strip().split()) >= _PROSE_WORD_THRESHOLD
        )
        for lit in literals
    )


def _cannot_drop_param_none_default(func: FunctionInfo, param: ParamInfo) -> bool:
    """Return True if dropping `= None` on `param` would violate Python default ordering."""
    if not param.default_is_none or param.kind == "kwonly":
        return False
    for prev in func.parameters:
        if prev is param:
            break
        if prev.kind in {"posonly", "regular"} and prev.has_default:
            return True
    return False


def _cannot_drop_field_none_default(cls: ClassInfo, fld: FieldInfo) -> bool:
    """Return True if dropping `= None` on dataclass/NamedTuple `fld` breaks field ordering."""
    if not fld.default_is_none or fld.is_kw_only:
        return False
    for prev in cls.fields:
        if prev is fld:
            break
        if not prev.is_kw_only and prev.has_default:
            return True
    return False


def _class_has_subclasses(cls: ClassInfo, index: CodebaseIndex) -> bool:
    """Return True if any indexed class inherits from `cls`."""
    for other in index.classes.values():
        if other.qualname == cls.qualname:
            continue
        for base in other.bases:
            if base in {cls.name, cls.qualname} or base.rsplit(".", maxsplit=1)[-1] == cls.name:
                return True
    return False


def _skip_param_narrowing_for_hierarchy(func: FunctionInfo) -> bool:
    """Skip parameter narrowing on overrides, ABCs, Protocols, abstract, and data-model methods."""
    return (
        func.is_override
        or func.is_abstract
        or func.in_protocol_or_abc
        or _is_data_model_dunder(func)
    )


def _skip_return_narrowing_for_protocol(func: FunctionInfo, config: NarrowerConfig) -> bool:
    """Skip return narrowing on abstract, Protocol/ABC, data-model, or overridden base methods."""
    return (
        func.is_abstract
        or func.in_protocol_or_abc
        or func.is_overridden
        or (func.in_base_class and not func.is_private and not config.closed_world)
        or _is_data_model_dunder(func)
    )


def _is_data_model_dunder(func: FunctionInfo) -> bool:
    """Return True if func is a fixed Python data-model method or PEP 562 module hook."""
    if func.is_method:
        return func.name in DATA_MODEL_DUNDER_METHODS
    return func.name in MODULE_DUNDER_FUNCTIONS


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
    dist_parts = [
        f"{lit.value!r} ({count}x)" if count > 0 else f"{lit.value!r} (default)"
        for lit, count in counts.most_common()
    ]
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
    index: CodebaseIndex,
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
