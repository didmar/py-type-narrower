"""Narrowing rule implementations (Tier 1 local rules and Tier 2 whole-repo rules)."""

from __future__ import annotations

import keyword
import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import libcst as cst

from py_type_narrower.body_analysis import filter_literals_by_kind
from py_type_narrower.cst_utils import (
    DictAnnotationSpec,
    ExtractedLiteral,
    extract_optional_container_return,
    extract_required_inner,
    extract_str_key_dict_annotation,
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
    TypedDictScaffold,
)
from py_type_narrower.scorer import (
    CallSiteScoreContext,
    ConstantBoolScoreContext,
    OptionalParamScoreContext,
    score_tier1_always_initialized_attr,
    score_tier1_concrete_any_return,
    score_tier1_container_non_none,
    score_tier1_fixed_tuple_return,
    score_tier1_guarded_param,
    score_tier1_literal_return,
    score_tier1_structured_dict_return,
    score_tier1_unused_optional_return,
    score_tier1_unused_union_return,
    score_tier2_callsite_literal,
    score_tier2_constant_bool_param,
    score_tier2_unused_optional_param,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from py_type_narrower.body_analysis import ParamGuardInfo, ReturnSummary
    from py_type_narrower.class_analysis import ClassAttrFact
    from py_type_narrower.config import NarrowerConfig
    from py_type_narrower.indexer import (
        ClassInfo,
        CodebaseIndex,
        FieldInfo,
        FunctionInfo,
        ParamCallStats,
        ParamInfo,
    )

    Tier1FunctionRule = Callable[[FunctionInfo, "_RuleEvalContext"], list[NarrowingCandidate]]
    Tier2ParamRule = Callable[
        [FunctionInfo, ParamInfo, ParamCallStats, "_RuleEvalContext"],
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


@dataclass(frozen=True)
class _RuleEvalContext:
    """Shared read-only evaluation context threaded across rule evaluators."""

    index: CodebaseIndex
    config: NarrowerConfig
    active: set[RuleCode]


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
    ctx = _RuleEvalContext(index=index, config=config, active=config.active_rules())
    resolved_targets = {p.resolve() for p in target_files} if target_files is not None else None
    candidates: list[NarrowingCandidate] = []

    for func in sorted(index.functions.values(), key=lambda f: (str(f.file), f.line, f.col)):
        if resolved_targets is not None and func.file.resolve() not in resolved_targets:
            continue
        if func.is_test_file and not config.include_test_files:
            continue
        _evaluate_function_rules(func, ctx, candidates)

    for cls in sorted(index.classes.values(), key=lambda c: (str(c.file), c.line, c.col)):
        if resolved_targets is not None and cls.file.resolve() not in resolved_targets:
            continue
        if cls.is_test_file and not config.include_test_files:
            continue
        _evaluate_class_rules(cls, ctx, candidates)

    return _filter_ignored_and_scored(candidates, index, config)


def _evaluate_function_rules(
    func: FunctionInfo,
    ctx: _RuleEvalContext,
    out: list[NarrowingCandidate],
) -> None:
    """Run Tier 1 and Tier 2 rules on a single FunctionInfo."""
    if ctx.config.max_tier >= Tier.LOCAL:
        _run_tier1_function_rules(func, ctx, out)
    if ctx.config.max_tier >= Tier.WHOLE_REPO:
        _run_tier2_function_rules(func, ctx, out)


def _evaluate_class_rules(
    cls: ClassInfo,
    ctx: _RuleEvalContext,
    out: list[NarrowingCandidate],
) -> None:
    """Run Tier 1 (NT104) and Tier 2 (NT203) rules on a single ClassInfo."""
    if ctx.config.max_tier >= Tier.LOCAL and RuleCode.NT104 in ctx.active:
        out.extend(_rule_nt104_always_initialized_attrs(cls, ctx))
    if ctx.config.max_tier >= Tier.WHOLE_REPO and RuleCode.NT203 in ctx.active:
        out.extend(_rule_nt203_structured_fields(cls, ctx.index, ctx.config))


def _run_tier1_function_rules(
    func: FunctionInfo,
    ctx: _RuleEvalContext,
    out: list[NarrowingCandidate],
) -> None:
    """Evaluate local high-confidence function rules in priority order."""
    rules: tuple[tuple[RuleCode, Tier1FunctionRule], ...] = (
        (RuleCode.NT101, _rule_nt101_guarded_params),
        (RuleCode.NT105, _rule_nt105_unused_union_return),
        (RuleCode.NT102, _rule_nt102_unused_optional_return),
        (RuleCode.NT103, _rule_nt103_literal_return),
        (RuleCode.NT106, _rule_nt106_fixed_tuple_return),
        (RuleCode.NT107, _rule_nt107_container_non_none_return),
        (RuleCode.NT108, _rule_nt108_concrete_any_return),
        (RuleCode.NT109, _rule_nt109_structured_dict_return),
    )
    seen_targets: set[tuple[TargetKind, str | None]] = set()
    for code, rule_fn in rules:
        if code not in ctx.active:
            continue
        for cand in rule_fn(func, ctx):
            target = (cand.target_kind, cand.target_name)
            if target not in seen_targets:
                seen_targets.add(target)
                out.append(cand)


def _run_tier2_function_rules(
    func: FunctionInfo,
    ctx: _RuleEvalContext,
    out: list[NarrowingCandidate],
) -> None:
    """Evaluate whole-repo call-site function rules (NT201, NT204, NT202, NT205, NT206)."""
    if _skip_param_narrowing_for_hierarchy(func):
        return
    nt101_params = {
        c.target_name for c in out if c.symbol == func.qualname and c.code == RuleCode.NT101
    }
    for param in _non_receiver_params(func):
        if param.name in nt101_params:
            continue
        guarded = _extract_nt101_matching_literals(func, param, ctx.index)
        if guarded is not None and _has_incompatible_nt101_caller(
            func, param, guarded[0], ctx.index
        ):
            continue
        stats = ctx.index.call_stats.get((func.qualname, param.name))
        if stats is None or not stats.observations or stats.has_dynamic_star_call:
            continue
        cand = _evaluate_param_tier2(func, param, stats, ctx)
        if cand is not None:
            out.append(cand)


def _evaluate_param_tier2(
    func: FunctionInfo,
    param: ParamInfo,
    stats: ParamCallStats,
    ctx: _RuleEvalContext,
) -> NarrowingCandidate | None:
    """Evaluate Tier 2 parameter rules in priority order (NT201, NT204, NT202, NT205, NT206)."""
    rules: tuple[tuple[RuleCode, Tier2ParamRule], ...] = (
        (RuleCode.NT201, _rule_nt201_callsite_literal),
        (RuleCode.NT204, _rule_nt204_unused_union_param),
        (RuleCode.NT202, _rule_nt202_unused_optional_param),
        (RuleCode.NT205, _rule_nt205_constant_bool_param),
        (RuleCode.NT206, _rule_nt206_callsite_structured_dict),
    )
    for code, rule_fn in rules:
        if code not in ctx.active:
            continue
        cand = rule_fn(func, param, stats, ctx)
        if cand is not None:
            return cand
    return None


def _rule_nt101_guarded_params(
    func: FunctionInfo,
    ctx: _RuleEvalContext,
) -> list[NarrowingCandidate]:
    """NT101: Parameter typed str/int is exhaustively guarded to specific literals in body."""
    if _skip_param_narrowing_for_hierarchy(func):
        return []

    results: list[NarrowingCandidate] = []
    for param in _non_receiver_params(func):
        extracted = _extract_nt101_matching_literals(func, param, ctx.index)
        if extracted is None:
            continue
        matching, reasons = extracted
        if len(matching) > ctx.config.max_literals:
            continue
        if _has_incompatible_nt101_caller(func, param, matching, ctx.index):
            continue

        suggested = format_literal_type(matching)
        evidence = [
            EvidenceItem(kind="body_guard", detail=reason, file=func.file, line=func.line)
            for reason in reasons
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
                current_type=param.annotation_str or param.primitive_kind or "str",
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


def _extract_nt101_matching_literals(
    func: FunctionInfo,
    param: ParamInfo,
    index: CodebaseIndex,
) -> tuple[set[ExtractedLiteral], list[str]] | None:
    """Return `(matching_literals, reasons)` if `param` has a strict body guard, else None."""
    if param.primitive_kind is None:
        return None
    guard = func.body_facts.param_guards.get(param.name)
    if guard is None:
        return None
    matching, reasons = _collect_guarded_literals_for_param(func, param, guard, index)
    if not matching:
        return None
    if param.has_default and param.default_literal is not None:
        if param.default_literal.kind != param.primitive_kind:
            return None
        matching = matching | {param.default_literal}
    return (matching, reasons)


def _collect_guarded_literals_for_param(
    func: FunctionInfo,
    param: ParamInfo,
    guard: ParamGuardInfo,
    index: CodebaseIndex,
) -> tuple[set[ExtractedLiteral], list[str]]:
    """Combine explicit body-guard literals with values from in-scope StrEnum/IntEnum coercions."""
    if param.primitive_kind is None:
        return (set(), [])
    matching = set(filter_literals_by_kind(guard.strict_literals, param.primitive_kind))
    reasons = list(guard.strict_evidence)
    if matching:
        return (matching, reasons)

    in_scope_enums = _in_scope_enums(func.module_name, index)
    for callee in sorted(guard.coerced_enum_names):
        enum_cls = in_scope_enums.get(callee)
        if (
            enum_cls is not None
            and enum_cls.enum_kind == param.primitive_kind
            and enum_cls.enum_values
        ):
            matching.update(enum_cls.enum_values)
            reasons.append(f"Coerced via `{callee}({param.name})`")
    return (matching, reasons)


def _has_incompatible_nt101_caller(
    func: FunctionInfo,
    param: ParamInfo,
    matching: set[ExtractedLiteral],
    index: CodebaseIndex,
) -> bool:
    """Return True if any indexed caller passes a non-literal or out-of-set argument to `param`."""
    stats = index.call_stats.get((func.qualname, param.name))
    if stats is None:
        return False
    if stats.has_dynamic_star_call:
        return True
    return any(
        obs.is_unknown
        or obs.is_none
        or not obs.literals
        or any(lit not in matching for lit in obs.literals)
        for obs in stats.observations
    )


def _rule_nt102_unused_optional_return(
    func: FunctionInfo,
    ctx: _RuleEvalContext,
) -> list[NarrowingCandidate]:
    """NT102: Return type `T | None` never returns `None` along any path."""
    if (
        func.return_optional_inner is None
        or func.return_annotation_str is None
        or _skip_return_narrowing_for_protocol(func, ctx.config)
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
    ctx: _RuleEvalContext,
) -> list[NarrowingCandidate]:
    """NT103: Return type `str`/`int` only returns constant literals or members of an Enum."""
    ret = func.body_facts.returns
    if (
        func.return_primitive_kind is None
        or func.return_annotation_str is None
        or func.is_override
        or _skip_return_narrowing_for_protocol(func, ctx.config)
        or ret.is_stub
        or ret.is_generator
        or not ret.has_returns
        or ret.can_fallthrough
    ):
        return []

    enum_cand = _try_enum_member_return_candidate(func, ctx)
    if enum_cand is not None:
        return [enum_cand]

    if ret.has_non_literal_return or not ret.returned_literals:
        return []

    matching = filter_literals_by_kind(ret.returned_literals, func.return_primitive_kind)
    if (
        len(matching) != len(ret.returned_literals)
        or not matching
        or len(matching) > ctx.config.max_literals
        or _has_prose_or_overlong_literal(matching)
        or (len(matching) == 1 and func.in_external_subclass and not func.is_private)
    ):
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
                is_private=func.is_private, closed_world=ctx.config.closed_world
            ),
            is_public=not func.is_private,
            is_single_constant=len(matching) == 1,
        )
    ]


def _try_enum_member_return_candidate(
    func: FunctionInfo,
    ctx: _RuleEvalContext,
) -> NarrowingCandidate | None:
    """Narrow `-> str` / `-> int` to an in-scope StrEnum/IntEnum when all returns are its members."""
    ret = func.body_facts.returns
    if (
        not ret.all_returns_enum_attr
        or not ret.returned_enum_attrs
        or func.return_annotation_str is None
    ):
        return None
    cls_names = {cls_name for cls_name, _member in ret.returned_enum_attrs}
    if len(cls_names) != 1:
        return None
    enum_name = next(iter(cls_names))
    enum_cls = _in_scope_enums(func.module_name, ctx.index).get(enum_name)
    if enum_cls is None or enum_cls.enum_kind != func.return_primitive_kind:
        return None
    if not all(member in enum_cls.enum_members for _cls, member in ret.returned_enum_attrs):
        return None
    if len(ret.returned_enum_attrs) == 1 and func.in_external_subclass and not func.is_private:
        return None

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
        suggested_type=enum_name,
        needs_literal_import=False,
        message=(
            f"Return type of `{func.name}` is `{func.return_annotation_str}`, "
            f"but every return yields a `{enum_name}` enum member"
        ),
        evidence=[
            EvidenceItem(
                kind="return_enum_members",
                detail=f"All return statements yield members of `{enum_name}`",
                file=func.file,
                line=func.line,
            )
        ],
        score=score_tier1_literal_return(
            is_private=func.is_private, closed_world=ctx.config.closed_world
        ),
        is_public=not func.is_private,
    )


def _rule_nt104_always_initialized_attrs(
    cls: ClassInfo,
    ctx: _RuleEvalContext,
) -> list[NarrowingCandidate]:
    """NT104: Class attributes narrowable via class-local Optional, Literal, or Union analysis."""
    has_subclasses = _class_has_subclasses(cls, ctx.index)
    results: list[NarrowingCandidate] = []
    for fact in cls.attr_facts:
        cand = _evaluate_single_attr_fact(cls, fact, ctx.config, has_subclasses=has_subclasses)
        if cand is not None:
            results.append(cand)
    return results


def _evaluate_single_attr_fact(
    cls: ClassInfo,
    fact: ClassAttrFact,
    config: NarrowerConfig,
    *,
    has_subclasses: bool,
) -> NarrowingCandidate | None:
    """Turn a ClassAttrFact (optional, literal, or union) into an NT104 candidate if eligible."""
    if fact.kind == "optional" and fact.optional_inner is not None:
        loc_note = "in `__init__`" if fact.declared_in_init else "on class body and `__init__`"
        return NarrowingCandidate(
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
    if has_subclasses:
        return None
    if fact.kind == "literal":
        return _nt104_literal_attr_candidate(cls, fact, config)
    if fact.kind == "union":
        return _nt104_union_attr_candidate(cls, fact)
    return None


def _nt104_literal_attr_candidate(
    cls: ClassInfo,
    fact: ClassAttrFact,
    config: NarrowerConfig,
) -> NarrowingCandidate | None:
    """Build an NT104 candidate for a class attribute only assigned constant str/int literals."""
    if fact.primitive_kind is None:
        return None
    matching = filter_literals_by_kind(set(fact.observed_literals), fact.primitive_kind)
    if (
        len(matching) != len(fact.observed_literals)
        or not matching
        or len(matching) > config.max_literals
        or _has_prose_or_overlong_literal(matching)
    ):
        return None
    # Avoid narrowing a single-constant `int` attribute (often a sentinel like `-1`/`0` mutated by
    # friend classes), or a public mutable attribute initialized to a single sentinel in open-world.
    is_attr_private = fact.name.startswith("_") or cls.is_private
    if len(matching) == 1 and (
        fact.primitive_kind == "int" or (not is_attr_private and not config.closed_world)
    ):
        return None

    suggested = format_literal_type(matching)
    return NarrowingCandidate(
        code=RuleCode.NT104,
        tier=Tier.LOCAL,
        file=cls.file,
        line=fact.line,
        col=fact.col,
        symbol=cls.qualname,
        target_kind=TargetKind.ATTRIBUTE,
        target_name=fact.name,
        current_type=fact.annotation_str,
        suggested_type=suggested,
        needs_literal_import=True,
        message=(
            f"Attribute `{cls.name}.{fact.name}` is `{fact.annotation_str}`, "
            f"but is only assigned {suggested} across class methods"
        ),
        evidence=[
            EvidenceItem(
                kind="class_attr_literals",
                detail=f"All assignments to `self.{fact.name}` use literals in {suggested}",
                file=cls.file,
                line=fact.line,
            )
        ],
        score=score_tier1_always_initialized_attr(is_private=is_attr_private),
        is_public=not is_attr_private,
        is_single_constant=len(matching) == 1,
    )


def _nt104_union_attr_candidate(
    cls: ClassInfo,
    fact: ClassAttrFact,
) -> NarrowingCandidate | None:
    """Build an NT104 candidate for a union class attribute with unused non-None branch(es)."""
    used, unused = _partition_union_members(fact.union_members, set(fact.observed_types))
    if not used or not unused or unused == ["None"] or used == ["None"]:
        return None
    suggested = format_union_type(used)
    drops_none = "None" in unused and fact.has_none_default
    return NarrowingCandidate(
        code=RuleCode.NT104,
        tier=Tier.LOCAL,
        file=cls.file,
        line=fact.line,
        col=fact.col,
        symbol=cls.qualname,
        target_kind=TargetKind.ATTRIBUTE,
        target_name=fact.name,
        current_type=fact.annotation_str,
        suggested_type=suggested,
        needs_literal_import=False,
        message=(
            f"Attribute `{cls.name}.{fact.name}` is `{fact.annotation_str}`, "
            f"but branch(es) {', '.join(unused)} are never assigned; narrow to `{suggested}`"
        ),
        evidence=[
            EvidenceItem(
                kind="class_attr_union",
                detail=f"Observed assigned types: {', '.join(sorted(fact.observed_types))}",
                file=cls.file,
                line=fact.line,
            )
        ],
        score=score_tier1_always_initialized_attr(
            is_private=fact.name.startswith("_") or cls.is_private
        ),
        is_public=not (fact.name.startswith("_") or cls.is_private),
        drop_none_default=drops_none,
    )


def _rule_nt105_unused_union_return(
    func: FunctionInfo,
    ctx: _RuleEvalContext,
) -> list[NarrowingCandidate]:
    """NT105: Return union `A | B | C` has unused non-None branch(es)."""
    members = func.return_union_members
    ret = func.body_facts.returns
    if (
        members is None
        or func.return_annotation_str is None
        or _skip_return_narrowing_for_protocol(func, ctx.config)
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
    ctx: _RuleEvalContext,
) -> list[NarrowingCandidate]:
    """NT106: Variadic `tuple[T, ...]` return always yields a fixed-length tuple literal."""
    if func.return_annotation_str is None or _skip_return_narrowing_for_protocol(func, ctx.config):
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


def _rule_nt107_container_non_none_return(
    func: FunctionInfo,
    ctx: _RuleEvalContext,
) -> list[NarrowingCandidate]:
    """NT107: Container/generator return `list[T | None]` / `Iterator[T | None]` never yields None."""
    if func.return_annotation_str is None:
        return []
    spec = extract_optional_container_return(func.return_annotation_str)
    if spec is None or _skip_return_narrowing_for_protocol(
        func, ctx.config, invariant=spec.is_mutable
    ):
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


_BARE_GENERIC_CONTAINERS: frozenset[str] = frozenset({"list", "dict", "set", "frozenset", "tuple"})
_MAX_ANY_RETURN_UNION_MEMBERS = 3


def _rule_nt108_concrete_any_return(
    func: FunctionInfo,
    ctx: _RuleEvalContext,
) -> list[NarrowingCandidate]:
    """NT108: Return type `Any` / `object` only returns concrete inferable type(s)."""
    if (
        func.return_annotation_str is None
        or func.return_annotation_str.strip() not in _ANY_OR_OBJECT_TYPES
        or _skip_return_narrowing_for_protocol(func, ctx.config)
    ):
        return []
    ret = func.body_facts.returns
    if ret.is_stub or ret.is_generator or not ret.has_returns or ret.has_unknown_return_type:
        return []

    observed = set(ret.returned_types)
    if ret.can_fallthrough:
        observed.add("None")
    if (
        not observed
        or observed == {"None"}
        or len(observed) > _MAX_ANY_RETURN_UNION_MEMBERS
        or any(t in _BARE_GENERIC_CONTAINERS or t in _ANY_OR_OBJECT_TYPES for t in observed)
    ):
        return []

    ordered = sorted(t for t in observed if t != "None")
    if "None" in observed:
        ordered.append("None")
    suggested = format_union_type(ordered)
    return [
        NarrowingCandidate(
            code=RuleCode.NT108,
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
                f"but all return paths produce `{suggested}`"
            ),
            evidence=[
                EvidenceItem(
                    kind="concrete_return_types",
                    detail=f"Observed concrete return types: {', '.join(ordered)}",
                    file=func.file,
                    line=func.line,
                )
            ],
            score=score_tier1_concrete_any_return(is_private=func.is_private),
            is_public=not func.is_private,
        )
    ]


@dataclass(frozen=True)
class _DictShape:
    """Observed constant-string-key shape aggregated across dict literals."""

    sample_count: int
    key_presence: dict[str, int]
    key_types: dict[str, set[str]]
    key_unknown: set[str]


@dataclass(frozen=True)
class _DictNarrowingResolution:
    """Resolved target type and metadata for a structured dict narrowing."""

    suggested_type: str
    is_fixable: bool
    needs_literal_import: bool
    matched_existing_td: bool
    detail: str
    scaffold: TypedDictScaffold | None = None


_MAX_TYPEDDICT_KEYS = 12
_BARE_CONTAINER_DEFAULTS: dict[str, str] = {
    "list": "list[Any]",
    "dict": "dict[str, Any]",
    "set": "set[Any]",
    "frozenset": "frozenset[Any]",
    "tuple": "tuple[Any, ...]",
}
_BUILTIN_TYPE_NAMES: frozenset[str] = frozenset(
    {
        "str",
        "int",
        "float",
        "bool",
        "bytes",
        "complex",
        "None",
        "Any",
        "object",
        "list",
        "dict",
        "set",
        "frozenset",
        "tuple",
    }
)
_IDENT_TOKEN_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


def _rule_nt109_structured_dict_return(
    func: FunctionInfo,
    ctx: _RuleEvalContext,
) -> list[NarrowingCandidate]:
    """NT109: Structured dict return narrows to an existing TypedDict, dict[Literal, V], or TypedDict schema."""
    if func.return_annotation_str is None:
        return []
    dict_spec = extract_str_key_dict_annotation(func.return_annotation_str)
    if dict_spec is None or _skip_return_narrowing_for_protocol(
        func, ctx.config, invariant=dict_spec.is_mutable
    ):
        return []
    ret = func.body_facts.returns
    if (
        ret.is_stub
        or ret.is_generator
        or not ret.has_returns
        or ret.can_fallthrough
        or not ret.all_returns_const_str_dict
        or ret.dict_return_count == 0
        or not ret.dict_key_presence
    ):
        return []

    shape = _DictShape(
        sample_count=ret.dict_return_count,
        key_presence=ret.dict_key_presence,
        key_types=ret.dict_key_types,
        key_unknown=ret.dict_key_unknown,
    )
    resolution = _resolve_structured_dict_narrowing(
        dict_spec=dict_spec,
        shape=shape,
        func=func,
        param_name=None,
        ctx=ctx,
    )
    if resolution is None:
        return []

    return [
        NarrowingCandidate(
            code=RuleCode.NT109,
            tier=Tier.LOCAL,
            file=func.file,
            line=func.return_line,
            col=func.return_col,
            symbol=func.qualname,
            target_kind=TargetKind.RETURN,
            target_name=None,
            current_type=func.return_annotation_str,
            suggested_type=resolution.suggested_type,
            needs_literal_import=resolution.needs_literal_import,
            message=(
                f"Return type of `{func.name}` is `{func.return_annotation_str}`, "
                f"but all returns have fixed keys; narrow to `{resolution.suggested_type}`"
            ),
            evidence=[
                EvidenceItem(
                    kind="structured_dict_return",
                    detail=resolution.detail,
                    file=func.file,
                    line=func.line,
                )
            ],
            score=score_tier1_structured_dict_return(
                is_private=func.is_private,
                matched_existing=resolution.matched_existing_td,
            ),
            is_public=not func.is_private,
            is_fixable=resolution.is_fixable,
            typeddict_scaffold=resolution.scaffold,
        )
    ]


def _rule_nt206_callsite_structured_dict(
    func: FunctionInfo,
    param: ParamInfo,
    stats: ParamCallStats,
    ctx: _RuleEvalContext,
) -> NarrowingCandidate | None:
    """NT206: Call-site structured dict parameter narrows to TypedDict or dict[Literal, V]."""
    dict_spec = extract_str_key_dict_annotation(param.annotation_str)
    if dict_spec is None or param.annotation_str is None or param.has_default:
        return None
    if any(obs.dict_literal_keys is None for obs in stats.observations):
        return None

    shape = _aggregate_callsite_dict_shape(stats)
    if shape is None:
        return None

    resolution = _resolve_structured_dict_narrowing(
        dict_spec=dict_spec,
        shape=shape,
        func=func,
        param_name=param.name,
        ctx=ctx,
    )
    if resolution is None:
        return None

    score, notes = score_tier2_unused_optional_param(
        OptionalParamScoreContext(
            is_private=func.is_private,
            closed_world=ctx.config.closed_world,
            call_count=len(stats.observations),
            min_call_sites=ctx.config.min_call_sites,
            only_test_callers=all(o.is_test_caller for o in stats.observations),
            has_none_default=False,
        )
    )
    evidence = [
        EvidenceItem(
            kind="callsite_structured_dict",
            detail=f"{len(stats.observations)} call(s); {resolution.detail}",
            file=func.file,
            line=param.line,
        ),
        *[EvidenceItem(kind="score_factor", detail=note) for note in notes],
    ]
    return NarrowingCandidate(
        code=RuleCode.NT206,
        tier=Tier.WHOLE_REPO,
        file=func.file,
        line=param.line,
        col=param.col,
        symbol=func.qualname,
        target_kind=TargetKind.PARAMETER,
        target_name=param.name,
        current_type=param.annotation_str,
        suggested_type=resolution.suggested_type,
        needs_literal_import=resolution.needs_literal_import,
        message=(
            f"Parameter `{param.name}` in `{func.name}` is `{param.annotation_str}`, "
            f"but all {len(stats.observations)} call(s) pass structured dict literals; "
            f"narrow to `{resolution.suggested_type}`"
        ),
        evidence=evidence,
        score=score,
        is_public=not func.is_private,
        is_fixable=resolution.is_fixable,
        typeddict_scaffold=resolution.scaffold,
    )


def _aggregate_callsite_dict_shape(stats: ParamCallStats) -> _DictShape | None:
    """Aggregate constant-string-key dict observations across call sites."""
    key_presence: dict[str, int] = {}
    key_types: dict[str, set[str]] = {}
    key_unknown: set[str] = set()
    for obs in stats.observations:
        entries = obs.dict_literal_keys
        if entries is None:
            return None
        seen: set[str] = set()
        for key, val_type in entries:
            seen.add(key)
            if val_type is None:
                key_unknown.add(key)
            else:
                key_types.setdefault(key, set()).add(val_type)
        for key in seen:
            key_presence[key] = key_presence.get(key, 0) + 1
    if not key_presence:
        return None
    return _DictShape(
        sample_count=len(stats.observations),
        key_presence=key_presence,
        key_types=key_types,
        key_unknown=key_unknown,
    )


def _resolve_structured_dict_narrowing(
    *,
    dict_spec: DictAnnotationSpec,
    shape: _DictShape,
    func: FunctionInfo,
    param_name: str | None,
    ctx: _RuleEvalContext,
) -> _DictNarrowingResolution | None:
    """Choose best narrowing: existing in-scope TypedDict, dict[Literal, V], or TypedDict schema/scaffold."""
    max_literals = ctx.config.max_literals
    key_count = len(shape.key_presence)
    if key_count == 0 or key_count > max(max_literals, _MAX_TYPEDDICT_KEYS):
        return None

    matched_td = _find_matching_typeddict(shape, func.module_name, ctx.index)
    if matched_td is not None:
        keys_str = ", ".join(repr(k) for k in sorted(shape.key_presence))
        return _DictNarrowingResolution(
            suggested_type=matched_td,
            is_fixable=True,
            needs_literal_import=False,
            matched_existing_td=True,
            detail=f"Matches existing TypedDict `{matched_td}` for keys {{{keys_str}}}",
        )

    if (
        not dict_spec.is_broad_value
        and dict_spec.value_type is not None
        and key_count <= max_literals
    ):
        key_lits = [ExtractedLiteral(kind="str", value=k) for k in sorted(shape.key_presence)]
        lit_str = format_literal_type(key_lits)
        suggested = f"{dict_spec.outer}[{lit_str}, {dict_spec.value_type}]"
        return _DictNarrowingResolution(
            suggested_type=suggested,
            is_fixable=True,
            needs_literal_import=True,
            matched_existing_td=False,
            detail=f"All dict keys belong to {lit_str}",
        )

    if dict_spec.is_broad_value:
        schema = _format_typeddict_suggestion(shape)
        if ctx.config.scaffold_typeddict:
            scaffold = _build_typeddict_scaffold(shape, func, param_name, ctx.index)
            return _DictNarrowingResolution(
                suggested_type=scaffold.class_name,
                is_fixable=True,
                needs_literal_import=False,
                matched_existing_td=False,
                detail=f"Scaffold TypedDict `{scaffold.class_name}` with fields: {schema}",
                scaffold=scaffold,
            )
        return _DictNarrowingResolution(
            suggested_type=schema,
            is_fixable=False,
            needs_literal_import=False,
            matched_existing_td=False,
            detail=f"Consider defining a TypedDict with fields: {schema}",
        )
    return None


def _build_typeddict_scaffold(
    shape: _DictShape,
    func: FunctionInfo,
    param_name: str | None,
    index: CodebaseIndex,
) -> TypedDictScaffold:
    """Construct a module-level TypedDictScaffold specification from an observed _DictShape."""
    visible_names = _module_visible_type_names(func.module_name, index)
    fields: list[tuple[str, str, bool]] = []
    for key in sorted(shape.key_presence):
        val_str = _resolve_scaffold_field_type(key, shape, visible_names)
        is_required = shape.key_presence[key] >= shape.sample_count
        fields.append((key, val_str, is_required))

    all_optional = all(not is_req for _, _, is_req in fields)
    total = not all_optional
    needs_not_required = total and any(not is_req for _, _, is_req in fields)
    needs_any = any("Any" in _IDENT_TOKEN_RE.findall(val_str) for _, val_str, _ in fields)
    use_functional = any(not key.isidentifier() or keyword.iskeyword(key) for key, _, _ in fields)
    class_name = _make_scaffold_class_name(func, param_name, index)
    anchor_symbol = (
        func.class_name.split(".", maxsplit=1)[0]
        if (func.is_method and func.class_name is not None)
        else func.name
    )
    return TypedDictScaffold(
        class_name=class_name,
        fields=tuple(fields),
        anchor_symbol=anchor_symbol,
        use_functional_syntax=use_functional,
        total=total,
        needs_not_required_import=needs_not_required,
        needs_any_import=needs_any,
    )


def _resolve_scaffold_field_type(
    key: str,
    shape: _DictShape,
    visible_names: set[str],
) -> str:
    """Resolve a concrete, module-visible type string for a scaffolded TypedDict field."""
    if key in shape.key_unknown or not shape.key_types.get(key):
        return "Any"
    normalized: set[str] = set()
    for raw in shape.key_types[key]:
        mapped = _BARE_CONTAINER_DEFAULTS.get(raw.strip(), raw.strip())
        if _all_type_tokens_visible(mapped, visible_names):
            normalized.add(mapped)
        else:
            return "Any"
    ordered = sorted(t for t in normalized if t != "None")
    if "None" in normalized:
        ordered.append("None")
    return format_union_type(ordered)


def _all_type_tokens_visible(type_str: str, visible_names: set[str]) -> bool:
    """Return True if every identifier in `type_str` is a builtin or visible in the module."""
    tokens = _IDENT_TOKEN_RE.findall(type_str)
    return all(tok in _BUILTIN_TYPE_NAMES or tok in visible_names for tok in tokens)


def _make_scaffold_class_name(
    func: FunctionInfo,
    param_name: str | None,
    index: CodebaseIndex,
) -> str:
    """Generate a collision-free PascalCase TypedDict class name for a function or parameter."""
    raw_parts: list[str] = []
    if func.class_name is not None:
        raw_parts.extend(func.class_name.split("."))
    raw_parts.append(func.name)
    if param_name is not None:
        raw_parts.append(param_name)

    camel = "".join(_to_pascal_case(part) for part in raw_parts) or "Record"
    prefix = "_" if (func.is_private or func.name.startswith("_")) else ""
    base_name = f"{prefix}{camel}Dict"

    used = _module_visible_type_names(func.module_name, index)
    for fn in index.functions.values():
        if fn.module_name == func.module_name and not fn.is_method:
            used.add(fn.name)
    if base_name not in used:
        return base_name
    suffix = 2
    while f"{base_name}{suffix}" in used:
        suffix += 1
    return f"{base_name}{suffix}"


def _module_visible_type_names(module_name: str, index: CodebaseIndex) -> set[str]:
    """Collect class names and imported symbols visible at top level in `module_name`."""
    names: set[str] = set(index.module_imports.get(module_name, {}).keys())
    for cls in index.classes.values():
        if cls.module_name == module_name:
            names.add(cls.name)
    return names


def _to_pascal_case(raw: str) -> str:
    """Convert snake_case or mixed identifiers to PascalCase while preserving internal capitals."""
    chunks = [chunk for chunk in raw.strip("_").split("_") if chunk]
    return "".join(chunk[0].upper() + chunk[1:] for chunk in chunks)


def _find_matching_typeddict(
    shape: _DictShape,
    module_name: str,
    index: CodebaseIndex,
) -> str | None:
    """Return the unique in-scope TypedDict name matching `shape`, or None if 0 or >1 match."""
    in_scope: dict[str, ClassInfo] = {
        cls.name: cls
        for cls in index.classes.values()
        if cls.module_name == module_name and cls.is_typeddict
    }
    for local_name, target_qual in index.module_imports.get(module_name, {}).items():
        target_cls = index.classes.get(target_qual)
        if target_cls is not None and target_cls.is_typeddict:
            in_scope[local_name] = target_cls

    matches = [
        name for name, td_cls in sorted(in_scope.items()) if _typeddict_matches_shape(td_cls, shape)
    ]
    return matches[0] if len(matches) == 1 else None


def _typeddict_matches_shape(td_cls: ClassInfo, shape: _DictShape) -> bool:
    """Return True if every observed key and value type satisfies `td_cls`'s TypedDict contract."""
    if not td_cls.fields:
        return False
    field_map = {f.name: f for f in td_cls.fields}
    if not set(shape.key_presence.keys()).issubset(field_map.keys()):
        return False
    for f in td_cls.fields:
        if not f.is_not_required and shape.key_presence.get(f.name, 0) != shape.sample_count:
            return False
    return all(_typeddict_field_accepts_shape(field_map[key], shape) for key in shape.key_presence)


def _typeddict_field_accepts_shape(field_info: FieldInfo, shape: _DictShape) -> bool:
    """Return True if all observed value types for `field_info.name` match the declared field type."""
    declared = _unwrap_typeddict_field_type(field_info)
    if declared in _ANY_OR_OBJECT_TYPES:
        return True
    if field_info.name in shape.key_unknown:
        return False
    observed = shape.key_types.get(field_info.name, set())
    if not observed:
        return False
    members = _parse_union_members(declared) or (declared,)
    union_has_bool = any(m.strip() in {"bool", "builtins.bool"} for m in members)
    union_has_int = any(m.strip() in {"int", "builtins.int"} for m in members)
    return all(
        any(
            matches_union_member(
                obs,
                member,
                union_has_bool=union_has_bool,
                union_has_int=union_has_int,
            )
            for member in members
        )
        for obs in observed
    )


def _parse_union_members(type_str: str) -> tuple[str, ...] | None:
    """Parse a type string as a top-level union if possible."""
    try:
        expr = cst.parse_expression(type_str)
    except cst.ParserSyntaxError:
        return None
    return extract_union_members(cst.Annotation(annotation=expr))


def _unwrap_typeddict_field_type(field_info: FieldInfo) -> str:
    """Strip NotRequired[...] / Required[...] wrapper from a TypedDict field annotation."""
    if field_info.not_required_inner is not None:
        return field_info.not_required_inner
    try:
        expr = cst.parse_expression(field_info.annotation_str)
    except cst.ParserSyntaxError:
        return field_info.annotation_str
    req_inner = extract_required_inner(cst.Annotation(annotation=expr))
    return req_inner if req_inner is not None else field_info.annotation_str


def _format_typeddict_suggestion(shape: _DictShape) -> str:
    """Format a compact TypedDict schema preview showing required and NotRequired fields."""
    parts: list[str] = []
    for key in sorted(shape.key_presence):
        if key in shape.key_unknown or not shape.key_types.get(key):
            val_str = "Any"
        else:
            obs = shape.key_types[key]
            ordered = sorted(t for t in obs if t != "None")
            if "None" in obs:
                ordered.append("None")
            val_str = format_union_type(ordered)
        if shape.key_presence[key] < shape.sample_count:
            val_str = f"NotRequired[{val_str}]"
        parts.append(f"{key!r}: {val_str}")
    return f"TypedDict({{{', '.join(parts)}}})"


def _rule_nt201_callsite_literal(
    func: FunctionInfo,
    param: ParamInfo,
    stats: ParamCallStats,
    ctx: _RuleEvalContext,
) -> NarrowingCandidate | None:
    """NT201: Call-site `str`/`int` parameter only receives literals or members of an Enum."""
    if param.primitive_kind is None or param.annotation_str is None:
        return None

    enum_cand = _try_callsite_enum_candidate(func, param, stats, ctx)
    if enum_cand is not None:
        return enum_cand

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
        or len(unique_lits) > ctx.config.max_literals
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
            closed_world=ctx.config.closed_world,
            call_count=total_calls,
            unique_value_count=len(unique_lits),
            min_call_sites=ctx.config.min_call_sites,
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


def _try_callsite_enum_candidate(
    func: FunctionInfo,
    param: ParamInfo,
    stats: ParamCallStats,
    ctx: _RuleEvalContext,
) -> NarrowingCandidate | None:
    """Narrow `param: str/int` to an in-scope StrEnum/IntEnum when all callers pass its members."""
    if param.annotation_str is None or param.has_default:
        return None
    enum_quals = {obs.enum_qualname for obs in stats.observations}
    if len(enum_quals) != 1 or None in enum_quals:
        return None
    target_qual = next(iter(enum_quals))
    in_scope = _in_scope_enums(func.module_name, ctx.index)
    visible_matches = [
        name
        for name, cls in sorted(in_scope.items())
        if cls.qualname == target_qual and cls.enum_kind == param.primitive_kind
    ]
    if len(visible_matches) != 1:
        return None
    enum_name = visible_matches[0]
    total_calls = len(stats.observations) + stats.omitted_count
    score, notes = score_tier2_unused_optional_param(
        OptionalParamScoreContext(
            is_private=func.is_private,
            closed_world=ctx.config.closed_world,
            call_count=total_calls,
            min_call_sites=ctx.config.min_call_sites,
            only_test_callers=all(o.is_test_caller for o in stats.observations),
            has_none_default=False,
        )
    )
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
        suggested_type=enum_name,
        needs_literal_import=False,
        message=(
            f"Parameter `{param.name}` in `{func.name}` only receives `{enum_name}` "
            f"enum values across {total_calls} call(s)"
        ),
        evidence=[
            EvidenceItem(
                kind="callsite_enum",
                detail=f"All {total_calls} call(s) pass `{enum_name}` enum values",
            ),
            *(EvidenceItem(kind="score_factor", detail=note) for note in notes),
        ],
        score=score,
        is_public=not func.is_private,
    )


def _in_scope_enums(module_name: str, index: CodebaseIndex) -> dict[str, ClassInfo]:
    """Return mapping of visible local name -> StrEnum/IntEnum ClassInfo for `module_name`."""
    in_scope: dict[str, ClassInfo] = {
        cls.name: cls
        for cls in index.classes.values()
        if cls.module_name == module_name and cls.enum_kind is not None
    }
    for local_name, target_qual in index.module_imports.get(module_name, {}).items():
        target_cls = index.classes.get(target_qual)
        if target_cls is not None and target_cls.enum_kind is not None:
            in_scope[local_name] = target_cls
    return in_scope


def _rule_nt202_unused_optional_param(
    func: FunctionInfo,
    param: ParamInfo,
    stats: ParamCallStats,
    ctx: _RuleEvalContext,
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
            closed_world=ctx.config.closed_world,
            call_count=call_count,
            min_call_sites=ctx.config.min_call_sites,
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
    """NT203: Dataclass / NamedTuple / attrs / Pydantic / TypedDict field narrowing."""
    if not cls.is_structured_model or cls.is_protocol_or_abc or not cls.fields:
        return []
    if _class_has_subclasses(cls, index):
        return []

    results: list[NarrowingCandidate] = []
    for fld in cls.fields:
        stats = index.field_stats.get((cls.qualname, fld.name))
        if stats is None or not stats.observations or stats.has_dynamic_star_call:
            continue
        for checker in (_check_field_not_required, _check_field_literal, _check_field_optional):
            cand = checker(cls, fld, stats, config)
            if cand is not None:
                results.append(cand)
                break
    return results


def _check_field_not_required(
    cls: ClassInfo,
    fld: FieldInfo,
    stats: ParamCallStats,
    config: NarrowerConfig,
) -> NarrowingCandidate | None:
    """Check if a TypedDict `NotRequired[T]` field is always supplied across instantiations."""
    if not cls.is_typeddict or fld.not_required_inner is None or stats.omitted_count > 0:
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
            has_none_default=False,
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
        suggested_type=fld.not_required_inner,
        needs_literal_import=False,
        message=(
            f"TypedDict field `{cls.name}.{fld.name}` is `{fld.annotation_str}`, "
            f"but is always provided across {call_count} instantiation(s); "
            f"narrow to `{fld.not_required_inner}`"
        ),
        evidence=[
            EvidenceItem(
                kind="call_sites",
                detail=f"Observed {call_count} instantiation(s), never omitted",
            ),
            *(EvidenceItem(kind="score_factor", detail=note) for note in notes),
        ],
        score=score,
        is_public=not cls.is_private,
    )


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
    ctx: _RuleEvalContext,
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
            closed_world=ctx.config.closed_world,
            call_count=call_count,
            min_call_sites=ctx.config.min_call_sites,
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
    ctx: _RuleEvalContext,
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
            closed_world=ctx.config.closed_world,
            call_count=total_calls,
            min_call_sites=ctx.config.min_call_sites,
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
    """Skip contravariant parameter narrowing on overrides, external subclasses, ABCs, or dunders."""
    return (
        _may_override_base_contract(func)
        or func.is_abstract
        or func.in_protocol_or_abc
        or _is_data_model_dunder(func)
    )


def _skip_return_narrowing_for_protocol(
    func: FunctionInfo,
    config: NarrowerConfig,
    *,
    invariant: bool = False,
) -> bool:
    """Skip return narrowing on protocol/ABC/base hooks, and on overrides when `invariant=True`."""
    return (
        func.is_abstract
        or func.in_protocol_or_abc
        or func.is_overridden
        or (func.in_base_class and not func.is_private and not config.closed_world)
        or (invariant and _may_override_base_contract(func))
        or _is_data_model_dunder(func)
    )


def _may_override_base_contract(func: FunctionInfo) -> bool:
    """Return True if `func` overrides an indexed base method or is a public external-subclass method."""
    return func.is_override or (
        func.in_external_subclass and not func.is_private and func.name != "__init__"
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
