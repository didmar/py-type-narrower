"""Confidence scoring heuristics for narrowing candidates."""

from __future__ import annotations

from dataclasses import dataclass

OPEN_DOMAIN_PARAM_NAMES: frozenset[str] = frozenset(
    {
        "name",
        "username",
        "filename",
        "filepath",
        "path",
        "url",
        "uri",
        "message",
        "msg",
        "text",
        "body",
        "content",
        "query",
        "title",
        "description",
        "email",
        "host",
        "hostname",
        "password",
        "token",
        "secret",
        "label",
        "comment",
        "sql",
        "prompt",
        "question",
        "reason",
        "error",
        "warning",
        "prefix",
        "suffix",
    }
)


_MULTI_CALL_THRESHOLD = 3


@dataclass(frozen=True)
class CallSiteScoreContext:
    """Inputs used to compute a transparent 0-100 confidence score for Tier 2 literal findings."""

    is_private: bool
    closed_world: bool
    call_count: int
    unique_value_count: int
    min_call_sites: int
    only_test_callers: bool
    has_body_corroboration: bool
    param_name: str


@dataclass(frozen=True)
class OptionalParamScoreContext:
    """Inputs used to compute a transparent 0-100 confidence score for NT202 / NT203 / NT204."""

    is_private: bool
    closed_world: bool
    call_count: int
    min_call_sites: int
    only_test_callers: bool
    has_none_default: bool


@dataclass(frozen=True)
class ConstantBoolScoreContext:
    """Inputs used to compute a transparent 0-100 confidence score for NT205."""

    is_private: bool
    closed_world: bool
    call_count: int
    min_call_sites: int
    only_test_callers: bool
    has_default: bool


def score_tier1_guarded_param(*, is_private: bool) -> int:
    """Score NT101 guarded parameter narrowing (always very high confidence)."""
    return 95 if is_private else 92


def score_tier1_unused_optional_return(*, is_private: bool) -> int:
    """Score NT102 unused None in return type."""
    return 95 if is_private else 90


def score_tier1_literal_return(*, is_private: bool, closed_world: bool) -> int:
    """Score NT103 literal return type narrowing."""
    if is_private or closed_world:
        return 90
    return 80


def score_tier1_always_initialized_attr(*, is_private: bool) -> int:
    """Score NT104 always-initialized class attribute narrowing."""
    return 92 if is_private else 88


def score_tier1_unused_union_return(*, is_private: bool) -> int:
    """Score NT105 unused union branch in return type."""
    return 92 if is_private else 88


def score_tier2_callsite_literal(ctx: CallSiteScoreContext) -> tuple[int, list[str]]:
    """Compute a 0-100 confidence score and explanation notes for NT201 / NT203 literals."""
    score = 55
    notes: list[str] = []

    if ctx.is_private or ctx.closed_world:
        score += 15
        notes.append("+15 internal/closed-world symbol")
    else:
        score -= 25
        notes.append("-25 public API without --closed-world")

    if ctx.call_count >= max(_MULTI_CALL_THRESHOLD, 2 * ctx.unique_value_count):
        score += 15
        notes.append(
            f"+15 strong repetition ({ctx.call_count} calls / {ctx.unique_value_count} values)"
        )
    elif ctx.call_count >= _MULTI_CALL_THRESHOLD:
        score += 8
        notes.append(f"+8 multiple call sites ({ctx.call_count} calls)")

    if ctx.has_body_corroboration:
        score += 10
        notes.append("+10 body also compares against these literals")

    if ctx.only_test_callers:
        score -= 20
        notes.append("-20 only observed in test callers")

    if ctx.call_count < ctx.min_call_sites:
        score -= 15
        notes.append(f"-15 fewer than {ctx.min_call_sites} call sites")

    if ctx.param_name.lower() in OPEN_DOMAIN_PARAM_NAMES:
        score -= 15
        notes.append(f"-15 parameter name `{ctx.param_name}` suggests open-ended data")

    return (_clamp_score(score), notes)


def score_tier2_unused_optional_param(ctx: OptionalParamScoreContext) -> tuple[int, list[str]]:
    """Compute a 0-100 confidence score and explanation notes for NT202 / NT203 / NT204."""
    score = 60
    notes: list[str] = []

    if ctx.is_private or ctx.closed_world:
        score += 15
        notes.append("+15 internal/closed-world symbol")
    else:
        score -= 20
        notes.append("-20 public API without --closed-world")

    if ctx.call_count >= _MULTI_CALL_THRESHOLD:
        score += 10
        notes.append(f"+10 observed across {ctx.call_count} call sites")
    elif ctx.call_count < ctx.min_call_sites:
        score -= 15
        notes.append(f"-15 fewer than {ctx.min_call_sites} call sites")

    if ctx.only_test_callers:
        score -= 20
        notes.append("-20 only observed in test callers")

    if ctx.has_none_default:
        score -= 5
        notes.append("-5 has `= None` default (never omitted by known callers)")

    return (_clamp_score(score), notes)


def score_tier2_constant_bool_param(ctx: ConstantBoolScoreContext) -> tuple[int, list[str]]:
    """Compute a 0-100 confidence score and explanation notes for NT205."""
    score = 55
    notes: list[str] = []

    if ctx.is_private or ctx.closed_world:
        score += 15
        notes.append("+15 internal/closed-world symbol")
    else:
        score -= 25
        notes.append("-25 public API without --closed-world")

    if ctx.call_count >= _MULTI_CALL_THRESHOLD:
        score += 15
        notes.append(f"+15 constant across {ctx.call_count} call sites")
    elif ctx.call_count < ctx.min_call_sites:
        score -= 15
        notes.append(f"-15 fewer than {ctx.min_call_sites} call sites")

    if ctx.only_test_callers:
        score -= 20
        notes.append("-20 only observed in test callers")

    if ctx.has_default:
        score -= 5
        notes.append("-5 has boolean default value")

    return (_clamp_score(score), notes)


def _clamp_score(value: int) -> int:
    """Clamp confidence score to [0, 100]."""
    return max(0, min(100, value))
