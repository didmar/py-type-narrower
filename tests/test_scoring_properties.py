from dataclasses import replace

from hypothesis import given
from hypothesis import strategies as st

from py_type_narrower.scorer import (
    OPEN_DOMAIN_PARAM_NAMES,
    CallSiteScoreContext,
    ConfidenceBand,
    OptionalParamScoreContext,
    confidence_band,
    score_tier1_guarded_param,
    score_tier1_literal_return,
    score_tier1_unused_optional_return,
    score_tier2_callsite_literal,
    score_tier2_unused_optional_param,
)

_SAFE_PARAM_NAMES = st.sampled_from(["mode", "level", "env", "fmt", "kind", "action", "strategy"])
_OPEN_PARAM_NAMES = st.sampled_from(sorted(OPEN_DOMAIN_PARAM_NAMES))
_ANY_PARAM_NAMES = st.one_of(_SAFE_PARAM_NAMES, _OPEN_PARAM_NAMES)

_CALLSITE_CTX = st.builds(
    CallSiteScoreContext,
    is_private=st.booleans(),
    closed_world=st.booleans(),
    call_count=st.integers(min_value=1, max_value=50),
    unique_value_count=st.integers(min_value=1, max_value=8),
    min_call_sites=st.integers(min_value=1, max_value=5),
    only_test_callers=st.booleans(),
    has_body_corroboration=st.booleans(),
    param_name=_ANY_PARAM_NAMES,
)

_OPTIONAL_CTX = st.builds(
    OptionalParamScoreContext,
    is_private=st.booleans(),
    closed_world=st.booleans(),
    call_count=st.integers(min_value=1, max_value=50),
    min_call_sites=st.integers(min_value=1, max_value=5),
    only_test_callers=st.booleans(),
    has_none_default=st.booleans(),
)


@given(is_private=st.booleans(), closed_world=st.booleans())
def test_tier1_scores_always_in_high_band(*, is_private: bool, closed_world: bool) -> None:
    s1 = score_tier1_guarded_param(is_private=is_private)
    s2 = score_tier1_unused_optional_return(is_private=is_private)
    s3 = score_tier1_literal_return(is_private=is_private, closed_world=closed_world)

    for score in (s1, s2, s3):
        assert 0 <= score <= 100
        assert confidence_band(score) == ConfidenceBand.HIGH


@given(ctx=_CALLSITE_CTX)
def test_nt201_score_bounds_and_monotonicity(ctx: CallSiteScoreContext) -> None:
    score, notes = score_tier2_callsite_literal(ctx)
    assert 0 <= score <= 100
    assert isinstance(notes, list)

    # Visibility & closed-world monotonicity: private / closed-world never lowers score
    priv_score, _ = score_tier2_callsite_literal(replace(ctx, is_private=True))
    pub_score, _ = score_tier2_callsite_literal(replace(ctx, is_private=False))
    assert priv_score >= pub_score

    cw_score, _ = score_tier2_callsite_literal(replace(ctx, closed_world=True))
    ow_score, _ = score_tier2_callsite_literal(replace(ctx, closed_world=False))
    assert cw_score >= ow_score

    # Caller evidence monotonicity: more calls never lowers score
    more_calls, _ = score_tier2_callsite_literal(replace(ctx, call_count=ctx.call_count + 5))
    assert more_calls >= score

    # Body corroboration never lowers score
    with_body, _ = score_tier2_callsite_literal(replace(ctx, has_body_corroboration=True))
    without_body, _ = score_tier2_callsite_literal(replace(ctx, has_body_corroboration=False))
    assert with_body >= without_body

    # Test-only callers never increases score
    prod_score, _ = score_tier2_callsite_literal(replace(ctx, only_test_callers=False))
    test_score, _ = score_tier2_callsite_literal(replace(ctx, only_test_callers=True))
    assert prod_score >= test_score

    # Open-domain param name never outranks domain-neutral param name
    neutral_score, _ = score_tier2_callsite_literal(replace(ctx, param_name="mode"))
    open_score, _ = score_tier2_callsite_literal(replace(ctx, param_name="filename"))
    assert neutral_score >= open_score


@given(ctx=_OPTIONAL_CTX)
def test_nt202_score_bounds_and_monotonicity(ctx: OptionalParamScoreContext) -> None:
    score, _ = score_tier2_unused_optional_param(ctx)
    assert 0 <= score <= 100

    priv_score, _ = score_tier2_unused_optional_param(replace(ctx, is_private=True))
    pub_score, _ = score_tier2_unused_optional_param(replace(ctx, is_private=False))
    assert priv_score >= pub_score

    cw_score, _ = score_tier2_unused_optional_param(replace(ctx, closed_world=True))
    ow_score, _ = score_tier2_unused_optional_param(replace(ctx, closed_world=False))
    assert cw_score >= ow_score

    more_calls, _ = score_tier2_unused_optional_param(replace(ctx, call_count=ctx.call_count + 5))
    assert more_calls >= score

    prod_score, _ = score_tier2_unused_optional_param(replace(ctx, only_test_callers=False))
    test_score, _ = score_tier2_unused_optional_param(replace(ctx, only_test_callers=True))
    assert prod_score >= test_score

    no_default, _ = score_tier2_unused_optional_param(replace(ctx, has_none_default=False))
    with_default, _ = score_tier2_unused_optional_param(replace(ctx, has_none_default=True))
    assert no_default >= with_default


def test_canonical_band_expectations() -> None:
    # Strong internal repeated literal param with body check => HIGH
    strong_score, _ = score_tier2_callsite_literal(
        CallSiteScoreContext(
            is_private=True,
            closed_world=False,
            call_count=10,
            unique_value_count=3,
            min_call_sites=2,
            only_test_callers=False,
            has_body_corroboration=True,
            param_name="mode",
        )
    )
    assert confidence_band(strong_score) == ConfidenceBand.HIGH

    # Public API with open-domain name called once in tests => LOW
    weak_score, _ = score_tier2_callsite_literal(
        CallSiteScoreContext(
            is_private=False,
            closed_world=False,
            call_count=1,
            unique_value_count=1,
            min_call_sites=2,
            only_test_callers=True,
            has_body_corroboration=False,
            param_name="filename",
        )
    )
    assert confidence_band(weak_score) == ConfidenceBand.LOW
