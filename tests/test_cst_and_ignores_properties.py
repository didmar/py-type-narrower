import ast
from pathlib import Path

import libcst as cst
from hypothesis import example, given
from hypothesis import strategies as st

from py_type_narrower.config import NarrowerConfig
from py_type_narrower.cst_utils import (
    ExtractedLiteral,
    extract_literal,
    extract_literal_values_from_annotation,
    extract_optional_inner,
    format_literal_type,
)
from py_type_narrower.ignores import is_ignored, parse_file_ignores
from py_type_narrower.indexer import index_files
from py_type_narrower.rewriter import rewrite_module_source
from py_type_narrower.rules import run_rules

# Arbitrary UTF-8 strings including newlines, control chars, quotes, and backslashes
_STR_LITERALS = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),
    min_size=0,
    max_size=20,
).map(lambda s: ExtractedLiteral(kind="str", value=s))

_INT_LITERALS = st.integers(min_value=-10_000, max_value=10_000).map(
    lambda n: ExtractedLiteral(kind="int", value=n)
)

_LITERAL_LISTS = st.one_of(
    st.lists(_STR_LITERALS, min_size=1, max_size=6, unique=True),
    st.lists(_INT_LITERALS, min_size=1, max_size=6, unique=True),
)


@given(values=_LITERAL_LISTS)
@example(
    values=[
        ExtractedLiteral(kind="str", value="\n"),
        ExtractedLiteral(kind="str", value="\r\n\t"),
        ExtractedLiteral(kind="str", value="a'b\"c\\d\\'e\\\"f"),
        ExtractedLiteral(kind="str", value="\x00"),
    ]
)
def test_format_literal_type_roundtrips_and_is_order_independent(
    values: list[ExtractedLiteral],
) -> None:
    rendered_fwd = format_literal_type(values)
    rendered_rev = format_literal_type(list(reversed(values)))

    # Order-independent formatting
    assert rendered_fwd == rendered_rev

    # Always valid Python syntax
    ast.parse(f"x: {rendered_fwd}")

    # Exact round-trip through extract_literal_values_from_annotation
    ann = cst.Annotation(annotation=cst.parse_expression(rendered_fwd))
    extracted = extract_literal_values_from_annotation(ann)
    assert extracted is not None
    assert set(extracted) == set(values)


@given(n=st.integers(min_value=-100_000, max_value=100_000))
def test_extract_literal_int_preserves_kind_and_value(n: int) -> None:
    expr = cst.parse_expression(repr(n))
    lit = extract_literal(expr)
    assert lit == ExtractedLiteral(kind="int", value=n)
    assert not isinstance(lit.value, bool)


@given(b=st.booleans())
def test_extract_literal_bool_never_confused_with_int(*, b: bool) -> None:
    expr = cst.parse_expression(repr(b))
    lit = extract_literal(expr)
    assert lit == ExtractedLiteral(kind="bool", value=b)
    assert lit.kind == "bool"


@given(
    inner=st.sampled_from(["int", "str", "Path", "list[int]", "dict[str, Any]"]),
    style=st.sampled_from(["bitor_right", "bitor_left", "optional", "union"]),
)
def test_extract_optional_inner_normalizes_all_forms(inner: str, style: str) -> None:
    if style == "bitor_right":
        expr_src = f"{inner} | None"
    elif style == "bitor_left":
        expr_src = f"None | {inner}"
    elif style == "optional":
        expr_src = f"Optional[{inner}]"
    else:
        expr_src = f"Union[{inner}, None]"

    ann = cst.Annotation(annotation=cst.parse_expression(expr_src))
    assert extract_optional_inner(ann) == inner


@given(text=st.text(max_size=200))
def test_parse_file_ignores_never_crashes(text: str) -> None:
    parsed = parse_file_ignores(text)
    assert isinstance(parsed, dict)


@given(
    code=st.sampled_from(["NT101", "NT102", "NT103", "NT201", "NT202"]),
    other=st.sampled_from(["NT101", "NT102", "NT103", "NT201", "NT202"]),
    lowercase=st.booleans(),
)
def test_inline_ignore_specificity_and_case_insensitivity(
    code: str,
    other: str,
    *,
    lowercase: bool,
) -> None:
    rendered_code = code.lower() if lowercase else code
    src = f"def f(x: str) -> int:  # type-narrow: ignore[{rendered_code}]\n    return 1\n"
    ignores = parse_file_ignores(src)
    assert is_ignored(ignores, 1, code) is True
    assert is_ignored(ignores, 1, other) is (code == other)


@given(
    values=st.lists(
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-", min_size=1, max_size=10),
        min_size=1,
        max_size=4,
        unique=True,
    )
)
def test_rewriter_produces_valid_idempotent_python_for_guarded_functions(
    values: list[str],
) -> None:
    branches = []
    for idx, val in enumerate(values):
        kw = "if" if idx == 0 else "elif"
        branches.append(f"    {kw} mode == {val!r}:\n        return len(mode)")
    branches.append("    else:\n        raise ValueError(mode)")
    source = "def choose(mode: str) -> int:\n" + "\n".join(branches) + "\n"

    fake_path = Path("/tmp/opencode/prop_sample.py")  # noqa: S108
    index = index_files([fake_path], project_root=fake_path.parent, overrides={fake_path: source})
    cands = run_rules(index, NarrowerConfig(max_tier=1))
    assert len(cands) == 1

    rewritten = rewrite_module_source(source, cands)
    ast.parse(rewritten)

    # Idempotency: re-analyzing the rewritten source yields 0 candidates.
    index2 = index_files(
        [fake_path], project_root=fake_path.parent, overrides={fake_path: rewritten}
    )
    assert run_rules(index2, NarrowerConfig(max_tier=1)) == []
