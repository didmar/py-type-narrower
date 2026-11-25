"""Pure LibCST helpers for extracting literals, parsing annotations, and formatting types."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Literal as TypingLiteral

import libcst as cst

type LiteralScalar = str | int | bool


@dataclass(frozen=True)
class ExtractedLiteral:
    """Scalar literal value with an explicit type tag distinguishing bool from int."""

    kind: TypingLiteral["str", "int", "bool"]
    value: LiteralScalar


def base_primitive_annotation(
    annotation: cst.Annotation | None,
) -> TypingLiteral["str", "int"] | None:
    """Return 'str' or 'int' if annotation is plain str/int, else None."""
    if is_str_annotation(annotation):
        return "str"
    if is_int_annotation(annotation):
        return "int"
    return None


def is_str_annotation(annotation: cst.Annotation | None) -> bool:
    """Return True if the annotation is plain `str` (or `builtins.str`)."""
    if annotation is None:
        return False
    return _annotation_simple_name(annotation.annotation) in {"str", "builtins.str"}


def is_int_annotation(annotation: cst.Annotation | None) -> bool:
    """Return True if the annotation is plain `int` (or `builtins.int`)."""
    if annotation is None:
        return False
    return _annotation_simple_name(annotation.annotation) in {"int", "builtins.int"}


def extract_optional_inner(annotation: cst.Annotation | None) -> str | None:
    """If annotation is Optional[T] or a Union containing None, return the non-None type string."""
    if annotation is None:
        return None
    expr = annotation.annotation
    if isinstance(expr, cst.BinaryOperation) and isinstance(expr.operator, cst.BitOr):
        return _strip_none_from_bitor_union(expr)
    if isinstance(expr, cst.Subscript):
        return _strip_none_from_subscript_union(expr)
    return None


def extract_literal_values_from_annotation(
    annotation: cst.Annotation | None,
) -> tuple[ExtractedLiteral, ...] | None:
    """Extract literal values if annotation is `Literal[...]` or `typing.Literal[...]`."""
    if annotation is None:
        return None
    expr = annotation.annotation
    if not isinstance(expr, cst.Subscript):
        return None
    base_name = _annotation_simple_name(expr.value)
    if base_name not in {"Literal", "typing.Literal", "typing_extensions.Literal"}:
        return None
    values: list[ExtractedLiteral] = []
    for elem in expr.slice:
        if not isinstance(elem.slice, cst.Index):
            return None
        lit = extract_literal(elem.slice.value)
        if lit is None:
            return None
        values.append(lit)
    return tuple(values) if values else None


def format_literal_type(values: set[ExtractedLiteral] | list[ExtractedLiteral]) -> str:
    """Format a set of ExtractedLiteral values as a deterministic `Literal[...]` type."""
    sorted_vals = sorted(set(values), key=_literal_sort_key)
    rendered = [_format_single_literal(val) for val in sorted_vals]
    return f"Literal[{', '.join(rendered)}]"


def is_definitely_non_none_expr(expr: cst.BaseExpression) -> bool:
    """Return True if a CST expression is syntactically guaranteed not to evaluate to None."""
    if is_none_expr(expr):
        return False
    if extract_literal(expr) is not None:
        return True
    non_none_nodes = (
        cst.Float,
        cst.Imaginary,
        cst.FormattedString,
        cst.List,
        cst.Tuple,
        cst.Set,
        cst.Dict,
        cst.ListComp,
        cst.SetComp,
        cst.DictComp,
        cst.GeneratorExp,
        cst.Lambda,
        cst.BinaryOperation,
        cst.UnaryOperation,
        cst.Comparison,
    )
    if isinstance(expr, non_none_nodes):
        return True
    if isinstance(expr, cst.BooleanOperation):
        return is_definitely_non_none_expr(expr.left) and is_definitely_non_none_expr(expr.right)
    if isinstance(expr, cst.IfExp):
        return is_definitely_non_none_expr(expr.body) and is_definitely_non_none_expr(expr.orelse)
    return False


def extract_literal(expr: cst.BaseExpression) -> ExtractedLiteral | None:
    """Extract a str, int, or bool literal value from a CST expression."""
    if isinstance(expr, cst.SimpleString):
        return _eval_string_literal(expr.value)
    if isinstance(expr, cst.ConcatenatedString):
        return _eval_string_literal(expr_to_code(expr))
    if isinstance(expr, cst.Integer):
        return _eval_int_literal(expr.value)
    if isinstance(expr, cst.UnaryOperation) and isinstance(expr.operator, (cst.Minus, cst.Plus)):
        return _eval_signed_int(expr)
    if isinstance(expr, cst.Name) and expr.value in {"True", "False"}:
        return ExtractedLiteral(kind="bool", value=expr.value == "True")
    return None


def _eval_string_literal(raw: str) -> ExtractedLiteral | None:
    """Safely evaluate a string literal token."""
    try:
        val = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return None
    if isinstance(val, str):
        return ExtractedLiteral(kind="str", value=val)
    return None


def _eval_signed_int(expr: cst.UnaryOperation) -> ExtractedLiteral | None:
    """Evaluate a unary +/- integer literal."""
    if not isinstance(expr.expression, cst.Integer):
        return None
    inner = _eval_int_literal(expr.expression.value)
    if inner is None or not isinstance(inner.value, int):
        return None
    value = -inner.value if isinstance(expr.operator, cst.Minus) else inner.value
    return ExtractedLiteral(kind="int", value=value)


def _eval_int_literal(raw: str) -> ExtractedLiteral | None:
    """Safely evaluate an integer literal token."""
    try:
        val = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return None
    if isinstance(val, int) and not isinstance(val, bool):
        return ExtractedLiteral(kind="int", value=val)
    return None


def _strip_none_from_bitor_union(expr: cst.BinaryOperation) -> str | None:
    """Strip None from a PEP 604 `A | B | None` union annotation."""
    members = _flatten_bitor(expr)
    non_none = [m for m in members if not is_none_expr(m)]
    if len(non_none) == len(members) or not non_none:
        return None
    return " | ".join(expr_to_code(member) for member in non_none)


def _flatten_bitor(expr: cst.BaseExpression) -> list[cst.BaseExpression]:
    """Flatten nested `|` BinaryOperations into a list of union members."""
    if isinstance(expr, cst.BinaryOperation) and isinstance(expr.operator, cst.BitOr):
        return _flatten_bitor(expr.left) + _flatten_bitor(expr.right)
    return [expr]


def _strip_none_from_subscript_union(expr: cst.Subscript) -> str | None:
    """Strip None from `Optional[T]` or `Union[T, None]`."""
    base_name = _annotation_simple_name(expr.value)
    if base_name in {"Optional", "typing.Optional"}:
        return _strip_none_from_optional_subscript(expr)
    if base_name in {"Union", "typing.Union"}:
        return _strip_none_from_union_subscript(expr)
    return None


def _strip_none_from_optional_subscript(expr: cst.Subscript) -> str | None:
    """Extract `T` from `Optional[T]`."""
    if len(expr.slice) == 1 and isinstance(expr.slice[0].slice, cst.Index):
        inner = expr.slice[0].slice.value
        if not is_none_expr(inner):
            return expr_to_code(inner)
    return None


def _strip_none_from_union_subscript(expr: cst.Subscript) -> str | None:
    """Strip `None` from `Union[T, None]`."""
    members: list[cst.BaseExpression] = []
    for elem in expr.slice:
        if not isinstance(elem.slice, cst.Index):
            return None
        members.append(elem.slice.value)
    non_none = [m for m in members if not is_none_expr(m)]
    if len(non_none) == len(members) or not non_none:
        return None
    return " | ".join(expr_to_code(member) for member in non_none)


def expr_to_code(node: cst.CSTNode) -> str:
    """Render a CST node back to source code."""
    return cst.Module(body=[]).code_for_node(node).strip()


def is_none_expr(expr: cst.BaseExpression) -> bool:
    """Return True if the CST expression is the literal `None`."""
    return isinstance(expr, cst.Name) and expr.value == "None"


def _annotation_simple_name(expr: cst.BaseExpression) -> str | None:
    """Return dotted name for simple Name/Attribute expressions, else None."""
    if isinstance(expr, cst.Name):
        return expr.value
    if isinstance(expr, cst.Attribute):
        prefix = _annotation_simple_name(expr.value)
        if prefix is not None:
            return f"{prefix}.{expr.attr.value}"
    return None


def _literal_sort_key(lit: ExtractedLiteral) -> tuple[int, str | int]:
    """Return a deterministic sort key across str/int/bool literals."""
    kind_order = {"str": 0, "int": 1, "bool": 2}[lit.kind]
    if isinstance(lit.value, bool):
        return (kind_order, int(lit.value))
    return (kind_order, lit.value)


def _format_single_literal(lit: ExtractedLiteral) -> str:
    """Format a single literal value using double quotes for strings."""
    if lit.kind == "str" and isinstance(lit.value, str):
        escaped = lit.value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return repr(lit.value)
