"""Pure LibCST helpers for extracting literals, parsing annotations, and formatting types."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Literal as TypingLiteral

import libcst as cst

if TYPE_CHECKING:
    from collections.abc import Sequence

type LiteralScalar = str | int | bool

_BUILTIN_TYPE_CONSTRUCTORS: frozenset[str] = frozenset(
    {
        "str",
        "int",
        "float",
        "bool",
        "bytes",
        "list",
        "dict",
        "set",
        "frozenset",
        "tuple",
        "complex",
    }
)

_CONTAINER_ALIASES: dict[str, set[str]] = {
    "list": {"list", "Sequence", "MutableSequence", "Iterable", "Collection"},
    "tuple": {"tuple", "Sequence", "Iterable", "Collection"},
    "set": {"set", "AbstractSet", "MutableSet", "Iterable", "Collection"},
    "frozenset": {"frozenset", "AbstractSet", "Iterable", "Collection"},
    "dict": {"dict", "Mapping", "MutableMapping"},
}


@dataclass(frozen=True)
class ExtractedLiteral:
    """Scalar literal value with an explicit type tag distinguishing bool from int."""

    kind: TypingLiteral["str", "int", "bool"]
    value: LiteralScalar


def is_bool_annotation(annotation: cst.Annotation | None) -> bool:
    """Return True if the annotation is plain `bool` (or `builtins.bool`)."""
    if annotation is None:
        return False
    return _annotation_simple_name(annotation.annotation) in {"bool", "builtins.bool"}


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


def format_union_type(members: Sequence[str]) -> str:
    """Format one or more union member strings using PEP 604 `|` syntax."""
    return " | ".join(members)


def format_literal_type(values: set[ExtractedLiteral] | list[ExtractedLiteral]) -> str:
    """Format a set of ExtractedLiteral values as a deterministic `Literal[...]` type."""
    sorted_vals = sorted(set(values), key=_literal_sort_key)
    rendered = [_format_single_literal(val) for val in sorted_vals]
    return f"Literal[{', '.join(rendered)}]"


def is_definitely_non_none_expr(expr: cst.BaseExpression) -> bool:
    """Return True if a CST expression is syntactically guaranteed not to evaluate to None."""
    if is_none_expr(expr):
        return False
    if extract_literal(expr) is not None or _is_bytes_literal(expr):
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


def infer_simple_expr_type(expr: cst.BaseExpression) -> str | None:
    """Infer a conservative static type name for simple syntactic expressions."""
    if is_none_expr(expr):
        return "None"
    lit = extract_literal(expr)
    if lit is not None:
        return lit.kind
    if _is_bytes_literal(expr):
        return "bytes"
    return _infer_compound_or_call_type(expr)


def matches_union_member(
    observed_type: str,
    member: str,
    *,
    union_has_bool: bool = False,
    union_has_int: bool = False,
) -> bool:
    """Return True if an observed type name can satisfy a declared union member."""
    obs = observed_type.strip()
    mem = member.strip()
    if obs == mem:
        return True
    if obs == "None" or mem == "None":
        return obs == mem

    obs_base = _strip_type_args(obs)
    mem_base = _strip_type_args(mem)
    if obs_base == mem_base or mem_base.rsplit(".", maxsplit=1)[-1] == obs_base:
        return True

    # PEP 484 numeric tower: bool -> int -> float (only when the narrower numeric isn't listed).
    if _matches_numeric_tower(
        obs_base,
        mem_base,
        union_has_bool=union_has_bool,
        union_has_int=union_has_int,
    ):
        return True

    # Literal[...] satisfies its underlying primitive union branch.
    if obs.startswith("Literal["):
        return _literal_type_str_matches(obs, mem_base)

    aliases = _CONTAINER_ALIASES.get(obs_base)
    return aliases is not None and mem_base.rsplit(".", maxsplit=1)[-1] in aliases


def _matches_numeric_tower(
    obs_base: str,
    mem_base: str,
    *,
    union_has_bool: bool,
    union_has_int: bool,
) -> bool:
    """Return True if `obs_base` widens to `mem_base` via PEP 484 numeric tower."""
    if obs_base == "bool" and mem_base == "int" and not union_has_bool:
        return True
    if obs_base == "int" and mem_base == "float" and not union_has_int:
        return True
    return obs_base == "bool" and mem_base == "float" and not union_has_bool and not union_has_int


_MIN_UNION_MEMBERS = 2


@dataclass(frozen=True)
class DictAnnotationSpec:
    """Parsed `dict[str, V]` / `Mapping[str, V]` / bare `dict` annotation."""

    outer: str
    value_type: str | None  # None for bare `dict`, or `Any`/`object`/concrete type string
    is_broad_value: bool  # True when bare `dict` or value is `Any`/`object`


_BROAD_VALUE_NAMES: frozenset[str] = frozenset({"Any", "typing.Any", "object", "builtins.object"})
_NOT_REQUIRED_NAMES: frozenset[str] = frozenset(
    {"NotRequired", "typing.NotRequired", "typing_extensions.NotRequired"}
)
_REQUIRED_NAMES: frozenset[str] = frozenset(
    {"Required", "typing.Required", "typing_extensions.Required"}
)


def extract_not_required_inner(annotation: cst.Annotation | None) -> str | None:
    """If annotation is `NotRequired[T]`, return `T`; else None."""
    return _extract_single_wrapper_inner(annotation, _NOT_REQUIRED_NAMES)


def extract_required_inner(annotation: cst.Annotation | None) -> str | None:
    """If annotation is `Required[T]`, return `T`; else None."""
    return _extract_single_wrapper_inner(annotation, _REQUIRED_NAMES)


def _extract_single_wrapper_inner(
    annotation: cst.Annotation | None,
    wrapper_names: frozenset[str],
) -> str | None:
    """Extract `T` from a 1-arg wrapper like `NotRequired[T]` or `Required[T]`."""
    if annotation is None or not isinstance(annotation.annotation, cst.Subscript):
        return None
    expr = annotation.annotation
    if _annotation_simple_name(expr.value) not in wrapper_names:
        return None
    args = _extract_subscript_index_args(expr)
    if args is None or len(args) != 1:
        return None
    return expr_to_code(args[0])


@dataclass(frozen=True)
class ContainerOptionalSpec:
    """Parsed container or generator return annotation whose element/value/yield is Optional."""

    kind: TypingLiteral["seq", "dict", "generator", "iterable"]
    narrowed_annotation: str


_SEQ_CONTAINER_NAMES: frozenset[str] = frozenset(
    {
        "list",
        "builtins.list",
        "List",
        "typing.List",
        "Sequence",
        "typing.Sequence",
        "collections.abc.Sequence",
        "MutableSequence",
        "typing.MutableSequence",
        "collections.abc.MutableSequence",
        "Collection",
        "typing.Collection",
        "collections.abc.Collection",
        "set",
        "builtins.set",
        "Set",
        "typing.Set",
        "AbstractSet",
        "typing.AbstractSet",
        "collections.abc.Set",
        "frozenset",
        "builtins.frozenset",
        "FrozenSet",
        "typing.FrozenSet",
    }
)
_DICT_CONTAINER_NAMES: frozenset[str] = frozenset(
    {
        "dict",
        "builtins.dict",
        "Dict",
        "typing.Dict",
        "Mapping",
        "typing.Mapping",
        "collections.abc.Mapping",
        "MutableMapping",
        "typing.MutableMapping",
        "collections.abc.MutableMapping",
    }
)


def extract_str_key_dict_annotation(annotation_str: str | None) -> DictAnnotationSpec | None:
    """Return DictAnnotationSpec if annotation is `dict[str, V]`, `Mapping[str, V]`, or `dict`."""
    if annotation_str is None:
        return None
    stripped = annotation_str.strip()
    if stripped in _DICT_CONTAINER_NAMES:
        return DictAnnotationSpec(outer=stripped, value_type=None, is_broad_value=True)
    expr = _parse_subscript_annotation(stripped)
    args = _extract_subscript_index_args(expr) if expr is not None else None
    if (
        expr is None
        or args is None
        or len(args) != _MIN_UNION_MEMBERS
        or _annotation_simple_name(expr.value) not in _DICT_CONTAINER_NAMES
        or _annotation_simple_name(args[0]) not in {"str", "builtins.str"}
    ):
        return None
    val_str = expr_to_code(args[1])
    return DictAnnotationSpec(
        outer=expr_to_code(expr.value),
        value_type=val_str,
        is_broad_value=val_str in _BROAD_VALUE_NAMES,
    )


_GENERATOR_ONLY_NAMES: frozenset[str] = frozenset(
    {
        "Iterator",
        "typing.Iterator",
        "collections.abc.Iterator",
        "AsyncIterator",
        "typing.AsyncIterator",
        "collections.abc.AsyncIterator",
        "Generator",
        "typing.Generator",
        "collections.abc.Generator",
        "AsyncGenerator",
        "typing.AsyncGenerator",
        "collections.abc.AsyncGenerator",
    }
)
_ITERABLE_NAMES: frozenset[str] = frozenset(
    {
        "Iterable",
        "typing.Iterable",
        "collections.abc.Iterable",
        "AsyncIterable",
        "typing.AsyncIterable",
        "collections.abc.AsyncIterable",
    }
)


def extract_optional_container_return(annotation_str: str | None) -> ContainerOptionalSpec | None:
    """If annotation is a container/generator with an Optional element/value/yield, return spec."""
    expr = _parse_subscript_annotation(annotation_str)
    if expr is None:
        return None
    base_name = _annotation_simple_name(expr.value)
    if base_name is None:
        return None
    outer = expr_to_code(expr.value)
    args = _extract_subscript_index_args(expr)
    if args is None:
        return None
    return _build_optional_container_spec(base_name, outer, args)


def _extract_subscript_index_args(expr: cst.Subscript) -> list[cst.BaseExpression] | None:
    """Return index expressions of a subscript if every slice is a plain cst.Index."""
    args: list[cst.BaseExpression] = []
    for elem in expr.slice:
        if not isinstance(elem.slice, cst.Index):
            return None
        args.append(elem.slice.value)
    return args or None


def _build_optional_container_spec(
    base_name: str,
    outer: str,
    args: list[cst.BaseExpression],
) -> ContainerOptionalSpec | None:
    """Classify a subscripted container/generator annotation and strip Optional from its payload."""
    if base_name in _SEQ_CONTAINER_NAMES and len(args) == 1:
        inner = extract_optional_inner(cst.Annotation(annotation=args[0]))
        return ContainerOptionalSpec("seq", f"{outer}[{inner}]") if inner else None
    if (
        base_name in {"tuple", "builtins.tuple", "Tuple", "typing.Tuple"}
        and len(args) == _MIN_UNION_MEMBERS
        and isinstance(args[1], cst.Ellipsis)
    ):
        inner = extract_optional_inner(cst.Annotation(annotation=args[0]))
        return ContainerOptionalSpec("seq", f"{outer}[{inner}, ...]") if inner else None
    if base_name in _DICT_CONTAINER_NAMES and len(args) == _MIN_UNION_MEMBERS:
        inner = extract_optional_inner(cst.Annotation(annotation=args[1]))
        if inner:
            return ContainerOptionalSpec("dict", f"{outer}[{expr_to_code(args[0])}, {inner}]")
        return None
    if base_name in _GENERATOR_ONLY_NAMES or base_name in _ITERABLE_NAMES:
        inner = extract_optional_inner(cst.Annotation(annotation=args[0]))
        if inner:
            rest = [expr_to_code(a) for a in args[1:]]
            joined = ", ".join([inner, *rest])
            kind: TypingLiteral["generator", "iterable"] = (
                "iterable" if base_name in _ITERABLE_NAMES else "generator"
            )
            return ContainerOptionalSpec(kind, f"{outer}[{joined}]")
    return None


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


def extract_variadic_tuple_element(annotation_str: str | None) -> tuple[str, str] | None:
    """If annotation is `tuple[T, ...]` or `Tuple[T, ...]`, return `(outer, elem_type)`."""
    expr = _parse_subscript_annotation(annotation_str)
    if expr is None or len(expr.slice) != _MIN_UNION_MEMBERS:
        return None
    base_name = _annotation_simple_name(expr.value)
    if base_name not in {"tuple", "builtins.tuple", "Tuple", "typing.Tuple"}:
        return None
    first, second = expr.slice[0].slice, expr.slice[1].slice
    if (
        isinstance(first, cst.Index)
        and isinstance(second, cst.Index)
        and isinstance(second.value, cst.Ellipsis)
    ):
        return (expr_to_code(expr.value), expr_to_code(first.value))
    return None


def _parse_subscript_annotation(annotation_str: str | None) -> cst.Subscript | None:
    """Parse a type annotation string and return it if it is a Subscript expression."""
    if annotation_str is None:
        return None
    try:
        expr = cst.parse_expression(annotation_str)
    except cst.ParserSyntaxError:
        return None
    return expr if isinstance(expr, cst.Subscript) else None


def extract_union_members(annotation: cst.Annotation | None) -> tuple[str, ...] | None:
    """Extract top-level union member strings from `A | B`, `Union[A, B]`, or `Optional[T]`."""
    if annotation is None:
        return None
    expr = annotation.annotation
    if isinstance(expr, cst.BinaryOperation) and isinstance(expr.operator, cst.BitOr):
        members = [expr_to_code(m) for m in _flatten_bitor(expr)]
        return tuple(members) if len(members) >= _MIN_UNION_MEMBERS else None
    if isinstance(expr, cst.Subscript):
        return _extract_subscript_union_members(expr)
    return None


def _infer_compound_or_call_type(expr: cst.BaseExpression) -> str | None:
    """Infer type names for floats, f-strings, container literals, comparisons, and builtin calls."""
    simple_node_map: tuple[tuple[type[cst.CSTNode] | tuple[type[cst.CSTNode], ...], str], ...] = (
        (cst.Float, "float"),
        (cst.Imaginary, "complex"),
        (cst.FormattedString, "str"),
        ((cst.List, cst.ListComp), "list"),
        ((cst.Dict, cst.DictComp), "dict"),
        ((cst.Set, cst.SetComp), "set"),
        (cst.Tuple, "tuple"),
        (cst.Comparison, "bool"),
    )
    for node_types, type_name in simple_node_map:
        if isinstance(expr, node_types):
            return type_name
    if isinstance(expr, cst.UnaryOperation) and isinstance(expr.operator, cst.Not):
        return "bool"
    if isinstance(expr, cst.Call) and isinstance(expr.func, cst.Name):
        if expr.func.value in _BUILTIN_TYPE_CONSTRUCTORS:
            return expr.func.value
        if expr.func.value == "len":
            return "int"
    return None


def _is_bytes_literal(expr: cst.BaseExpression) -> bool:
    """Return True if expr is a bytes literal such as `b'abc'`."""
    if not isinstance(expr, (cst.SimpleString, cst.ConcatenatedString)):
        return False
    try:
        val = ast.literal_eval(expr_to_code(expr))
    except (SyntaxError, ValueError):
        return False
    return isinstance(val, bytes)


def _strip_type_args(type_str: str) -> str:
    """Return the outer name of a generic type string (e.g. `list[int]` -> `list`)."""
    bracket = type_str.find("[")
    return type_str[:bracket].strip() if bracket != -1 else type_str.strip()


def _literal_type_str_matches(obs_literal: str, mem_base: str) -> bool:
    """Check if a `Literal[...]` string matches `str`, `int`, or `bool`."""
    try:
        ann = cst.Annotation(annotation=cst.parse_expression(obs_literal))
    except cst.ParserSyntaxError:
        return False
    lits = extract_literal_values_from_annotation(ann)
    if not lits:
        return False
    return any(lit.kind == mem_base for lit in lits)


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


def _extract_subscript_union_members(expr: cst.Subscript) -> tuple[str, ...] | None:
    """Extract member type strings from `Optional[T]` or `Union[A, B, ...]`."""
    base_name = _annotation_simple_name(expr.value)
    if base_name in {"Optional", "typing.Optional"}:
        inner = _strip_none_from_optional_subscript(expr)
        return (inner, "None") if inner is not None else None
    if base_name in {"Union", "typing.Union"}:
        members: list[str] = []
        for elem in expr.slice:
            if not isinstance(elem.slice, cst.Index):
                return None
            members.append(expr_to_code(elem.slice.value))
        return tuple(members) if len(members) >= _MIN_UNION_MEMBERS else None
    return None


def _annotation_simple_name(expr: cst.BaseExpression) -> str | None:
    """Return dotted name for simple Name/Attribute expressions, else None."""
    if isinstance(expr, cst.Name):
        return expr.value
    if isinstance(expr, cst.Attribute):
        prefix = _annotation_simple_name(expr.value)
        if prefix is not None:
            return f"{prefix}.{expr.attr.value}"
    return None


def _strip_none_from_optional_subscript(expr: cst.Subscript) -> str | None:
    """Extract `T` from `Optional[T]`."""
    if len(expr.slice) == 1 and isinstance(expr.slice[0].slice, cst.Index):
        inner = expr.slice[0].slice.value
        if not is_none_expr(inner):
            return expr_to_code(inner)
    return None


def expr_to_code(node: cst.CSTNode) -> str:
    """Render a CST node back to source code."""
    return cst.Module(body=[]).code_for_node(node).strip()


def is_none_expr(expr: cst.BaseExpression) -> bool:
    """Return True if the CST expression is the literal `None`."""
    return isinstance(expr, cst.Name) and expr.value == "None"


def _literal_sort_key(lit: ExtractedLiteral) -> tuple[int, str | int]:
    """Return a deterministic sort key across str/int/bool literals."""
    kind_order = {"str": 0, "int": 1, "bool": 2}[lit.kind]
    if isinstance(lit.value, bool):
        return (kind_order, int(lit.value))
    return (kind_order, lit.value)


def _format_single_literal(lit: ExtractedLiteral) -> str:
    """Format a single literal value using double quotes for strings."""
    if lit.kind == "str" and isinstance(lit.value, str):
        raw = repr(lit.value)
        if raw.startswith('"'):
            return raw
        inner = raw[1:-1].replace('"', '\\"').replace("\\'", "'")
        return f'"{inner}"'
    return repr(lit.value)
