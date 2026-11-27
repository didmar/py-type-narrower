"""Class-local LibCST analysis for narrowable class attributes (NT104)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from typing import Literal as TypingLiteral

import libcst as cst

from py_type_narrower.cst_utils import (
    ExtractedLiteral,
    base_primitive_annotation,
    expr_to_code,
    extract_literal,
    extract_optional_inner,
    extract_union_members,
    infer_simple_expr_type,
    is_definitely_non_none_expr,
    is_none_expr,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_NON_NONE_BUILTINS: frozenset[str] = frozenset(
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
        "Path",
    }
)


@dataclass(frozen=True)
class ClassAttrFact:
    """Metadata for a class attribute narrowable via class-local analysis (NT104)."""

    name: str
    line: int
    col: int
    annotation_str: str
    kind: TypingLiteral["optional", "literal", "union"]
    declared_in_init: bool
    has_none_default: bool = False
    optional_inner: str | None = None
    primitive_kind: TypingLiteral["str", "int"] | None = None
    observed_literals: frozenset[ExtractedLiteral] = field(default_factory=frozenset)
    union_members: tuple[str, ...] = ()
    observed_types: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class _AttrDecl:
    """Candidate attribute declaration found on class body or in `__init__`."""

    name: str
    annotation_str: str
    optional_inner: str | None
    primitive_kind: TypingLiteral["str", "int"] | None
    union_members: tuple[str, ...] | None
    declared_in_init: bool
    has_none_default: bool
    default_expr: cst.BaseExpression | None


@dataclass
class _AttrWriteSummary:
    """Aggregated writes to a single `self.<attr>` across all methods and class body."""

    has_write: bool = False
    disqualified_non_none: bool = False
    deleted: bool = False
    has_non_literal: bool = False
    literals: set[ExtractedLiteral] = field(default_factory=set)
    has_unknown_type: bool = False
    observed_types: set[str] = field(default_factory=set)


def analyze_class_attributes(
    cls_node: cst.ClassDef,
    known_class_names: set[str],
    get_position: Callable[[str, bool], tuple[int, int]],
) -> list[ClassAttrFact]:
    """Find class attributes narrowable from class-body and method assignments."""
    stmts = _extract_suite_stmts(cls_node.body)
    init_fn = _find_init_method(stmts)
    if init_fn is None:
        return []

    decls = _collect_attr_decls(stmts, init_fn)
    if not decls:
        return []

    init_non_none_names = _collect_init_non_none_names(init_fn, known_class_names)
    guaranteed_in_init = _collect_guaranteed_init_attrs(
        _extract_suite_stmts(init_fn.body), init_non_none_names, known_class_names
    )
    writes = _collect_attr_writes(stmts, decls, known_class_names)

    results: list[ClassAttrFact] = []
    for attr_name, decl in decls.items():
        summary = writes.get(attr_name)
        if summary is None or summary.deleted:
            continue
        fact = _build_attr_fact(
            decl=decl,
            summary=summary,
            guaranteed_in_init=attr_name in guaranteed_in_init,
            get_position=get_position,
        )
        if fact is not None:
            results.append(fact)
    return results


def _build_attr_fact(
    *,
    decl: _AttrDecl,
    summary: _AttrWriteSummary,
    guaranteed_in_init: bool,
    get_position: Callable[[str, bool], tuple[int, int]],
) -> ClassAttrFact | None:
    """Convert one attribute declaration + write summary into a ClassAttrFact if narrowable."""
    if decl.optional_inner is not None and guaranteed_in_init and not summary.disqualified_non_none:
        line, col = get_position(decl.name, decl.declared_in_init)
        return ClassAttrFact(
            name=decl.name,
            line=line,
            col=col,
            annotation_str=decl.annotation_str,
            kind="optional",
            declared_in_init=decl.declared_in_init,
            has_none_default=decl.has_none_default,
            optional_inner=decl.optional_inner,
        )
    if (
        decl.primitive_kind is not None
        and summary.has_write
        and not summary.has_non_literal
        and summary.literals
    ):
        line, col = get_position(decl.name, decl.declared_in_init)
        return ClassAttrFact(
            name=decl.name,
            line=line,
            col=col,
            annotation_str=decl.annotation_str,
            kind="literal",
            declared_in_init=decl.declared_in_init,
            primitive_kind=decl.primitive_kind,
            observed_literals=frozenset(summary.literals),
        )
    if (
        decl.union_members is not None
        and summary.has_write
        and not summary.has_unknown_type
        and summary.observed_types
    ):
        line, col = get_position(decl.name, decl.declared_in_init)
        return ClassAttrFact(
            name=decl.name,
            line=line,
            col=col,
            annotation_str=decl.annotation_str,
            kind="union",
            declared_in_init=decl.declared_in_init,
            has_none_default=decl.has_none_default,
            union_members=decl.union_members,
            observed_types=frozenset(summary.observed_types),
        )
    return None


def _find_init_method(stmts: list[cst.BaseStatement]) -> cst.FunctionDef | None:
    """Return the `__init__` FunctionDef on a class body if present."""
    for stmt in stmts:
        if isinstance(stmt, cst.FunctionDef) and stmt.name.value == "__init__":
            return stmt
    return None


def _collect_guaranteed_init_attrs(
    stmts: list[cst.BaseStatement],
    non_none_names: set[str],
    known_class_names: set[str],
) -> set[str]:
    """Find `self.<attr>` names guaranteed to be assigned a non-None value along every path."""
    guaranteed: set[str] = set()
    for stmt in stmts:
        if isinstance(stmt, cst.SimpleStatementLine):
            for small in stmt.body:
                attr = _extract_non_none_self_assign(small, non_none_names, known_class_names)
                if attr is not None:
                    guaranteed.add(attr)
        elif isinstance(stmt, cst.If):
            guaranteed.update(_guaranteed_attrs_in_if(stmt, non_none_names, known_class_names))
    return guaranteed


def _guaranteed_attrs_in_if(
    stmt: cst.If,
    non_none_names: set[str],
    known_class_names: set[str],
) -> set[str]:
    """Return attrs guaranteed initialized across all branches of an if/elif/else."""
    if stmt.orelse is None:
        return set()
    then_attrs = _collect_guaranteed_init_attrs(
        _extract_suite_stmts(stmt.body), non_none_names, known_class_names
    )
    if isinstance(stmt.orelse, cst.Else):
        else_attrs = _collect_guaranteed_init_attrs(
            _extract_suite_stmts(stmt.orelse.body), non_none_names, known_class_names
        )
        return then_attrs & else_attrs
    if isinstance(stmt.orelse, cst.If):
        return then_attrs & _guaranteed_attrs_in_if(stmt.orelse, non_none_names, known_class_names)
    return set()


def _extract_non_none_self_assign(
    stmt: cst.BaseSmallStatement,
    non_none_names: set[str],
    known_class_names: set[str],
) -> str | None:
    """Return attr name if stmt assigns a provably non-None value to `self.<attr>`."""
    if isinstance(stmt, cst.Assign) and len(stmt.targets) == 1:
        attr = _self_attr_name(stmt.targets[0].target)
        if attr is not None and _expr_is_non_none(stmt.value, non_none_names, known_class_names):
            return attr
    if isinstance(stmt, cst.AnnAssign) and stmt.value is not None:
        attr = _self_attr_name(stmt.target)
        if attr is not None and _expr_is_non_none(stmt.value, non_none_names, known_class_names):
            return attr
    return None


class _InitAttrCollector(cst.CSTVisitor):
    """Find annotated `self.<attr>: T = ...` declarations inside `__init__`."""

    def __init__(self) -> None:
        self.decls: dict[str, _AttrDecl] = {}

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:  # noqa: ARG002, N802
        return False

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:  # noqa: ARG002, N802
        return False

    def visit_AnnAssign(self, node: cst.AnnAssign) -> bool:  # noqa: N802
        attr = _self_attr_name(node.target)
        if attr is None:
            return True
        decl = _make_attr_decl(attr, node.annotation, node.value, declared_in_init=True)
        if decl is not None:
            self.decls[attr] = decl
        return True


def _collect_attr_decls(
    class_stmts: list[cst.BaseStatement],
    init_fn: cst.FunctionDef,
) -> dict[str, _AttrDecl]:
    """Collect narrowable attribute annotations on class body and inside `__init__`."""
    decls: dict[str, _AttrDecl] = {}
    for stmt in class_stmts:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        for small in stmt.body:
            if not isinstance(small, cst.AnnAssign) or not isinstance(small.target, cst.Name):
                continue
            decl = _make_attr_decl(
                small.target.value,
                small.annotation,
                small.value,
                declared_in_init=False,
            )
            if decl is not None:
                decls[small.target.value] = decl

    init_collector = _InitAttrCollector()
    init_fn.body.visit(init_collector)
    for attr_name, decl in init_collector.decls.items():
        decls.setdefault(attr_name, decl)
    return decls


def _make_attr_decl(
    name: str,
    annotation: cst.Annotation,
    value: cst.BaseExpression | None,
    *,
    declared_in_init: bool,
) -> _AttrDecl | None:
    """Build an _AttrDecl if annotation is Optional, primitive str/int, or a Union."""
    ann_str = expr_to_code(annotation.annotation)
    if ann_str.startswith(("ClassVar", "typing.ClassVar")):
        return None
    opt_inner = extract_optional_inner(annotation)
    prim_kind = base_primitive_annotation(annotation)
    union_members = extract_union_members(annotation)
    if opt_inner is None and prim_kind is None and union_members is None:
        return None
    return _AttrDecl(
        name=name,
        annotation_str=ann_str,
        optional_inner=opt_inner,
        primitive_kind=prim_kind,
        union_members=union_members,
        declared_in_init=declared_in_init,
        has_none_default=not declared_in_init and value is not None and is_none_expr(value),
        default_expr=None if declared_in_init else value,
    )


class _SelfAttrWriteVisitor(cst.CSTVisitor):
    """Collect all writes and deletions of `self.<attr>` inside a method."""

    def __init__(
        self,
        non_none_names: set[str],
        local_types: dict[str, str],
        known_class_names: set[str],
        writes: dict[str, _AttrWriteSummary],
    ) -> None:
        self.non_none_names = non_none_names
        self.local_types = local_types
        self.known_class_names = known_class_names
        self.writes = writes

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:  # noqa: ARG002, N802
        return False

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:  # noqa: ARG002, N802
        return False

    def visit_Assign(self, node: cst.Assign) -> bool:  # noqa: N802
        for target in node.targets:
            attr = _self_attr_name(target.target)
            if attr is not None and attr in self.writes:
                _record_attr_write(
                    self.writes[attr],
                    node.value,
                    self.non_none_names,
                    self.local_types,
                    self.known_class_names,
                )
        return True

    def visit_AnnAssign(self, node: cst.AnnAssign) -> bool:  # noqa: N802
        attr = _self_attr_name(node.target)
        if attr is not None and attr in self.writes and node.value is not None:
            _record_attr_write(
                self.writes[attr],
                node.value,
                self.non_none_names,
                self.local_types,
                self.known_class_names,
            )
        return True

    def visit_AugAssign(self, node: cst.AugAssign) -> bool:  # noqa: N802
        attr = _self_attr_name(node.target)
        if attr is not None and attr in self.writes:
            summary = self.writes[attr]
            summary.has_write = True
            summary.has_non_literal = True
            summary.has_unknown_type = True
        return True

    def visit_Del(self, node: cst.Del) -> bool:  # noqa: N802
        attr = _self_attr_name(node.target)
        if attr is not None and attr in self.writes:
            self.writes[attr].deleted = True
        return True


def _collect_attr_writes(
    class_stmts: list[cst.BaseStatement],
    decls: dict[str, _AttrDecl],
    known_class_names: set[str],
) -> dict[str, _AttrWriteSummary]:
    """Collect write summaries for every declared attribute across class body and methods."""
    writes = {name: _AttrWriteSummary() for name in decls}
    # Record class-body default values for literal/union tracking (skip None default for Optional
    # because NT104 intentionally drops a class-body `= None` placeholder if __init__ overwrites it).
    for name, decl in decls.items():
        if decl.default_expr is None:
            continue
        if decl.has_none_default and decl.optional_inner is not None:
            continue
        _record_attr_write(writes[name], decl.default_expr, set(), {}, known_class_names)

    for stmt in class_stmts:
        if not isinstance(stmt, cst.FunctionDef):
            continue
        method_non_none = _collect_init_non_none_names(stmt, known_class_names)
        method_types = _collect_method_param_types(stmt)
        visitor = _SelfAttrWriteVisitor(
            non_none_names=method_non_none,
            local_types=method_types,
            known_class_names=known_class_names,
            writes=writes,
        )
        stmt.body.visit(visitor)
    return writes


def _record_attr_write(
    summary: _AttrWriteSummary,
    value: cst.BaseExpression,
    non_none_names: set[str],
    local_types: dict[str, str],
    known_class_names: set[str],
) -> None:
    """Update a single attribute's write summary for an assigned expression."""
    summary.has_write = True
    if not _expr_is_non_none(value, non_none_names, known_class_names):
        summary.disqualified_non_none = True
    lit = extract_literal(value)
    if lit is not None:
        summary.literals.add(lit)
    else:
        summary.has_non_literal = True
    inferred = _infer_attr_expr_type(value, local_types, known_class_names)
    if inferred is None:
        summary.has_unknown_type = True
    else:
        summary.observed_types.add(inferred)


def _infer_attr_expr_type(
    expr: cst.BaseExpression,
    local_types: dict[str, str],
    known_class_names: set[str],
) -> str | None:
    """Infer a conservative type string for an attribute assignment RHS."""
    simple = infer_simple_expr_type(expr)
    if simple is not None:
        return simple
    if isinstance(expr, cst.Name) and expr.value in local_types:
        typ = local_types[expr.value]
        return None if typ in {"Any", "typing.Any", "object", "builtins.object"} else typ
    if (
        isinstance(expr, cst.Call)
        and isinstance(expr.func, cst.Name)
        and expr.func.value in known_class_names
    ):
        return expr.func.value
    return None


def _collect_method_param_types(fn: cst.FunctionDef) -> dict[str, str]:
    """Collect annotated parameter types on a method (excluding `self`)."""
    types: dict[str, str] = {}
    params = (*fn.params.posonly_params, *fn.params.params, *fn.params.kwonly_params)
    for p in params[1:]:
        if p.annotation is not None:
            types[p.name.value] = expr_to_code(p.annotation.annotation)
    return types


def _collect_init_non_none_names(
    init_fn: cst.FunctionDef,
    known_class_names: set[str],
) -> set[str]:
    """Return parameter/local names in `__init__` that are provably non-None."""
    names: set[str] = set()
    params = (
        *init_fn.params.posonly_params,
        *init_fn.params.params,
        *init_fn.params.kwonly_params,
    )
    for p in params[1:]:
        if p.annotation is None:
            continue
        ann_str = expr_to_code(p.annotation.annotation)
        if extract_optional_inner(p.annotation) is None and ann_str != "None":
            names.add(p.name.value)
    # Include simple non-None local assignments in __init__.
    for stmt in _extract_suite_stmts(init_fn.body):
        if isinstance(stmt, cst.SimpleStatementLine):
            for small in stmt.body:
                if (
                    isinstance(small, cst.Assign)
                    and len(small.targets) == 1
                    and isinstance(small.targets[0].target, cst.Name)
                    and _expr_is_non_none(small.value, names, known_class_names)
                ):
                    names.add(small.targets[0].target.value)
    return names


def _self_attr_name(node: cst.BaseExpression) -> str | None:
    """Return `attr` if node is `self.<attr>`, else None."""
    if (
        isinstance(node, cst.Attribute)
        and isinstance(node.value, cst.Name)
        and node.value.value == "self"
    ):
        return node.attr.value
    return None


def _expr_is_non_none(
    expr: cst.BaseExpression,
    non_none_names: set[str],
    known_class_names: set[str],
) -> bool:
    """Return True if expr is provably non-None inside a method body."""
    if is_definitely_non_none_expr(expr):
        return True
    if isinstance(expr, cst.Name) and expr.value in non_none_names:
        return True
    if isinstance(expr, cst.Call) and isinstance(expr.func, cst.Name):
        return expr.func.value in _NON_NONE_BUILTINS or expr.func.value in known_class_names
    return False


def _extract_suite_stmts(body: cst.BaseSuite) -> list[cst.BaseStatement]:
    """Flatten a CST suite into a list of statements."""
    if isinstance(body, cst.IndentedBlock):
        return list(body.body)
    if isinstance(body, cst.SimpleStatementSuite):
        return [cst.SimpleStatementLine(body=list(body.body))]
    return []
