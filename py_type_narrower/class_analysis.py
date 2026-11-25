"""Class-local LibCST analysis for always-initialized Optional attributes (NT104)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import libcst as cst

from py_type_narrower.cst_utils import (
    expr_to_code,
    extract_optional_inner,
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
    """Metadata for a class attribute typed `T | None` that is always initialized non-None."""

    name: str
    line: int
    col: int
    annotation_str: str
    optional_inner: str
    declared_in_init: bool
    has_none_default: bool


@dataclass(frozen=True)
class _OptionalAttrDecl:
    """Candidate `T | None` attribute declaration found on class body or in `__init__`."""

    name: str
    node: cst.CSTNode
    annotation_str: str
    optional_inner: str
    declared_in_init: bool
    has_none_default: bool


def analyze_class_attributes(
    cls_node: cst.ClassDef,
    known_class_names: set[str],
    get_position: Callable[[cst.CSTNode], tuple[int, int]],
) -> list[ClassAttrFact]:
    """Find `T | None` attributes that are unconditionally initialized non-None in `__init__`."""
    stmts = _extract_suite_stmts(cls_node.body)
    init_fn = _find_init_method(stmts)
    if init_fn is None:
        return []

    decls = _collect_optional_attr_decls(stmts, init_fn)
    if not decls:
        return []

    init_non_none_names = _collect_init_non_none_names(init_fn, known_class_names)
    guaranteed_in_init = _collect_guaranteed_init_attrs(
        _extract_suite_stmts(init_fn.body), init_non_none_names, known_class_names
    )
    disqualified = _collect_possibly_none_or_deleted_attrs(stmts, known_class_names)

    results: list[ClassAttrFact] = []
    for attr_name, decl in decls.items():
        if attr_name not in guaranteed_in_init or attr_name in disqualified:
            continue
        line, col = get_position(decl.node)
        results.append(
            ClassAttrFact(
                name=attr_name,
                line=line,
                col=col,
                annotation_str=decl.annotation_str,
                optional_inner=decl.optional_inner,
                declared_in_init=decl.declared_in_init,
                has_none_default=decl.has_none_default,
            )
        )
    return results


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


class _InitOptionalAttrCollector(cst.CSTVisitor):
    """Find `self.<attr>: T | None = ...` annotations inside `__init__`."""

    def __init__(self) -> None:
        self.decls: dict[str, _OptionalAttrDecl] = {}

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:  # noqa: ARG002, N802
        return False

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:  # noqa: ARG002, N802
        return False

    def visit_AnnAssign(self, node: cst.AnnAssign) -> bool:  # noqa: N802
        attr = _self_attr_name(node.target)
        if attr is None:
            return True
        inner = extract_optional_inner(node.annotation)
        if inner is None:
            return True
        self.decls[attr] = _OptionalAttrDecl(
            name=attr,
            node=node.target,
            annotation_str=expr_to_code(node.annotation.annotation),
            optional_inner=inner,
            declared_in_init=True,
            has_none_default=False,
        )
        return True


def _collect_optional_attr_decls(
    class_stmts: list[cst.BaseStatement],
    init_fn: cst.FunctionDef,
) -> dict[str, _OptionalAttrDecl]:
    """Collect `T | None` attribute annotations on class body and inside `__init__`."""
    decls: dict[str, _OptionalAttrDecl] = {}
    for stmt in class_stmts:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        for small in stmt.body:
            if not isinstance(small, cst.AnnAssign) or not isinstance(small.target, cst.Name):
                continue
            inner = extract_optional_inner(small.annotation)
            if inner is None:
                continue
            ann_str = expr_to_code(small.annotation.annotation)
            decls[small.target.value] = _OptionalAttrDecl(
                name=small.target.value,
                node=small.target,
                annotation_str=ann_str,
                optional_inner=inner,
                declared_in_init=False,
                has_none_default=small.value is not None and is_none_expr(small.value),
            )

    init_collector = _InitOptionalAttrCollector()
    init_fn.body.visit(init_collector)
    for attr_name, decl in init_collector.decls.items():
        decls.setdefault(attr_name, decl)
    return decls


class _SelfAttrMutationVisitor(cst.CSTVisitor):
    """Flag any `self.<attr>` assigned None/unknown or deleted inside a method."""

    def __init__(
        self,
        non_none_names: set[str],
        known_class_names: set[str],
        disqualified: set[str],
    ) -> None:
        self.non_none_names = non_none_names
        self.known_class_names = known_class_names
        self.disqualified = disqualified

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:  # noqa: ARG002, N802
        return False

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:  # noqa: ARG002, N802
        return False

    def visit_Assign(self, node: cst.Assign) -> bool:  # noqa: N802
        for target in node.targets:
            attr = _self_attr_name(target.target)
            if attr is not None and not _expr_is_non_none(
                node.value, self.non_none_names, self.known_class_names
            ):
                self.disqualified.add(attr)
        return True

    def visit_AnnAssign(self, node: cst.AnnAssign) -> bool:  # noqa: N802
        attr = _self_attr_name(node.target)
        if (
            attr is not None
            and node.value is not None
            and not _expr_is_non_none(node.value, self.non_none_names, self.known_class_names)
        ):
            self.disqualified.add(attr)
        return True

    def visit_Del(self, node: cst.Del) -> bool:  # noqa: N802
        attr = _self_attr_name(node.target)
        if attr is not None:
            self.disqualified.add(attr)
        return True


def _collect_possibly_none_or_deleted_attrs(
    class_stmts: list[cst.BaseStatement],
    known_class_names: set[str],
) -> set[str]:
    """Collect any `self.<attr>` that is ever deleted or assigned None/unknown in any method."""
    disqualified: set[str] = set()
    for stmt in class_stmts:
        if not isinstance(stmt, cst.FunctionDef):
            continue
        method_non_none = _collect_init_non_none_names(stmt, known_class_names)
        visitor = _SelfAttrMutationVisitor(
            non_none_names=method_non_none,
            known_class_names=known_class_names,
            disqualified=disqualified,
        )
        stmt.body.visit(visitor)
    return disqualified


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
