"""LibCST rewriter for applying NarrowingCandidates and generating unified diffs."""

from __future__ import annotations

import difflib
from collections import defaultdict
from typing import TYPE_CHECKING

import libcst as cst

from py_type_narrower.models import NarrowingCandidate, TargetKind

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

_MIN_QUALIFIED_PARTS = 2


def rewrite_files(
    candidates: list[NarrowingCandidate],
    sources: dict[Path, str],
) -> dict[Path, str]:
    """Apply a set of NarrowingCandidates to in-memory sources.

    Returns a mapping containing only files whose content changed.
    """
    by_file: dict[Path, list[NarrowingCandidate]] = defaultdict(list)
    for cand in candidates:
        by_file[cand.file.resolve()].append(cand)

    updated: dict[Path, str] = {}
    for file_path, file_candidates in by_file.items():
        original = sources.get(file_path)
        if original is None:
            continue
        new_code = rewrite_module_source(original, file_candidates)
        if new_code != original:
            updated[file_path] = new_code
    return updated


def compute_unified_diffs(
    original_sources: dict[Path, str],
    updated_sources: dict[Path, str],
    project_root: Path,
) -> dict[Path, str]:
    """Generate unified diffs for every modified file."""
    diffs: dict[Path, str] = {}
    root = project_root.resolve()
    for file_path, new_code in sorted(updated_sources.items()):
        old_code = original_sources.get(file_path, "")
        try:
            display = str(file_path.relative_to(root))
        except ValueError:
            display = str(file_path)
        diff_lines = list(
            difflib.unified_diff(
                old_code.splitlines(keepends=True),
                new_code.splitlines(keepends=True),
                fromfile=f"a/{display}",
                tofile=f"b/{display}",
            )
        )
        if diff_lines:
            diffs[file_path] = "".join(diff_lines)
    return diffs


class _NarrowingTransformer(cst.CSTTransformer):
    """Apply parameter, return, and attribute narrowing edits to matching CST nodes."""

    def __init__(self, candidates: list[NarrowingCandidate]) -> None:
        self._class_stack: list[str] = []
        self._func_stack: list[str] = []
        self.added_literal = False
        self._by_func: dict[tuple[str | None, str], list[NarrowingCandidate]] = defaultdict(list)
        self._by_attr: dict[tuple[str, str], NarrowingCandidate] = {}

        for cand in candidates:
            parts = cand.symbol.split(".")
            if cand.target_kind == TargetKind.ATTRIBUTE and cand.target_name is not None:
                self._by_attr[(parts[-1], cand.target_name)] = cand
                continue
            self._by_func[(None, parts[-1])].append(cand)
            if len(parts) >= _MIN_QUALIFIED_PARTS:
                self._by_func[(parts[-2], parts[-1])].append(cand)

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:  # noqa: N802
        self._class_stack.append(node.name.value)
        return True

    def leave_ClassDef(  # noqa: N802
        self,
        original_node: cst.ClassDef,  # noqa: ARG002
        updated_node: cst.ClassDef,
    ) -> cst.ClassDef:
        self._class_stack.pop()
        return updated_node

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:  # noqa: N802
        self._func_stack.append(node.name.value)
        return True

    def leave_FunctionDef(  # noqa: N802
        self,
        original_node: cst.FunctionDef,  # noqa: ARG002
        updated_node: cst.FunctionDef,
    ) -> cst.FunctionDef:
        self._func_stack.pop()
        if self._func_stack:
            return updated_node

        cls_name = self._class_stack[-1] if self._class_stack else None
        matched = self._matching_candidates(cls_name, updated_node.name.value)
        if not matched:
            return updated_node

        return self._apply_to_function(updated_node, matched)

    def leave_AnnAssign(  # noqa: N802
        self,
        original_node: cst.AnnAssign,  # noqa: ARG002
        updated_node: cst.AnnAssign,
    ) -> cst.AnnAssign:
        if not self._class_stack:
            return updated_node
        cls_name = self._class_stack[-1]
        attr_name = self._match_attr_target(updated_node.target)
        if attr_name is None:
            return updated_node
        cand = self._by_attr.get((cls_name, attr_name))
        if cand is None:
            return updated_node
        return self._apply_to_ann_assign(updated_node, cand)

    def _match_attr_target(self, target: cst.BaseAssignTargetExpression) -> str | None:
        """Return attribute name for class-body `attr: T` or `__init__`'s `self.attr: T`."""
        if not self._func_stack and isinstance(target, cst.Name):
            return target.value
        if (
            self._func_stack == ["__init__"]
            and isinstance(target, cst.Attribute)
            and isinstance(target.value, cst.Name)
            and target.value.value == "self"
        ):
            return target.attr.value
        return None

    def _apply_to_ann_assign(
        self,
        node: cst.AnnAssign,
        cand: NarrowingCandidate,
    ) -> cst.AnnAssign:
        """Rewrite the annotation (and optionally drop `= None`) on an AnnAssign."""
        new_expr = cst.parse_expression(cand.suggested_type)
        new_ann = node.annotation.with_changes(annotation=new_expr)
        if cand.needs_literal_import:
            self.added_literal = True
        if cand.drop_none_default and not self._func_stack:
            return node.with_changes(
                annotation=new_ann,
                equal=cst.MaybeSentinel.DEFAULT,
                value=None,
            )
        return node.with_changes(annotation=new_ann)

    def _matching_candidates(
        self,
        cls_name: str | None,
        func_name: str,
    ) -> list[NarrowingCandidate]:
        """Return candidates whose symbol matches the current `(class_name, func_name)`."""
        if cls_name is not None:
            raw = self._by_func.get((cls_name, func_name), [])
            return [c for c in raw if c.symbol.endswith(f".{cls_name}.{func_name}")]
        raw = self._by_func.get((None, func_name), [])
        return [
            c
            for c in raw
            if c.symbol == func_name
            or (
                c.symbol.endswith(f".{func_name}")
                and len(c.symbol.split(".")) >= _MIN_QUALIFIED_PARTS
            )
        ]

    def _apply_to_function(
        self,
        node: cst.FunctionDef,
        candidates: list[NarrowingCandidate],
    ) -> cst.FunctionDef:
        """Apply parameter and return annotation updates to a FunctionDef."""
        param_edits = {
            c.target_name: c
            for c in candidates
            if c.target_kind == TargetKind.PARAMETER and c.target_name is not None
        }
        return_edit = next(
            (c for c in candidates if c.target_kind == TargetKind.RETURN),
            None,
        )

        new_params = self._update_params(node.params, param_edits) if param_edits else node.params
        new_returns = node.returns
        if return_edit is not None and node.returns is not None:
            new_expr = cst.parse_expression(return_edit.suggested_type)
            new_returns = node.returns.with_changes(annotation=new_expr)
            if return_edit.needs_literal_import:
                self.added_literal = True

        return node.with_changes(params=new_params, returns=new_returns)

    def _update_params(
        self,
        params: cst.Parameters,
        edits: dict[str, NarrowingCandidate],
    ) -> cst.Parameters:
        """Update matching parameters in posonly, regular, and kwonly lists."""
        return params.with_changes(
            posonly_params=[self._update_single_param(p, edits) for p in params.posonly_params],
            params=[self._update_single_param(p, edits) for p in params.params],
            kwonly_params=[self._update_single_param(p, edits) for p in params.kwonly_params],
        )

    def _update_single_param(
        self,
        param: cst.Param,
        edits: dict[str, NarrowingCandidate],
    ) -> cst.Param:
        """Rewrite annotation (and optionally drop `= None` default) on a single Param."""
        cand = edits.get(param.name.value)
        if cand is None or param.annotation is None:
            return param

        new_expr = cst.parse_expression(cand.suggested_type)
        new_ann = param.annotation.with_changes(annotation=new_expr)
        if cand.needs_literal_import:
            self.added_literal = True

        if cand.drop_none_default:
            return param.with_changes(
                annotation=new_ann,
                equal=cst.MaybeSentinel.DEFAULT,
                default=None,
            )
        return param.with_changes(annotation=new_ann)


def rewrite_module_source(source: str, candidates: list[NarrowingCandidate]) -> str:
    """Apply NarrowingCandidates belonging to a single module and return updated source."""
    if not candidates:
        return source

    module = cst.parse_module(source)
    transformer = _NarrowingTransformer(candidates)
    modified = module.visit(transformer)

    if transformer.added_literal and not _has_literal_import(modified):
        modified = _insert_literal_import(modified)
    return modified.code


def _has_literal_import(module: cst.Module) -> bool:
    """Return True if `Literal` is already imported in the module."""
    for stmt in module.body:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        for small in stmt.body:
            if not isinstance(small, cst.ImportFrom):
                continue
            if isinstance(small.names, cst.ImportStar):
                continue
            for alias in small.names:
                name = alias.name
                if isinstance(name, cst.Name) and name.value == "Literal":
                    return True
    return False


def _insert_literal_import(module: cst.Module) -> cst.Module:
    """Add `Literal` to an existing `from typing import ...` or insert a new import line."""
    new_body = list(module.body)
    for idx, stmt in enumerate(new_body):
        merged = _try_add_literal_to_typing_import(stmt)
        if merged is not None:
            new_body[idx] = merged
            return module.with_changes(body=new_body)

    insert_idx = _find_import_insertion_index(new_body)
    import_stmt = cst.parse_statement("from typing import Literal\n")
    new_body.insert(insert_idx, import_stmt)
    return module.with_changes(body=new_body)


def _try_add_literal_to_typing_import(
    stmt: cst.SimpleStatementLine | cst.BaseCompoundStatement,
) -> cst.SimpleStatementLine | None:
    """If stmt is `from typing import ...`, return a copy with `Literal` appended."""
    if not isinstance(stmt, cst.SimpleStatementLine) or len(stmt.body) != 1:
        return None
    small = stmt.body[0]
    if not isinstance(small, cst.ImportFrom) or isinstance(small.names, cst.ImportStar):
        return None
    if small.relative or not isinstance(small.module, cst.Name) or small.module.value != "typing":
        return None
    aliases = list(small.names)
    aliases.append(cst.ImportAlias(name=cst.Name("Literal")))
    return stmt.with_changes(body=[small.with_changes(names=aliases)])


def _find_import_insertion_index(body: Sequence[cst.BaseStatement]) -> int:
    """Find the proper top-of-module index after docstring and `from __future__` imports."""
    idx = 0
    if body and _is_docstring(body[0]):
        idx = 1
    while idx < len(body) and _is_future_import(body[idx]):
        idx += 1
    return idx


def _is_docstring(stmt: cst.BaseStatement) -> bool:
    """Return True if stmt is a module docstring."""
    if not isinstance(stmt, cst.SimpleStatementLine) or len(stmt.body) != 1:
        return False
    inner = stmt.body[0]
    return isinstance(inner, cst.Expr) and isinstance(
        inner.value, (cst.SimpleString, cst.ConcatenatedString)
    )


def _is_future_import(stmt: cst.BaseStatement) -> bool:
    """Return True if stmt is a `from __future__ import ...` statement."""
    if not isinstance(stmt, cst.SimpleStatementLine) or len(stmt.body) != 1:
        return False
    inner = stmt.body[0]
    return (
        isinstance(inner, cst.ImportFrom)
        and isinstance(inner.module, cst.Name)
        and inner.module.value == "__future__"
    )
