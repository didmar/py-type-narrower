"""LibCST rewriter for applying NarrowingCandidates and generating unified diffs."""

from __future__ import annotations

import difflib
from collections import defaultdict
from typing import TYPE_CHECKING

import libcst as cst

from py_type_narrower.models import NarrowingCandidate, TargetKind, TypedDictScaffold

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
        if not cand.is_fixable:
            continue
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
        self.added_scaffolds: list[TypedDictScaffold] = []
        self._seen_scaffold_names: set[str] = set()
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
            self._record_scaffold(return_edit)

        return node.with_changes(params=new_params, returns=new_returns)

    def _update_params(
        self,
        params: cst.Parameters,
        edits: dict[str, NarrowingCandidate],
    ) -> cst.Parameters:
        """Update matching parameters while preserving valid positional default ordering."""
        seen_default = False
        new_posonly: list[cst.Param] = []
        for param in params.posonly_params:
            updated = self._update_single_param(param, edits, allow_drop_default=not seen_default)
            if updated.default is not None:
                seen_default = True
            new_posonly.append(updated)

        new_regular: list[cst.Param] = []
        for param in params.params:
            updated = self._update_single_param(param, edits, allow_drop_default=not seen_default)
            if updated.default is not None:
                seen_default = True
            new_regular.append(updated)

        new_kwonly = [
            self._update_single_param(p, edits, allow_drop_default=True)
            for p in params.kwonly_params
        ]
        return params.with_changes(
            posonly_params=new_posonly,
            params=new_regular,
            kwonly_params=new_kwonly,
        )

    def _update_single_param(
        self,
        param: cst.Param,
        edits: dict[str, NarrowingCandidate],
        *,
        allow_drop_default: bool,
    ) -> cst.Param:
        """Rewrite annotation (and optionally drop `= None` default) on a single Param."""
        cand = edits.get(param.name.value)
        if cand is None or param.annotation is None:
            return param

        new_expr = cst.parse_expression(cand.suggested_type)
        new_ann = param.annotation.with_changes(annotation=new_expr)
        if cand.needs_literal_import:
            self.added_literal = True
        self._record_scaffold(cand)

        if cand.drop_none_default and allow_drop_default:
            return param.with_changes(
                annotation=new_ann,
                equal=cst.MaybeSentinel.DEFAULT,
                default=None,
            )
        return param.with_changes(annotation=new_ann)

    def _record_scaffold(self, cand: NarrowingCandidate) -> None:
        """Record a TypedDictScaffold once when its candidate is actually applied."""
        scaffold = cand.typeddict_scaffold
        if scaffold is not None and scaffold.class_name not in self._seen_scaffold_names:
            self._seen_scaffold_names.add(scaffold.class_name)
            self.added_scaffolds.append(scaffold)


def rewrite_module_source(source: str, candidates: list[NarrowingCandidate]) -> str:
    """Apply fixable NarrowingCandidates belonging to a single module and return updated source."""
    fixable = [c for c in candidates if c.is_fixable]
    if not fixable:
        return source

    module = cst.parse_module(source)
    transformer = _NarrowingTransformer(fixable)
    modified = module.visit(transformer)

    if transformer.added_scaffolds:
        modified = _insert_typeddict_scaffolds(modified, transformer.added_scaffolds)

    needed_imports = _collect_needed_typing_imports(transformer)
    if needed_imports:
        modified = _ensure_typing_imports(modified, needed_imports)
    return modified.code


def _collect_needed_typing_imports(transformer: _NarrowingTransformer) -> list[str]:
    """Collect `typing` symbol names required by applied edits in deterministic order."""
    needed: list[str] = []
    if transformer.added_literal:
        needed.append("Literal")
    if transformer.added_scaffolds:
        needed.append("TypedDict")
        if any(s.needs_not_required_import for s in transformer.added_scaffolds):
            needed.append("NotRequired")
        if any(s.needs_any_import for s in transformer.added_scaffolds):
            needed.append("Any")
    return needed


def _insert_typeddict_scaffolds(
    module: cst.Module,
    scaffolds: list[TypedDictScaffold],
) -> cst.Module:
    """Insert scaffolded TypedDict definitions above their anchor top-level definitions."""
    new_body = list(module.body)
    for scaffold in scaffolds:
        stmt = _build_typeddict_statement(scaffold)
        anchor_idx = _find_anchor_index(new_body, scaffold.anchor_symbol)
        if anchor_idx is None:
            anchor_idx = _find_import_insertion_index(new_body)
        new_body.insert(anchor_idx, stmt)
    return module.with_changes(body=new_body)


def _find_anchor_index(
    body: Sequence[cst.BaseStatement],
    anchor_symbol: str,
) -> int | None:
    """Return index of the top-level ClassDef or FunctionDef named `anchor_symbol`."""
    for idx, stmt in enumerate(body):
        if isinstance(stmt, (cst.ClassDef, cst.FunctionDef)) and stmt.name.value == anchor_symbol:
            return idx
    return None


def _build_typeddict_statement(
    scaffold: TypedDictScaffold,
) -> cst.SimpleStatementLine | cst.BaseCompoundStatement:
    """Render and parse a TypedDict class or functional definition statement."""
    if scaffold.use_functional_syntax:
        entries: list[str] = []
        for key, val_type, is_req in scaffold.fields:
            field_type = f"NotRequired[{val_type}]" if (scaffold.total and not is_req) else val_type
            entries.append(f"        {key!r}: {field_type},")
        total_arg = ",\n    total=False" if not scaffold.total else ""
        code = (
            f"{scaffold.class_name} = TypedDict(\n"
            f"    {scaffold.class_name!r},\n"
            "    {\n" + "\n".join(entries) + "\n    }" + total_arg + ",\n)\n"
        )
    else:
        header = (
            f"class {scaffold.class_name}(TypedDict):"
            if scaffold.total
            else f"class {scaffold.class_name}(TypedDict, total=False):"
        )
        lines = [header]
        for key, val_type, is_req in scaffold.fields:
            field_type = f"NotRequired[{val_type}]" if (scaffold.total and not is_req) else val_type
            lines.append(f"    {key}: {field_type}")
        code = "\n".join(lines) + "\n"

    stmt = cst.parse_statement(code)
    return stmt.with_changes(leading_lines=[cst.EmptyLine(), cst.EmptyLine()])


def _ensure_typing_imports(module: cst.Module, needed_names: list[str]) -> cst.Module:
    """Ensure every name in `needed_names` is imported from `typing`."""
    existing = _existing_imported_names(module)
    missing = [name for name in needed_names if name not in existing]
    if not missing:
        return module

    new_body = list(module.body)
    for idx, stmt in enumerate(new_body):
        merged = _try_add_names_to_typing_import(stmt, missing)
        if merged is not None:
            new_body[idx] = merged
            return module.with_changes(body=new_body)

    insert_idx = _find_import_insertion_index(new_body)
    import_stmt = cst.parse_statement(f"from typing import {', '.join(missing)}\n")
    new_body.insert(insert_idx, import_stmt)
    return module.with_changes(body=new_body)


def _existing_imported_names(module: cst.Module) -> set[str]:
    """Collect symbol names already imported at module top level via `from ... import ...`."""
    imported: set[str] = set()
    for stmt in module.body:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        for small in stmt.body:
            if not isinstance(small, cst.ImportFrom) or isinstance(small.names, cst.ImportStar):
                continue
            for alias in small.names:
                asname = alias.asname.name if alias.asname is not None else alias.name
                if isinstance(asname, cst.Name):
                    imported.add(asname.value)
    return imported


def _try_add_names_to_typing_import(
    stmt: cst.SimpleStatementLine | cst.BaseCompoundStatement,
    names_to_add: list[str],
) -> cst.SimpleStatementLine | None:
    """If stmt is `from typing import ...`, return a copy with `names_to_add` appended."""
    if not isinstance(stmt, cst.SimpleStatementLine) or len(stmt.body) != 1:
        return None
    small = stmt.body[0]
    if not isinstance(small, cst.ImportFrom) or isinstance(small.names, cst.ImportStar):
        return None
    if small.relative or not isinstance(small.module, cst.Name) or small.module.value != "typing":
        return None
    aliases = list(small.names)
    aliases.extend(cst.ImportAlias(name=cst.Name(name)) for name in names_to_add)
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
