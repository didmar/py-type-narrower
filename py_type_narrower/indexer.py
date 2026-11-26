"""Whole-repository LibCST indexer collecting definitions, hierarchy, body facts, and calls."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal as TypingLiteral

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider

from py_type_narrower.body_analysis import (
    BodyFacts,
    analyze_function_body,
    extract_module_const_dicts,
)
from py_type_narrower.class_analysis import ClassAttrFact, analyze_class_attributes
from py_type_narrower.cst_utils import (
    ExtractedLiteral,
    base_primitive_annotation,
    expr_to_code,
    extract_literal,
    extract_literal_values_from_annotation,
    extract_not_required_inner,
    extract_optional_inner,
    extract_required_inner,
    extract_union_members,
    infer_simple_expr_type,
    is_bool_annotation,
    is_definitely_non_none_expr,
    is_none_expr,
)
from py_type_narrower.ignores import LineIgnore, parse_file_ignores

logger = logging.getLogger(__name__)

ABSTRACT_BASE_NAMES: frozenset[str] = frozenset(
    {"ABC", "Protocol", "abc.ABC", "typing.Protocol", "typing_extensions.Protocol"}
)
NAMEDTUPLE_BASE_NAMES: frozenset[str] = frozenset({"NamedTuple", "typing.NamedTuple"})
PYDANTIC_BASE_NAMES: frozenset[str] = frozenset({"BaseModel", "pydantic.BaseModel"})
TYPEDDICT_BASE_NAMES: frozenset[str] = frozenset(
    {"TypedDict", "typing.TypedDict", "typing_extensions.TypedDict"}
)
STRENUM_BASE_NAMES: frozenset[str] = frozenset({"StrEnum", "enum.StrEnum"})
INTENUM_BASE_NAMES: frozenset[str] = frozenset({"IntEnum", "enum.IntEnum"})
ENUM_BASE_NAMES: frozenset[str] = frozenset({"Enum", "enum.Enum"})
TRIVIAL_BASE_NAMES: frozenset[str] = (
    ABSTRACT_BASE_NAMES
    | NAMEDTUPLE_BASE_NAMES
    | PYDANTIC_BASE_NAMES
    | TYPEDDICT_BASE_NAMES
    | STRENUM_BASE_NAMES
    | INTENUM_BASE_NAMES
    | ENUM_BASE_NAMES
    | frozenset({"object", "builtins.object", "Generic", "typing.Generic"})
)


@dataclass
class ParamInfo:
    """Indexed metadata for a single function parameter."""

    name: str
    line: int
    col: int
    annotation_str: str | None
    primitive_kind: TypingLiteral["str", "int"] | None
    optional_inner: str | None
    literal_values: tuple[ExtractedLiteral, ...] | None
    has_default: bool
    default_is_none: bool
    default_literal: ExtractedLiteral | None
    kind: TypingLiteral["posonly", "regular", "kwonly"]
    is_bool: bool = False
    union_members: tuple[str, ...] | None = None
    default_type: str | None = None


@dataclass
class FieldInfo:
    """Indexed metadata for a dataclass or NamedTuple field."""

    name: str
    line: int
    col: int
    annotation_str: str
    primitive_kind: TypingLiteral["str", "int"] | None
    optional_inner: str | None
    has_default: bool
    default_is_none: bool
    default_literal: ExtractedLiteral | None
    is_kw_only: bool = False
    is_not_required: bool = False
    not_required_inner: str | None = None


@dataclass
class FunctionInfo:
    """Indexed metadata and body facts for a function or method definition."""

    qualname: str
    name: str
    module_name: str
    class_name: str | None
    file: Path
    line: int
    col: int
    is_private: bool
    is_method: bool
    is_staticmethod: bool
    is_classmethod: bool
    is_abstract: bool
    is_override: bool
    in_protocol_or_abc: bool
    is_test_file: bool
    parameters: list[ParamInfo]
    has_vararg: bool
    has_kwarg: bool
    return_annotation_str: str | None
    return_primitive_kind: TypingLiteral["str", "int"] | None
    return_optional_inner: str | None
    return_line: int
    return_col: int
    body_facts: BodyFacts
    return_union_members: tuple[str, ...] | None = None
    in_external_subclass: bool = False
    is_overridden: bool = False
    in_base_class: bool = False


@dataclass
class ObservedArg:
    """Single observed argument passed to a parameter or field at a call/write site."""

    file: Path
    line: int
    is_test_caller: bool
    is_none: bool = False
    is_non_none: bool = False
    literals: tuple[ExtractedLiteral, ...] | None = None
    is_unknown: bool = False
    observed_type: str | None = None
    dict_literal_keys: tuple[tuple[str, str | None], ...] | None = None
    enum_qualname: str | None = None


@dataclass
class ParamCallStats:
    """Aggregated call-site observations for one parameter or dataclass field."""

    observations: list[ObservedArg] = field(default_factory=list)
    omitted_count: int = 0
    has_dynamic_star_call: bool = False


@dataclass
class ClassInfo:
    """Indexed class hierarchy, fields, and local attribute facts."""

    qualname: str
    name: str
    module_name: str
    bases: list[str]
    file: Path
    line: int
    col: int
    is_private: bool = False
    is_test_file: bool = False
    methods: set[str] = field(default_factory=set)
    is_protocol_or_abc: bool = False
    is_dataclass: bool = False
    is_namedtuple: bool = False
    is_attrs: bool = False
    is_pydantic: bool = False
    is_typeddict: bool = False
    typeddict_total: bool = True
    is_kw_only_dataclass: bool = False
    enum_kind: TypingLiteral["str", "int"] | None = None
    enum_members: frozenset[str] = field(default_factory=frozenset)
    enum_values: frozenset[ExtractedLiteral] = field(default_factory=frozenset)
    fields: list[FieldInfo] = field(default_factory=list)
    attr_facts: list[ClassAttrFact] = field(default_factory=list)

    @property
    def is_structured_model(self) -> bool:
        """Return True if this class is a dataclass, NamedTuple, attrs, Pydantic, or TypedDict."""
        return (
            self.is_dataclass
            or self.is_namedtuple
            or self.is_attrs
            or self.is_pydantic
            or self.is_typeddict
        )


@dataclass
class CodebaseIndex:
    """Complete static index for a set of Python files."""

    files: dict[Path, str] = field(default_factory=dict)
    ignores: dict[Path, dict[int, LineIgnore]] = field(default_factory=dict)
    functions: dict[str, FunctionInfo] = field(default_factory=dict)
    classes: dict[str, ClassInfo] = field(default_factory=dict)
    call_stats: dict[tuple[str, str], ParamCallStats] = field(default_factory=dict)
    field_stats: dict[tuple[str, str], ParamCallStats] = field(default_factory=dict)
    module_imports: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass
class _ModuleContext:
    """Intermediate per-module parse data kept between definition and call-site passes."""

    file: Path
    module_name: str
    is_test_file: bool
    wrapper: MetadataWrapper
    imports: dict[str, str]
    module_imports: dict[str, str]
    class_names: set[str]


def index_paths(
    paths: list[Path],
    project_root: Path,
    exclude: list[str],
    *,
    overrides: dict[Path, str] | None = None,
) -> CodebaseIndex:
    """Discover and index Python files under paths into a CodebaseIndex."""
    python_files = discover_python_files(paths, exclude)
    return index_files(python_files, project_root, overrides=overrides)


def index_files(
    files: list[Path],
    project_root: Path,
    *,
    overrides: dict[Path, str] | None = None,
) -> CodebaseIndex:
    """Index an explicit list of Python files (optionally with in-memory content overrides)."""
    index = CodebaseIndex()
    module_contexts: list[_ModuleContext] = []

    # Pass 1: parse modules, collect classes, functions, imports, and local body facts.
    for path in sorted(files):
        source = _read_source(path, overrides)
        if source is None:
            continue
        ctx = _index_module_definitions(path, source, project_root, index)
        if ctx is not None:
            module_contexts.append(ctx)

    # Resolve class inheritance overrides and module re-exports across modules.
    classes_by_short_name = _mark_method_overrides(index)
    _resolve_module_reexports(module_contexts, index)
    for ctx in module_contexts:
        index.module_imports[ctx.module_name] = dict(ctx.imports)

    # Pass 2: walk calls across all modules and bind arguments to target FunctionInfo params.
    for ctx in module_contexts:
        _collect_module_calls(ctx, index, classes_by_short_name)

    return index


def discover_python_files(paths: list[Path], exclude: list[str]) -> list[Path]:
    """Recursively collect `.py` files from files/directories while honoring excludes."""
    exclude_set = set(exclude)
    found: set[Path] = set()
    for raw_path in paths:
        path = raw_path.resolve()
        if path.is_file() and path.suffix == ".py":
            found.add(path)
        elif path.is_dir():
            for py_file in path.rglob("*.py"):
                if not _is_excluded(py_file, path, exclude_set):
                    found.add(py_file.resolve())
    return sorted(found)


def _is_excluded(file_path: Path, base: Path, exclude_set: set[str]) -> bool:
    """Return True if any relative path component matches the exclude list."""
    try:
        rel_parts = file_path.relative_to(base).parts
    except ValueError:
        rel_parts = file_path.parts
    return any(part in exclude_set for part in rel_parts)


def _read_source(path: Path, overrides: dict[Path, str] | None) -> str | None:
    """Read source text from overrides or disk."""
    resolved = path.resolve()
    if overrides is not None:
        if resolved in overrides:
            return overrides[resolved]
        if path in overrides:
            return overrides[path]
    try:
        return resolved.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Skipping unreadable file %s: %s", path, exc)
        return None


_MAX_REEXPORT_HOPS = 8


def _resolve_module_reexports(
    module_contexts: list[_ModuleContext],
    index: CodebaseIndex,
) -> None:
    """Resolve package `__init__.py` and module re-exports transitively in `ctx.imports`."""
    reexports: dict[str, str] = {}
    for ctx in module_contexts:
        for local_name, target_qual in ctx.imports.items():
            alias_qual = f"{ctx.module_name}.{local_name}"
            if alias_qual not in index.functions and alias_qual not in index.classes:
                reexports[alias_qual] = target_qual

    for ctx in module_contexts:
        for local_name, target_qual in list(ctx.imports.items()):
            curr = target_qual
            for _ in range(_MAX_REEXPORT_HOPS):
                nxt = reexports.get(curr)
                if nxt is None or nxt == curr:
                    break
                curr = nxt
            ctx.imports[local_name] = curr


def _mark_method_overrides(index: CodebaseIndex) -> dict[str, list[ClassInfo]]:
    """Mark override relationships and base/external-subclass flags across indexed classes."""
    classes_by_short_name: dict[str, list[ClassInfo]] = {}
    for cls in index.classes.values():
        classes_by_short_name.setdefault(cls.name, []).append(cls)

    external_subclasses, base_classes, overridden_methods = _collect_class_hierarchy_facts(
        index.classes, classes_by_short_name
    )

    for func in index.functions.values():
        if not func.is_method or func.class_name is None:
            continue
        cls_qualname = f"{func.module_name}.{func.class_name}"
        cls_info = index.classes.get(cls_qualname)
        if cls_info is None:
            continue
        func.in_external_subclass = cls_qualname in external_subclasses
        func.in_base_class = cls_qualname in base_classes
        func.is_overridden = (cls_qualname, func.name) in overridden_methods
        if _method_exists_in_bases(func.name, cls_info, index.classes, classes_by_short_name):
            func.is_override = True
    return classes_by_short_name


def _collect_class_hierarchy_facts(
    all_classes: dict[str, ClassInfo],
    by_short_name: dict[str, list[ClassInfo]],
) -> tuple[set[str], set[str], set[tuple[str, str]]]:
    """Compute `(external_subclasses, base_classes_with_subclasses, overridden_methods)`."""
    external_subclasses: set[str] = set()
    base_classes: set[str] = set()
    overridden_methods: set[tuple[str, str]] = set()
    for qual, cls in all_classes.items():
        if _has_external_base(cls, all_classes, by_short_name):
            external_subclasses.add(qual)
        for ancestor in _collect_indexed_ancestors(cls, all_classes, by_short_name):
            base_classes.add(ancestor.qualname)
            for method_name in cls.methods & ancestor.methods:
                overridden_methods.add((ancestor.qualname, method_name))
    return (external_subclasses, base_classes, overridden_methods)


def _has_external_base(
    cls_info: ClassInfo,
    all_classes: dict[str, ClassInfo],
    by_short_name: dict[str, list[ClassInfo]],
) -> bool:
    """Return True if `cls_info` (or an indexed ancestor) inherits from an unindexed base class."""
    visited: set[str] = set()
    queue = list(cls_info.bases)
    while queue:
        raw_base = queue.pop(0)
        base_name = _strip_base_type_args(raw_base)
        if base_name in visited:
            continue
        visited.add(base_name)
        if base_name in TRIVIAL_BASE_NAMES:
            continue
        matches = _lookup_base_classes(base_name, cls_info.module_name, all_classes, by_short_name)
        if not matches:
            return True
        for base_cls in matches:
            queue.extend(base_cls.bases)
    return False


def _method_exists_in_bases(
    method_name: str,
    cls_info: ClassInfo,
    all_classes: dict[str, ClassInfo],
    by_short_name: dict[str, list[ClassInfo]],
) -> bool:
    """Walk base classes to see if method_name is defined on any ancestor."""
    visited: set[str] = set()
    queue = list(cls_info.bases)
    while queue:
        raw_base = queue.pop(0)
        base_name = _strip_base_type_args(raw_base)
        if base_name in visited:
            continue
        visited.add(base_name)
        if base_name in ABSTRACT_BASE_NAMES:
            continue
        matches = _lookup_base_classes(base_name, cls_info.module_name, all_classes, by_short_name)
        for base_cls in matches:
            if method_name in base_cls.methods:
                return True
            queue.extend(base_cls.bases)
    return False


@dataclass(frozen=True)
class _DefCollectionConfig:
    """Module-level context passed to _DefinitionCollector."""

    file: Path
    module_name: str
    is_test_file: bool
    exported_all: set[str] | None
    const_dicts: dict[str, set[ExtractedLiteral]]
    known_class_names: set[str]


class _DefinitionCollector(cst.CSTVisitor):
    """Collect classes, functions, parameters, and local body facts in a single module."""

    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, cfg: _DefCollectionConfig, index: CodebaseIndex) -> None:
        self.file = cfg.file
        self.module_name = cfg.module_name
        self.is_test_file = cfg.is_test_file
        self.exported_all = cfg.exported_all
        self.const_dicts = cfg.const_dicts
        self.known_class_names = cfg.known_class_names
        self.index = index
        self._class_stack: list[ClassInfo] = []
        self._func_depth = 0

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:  # noqa: N802
        if _has_decorator(node, {"no_type_check"}):
            return False
        pos = self.get_metadata(PositionProvider, node.name)
        bases = [expr_to_code(arg.value) for arg in node.bases]
        is_proto_abc = any(base in ABSTRACT_BASE_NAMES for base in bases)
        is_dataclass, is_kw_only_dc = _inspect_dataclass_decorator(node)
        is_attrs, is_kw_only_attrs = _inspect_attrs_decorator(node)
        is_namedtuple = any(base in NAMEDTUPLE_BASE_NAMES for base in bases)
        is_pydantic = any(base in PYDANTIC_BASE_NAMES for base in bases)
        is_typeddict = any(base in TYPEDDICT_BASE_NAMES for base in bases)
        is_structured = is_dataclass or is_attrs or is_namedtuple or is_pydantic or is_typeddict
        typeddict_total = _extract_typeddict_total(node) if is_typeddict else True
        enum_kind = _detect_enum_kind(bases)
        enum_members, enum_values = (
            _extract_enum_members(node, enum_kind)
            if enum_kind is not None
            else (frozenset(), frozenset())
        )
        prefix = ".".join(c.name for c in self._class_stack)
        cls_name = f"{prefix}.{node.name.value}" if prefix else node.name.value
        qualname = f"{self.module_name}.{cls_name}"
        is_private = _is_symbol_private(
            node.name.value,
            class_name=None,
            exported_all=self.exported_all,
            in_nested_function=self._func_depth > 0,
        )
        fields = (
            self._extract_structured_fields(
                node,
                is_kw_only_dc=is_kw_only_dc or is_kw_only_attrs or is_pydantic,
                is_typeddict=is_typeddict,
                typeddict_total=typeddict_total,
            )
            if is_structured
            else []
        )
        attr_facts = (
            analyze_class_attributes(node, self.known_class_names, self._node_position)
            if not (is_proto_abc or is_structured or enum_kind)
            else []
        )
        cls_info = ClassInfo(
            qualname=qualname,
            name=node.name.value,
            module_name=self.module_name,
            bases=bases,
            file=self.file,
            line=pos.start.line,
            col=pos.start.column,
            is_private=is_private,
            is_test_file=self.is_test_file,
            is_protocol_or_abc=is_proto_abc,
            is_dataclass=is_dataclass,
            is_namedtuple=is_namedtuple,
            is_attrs=is_attrs,
            is_pydantic=is_pydantic,
            is_typeddict=is_typeddict,
            typeddict_total=typeddict_total,
            is_kw_only_dataclass=is_kw_only_dc or is_kw_only_attrs or is_pydantic,
            enum_kind=enum_kind,
            enum_members=enum_members,
            enum_values=enum_values,
            fields=fields,
            attr_facts=attr_facts,
        )
        self.index.classes[qualname] = cls_info
        self._class_stack.append(cls_info)
        return True

    def _node_position(self, node: cst.CSTNode) -> tuple[int, int]:
        """Return 1-based line and 0-based column for a CST node."""
        pos = self.get_metadata(PositionProvider, node)
        return (pos.start.line, pos.start.column)

    def _extract_structured_fields(
        self,
        cls_node: cst.ClassDef,
        *,
        is_kw_only_dc: bool,
        is_typeddict: bool = False,
        typeddict_total: bool = True,
    ) -> list[FieldInfo]:
        """Extract dataclass / NamedTuple / TypedDict fields from class body annotations."""
        if not isinstance(cls_node.body, cst.IndentedBlock):
            return []
        fields: list[FieldInfo] = []
        kw_only_active = is_kw_only_dc or is_typeddict
        for stmt in cls_node.body.body:
            if not isinstance(stmt, cst.SimpleStatementLine):
                continue
            for small in stmt.body:
                if not isinstance(small, cst.AnnAssign) or not isinstance(small.target, cst.Name):
                    continue
                ann_str = expr_to_code(small.annotation.annotation)
                if ann_str in {"KW_ONLY", "dataclasses.KW_ONLY"}:
                    kw_only_active = True
                    continue
                if ann_str.startswith(("ClassVar", "typing.ClassVar")):
                    continue
                field_info = self._build_field_info(
                    small,
                    ann_str,
                    kw_only_active=kw_only_active,
                    is_typeddict=is_typeddict,
                    typeddict_total=typeddict_total,
                )
                if field_info is not None:
                    fields.append(field_info)
        return fields

    def _build_field_info(
        self,
        stmt: cst.AnnAssign,
        ann_str: str,
        *,
        kw_only_active: bool,
        is_typeddict: bool = False,
        typeddict_total: bool = True,
    ) -> FieldInfo | None:
        """Build a FieldInfo from a single class-body AnnAssign, skipping init=False fields."""
        if not isinstance(stmt.target, cst.Name):
            return None
        val = stmt.value
        field_kw_only = kw_only_active
        has_default = val is not None
        default_is_none = False
        default_lit: ExtractedLiteral | None = None

        if val is not None and _is_structured_field_call(val):
            init_flag, dc_has_default, dc_kw_only, default_is_none, default_lit = (
                _parse_structured_field_call(val)
            )
            if not init_flag:
                return None
            has_default = dc_has_default
            field_kw_only = field_kw_only or dc_kw_only
        elif val is not None:
            default_is_none = is_none_expr(val)
            default_lit = extract_literal(val)

        not_req_inner = extract_not_required_inner(stmt.annotation) if is_typeddict else None
        req_inner = extract_required_inner(stmt.annotation) if is_typeddict else None
        is_not_required = not_req_inner is not None or (
            is_typeddict and not typeddict_total and req_inner is None
        )
        pos = self.get_metadata(PositionProvider, stmt.target)
        return FieldInfo(
            name=stmt.target.value,
            line=pos.start.line,
            col=pos.start.column,
            annotation_str=ann_str,
            primitive_kind=base_primitive_annotation(stmt.annotation),
            optional_inner=extract_optional_inner(stmt.annotation),
            has_default=has_default or is_not_required,
            default_is_none=default_is_none,
            default_literal=default_lit,
            is_kw_only=field_kw_only,
            is_not_required=is_not_required,
            not_required_inner=not_req_inner,
        )

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:  # noqa: N802
        if not _has_decorator(original_node, {"no_type_check"}):
            self._class_stack.pop()

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:  # noqa: N802
        in_nested = self._func_depth > 0
        self._func_depth += 1
        if in_nested or _has_decorator(node, {"no_type_check"}):
            return True

        pos = self.get_metadata(PositionProvider, node.name)
        current_cls = self._class_stack[-1] if self._class_stack else None
        is_method = current_cls is not None
        if current_cls is not None:
            current_cls.methods.add(node.name.value)

        cls_name = current_cls.name if current_cls is not None else None
        qualname = (
            f"{self.module_name}.{cls_name}.{node.name.value}"
            if cls_name is not None
            else f"{self.module_name}.{node.name.value}"
        )

        params = self._extract_params(node.params)
        param_non_none = {
            p.name
            for p in params
            if p.annotation_str is not None
            and p.optional_inner is None
            and p.annotation_str != "None"
        }
        param_types = {p.name: p.annotation_str for p in params if p.annotation_str is not None}
        body_facts = analyze_function_body(
            node,
            param_non_none_names=param_non_none,
            module_const_dicts=self.const_dicts,
            known_class_names=self.known_class_names,
            param_types=param_types,
        )

        ret_pos = (
            self.get_metadata(PositionProvider, node.returns) if node.returns is not None else pos
        )

        info = FunctionInfo(
            qualname=qualname,
            name=node.name.value,
            module_name=self.module_name,
            class_name=cls_name,
            file=self.file,
            line=pos.start.line,
            col=pos.start.column,
            is_private=_is_symbol_private(
                node.name.value, cls_name, self.exported_all, in_nested_function=False
            ),
            is_method=is_method,
            is_staticmethod=_has_decorator(node, {"staticmethod"}),
            is_classmethod=_has_decorator(node, {"classmethod"}),
            is_abstract=_has_decorator(node, {"abstractmethod"}),
            is_override=_has_decorator(node, {"override"}),
            in_protocol_or_abc=current_cls.is_protocol_or_abc if current_cls else False,
            is_test_file=self.is_test_file,
            parameters=params,
            has_vararg=isinstance(node.params.star_arg, cst.Param),
            has_kwarg=node.params.star_kwarg is not None,
            return_annotation_str=(
                expr_to_code(node.returns.annotation) if node.returns is not None else None
            ),
            return_primitive_kind=base_primitive_annotation(node.returns),
            return_optional_inner=extract_optional_inner(node.returns),
            return_line=ret_pos.start.line,
            return_col=ret_pos.start.column,
            body_facts=body_facts,
            return_union_members=extract_union_members(node.returns),
        )
        self.index.functions[qualname] = info
        return True

    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:  # noqa: ARG002, N802
        self._func_depth -= 1

    def _extract_params(self, params: cst.Parameters) -> list[ParamInfo]:
        """Extract ParamInfo records for posonly, regular, and kwonly parameters."""
        result: list[ParamInfo] = []
        groups: list[tuple[TypingLiteral["posonly", "regular", "kwonly"], list[cst.Param]]] = [
            ("posonly", list(params.posonly_params)),
            ("regular", list(params.params)),
            ("kwonly", list(params.kwonly_params)),
        ]
        for kind, group in groups:
            for p in group:
                pos = self.get_metadata(PositionProvider, p.name)
                default = p.default
                result.append(
                    ParamInfo(
                        name=p.name.value,
                        line=pos.start.line,
                        col=pos.start.column,
                        annotation_str=(
                            expr_to_code(p.annotation.annotation)
                            if p.annotation is not None
                            else None
                        ),
                        primitive_kind=base_primitive_annotation(p.annotation),
                        optional_inner=extract_optional_inner(p.annotation),
                        literal_values=extract_literal_values_from_annotation(p.annotation),
                        has_default=default is not None,
                        default_is_none=default is not None and is_none_expr(default),
                        default_literal=extract_literal(default) if default is not None else None,
                        kind=kind,
                        is_bool=is_bool_annotation(p.annotation),
                        union_members=extract_union_members(p.annotation),
                        default_type=infer_simple_expr_type(default)
                        if default is not None
                        else None,
                    )
                )
        return result


def _index_module_definitions(
    path: Path,
    source: str,
    project_root: Path,
    index: CodebaseIndex,
) -> _ModuleContext | None:
    """Parse one module and populate its classes, functions, and body facts in the index."""
    resolved = path.resolve()
    try:
        module = cst.parse_module(source)
    except cst.ParserSyntaxError as exc:
        logger.warning("Skipping file with syntax error %s: %s", path, exc)
        return None

    module_name = _compute_module_name(resolved, project_root)
    index.files[resolved] = source
    index.ignores[resolved] = parse_file_ignores(source)

    wrapper = MetadataWrapper(module)
    const_dicts = extract_module_const_dicts(module)
    exported_all = _extract_dunder_all(module)
    imports, module_imports = _extract_imports(module, module_name)
    class_names = _collect_top_level_class_names(module)
    test_file = is_test_path(resolved)

    collector = _DefinitionCollector(
        cfg=_DefCollectionConfig(
            file=resolved,
            module_name=module_name,
            is_test_file=test_file,
            exported_all=exported_all,
            const_dicts=const_dicts,
            known_class_names=class_names,
        ),
        index=index,
    )
    wrapper.visit(collector)

    return _ModuleContext(
        file=resolved,
        module_name=module_name,
        is_test_file=test_file,
        wrapper=wrapper,
        imports=imports,
        module_imports=module_imports,
        class_names=class_names,
    )


def is_test_path(path: Path) -> bool:
    """Return True if a file path appears to be a test module."""
    name = path.name
    if name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py":
        return True
    return any(part in {"tests", "test"} for part in path.parts)


def _compute_module_name(path: Path, project_root: Path) -> str:
    """Compute a dotted Python module name relative to project_root (stripping src/)."""
    resolved = path.resolve()
    root = project_root.resolve()
    try:
        rel = resolved.relative_to(root)
    except ValueError:
        rel = Path(resolved.name)

    parts = list(rel.parts)
    if parts and parts[0] == "src" and len(parts) > 1:
        parts = parts[1:]
    if parts and parts[-1] == "__init__.py":
        parts = parts[:-1]
    elif parts:
        parts[-1] = Path(parts[-1]).stem
    return ".".join(parts) if parts else resolved.stem


def _extract_dunder_all(module: cst.Module) -> set[str] | None:
    """Extract string names from `__all__ = [...]` if defined at module top level."""
    for stmt in module.body:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        for small in stmt.body:
            if not isinstance(small, cst.Assign) or len(small.targets) != 1:
                continue
            target = small.targets[0].target
            if not isinstance(target, cst.Name) or target.value != "__all__":
                continue
            if isinstance(small.value, (cst.List, cst.Tuple)):
                names: set[str] = set()
                for elem in small.value.elements:
                    if isinstance(elem, cst.Element):
                        lit = extract_literal(elem.value)
                        if lit is not None and lit.kind == "str" and isinstance(lit.value, str):
                            names.add(lit.value)
                return names
    return None


def _extract_imports(
    module: cst.Module,
    module_name: str,
) -> tuple[dict[str, str], dict[str, str]]:
    """Extract symbol imports (`local -> qualname`) and module imports (`alias -> mod`)."""
    symbol_imports: dict[str, str] = {}
    module_imports: dict[str, str] = {}
    for stmt in module.body:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        for small in stmt.body:
            if isinstance(small, cst.Import):
                _record_import(small, module_imports)
            elif isinstance(small, cst.ImportFrom):
                _record_import_from(small, module_name, symbol_imports, module_imports)
    return symbol_imports, module_imports


def _record_import(node: cst.Import, module_imports: dict[str, str]) -> None:
    """Record `import a.b` or `import a.b as c`."""
    for alias in node.names:
        full_mod = expr_to_code(alias.name)
        if alias.asname is not None and isinstance(alias.asname.name, cst.Name):
            module_imports[alias.asname.name.value] = full_mod
        else:
            top_name = full_mod.split(".")[0]
            module_imports[top_name] = top_name
            module_imports[full_mod] = full_mod


def _record_import_from(
    node: cst.ImportFrom,
    current_module: str,
    symbol_imports: dict[str, str],
    module_imports: dict[str, str],
) -> None:
    """Record `from pkg.mod import fn as alias` including relative imports."""
    if isinstance(node.names, cst.ImportStar):
        return
    base_mod = _resolve_relative_module(node, current_module)
    if not base_mod:
        return
    for alias in node.names:
        imported_name = expr_to_code(alias.name)
        local_name = (
            alias.asname.name.value
            if alias.asname is not None and isinstance(alias.asname.name, cst.Name)
            else imported_name
        )
        qualified = f"{base_mod}.{imported_name}"
        symbol_imports[local_name] = qualified
        module_imports[local_name] = qualified


def _resolve_relative_module(node: cst.ImportFrom, current_module: str) -> str:
    """Resolve relative dots on an ImportFrom statement into a dotted module path."""
    rel_count = len(node.relative)
    mod_part = expr_to_code(node.module) if node.module is not None else ""
    if rel_count == 0:
        return mod_part
    parts = current_module.split(".")
    # Keep package prefix by popping `rel_count` segments.
    prefix = parts[:-rel_count] if rel_count <= len(parts) else []
    if mod_part:
        prefix.append(mod_part)
    return ".".join(prefix)


def _collect_top_level_class_names(module: cst.Module) -> set[str]:
    """Return names of classes defined at top level in a module."""
    return {stmt.name.value for stmt in module.body if isinstance(stmt, cst.ClassDef)}


def _has_decorator(node: cst.FunctionDef | cst.ClassDef, names: set[str]) -> bool:
    """Return True if a function or class has any decorator matching one of `names`."""
    for dec in node.decorators:
        expr = dec.decorator
        if isinstance(expr, cst.Call):
            expr = expr.func
        code = expr_to_code(expr)
        if code in names or code.split(".")[-1] in names:
            return True
    return False


def _is_symbol_private(
    name: str,
    class_name: str | None,
    exported_all: set[str] | None,
    *,
    in_nested_function: bool,
) -> bool:
    """Determine whether a function or method is internal/private."""
    if in_nested_function:
        return True
    if name.startswith("_") and not (name.startswith("__") and name.endswith("__")):
        return True
    if class_name is not None and class_name.startswith("_"):
        return True
    if exported_all is not None:
        top_symbol = class_name if class_name is not None else name
        if top_symbol not in exported_all:
            return True
    return False


@dataclass
class _LocalVarBinding:
    """Lightweight per-scope type/literal knowledge for a local variable or parameter."""

    literals: tuple[ExtractedLiteral, ...] | None = None
    is_non_none: bool = False
    class_type: str | None = None
    inferred_type: str | None = None


def _detect_enum_kind(bases: list[str]) -> TypingLiteral["str", "int"] | None:
    """Return 'str' or 'int' if class bases define a StrEnum or IntEnum."""
    if any(b in STRENUM_BASE_NAMES for b in bases):
        return "str"
    if any(b in INTENUM_BASE_NAMES for b in bases):
        return "int"
    if any(b in ENUM_BASE_NAMES for b in bases):
        if "str" in bases or "builtins.str" in bases:
            return "str"
        if "int" in bases or "builtins.int" in bases:
            return "int"
    return None


def _extract_enum_members(
    node: cst.ClassDef,
    enum_kind: TypingLiteral["str", "int"],
) -> tuple[frozenset[str], frozenset[ExtractedLiteral]]:
    """Extract member names and constant literal values from a StrEnum/IntEnum body."""
    if not isinstance(node.body, cst.IndentedBlock):
        return (frozenset(), frozenset())
    members: set[str] = set()
    values: set[ExtractedLiteral] = set()
    has_non_literal_member = False
    for stmt in node.body.body:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        for small in stmt.body:
            if not isinstance(small, cst.Assign) or len(small.targets) != 1:
                continue
            target = small.targets[0].target
            if not isinstance(target, cst.Name) or target.value.startswith("_"):
                continue
            members.add(target.value)
            lit = extract_literal(small.value)
            if lit is not None and lit.kind == enum_kind:
                values.add(lit)
            else:
                has_non_literal_member = True
    final_values = frozenset() if has_non_literal_member else frozenset(values)
    return (frozenset(members), final_values)


def _extract_typeddict_total(node: cst.ClassDef) -> bool:
    """Return False if a TypedDict class declares `total=False`, else True."""
    for kw in node.keywords:
        if kw.keyword is not None and kw.keyword.value == "total":
            lit = extract_literal(kw.value)
            if lit is not None and lit.kind == "bool" and isinstance(lit.value, bool):
                return lit.value
    return True


def _inspect_dataclass_decorator(node: cst.ClassDef) -> tuple[bool, bool]:
    """Return `(is_dataclass, is_kw_only)` from class decorators."""
    for dec in node.decorators:
        expr = dec.decorator
        kw_only = False
        if isinstance(expr, cst.Call):
            for arg in expr.args:
                if (
                    arg.keyword is not None
                    and arg.keyword.value == "kw_only"
                    and isinstance(arg.value, cst.Name)
                    and arg.value.value == "True"
                ):
                    kw_only = True
            expr = expr.func
        code = expr_to_code(expr)
        if (
            code in {"dataclass", "dataclasses.dataclass"}
            or code.rsplit(".", maxsplit=1)[-1] == "dataclass"
        ):
            return (True, kw_only)
    return (False, False)


_ATTRS_DECORATOR_NAMES: frozenset[str] = frozenset(
    {
        "define",
        "attrs.define",
        "frozen",
        "attrs.frozen",
        "mutable",
        "attrs.mutable",
        "attr.s",
        "attr.attrs",
        "attrs.attrs",
    }
)


def _inspect_attrs_decorator(node: cst.ClassDef) -> tuple[bool, bool]:
    """Return `(is_attrs, is_kw_only)` from attrs class decorators."""
    for dec in node.decorators:
        expr = dec.decorator
        kw_only = False
        if isinstance(expr, cst.Call):
            for arg in expr.args:
                if (
                    arg.keyword is not None
                    and arg.keyword.value == "kw_only"
                    and isinstance(arg.value, cst.Name)
                    and arg.value.value == "True"
                ):
                    kw_only = True
            expr = expr.func
        if expr_to_code(expr) in _ATTRS_DECORATOR_NAMES:
            return (True, kw_only)
    return (False, False)


def _is_structured_field_call(expr: cst.BaseExpression) -> bool:
    """Return True if expr is a dataclass, attrs, or Pydantic field specification call."""
    if not isinstance(expr, cst.Call):
        return False
    code = expr_to_code(expr.func)
    return code in {
        "field",
        "dataclasses.field",
        "attrs.field",
        "attr.field",
        "attr.ib",
        "Field",
        "pydantic.Field",
    }


def _parse_structured_field_call(
    expr: cst.BaseExpression,
) -> tuple[bool, bool, bool, bool, ExtractedLiteral | None]:
    """Parse `(init, has_default, kw_only, default_is_none, default_lit)` from a field() call."""
    if not isinstance(expr, cst.Call):
        return (True, False, False, False, None)
    init_flag = True
    has_default = False
    kw_only = False
    default_is_none = False
    default_lit: ExtractedLiteral | None = None
    for idx, arg in enumerate(expr.args):
        if arg.keyword is None and idx == 0 and not isinstance(arg.value, cst.Ellipsis):
            has_default = True
            default_is_none = is_none_expr(arg.value)
            default_lit = extract_literal(arg.value)
            continue
        if arg.keyword is None:
            continue
        kw = arg.keyword.value
        if kw == "init" and isinstance(arg.value, cst.Name) and arg.value.value == "False":
            init_flag = False
        elif kw == "default" and not isinstance(arg.value, cst.Ellipsis):
            has_default = True
            default_is_none = is_none_expr(arg.value)
            default_lit = extract_literal(arg.value)
        elif kw in {"default_factory", "factory"}:
            has_default = True
        elif kw == "kw_only" and isinstance(arg.value, cst.Name) and arg.value.value == "True":
            kw_only = True
    return (init_flag, has_default, kw_only, default_is_none, default_lit)


class _CallSiteCollector(cst.CSTVisitor):
    """Resolve call expressions across a module and record argument observations."""

    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(
        self,
        ctx: _ModuleContext,
        index: CodebaseIndex,
        classes_by_short_name: dict[str, list[ClassInfo]],
    ) -> None:
        self.ctx = ctx
        self.index = index
        self._classes_by_short_name = classes_by_short_name
        self._class_stack: list[str] = []
        self._return_typeddict_stack: list[ClassInfo | None] = []
        # Each entry is `(is_class_scope, bindings)`; class scopes do not enclose methods in Python.
        self._scope_stack: list[tuple[bool, dict[str, _LocalVarBinding]]] = [(False, {})]

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:  # noqa: N802
        self._class_stack.append(node.name.value)
        self._scope_stack.append((True, {}))
        return True

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:  # noqa: ARG002, N802
        self._scope_stack.pop()
        self._class_stack.pop()

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:  # noqa: N802
        scope: dict[str, _LocalVarBinding] = {}
        for p in (*node.params.posonly_params, *node.params.params, *node.params.kwonly_params):
            ann = p.annotation
            lits = extract_literal_values_from_annotation(ann)
            ann_str = expr_to_code(ann.annotation) if ann is not None else None
            class_ref = ann_str.strip("'\"") if ann_str is not None else None
            is_non_none = (
                ann is not None and extract_optional_inner(ann) is None and ann_str != "None"
            )
            scope[p.name.value] = _LocalVarBinding(
                literals=lits,
                is_non_none=is_non_none,
                class_type=class_ref,
                inferred_type=ann_str,
            )
        self._scope_stack.append((False, scope))
        ret_td: ClassInfo | None = None
        if node.returns is not None:
            ret_ref = expr_to_code(node.returns.annotation).strip("'\"")
            candidate = self._lookup_class_by_ref(ret_ref)
            if candidate is not None and candidate.is_typeddict:
                ret_td = candidate
        self._return_typeddict_stack.append(ret_td)
        return True

    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:  # noqa: ARG002, N802
        self._scope_stack.pop()
        self._return_typeddict_stack.pop()

    def visit_Return(self, node: cst.Return) -> bool:  # noqa: N802
        if (
            self._return_typeddict_stack
            and self._return_typeddict_stack[-1] is not None
            and isinstance(node.value, cst.Dict)
        ):
            pos = self.get_metadata(PositionProvider, node)
            self._bind_typeddict_literal(
                self._return_typeddict_stack[-1], node.value, pos.start.line
            )
        return True

    def visit_Assign(self, node: cst.Assign) -> bool:  # noqa: N802
        if len(node.targets) == 1:
            target = node.targets[0].target
            if isinstance(target, cst.Name):
                self._record_var_value(target.value, node.value, annotation=None)
            elif isinstance(target, cst.Attribute):
                self._record_attr_write(target, node.value)
            elif isinstance(target, cst.Subscript):
                self._record_typeddict_subscript_write(target, node.value)
            else:
                self._bind_unknown_target(target)
        else:
            for assign_target in node.targets:
                self._bind_unknown_target(assign_target.target)
        return True

    def visit_AnnAssign(self, node: cst.AnnAssign) -> bool:  # noqa: N802
        if isinstance(node.target, cst.Name):
            self._record_var_value(node.target.value, node.value, annotation=node.annotation)
            if isinstance(node.value, cst.Dict):
                ann_ref = expr_to_code(node.annotation.annotation).strip("'\"")
                td_cls = self._lookup_class_by_ref(ann_ref)
                if td_cls is not None and td_cls.is_typeddict:
                    pos = self.get_metadata(PositionProvider, node)
                    self._bind_typeddict_literal(td_cls, node.value, pos.start.line)
        return True

    def visit_AugAssign(self, node: cst.AugAssign) -> bool:  # noqa: N802
        if isinstance(node.target, cst.Name):
            scope = self._scope_stack[-1][1]
            existing = scope.get(node.target.value)
            if existing is not None:
                scope[node.target.value] = _LocalVarBinding(
                    literals=None,
                    is_non_none=existing.is_non_none,
                    class_type=existing.class_type,
                    inferred_type=existing.inferred_type,
                )
        return True

    def visit_For(self, node: cst.For) -> bool:  # noqa: N802
        self._bind_unknown_target(node.target)
        return True

    def visit_WithItem(self, node: cst.WithItem) -> bool:  # noqa: N802
        if node.asname is not None:
            self._bind_unknown_target(node.asname.name)
        return True

    def _bind_unknown_target(self, target: cst.BaseExpression) -> None:
        """Bind simple or unpacked target names as unknown so they shadow outer scopes."""
        if isinstance(target, cst.Name):
            self._scope_stack[-1][1][target.value] = _LocalVarBinding()
        elif isinstance(target, (cst.Tuple, cst.List)):
            for elem in target.elements:
                if isinstance(elem, (cst.Element, cst.StarredElement)):
                    self._bind_unknown_target(elem.value)

    def visit_Call(self, node: cst.Call) -> bool:  # noqa: N802
        pos = self.get_metadata(PositionProvider, node)
        target_fn, is_bound_call = self._resolve_call_target(node.func)
        if target_fn is not None:
            self._bind_call_arguments(target_fn, node, pos.start.line, is_bound=is_bound_call)
            return True

        target_cls = self._resolve_structured_class_target(node.func)
        if target_cls is not None:
            self._bind_field_arguments(target_cls, node, pos.start.line)
        return True

    def _record_attr_write(self, target: cst.Attribute, value: cst.BaseExpression) -> None:
        """Record `obj.field = value` writes against indexed structured-model fields."""
        if not isinstance(target.value, cst.Name):
            return
        receiver = target.value.value
        cls_info = self._resolve_receiver_class(receiver)
        if cls_info is None or not cls_info.is_structured_model:
            return
        field_name = target.attr.value
        if any(f.name == field_name for f in cls_info.fields):
            pos = self.get_metadata(PositionProvider, target)
            stats = self.index.field_stats.setdefault(
                (cls_info.qualname, field_name), ParamCallStats()
            )
            stats.observations.append(self._classify_arg_expr(value, pos.start.line))

    def _record_typeddict_subscript_write(
        self,
        target: cst.Subscript,
        value: cst.BaseExpression,
    ) -> None:
        """Record `td_var["key"] = value` writes against indexed TypedDict fields."""
        if not isinstance(target.value, cst.Name) or len(target.slice) != 1:
            return
        cls_info = self._resolve_receiver_class(target.value.value)
        if cls_info is None or not cls_info.is_typeddict:
            return
        first = target.slice[0].slice
        if not isinstance(first, cst.Index):
            return
        key_lit = extract_literal(first.value)
        if key_lit is None or key_lit.kind != "str" or not isinstance(key_lit.value, str):
            return
        field_name = key_lit.value
        if any(f.name == field_name for f in cls_info.fields):
            pos = self.get_metadata(PositionProvider, target)
            stats = self.index.field_stats.setdefault(
                (cls_info.qualname, field_name), ParamCallStats()
            )
            stats.observations.append(self._classify_arg_expr(value, pos.start.line))

    def _bind_typeddict_literal(self, target: ClassInfo, expr: cst.Dict, line: int) -> None:
        """Bind keys/values of a dict literal typed as `target` TypedDict onto field_stats."""
        fields = target.fields
        by_name = {f.name: f for f in fields}
        if any(not isinstance(el, cst.DictElement) for el in expr.elements):
            for f in fields:
                stats = self.index.field_stats.setdefault(
                    (target.qualname, f.name), ParamCallStats()
                )
                stats.has_dynamic_star_call = True
            return

        bound: set[str] = set()
        for el in expr.elements:
            if not isinstance(el, cst.DictElement):
                continue
            key_lit = extract_literal(el.key)
            if key_lit is None or key_lit.kind != "str" or not isinstance(key_lit.value, str):
                for f in fields:
                    stats = self.index.field_stats.setdefault(
                        (target.qualname, f.name), ParamCallStats()
                    )
                    stats.has_dynamic_star_call = True
                return
            key = key_lit.value
            if key in by_name:
                bound.add(key)
                stats = self.index.field_stats.setdefault((target.qualname, key), ParamCallStats())
                stats.observations.append(self._classify_arg_expr(el.value, line))

        for fld in fields:
            if fld.name not in bound:
                stats = self.index.field_stats.setdefault(
                    (target.qualname, fld.name), ParamCallStats()
                )
                stats.omitted_count += 1

    def _resolve_receiver_class(self, receiver: str) -> ClassInfo | None:
        """Resolve `self` or a typed local variable to its ClassInfo if indexed."""
        if receiver == "self" and self._class_stack:
            qual = f"{self.ctx.module_name}.{self._class_stack[-1]}"
            return self.index.classes.get(qual)
        binding = self._lookup_local_var(receiver)
        if binding is None or binding.class_type is None:
            return None
        return self._lookup_class_by_ref(binding.class_type)

    def _lookup_class_by_ref(self, class_ref: str) -> ClassInfo | None:
        """Resolve a class name in current module or imports to a ClassInfo."""
        for qual in (f"{self.ctx.module_name}.{class_ref}", self.ctx.imports.get(class_ref, "")):
            if qual in self.index.classes:
                return self.index.classes[qual]
        return None

    def _resolve_structured_class_target(self, func_expr: cst.BaseExpression) -> ClassInfo | None:
        """Resolve a constructor call `MyDataclass(...)` or `mod.MyDataclass(...)`."""
        candidates: list[str] = []
        if isinstance(func_expr, cst.Name):
            candidates.append(f"{self.ctx.module_name}.{func_expr.value}")
            imported = self.ctx.imports.get(func_expr.value)
            if imported:
                candidates.append(imported)
        elif isinstance(func_expr, cst.Attribute) and isinstance(func_expr.value, cst.Name):
            mod = self.ctx.module_imports.get(func_expr.value.value)
            if mod:
                candidates.append(f"{mod}.{func_expr.attr.value}")
        for qual in candidates:
            cls = self.index.classes.get(qual)
            if cls is not None and cls.is_structured_model and cls.fields:
                return cls
        return None

    def _bind_field_arguments(self, target: ClassInfo, call: cst.Call, line: int) -> None:
        """Bind constructor call arguments onto structured-model fields."""
        if (
            target.is_typeddict
            and len(call.args) == 1
            and call.args[0].keyword is None
            and call.args[0].star == ""
            and isinstance(call.args[0].value, cst.Dict)
        ):
            self._bind_typeddict_literal(target, call.args[0].value, line)
            return
        fields = target.fields
        pos_fields = [f for f in fields if not f.is_kw_only]
        by_name = {f.name: f for f in fields}
        if any(arg.star in {"*", "**"} for arg in call.args):
            for f in fields:
                stats = self.index.field_stats.setdefault(
                    (target.qualname, f.name), ParamCallStats()
                )
                stats.has_dynamic_star_call = True
            return

        bound: set[str] = set()
        pos_idx = 0
        for arg in call.args:
            if arg.keyword is not None:
                kw_name = arg.keyword.value
                if kw_name in by_name:
                    bound.add(kw_name)
                    stats = self.index.field_stats.setdefault(
                        (target.qualname, kw_name), ParamCallStats()
                    )
                    stats.observations.append(self._classify_arg_expr(arg.value, line))
            elif pos_idx < len(pos_fields):
                fld = pos_fields[pos_idx]
                pos_idx += 1
                bound.add(fld.name)
                stats = self.index.field_stats.setdefault(
                    (target.qualname, fld.name), ParamCallStats()
                )
                stats.observations.append(self._classify_arg_expr(arg.value, line))

        for fld in fields:
            if fld.name not in bound:
                stats = self.index.field_stats.setdefault(
                    (target.qualname, fld.name), ParamCallStats()
                )
                stats.omitted_count += 1

    def _record_var_value(
        self,
        var_name: str,
        value: cst.BaseExpression | None,
        annotation: cst.Annotation | None,
    ) -> None:
        """Update local scope knowledge for a variable assignment."""
        scope = self._scope_stack[-1][1]
        ann_lits = extract_literal_values_from_annotation(annotation)
        ann_str = expr_to_code(annotation.annotation) if annotation is not None else None
        ann_non_none = (
            annotation is not None
            and extract_optional_inner(annotation) is None
            and ann_str != "None"
        )
        val_lit = extract_literal(value) if value is not None else None
        val_non_none = value is not None and is_definitely_non_none_expr(value)
        val_type = infer_simple_expr_type(value) if value is not None else None

        class_type = ann_str
        if (
            class_type is None
            and value is not None
            and self._is_known_class_ctor_call(value)
            and isinstance(value, cst.Call)
            and isinstance(value.func, cst.Name)
        ):
            class_type = value.func.value
            val_type = value.func.value
            val_non_none = True

        inferred_type = ann_str or val_type
        existing = scope.get(var_name)
        if existing is not None:
            merged_lits = ann_lits or (
                (val_lit,) if val_lit is not None and existing.literals == (val_lit,) else None
            )
            merged_type = (
                inferred_type
                if (inferred_type is not None and existing.inferred_type == inferred_type)
                else None
            )
            scope[var_name] = _LocalVarBinding(
                literals=merged_lits,
                is_non_none=existing.is_non_none and (ann_non_none or val_non_none),
                class_type=class_type or existing.class_type,
                inferred_type=merged_type,
            )
            return

        lits = ann_lits or ((val_lit,) if val_lit is not None else None)
        scope[var_name] = _LocalVarBinding(
            literals=lits,
            is_non_none=ann_non_none or val_non_none,
            class_type=class_type,
            inferred_type=inferred_type,
        )

    def _resolve_call_target(
        self, func_expr: cst.BaseExpression
    ) -> tuple[FunctionInfo | None, bool]:
        """Resolve a call target expression to `(FunctionInfo, is_bound_method_call)`."""
        if isinstance(func_expr, cst.Name):
            return self._resolve_name_call(func_expr.value)
        if isinstance(func_expr, cst.Attribute):
            return self._resolve_attr_call(func_expr)
        return (None, False)

    def _resolve_name_call(self, name: str) -> tuple[FunctionInfo | None, bool]:
        """Resolve a bare `name(...)` call to a function or class `__init__`."""
        candidates = [
            f"{self.ctx.module_name}.{name}",
            self.ctx.imports.get(name, ""),
        ]
        for qual in candidates:
            if qual in self.index.functions:
                return (self.index.functions[qual], False)
            init_qual = f"{qual}.__init__"
            if init_qual in self.index.functions:
                return (self.index.functions[init_qual], True)
        return (None, False)

    def _resolve_attr_call(self, attr: cst.Attribute) -> tuple[FunctionInfo | None, bool]:
        """Resolve `obj.method(...)`, `self.method(...)`, or `mod.func(...)`."""
        method_name = attr.attr.value
        if isinstance(attr.value, cst.Name):
            receiver = attr.value.value
            # 1. self.method(...) or cls.method(...)
            if receiver in {"self", "cls"} and self._class_stack:
                cls_name = self._class_stack[-1]
                fn = self._lookup_class_method(self.ctx.module_name, cls_name, method_name)
                if fn is not None:
                    return (fn, not fn.is_staticmethod)
            # 2. Imported module call: mod.func(...)
            if receiver in self.ctx.module_imports:
                mod_qual = f"{self.ctx.module_imports[receiver]}.{method_name}"
                if mod_qual in self.index.functions:
                    return (self.index.functions[mod_qual], False)
                init_qual = f"{mod_qual}.__init__"
                if init_qual in self.index.functions:
                    return (self.index.functions[init_qual], True)
            # 3. Direct class call: MyClass.method(...)
            fn = self._lookup_class_method(self.ctx.module_name, receiver, method_name)
            if fn is not None:
                return (fn, fn.is_classmethod)
            # 4. Typed local variable receiver: obj.method(...)
            binding = self._lookup_local_var(receiver)
            if binding is not None and binding.class_type is not None:
                fn = self._lookup_class_method(
                    self.ctx.module_name, binding.class_type, method_name
                )
                if fn is not None:
                    return (fn, not fn.is_staticmethod)
        return (None, False)

    def _lookup_class_method(
        self,
        module_name: str,
        class_ref: str,
        method_name: str,
    ) -> FunctionInfo | None:
        """Look up `class_ref.method_name` directly or on indexed ancestor classes."""
        cls_candidates = [f"{module_name}.{class_ref}"]
        imported = self.ctx.imports.get(class_ref)
        if imported is not None:
            cls_candidates.append(imported)

        for cls_qual in cls_candidates:
            direct = self.index.functions.get(f"{cls_qual}.{method_name}")
            if direct is not None:
                return direct

        for cls_qual in cls_candidates:
            cls_info = self.index.classes.get(cls_qual)
            if cls_info is None:
                continue
            for ancestor in _collect_indexed_ancestors(
                cls_info, self.index.classes, self._classes_by_short_name
            ):
                inherited = self.index.functions.get(f"{ancestor.qualname}.{method_name}")
                if inherited is not None:
                    return inherited
        return None

    def _lookup_local_var(self, name: str) -> _LocalVarBinding | None:
        """Look up a variable binding from innermost scope outward, skipping enclosing class scopes."""
        if name in self._scope_stack[-1][1]:
            return self._scope_stack[-1][1][name]
        for is_class_scope, scope in reversed(self._scope_stack[:-1]):
            if not is_class_scope and name in scope:
                return scope[name]
        return None

    def _bind_call_arguments(
        self,
        target: FunctionInfo,
        call: cst.Call,
        line: int,
        *,
        is_bound: bool,
    ) -> None:
        """Bind positional and keyword call arguments onto target function parameters."""
        params = target.parameters
        if is_bound and target.is_method and not target.is_staticmethod and params:
            params = params[1:]  # Skip self / cls

        pos_params = [p for p in params if p.kind in {"posonly", "regular"}]
        by_name = {p.name: p for p in params}
        has_star = any(arg.star in {"*", "**"} for arg in call.args)
        if has_star:
            for p in params:
                stats = self.index.call_stats.setdefault(
                    (target.qualname, p.name), ParamCallStats()
                )
                stats.has_dynamic_star_call = True
            return

        bound_params: set[str] = set()
        pos_idx = 0
        for arg in call.args:
            if arg.keyword is not None:
                kw_name = arg.keyword.value
                if kw_name in by_name:
                    bound_params.add(kw_name)
                    self._record_arg_observation(target.qualname, kw_name, arg.value, line)
            elif pos_idx < len(pos_params):
                param = pos_params[pos_idx]
                pos_idx += 1
                bound_params.add(param.name)
                self._record_arg_observation(target.qualname, param.name, arg.value, line)

        for param in params:
            if param.name not in bound_params:
                stats = self.index.call_stats.setdefault(
                    (target.qualname, param.name), ParamCallStats()
                )
                stats.omitted_count += 1

    def _record_arg_observation(
        self,
        func_qualname: str,
        param_name: str,
        value_expr: cst.BaseExpression,
        line: int,
    ) -> None:
        """Classify one argument expression and append it to ParamCallStats."""
        stats = self.index.call_stats.setdefault((func_qualname, param_name), ParamCallStats())
        obs = self._classify_arg_expr(value_expr, line)
        stats.observations.append(obs)

    def _classify_arg_expr(self, expr: cst.BaseExpression, line: int) -> ObservedArg:
        """Classify a call-site argument expression as None, literal(s), non-None, or unknown."""
        if is_none_expr(expr):
            return ObservedArg(
                file=self.ctx.file,
                line=line,
                is_test_caller=self.ctx.is_test_file,
                is_none=True,
                is_non_none=False,
                observed_type="None",
            )

        lit = extract_literal(expr)
        if lit is not None:
            return ObservedArg(
                file=self.ctx.file,
                line=line,
                is_test_caller=self.ctx.is_test_file,
                is_non_none=True,
                literals=(lit,),
                observed_type=lit.kind,
            )

        if isinstance(expr, cst.Name):
            return self._classify_name_arg(expr.value, line)

        if isinstance(expr, cst.Attribute) and isinstance(expr.value, cst.Name):
            enum_cls = self._resolve_in_scope_enum(expr.value.value)
            if enum_cls is not None and expr.attr.value in enum_cls.enum_members:
                return ObservedArg(
                    file=self.ctx.file,
                    line=line,
                    is_test_caller=self.ctx.is_test_file,
                    is_non_none=True,
                    is_unknown=True,
                    observed_type=enum_cls.name,
                    enum_qualname=enum_cls.qualname,
                )

        dict_keys = self._extract_const_str_dict_arg(expr) if isinstance(expr, cst.Dict) else None

        inferred = infer_simple_expr_type(expr)
        if (
            inferred is None
            and self._is_known_class_ctor_call(expr)
            and isinstance(expr, cst.Call)
            and isinstance(expr.func, cst.Name)
        ):
            inferred = expr.func.value

        if is_definitely_non_none_expr(expr) or self._is_known_class_ctor_call(expr):
            return ObservedArg(
                file=self.ctx.file,
                line=line,
                is_test_caller=self.ctx.is_test_file,
                is_non_none=True,
                is_unknown=True,
                observed_type=inferred,
                dict_literal_keys=dict_keys,
            )

        return ObservedArg(
            file=self.ctx.file,
            line=line,
            is_test_caller=self.ctx.is_test_file,
            is_unknown=True,
            observed_type=inferred,
        )

    def _classify_name_arg(self, name: str, line: int) -> ObservedArg:
        """Classify a bare variable name argument using local scope bindings."""
        binding = self._lookup_local_var(name)
        if binding is not None:
            if binding.literals is not None:
                obs_type = binding.inferred_type or binding.literals[0].kind
                return ObservedArg(
                    file=self.ctx.file,
                    line=line,
                    is_test_caller=self.ctx.is_test_file,
                    is_non_none=True,
                    literals=binding.literals,
                    observed_type=obs_type,
                )
            if binding.is_non_none:
                enum_cls = (
                    self._resolve_in_scope_enum(binding.inferred_type)
                    if binding.inferred_type is not None
                    else None
                )
                return ObservedArg(
                    file=self.ctx.file,
                    line=line,
                    is_test_caller=self.ctx.is_test_file,
                    is_non_none=True,
                    is_unknown=True,
                    observed_type=binding.inferred_type,
                    enum_qualname=enum_cls.qualname if enum_cls is not None else None,
                )
            if binding.inferred_type is not None:
                return ObservedArg(
                    file=self.ctx.file,
                    line=line,
                    is_test_caller=self.ctx.is_test_file,
                    is_unknown=True,
                    observed_type=binding.inferred_type,
                )
        return ObservedArg(
            file=self.ctx.file,
            line=line,
            is_test_caller=self.ctx.is_test_file,
            is_unknown=True,
        )

    def _extract_const_str_dict_arg(
        self,
        expr: cst.Dict,
    ) -> tuple[tuple[str, str | None], ...] | None:
        """Extract `(key, inferred_val_type)` pairs if expr is a constant-string-key dict literal."""
        entries: list[tuple[str, str | None]] = []
        for elem in expr.elements:
            if not isinstance(elem, cst.DictElement):
                return None
            key_lit = extract_literal(elem.key)
            if key_lit is None or key_lit.kind != "str" or not isinstance(key_lit.value, str):
                return None
            val_type = self._infer_call_expr_type(elem.value)
            entries.append((key_lit.value, val_type))
        return tuple(entries)

    def _infer_call_expr_type(self, expr: cst.BaseExpression) -> str | None:
        """Infer a conservative static type string for an argument or dict-value expression."""
        simple = infer_simple_expr_type(expr)
        if simple is not None:
            return simple
        if isinstance(expr, cst.Name):
            binding = self._lookup_local_var(expr.value)
            if binding is not None and binding.inferred_type is not None:
                return binding.inferred_type
        if (
            self._is_known_class_ctor_call(expr)
            and isinstance(expr, cst.Call)
            and isinstance(expr.func, cst.Name)
        ):
            return expr.func.value
        return None

    def _resolve_in_scope_enum(self, cls_name: str) -> ClassInfo | None:
        """Resolve a class name in the caller's module to an indexed StrEnum/IntEnum ClassInfo."""
        same_mod = self.index.classes.get(f"{self.ctx.module_name}.{cls_name}")
        if same_mod is not None and same_mod.enum_kind is not None:
            return same_mod
        imported_qual = self.ctx.imports.get(cls_name)
        if imported_qual is not None:
            imported_cls = self.index.classes.get(imported_qual)
            if imported_cls is not None and imported_cls.enum_kind is not None:
                return imported_cls
        return None

    def _is_known_class_ctor_call(self, expr: cst.BaseExpression) -> bool:
        """Return True if expr is a direct call to an indexed class constructor."""
        if not isinstance(expr, cst.Call) or not isinstance(expr.func, cst.Name):
            return False
        cls_candidates = {
            f"{self.ctx.module_name}.{expr.func.value}",
            self.ctx.imports.get(expr.func.value, ""),
        }
        return any(c in self.index.classes for c in cls_candidates)


def _collect_indexed_ancestors(
    cls_info: ClassInfo,
    all_classes: dict[str, ClassInfo],
    by_short_name: dict[str, list[ClassInfo]],
) -> list[ClassInfo]:
    """Return all distinct indexed ancestor ClassInfo objects of `cls_info`."""
    ancestors: list[ClassInfo] = []
    seen_quals: set[str] = {cls_info.qualname}
    queue = list(cls_info.bases)
    while queue:
        base_name = _strip_base_type_args(queue.pop(0))
        if base_name in TRIVIAL_BASE_NAMES:
            continue
        for base_cls in _lookup_base_classes(
            base_name, cls_info.module_name, all_classes, by_short_name
        ):
            if base_cls.qualname not in seen_quals:
                seen_quals.add(base_cls.qualname)
                ancestors.append(base_cls)
                queue.extend(base_cls.bases)
    return ancestors


def _strip_base_type_args(base_name: str) -> str:
    """Strip generic arguments from a base class expression (e.g. `IO[str]` -> `IO`)."""
    bracket = base_name.find("[")
    return base_name[:bracket].strip() if bracket != -1 else base_name.strip()


def _lookup_base_classes(
    base_name: str,
    current_module: str,
    all_classes: dict[str, ClassInfo],
    by_short_name: dict[str, list[ClassInfo]],
) -> list[ClassInfo]:
    """Resolve a base class name to indexed ClassInfo objects."""
    if base_name in all_classes:
        return [all_classes[base_name]]
    same_mod = f"{current_module}.{base_name}"
    if same_mod in all_classes:
        return [all_classes[same_mod]]
    short = base_name.rsplit(".", maxsplit=1)[-1]
    return by_short_name.get(short, [])


def _collect_module_calls(
    ctx: _ModuleContext,
    index: CodebaseIndex,
    classes_by_short_name: dict[str, list[ClassInfo]],
) -> None:
    """Visit a module to resolve call sites and aggregate per-parameter argument observations."""
    collector = _CallSiteCollector(
        ctx=ctx, index=index, classes_by_short_name=classes_by_short_name
    )
    ctx.wrapper.visit(collector)
