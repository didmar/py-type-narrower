"""Intra-procedural LibCST analysis of function bodies (returns and parameter guards)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal as TypingLiteral

import libcst as cst

from py_type_narrower.cst_utils import (
    ExtractedLiteral,
    extract_literal,
    extract_optional_inner,
    is_definitely_non_none_expr,
    is_none_expr,
)


@dataclass
class ReturnSummary:
    """Summary of all return statements and control flow in a function body."""

    is_stub: bool = False
    is_generator: bool = False
    has_returns: bool = False
    can_fallthrough: bool = True
    returns_none: bool = False
    all_returns_non_none: bool = True
    has_non_literal_return: bool = False
    returned_literals: set[ExtractedLiteral] = field(default_factory=set)


@dataclass
class ParamGuardInfo:
    """Strict and soft literal constraints discovered for a parameter inside a function body."""

    strict_literals: set[ExtractedLiteral] = field(default_factory=set)
    strict_evidence: list[str] = field(default_factory=list)
    soft_literals: set[ExtractedLiteral] = field(default_factory=set)


@dataclass
class BodyFacts:
    """Combined intra-procedural facts for a single function."""

    returns: ReturnSummary
    param_guards: dict[str, ParamGuardInfo] = field(default_factory=dict)


def analyze_function_body(
    func: cst.FunctionDef,
    param_non_none_names: set[str],
    module_const_dicts: dict[str, set[ExtractedLiteral]],
    known_class_names: set[str],
) -> BodyFacts:
    """Analyze return paths and parameter literal guards for a function definition."""
    stmts = _extract_body_statements(func.body)
    returns = _analyze_returns(stmts, param_non_none_names, known_class_names)
    param_names = _collect_param_names(func.params)
    guards = _analyze_param_guards(stmts, param_names, module_const_dicts)
    return BodyFacts(returns=returns, param_guards=guards)


def extract_module_const_dicts(module: cst.Module) -> dict[str, set[ExtractedLiteral]]:
    """Extract top-level constant dicts whose keys are all scalar literals."""
    const_dicts: dict[str, set[ExtractedLiteral]] = {}
    for stmt in module.body:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        for small in stmt.body:
            target_name, value_expr = _extract_simple_assignment(small)
            if target_name is None or not isinstance(value_expr, cst.Dict):
                continue
            keys = _extract_dict_literal_keys(value_expr)
            if keys is not None:
                const_dicts[target_name] = keys
    return const_dicts


def _collect_param_names(params: cst.Parameters) -> set[str]:
    """Return all parameter names defined on a function signature."""
    names: set[str] = set()
    for p in (*params.posonly_params, *params.params, *params.kwonly_params):
        names.add(p.name.value)
    return names


def _stmt_always_terminates(stmt: cst.BaseStatement) -> bool:
    """Return True if a single statement unconditionally returns or raises."""
    if isinstance(stmt, cst.SimpleStatementLine):
        return any(isinstance(small, (cst.Return, cst.Raise)) for small in stmt.body)
    if isinstance(stmt, cst.If):
        return _if_always_terminates(stmt)
    if isinstance(stmt, cst.With):
        return _block_always_terminates(_extract_body_statements(stmt.body))
    if isinstance(stmt, (cst.Try, cst.TryStar)):
        return _try_always_terminates(stmt)
    if isinstance(stmt, cst.Match):
        return _match_always_terminates(stmt)
    return False


def _if_always_terminates(stmt: cst.If) -> bool:
    """Return True if an if/elif/else chain unconditionally terminates."""
    if stmt.orelse is None:
        return False
    if not _block_always_terminates(_extract_body_statements(stmt.body)):
        return False
    if isinstance(stmt.orelse, cst.Else):
        return _block_always_terminates(_extract_body_statements(stmt.orelse.body))
    if isinstance(stmt.orelse, cst.If):
        return _if_always_terminates(stmt.orelse)
    return False


def _try_always_terminates(stmt: cst.Try | cst.TryStar) -> bool:
    """Return True if a try/except/else/finally block unconditionally terminates."""
    if stmt.finalbody is not None and _block_always_terminates(
        _extract_body_statements(stmt.finalbody.body)
    ):
        return True
    try_body_terminates = _block_always_terminates(_extract_body_statements(stmt.body))
    else_terminates = stmt.orelse is not None and _block_always_terminates(
        _extract_body_statements(stmt.orelse.body)
    )
    if not (try_body_terminates or else_terminates):
        return False
    return all(
        _block_always_terminates(_extract_body_statements(handler.body))
        for handler in stmt.handlers
    )


def _match_always_terminates(stmt: cst.Match) -> bool:
    """Return True if a match statement has a catch-all case and every case terminates."""
    has_wildcard = False
    for case in stmt.cases:
        if case.guard is None and _is_irrefutable_pattern(case.pattern):
            has_wildcard = True
        if not _block_always_terminates(_extract_body_statements(case.body)):
            return False
    return has_wildcard


def _analyze_param_guards(
    stmts: list[cst.BaseStatement],
    param_names: set[str],
    module_const_dicts: dict[str, set[ExtractedLiteral]],
) -> dict[str, ParamGuardInfo]:
    """Detect strict and soft literal guards on parameters in a function body."""
    guards: dict[str, ParamGuardInfo] = {name: ParamGuardInfo() for name in param_names}
    _scan_statements_for_guards(stmts, param_names, guards)
    _scan_const_dict_subscripts(stmts, param_names, module_const_dicts, guards)
    _scan_soft_comparisons(stmts, param_names, guards)
    return guards


def _scan_statements_for_guards(
    stmts: list[cst.BaseStatement],
    param_names: set[str],
    guards: dict[str, ParamGuardInfo],
) -> None:
    """Detect if/elif/else-raise, sequential if-return + raise, negative guards, and match."""
    _scan_sequential_if_returns(stmts, param_names, guards)
    for idx, stmt in enumerate(stmts):
        if isinstance(stmt, cst.If):
            _check_if_chain_guard(stmt, param_names, guards)
            _check_negative_guard(stmt, param_names, guards)
        elif isinstance(stmt, cst.Match):
            next_raises = idx + 1 < len(stmts) and _stmt_always_raises(stmts[idx + 1])
            _check_match_guard(stmt, param_names, guards, followed_by_raise=next_raises)


def _scan_sequential_if_returns(
    stmts: list[cst.BaseStatement],
    param_names: set[str],
    guards: dict[str, ParamGuardInfo],
) -> None:
    """Detect sequences like `if p == 'a': return ...; if p == 'b': return ...; raise ...`."""
    collected: dict[str, set[ExtractedLiteral]] = {}
    for stmt in stmts:
        if isinstance(stmt, cst.If) and stmt.orelse is None:
            match = _extract_positive_param_literals(stmt.test, param_names)
            if match is not None and _block_always_terminates(_extract_body_statements(stmt.body)):
                param, lits = match
                collected.setdefault(param, set()).update(lits)
                continue
        if _stmt_always_raises(stmt) and collected:
            for param, lits in collected.items():
                if lits:
                    _record_strict_guard(
                        guards[param], lits, "exhaustively checked by sequential if-guards + raise"
                    )
            collected.clear()
            continue
        collected.clear()


def _check_if_chain_guard(
    stmt: cst.If,
    param_names: set[str],
    guards: dict[str, ParamGuardInfo],
) -> None:
    """Check whether an if/elif/.../else chain constrains a single param and raises in else."""
    target_param: str | None = None
    lits: set[ExtractedLiteral] = set()
    curr: cst.If | cst.Else | None = stmt

    while isinstance(curr, cst.If):
        extracted = _extract_positive_param_literals(curr.test, param_names)
        if extracted is None:
            return
        param, branch_lits = extracted
        if target_param is None:
            target_param = param
        elif target_param != param:
            return
        lits.update(branch_lits)
        curr = curr.orelse

    if target_param is None or not lits or not isinstance(curr, cst.Else):
        return
    if _block_always_raises(_extract_body_statements(curr.body)):
        _record_strict_guard(guards[target_param], lits, "guarded by if/elif chain with else raise")


def _check_negative_guard(
    stmt: cst.If,
    param_names: set[str],
    guards: dict[str, ParamGuardInfo],
) -> None:
    """Check early guard like `if p not in ('a', 'b'): raise ValueError(...)`."""
    if not _block_always_raises(_extract_body_statements(stmt.body)):
        return
    extracted = _extract_negative_param_literals(stmt.test, param_names)
    if extracted is None:
        return
    param, lits = extracted
    if lits:
        _record_strict_guard(guards[param], lits, "guarded by negative check + raise")


def _check_match_guard(
    stmt: cst.Match,
    param_names: set[str],
    guards: dict[str, ParamGuardInfo],
    *,
    followed_by_raise: bool,
) -> None:
    """Check `match param:` with literal cases and raising wildcard (or trailing raise)."""
    if not isinstance(stmt.subject, cst.Name) or stmt.subject.value not in param_names:
        return
    param = stmt.subject.value
    lits: set[ExtractedLiteral] = set()
    has_raising_wildcard = False
    all_literal_cases_terminate = True

    for case in stmt.cases:
        if case.guard is None and _is_irrefutable_pattern(case.pattern):
            if _block_always_raises(_extract_body_statements(case.body)):
                has_raising_wildcard = True
                continue
            return
        case_lits = _extract_match_pattern_literals(case.pattern)
        if case_lits is None:
            return
        lits.update(case_lits)
        if not _block_always_terminates(_extract_body_statements(case.body)):
            all_literal_cases_terminate = False

    if lits and (has_raising_wildcard or (followed_by_raise and all_literal_cases_terminate)):
        _record_strict_guard(guards[param], lits, "guarded by exhaustive match statement")


def _extract_body_statements(body: cst.BaseSuite) -> list[cst.BaseStatement]:
    """Flatten a function suite into a list of top-level statements."""
    if isinstance(body, cst.IndentedBlock):
        return list(body.body)
    if isinstance(body, cst.SimpleStatementSuite):
        return [cst.SimpleStatementLine(body=list(body.body))]
    return []


def _is_irrefutable_pattern(pattern: cst.MatchPattern) -> bool:
    """Return True if a match pattern matches any value (e.g., `case _:` or `case x:`)."""
    if isinstance(pattern, cst.MatchAs) and pattern.pattern is None:
        return True
    if isinstance(pattern, cst.MatchOr):
        return any(_is_irrefutable_pattern(opt.pattern) for opt in pattern.patterns)
    return False


def _block_always_raises(stmts: list[cst.BaseStatement]) -> bool:
    """Return True if a block unconditionally raises an exception or calls assert_never."""
    return any(_stmt_always_raises(stmt) for stmt in stmts)


def _stmt_always_raises(stmt: cst.BaseStatement) -> bool:
    """Return True if a statement is a raise or a call to assert_never(...)."""
    if not isinstance(stmt, cst.SimpleStatementLine):
        return False
    for small in stmt.body:
        if isinstance(small, cst.Raise):
            return True
        if isinstance(small, cst.Expr) and _is_assert_never_call(small.value):
            return True
    return False


def _is_assert_never_call(expr: cst.BaseExpression) -> bool:
    """Return True if expression is a call to `assert_never(...)` or `typing.assert_never(...)`."""
    if not isinstance(expr, cst.Call):
        return False
    if isinstance(expr.func, cst.Name):
        return expr.func.value == "assert_never"
    if isinstance(expr.func, cst.Attribute):
        return expr.func.attr.value == "assert_never"
    return False


def _extract_simple_assignment(
    stmt: cst.BaseSmallStatement,
) -> tuple[str | None, cst.BaseExpression | None]:
    """Extract `(name, value)` from a simple `NAME = expr` or `NAME: T = expr`."""
    if isinstance(stmt, cst.Assign) and len(stmt.targets) == 1:
        target = stmt.targets[0].target
        if isinstance(target, cst.Name):
            return (target.value, stmt.value)
    if (
        isinstance(stmt, cst.AnnAssign)
        and isinstance(stmt.target, cst.Name)
        and stmt.value is not None
    ):
        return (stmt.target.value, stmt.value)
    return (None, None)


def filter_literals_by_kind(
    lits: set[ExtractedLiteral],
    kind: TypingLiteral["str", "int"],
) -> set[ExtractedLiteral]:
    """Keep only literals matching the target primitive kind ('str' or 'int')."""
    return {lit for lit in lits if lit.kind == kind}


class _ReturnCollector(cst.CSTVisitor):
    """Collect return statements and yield expressions without descending into nested defs."""

    def __init__(self, local_non_none: set[str], known_class_names: set[str]) -> None:
        self.local_non_none = local_non_none
        self.known_class_names = known_class_names
        self.is_generator = False
        self.has_returns = False
        self.returns_none = False
        self.all_returns_non_none = True
        self.has_non_literal_return = False
        self.returned_literals: set[ExtractedLiteral] = set()

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:  # noqa: ARG002, N802
        return False

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:  # noqa: ARG002, N802
        return False

    def visit_Yield(self, node: cst.Yield) -> bool:  # noqa: ARG002, N802
        self.is_generator = True
        return False

    def visit_Return(self, node: cst.Return) -> bool:  # noqa: N802
        self.has_returns = True
        if node.value is None or is_none_expr(node.value):
            self.returns_none = True
            self.all_returns_non_none = False
            self.has_non_literal_return = True
            return False

        if not _expr_is_non_none_in_scope(node.value, self.local_non_none, self.known_class_names):
            self.all_returns_non_none = False

        lit = extract_literal(node.value)
        if lit is not None:
            self.returned_literals.add(lit)
        else:
            self.has_non_literal_return = True
        return False


def _analyze_returns(
    stmts: list[cst.BaseStatement],
    param_non_none_names: set[str],
    known_class_names: set[str],
) -> ReturnSummary:
    """Inspect all return statements and control flow inside a function body."""
    summary = ReturnSummary()
    if _is_stub_body(stmts):
        summary.is_stub = True
        return summary

    local_non_none = _collect_local_non_none_names(stmts, param_non_none_names, known_class_names)
    collector = _ReturnCollector(local_non_none=local_non_none, known_class_names=known_class_names)
    for stmt in stmts:
        stmt.visit(collector)

    summary.is_generator = collector.is_generator
    summary.has_returns = collector.has_returns
    summary.returns_none = collector.returns_none
    summary.all_returns_non_none = collector.all_returns_non_none
    summary.has_non_literal_return = collector.has_non_literal_return
    summary.returned_literals = collector.returned_literals
    summary.can_fallthrough = not _block_always_terminates(stmts)
    return summary


def _is_stub_body(stmts: list[cst.BaseStatement]) -> bool:
    """Return True if the function body is only a docstring, pass, ..., or NotImplementedError."""
    meaningful: list[cst.BaseStatement] = []
    for idx, stmt in enumerate(stmts):
        if idx == 0 and _is_docstring_stmt(stmt):
            continue
        meaningful.append(stmt)
    if not meaningful:
        return True
    if len(meaningful) == 1:
        only = meaningful[0]
        if isinstance(only, cst.SimpleStatementLine) and len(only.body) == 1:
            inner = only.body[0]
            if isinstance(inner, cst.Pass):
                return True
            if isinstance(inner, cst.Expr) and isinstance(inner.value, cst.Ellipsis):
                return True
            if isinstance(inner, cst.Raise) and _is_not_implemented_raise(inner):
                return True
    return False


def _is_docstring_stmt(stmt: cst.BaseStatement) -> bool:
    """Return True if a statement is a standalone string literal expression."""
    if not isinstance(stmt, cst.SimpleStatementLine) or len(stmt.body) != 1:
        return False
    inner = stmt.body[0]
    return isinstance(inner, cst.Expr) and isinstance(
        inner.value, (cst.SimpleString, cst.ConcatenatedString)
    )


def _is_not_implemented_raise(stmt: cst.Raise) -> bool:
    """Return True if a raise statement raises NotImplementedError."""
    exc = stmt.exc
    if isinstance(exc, cst.Name) and exc.value == "NotImplementedError":
        return True
    if isinstance(exc, cst.Call) and isinstance(exc.func, cst.Name):
        return exc.func.value == "NotImplementedError"
    return False


def _block_always_terminates(stmts: list[cst.BaseStatement]) -> bool:
    """Return True if executing stmts cannot fall through normally."""
    return any(_stmt_always_terminates(stmt) for stmt in stmts)


class _LocalAssignmentVisitor(cst.CSTVisitor):
    """Track simple local variable assignments to identify variables that are never None."""

    def __init__(
        self,
        known_class_names: set[str],
        assigned_non_none: set[str],
        possibly_none: set[str],
    ) -> None:
        self.known_class_names = known_class_names
        self.assigned_non_none = assigned_non_none
        self.possibly_none = possibly_none

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:  # noqa: ARG002, N802
        return False

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:  # noqa: ARG002, N802
        return False

    def visit_Assign(self, node: cst.Assign) -> bool:  # noqa: N802
        for assign_target in node.targets:
            if isinstance(assign_target.target, cst.Name):
                name = assign_target.target.value
                if _expr_is_non_none_in_scope(
                    node.value, self.assigned_non_none, self.known_class_names
                ):
                    self.assigned_non_none.add(name)
                else:
                    self.possibly_none.add(name)
        return True

    def visit_AnnAssign(self, node: cst.AnnAssign) -> bool:  # noqa: N802
        if not isinstance(node.target, cst.Name):
            return True
        name = node.target.value
        ann_allows_none = extract_optional_inner(node.annotation) is not None
        if node.value is not None and not ann_allows_none:
            if _expr_is_non_none_in_scope(
                node.value, self.assigned_non_none, self.known_class_names
            ):
                self.assigned_non_none.add(name)
            else:
                self.possibly_none.add(name)
        elif ann_allows_none:
            self.possibly_none.add(name)
        return True


def _collect_local_non_none_names(
    stmts: list[cst.BaseStatement],
    param_non_none_names: set[str],
    known_class_names: set[str],
) -> set[str]:
    """Conservatively collect local names that are never assigned None."""
    assigned_non_none: set[str] = set(param_non_none_names)
    possibly_none: set[str] = set()
    visitor = _LocalAssignmentVisitor(
        known_class_names=known_class_names,
        assigned_non_none=assigned_non_none,
        possibly_none=possibly_none,
    )
    for stmt in stmts:
        stmt.visit(visitor)
    return assigned_non_none - possibly_none


def _expr_is_non_none_in_scope(
    expr: cst.BaseExpression,
    local_non_none: set[str],
    known_class_names: set[str],
) -> bool:
    """Return True if an expression is provably non-None given local scope facts."""
    if is_definitely_non_none_expr(expr):
        return True
    if isinstance(expr, cst.Name) and expr.value in local_non_none:
        return True
    if isinstance(expr, cst.Call):
        return _is_non_none_call(expr, known_class_names)
    if isinstance(expr, cst.IfExp):
        return _expr_is_non_none_in_scope(
            expr.body, local_non_none, known_class_names
        ) and _expr_is_non_none_in_scope(expr.orelse, local_non_none, known_class_names)
    return False


def _is_non_none_call(call: cst.Call, known_class_names: set[str]) -> bool:
    """Return True if a call is a known constructor or builtin that never returns None."""
    non_none_builtins = {
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
        "len",
        "range",
        "sorted",
        "repr",
        "type",
        "Path",
    }
    if isinstance(call.func, cst.Name):
        return call.func.value in non_none_builtins or call.func.value in known_class_names
    return False


class _ConstDictSubscriptVisitor(cst.CSTVisitor):
    """Detect `{'a': 1, 'b': 2}[param]` and `CONST_DICT[param]` lookups inside a function."""

    def __init__(
        self,
        param_names: set[str],
        module_const_dicts: dict[str, set[ExtractedLiteral]],
        guards: dict[str, ParamGuardInfo],
    ) -> None:
        self.param_names = param_names
        self.module_const_dicts = module_const_dicts
        self.guards = guards

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:  # noqa: ARG002, N802
        return False

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:  # noqa: ARG002, N802
        return False

    def visit_Subscript(self, node: cst.Subscript) -> bool:  # noqa: N802
        if len(node.slice) != 1 or not isinstance(node.slice[0].slice, cst.Index):
            return True
        index_expr = node.slice[0].slice.value
        if not isinstance(index_expr, cst.Name) or index_expr.value not in self.param_names:
            return True

        param = index_expr.value
        if isinstance(node.value, cst.Dict):
            keys = _extract_dict_literal_keys(node.value)
            if keys:
                _record_strict_guard(
                    self.guards[param], keys, "used as subscript key into constant dict literal"
                )
        elif isinstance(node.value, cst.Name) and node.value.value in self.module_const_dicts:
            keys = self.module_const_dicts[node.value.value]
            _record_strict_guard(
                self.guards[param],
                keys,
                f"used as subscript key into module constant dict `{node.value.value}`",
            )
        return True


def _scan_const_dict_subscripts(
    stmts: list[cst.BaseStatement],
    param_names: set[str],
    module_const_dicts: dict[str, set[ExtractedLiteral]],
    guards: dict[str, ParamGuardInfo],
) -> None:
    """Detect `{'a': ..., 'b': ...}[param]` or `CONST_DICT[param]` subscripts."""
    visitor = _ConstDictSubscriptVisitor(
        param_names=param_names,
        module_const_dicts=module_const_dicts,
        guards=guards,
    )
    for stmt in stmts:
        stmt.visit(visitor)


def _record_strict_guard(
    info: ParamGuardInfo,
    lits: set[ExtractedLiteral],
    reason: str,
) -> None:
    """Intersect or initialize strict literal constraints on a parameter."""
    if not info.strict_literals:
        info.strict_literals = set(lits)
    else:
        info.strict_literals &= lits
    info.strict_evidence.append(reason)


def _extract_dict_literal_keys(node: cst.Dict) -> set[ExtractedLiteral] | None:
    """Return all scalar literal keys of a dict literal, or None if any key is non-literal."""
    keys: set[ExtractedLiteral] = set()
    for elem in node.elements:
        if not isinstance(elem, cst.DictElement):
            return None
        lit = extract_literal(elem.key)
        if lit is None:
            return None
        keys.add(lit)
    return keys or None


class _SoftComparisonVisitor(cst.CSTVisitor):
    """Collect any literal comparisons against parameters as soft corroborating evidence."""

    def __init__(self, param_names: set[str], guards: dict[str, ParamGuardInfo]) -> None:
        self.param_names = param_names
        self.guards = guards

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:  # noqa: ARG002, N802
        return False

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:  # noqa: ARG002, N802
        return False

    def visit_Comparison(self, node: cst.Comparison) -> bool:  # noqa: N802
        pos = _extract_positive_param_literals(node, self.param_names)
        if pos is not None:
            self.guards[pos[0]].soft_literals.update(pos[1])
        neg = _extract_negative_param_literals(node, self.param_names)
        if neg is not None:
            self.guards[neg[0]].soft_literals.update(neg[1])
        return True

    def visit_Match(self, node: cst.Match) -> bool:  # noqa: N802
        if isinstance(node.subject, cst.Name) and node.subject.value in self.param_names:
            for case in node.cases:
                lits = _extract_match_pattern_literals(case.pattern)
                if lits:
                    self.guards[node.subject.value].soft_literals.update(lits)
        return True


def _scan_soft_comparisons(
    stmts: list[cst.BaseStatement],
    param_names: set[str],
    guards: dict[str, ParamGuardInfo],
) -> None:
    """Collect any non-exhaustive literal comparisons against parameters as soft evidence."""
    visitor = _SoftComparisonVisitor(param_names=param_names, guards=guards)
    for stmt in stmts:
        stmt.visit(visitor)


def _extract_negative_param_literals(
    expr: cst.BaseExpression,
    param_names: set[str],
) -> tuple[str, set[ExtractedLiteral]] | None:
    """Extract `(param, literals)` from `p not in (...)`, `p != lit`, or `and`-chains."""
    if isinstance(expr, cst.UnaryOperation) and isinstance(expr.operator, cst.Not):
        return _extract_positive_param_literals(expr.expression, param_names)
    if isinstance(expr, cst.BooleanOperation) and isinstance(expr.operator, cst.And):
        left = _extract_negative_param_literals(expr.left, param_names)
        right = _extract_negative_param_literals(expr.right, param_names)
        if left is not None and right is not None and left[0] == right[0]:
            return (left[0], left[1] | right[1])
        return None
    if isinstance(expr, cst.Comparison) and len(expr.comparisons) == 1:
        target = expr.comparisons[0]
        if isinstance(target.operator, cst.NotEqual):
            return _match_equality(expr.left, target.comparator, param_names)
        if isinstance(target.operator, cst.NotIn):
            return _match_membership(expr.left, target.comparator, param_names)
    return None


def _extract_positive_param_literals(
    expr: cst.BaseExpression,
    param_names: set[str],
) -> tuple[str, set[ExtractedLiteral]] | None:
    """Extract `(param, literals)` from `p == lit`, `p in (lit1, lit2)`, or `or`-chains."""
    if isinstance(expr, cst.BooleanOperation) and isinstance(expr.operator, cst.Or):
        left = _extract_positive_param_literals(expr.left, param_names)
        right = _extract_positive_param_literals(expr.right, param_names)
        if left is not None and right is not None and left[0] == right[0]:
            return (left[0], left[1] | right[1])
        return None
    if isinstance(expr, cst.Comparison) and len(expr.comparisons) == 1:
        target = expr.comparisons[0]
        if isinstance(target.operator, cst.Equal):
            return _match_equality(expr.left, target.comparator, param_names)
        if isinstance(target.operator, cst.In):
            return _match_membership(expr.left, target.comparator, param_names)
    return None


def _match_equality(
    left: cst.BaseExpression,
    right: cst.BaseExpression,
    param_names: set[str],
) -> tuple[str, set[ExtractedLiteral]] | None:
    """Match `param == lit` or `lit == param`."""
    if isinstance(left, cst.Name) and left.value in param_names:
        lit = extract_literal(right)
        if lit is not None:
            return (left.value, {lit})
    if isinstance(right, cst.Name) and right.value in param_names:
        lit = extract_literal(left)
        if lit is not None:
            return (right.value, {lit})
    return None


def _match_membership(
    left: cst.BaseExpression,
    container: cst.BaseExpression,
    param_names: set[str],
) -> tuple[str, set[ExtractedLiteral]] | None:
    """Match `param in (lit1, lit2)` / `[...]` / `{...}`."""
    if not isinstance(left, cst.Name) or left.value not in param_names:
        return None
    if not isinstance(container, (cst.Tuple, cst.List, cst.Set)):
        return None
    lits: set[ExtractedLiteral] = set()
    for elem in container.elements:
        if not isinstance(elem, cst.Element):
            return None
        lit = extract_literal(elem.value)
        if lit is None:
            return None
        lits.add(lit)
    return (left.value, lits) if lits else None


def _extract_match_pattern_literals(pattern: cst.MatchPattern) -> set[ExtractedLiteral] | None:
    """Extract scalar literal values from a MatchValue or MatchOr pattern."""
    if isinstance(pattern, cst.MatchValue):
        lit = extract_literal(pattern.value)
        return {lit} if lit is not None else None
    if isinstance(pattern, cst.MatchOr):
        lits: set[ExtractedLiteral] = set()
        for opt in pattern.patterns:
            sub = _extract_match_pattern_literals(opt.pattern)
            if sub is None:
                return None
            lits.update(sub)
        return lits
    return None
