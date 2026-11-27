from pathlib import Path

import libcst as cst
import pytest

from py_type_narrower.config import NarrowerConfig
from py_type_narrower.indexer import discover_python_files, index_files
from py_type_narrower.models import RuleCode
from py_type_narrower.rules import run_rules


def _run_on_code(tmp_path: Path, code: str, *, max_tier: int = 2, max_literals: int = 8):
    mod = tmp_path / "mod.py"
    mod.write_text(code, encoding="utf-8")
    index = index_files([mod], project_root=tmp_path)
    cfg = NarrowerConfig(max_tier=max_tier, max_literals=max_literals)
    return run_rules(index, cfg)


@pytest.mark.parametrize(
    ("case_name", "code"),
    [
        (
            "generator_return_optional",
            """
from collections.abc import Iterator

def gen(flag: bool) -> Iterator[int] | None:
    if flag:
        yield 1
    yield 2
""",
        ),
        (
            "abstract_method_param_and_return",
            """
from abc import ABC, abstractmethod

class Base(ABC):
    @abstractmethod
    def run(self, mode: str) -> str | None:
        if mode == "a":
            return "ok"
        else:
            raise ValueError(mode)
""",
        ),
        (
            "explicit_override_decorator",
            """
from typing import override

class Sub:
    @override
    def run(self, mode: str) -> int:
        if mode == "a":
            return len(mode)
        else:
            raise ValueError(mode)
""",
        ),
        (
            "star_args_call_prevents_nt201_and_nt202",
            """
def _helper(mode: str, limit: int | None) -> int:
    return len(mode) + (limit or 0)

def caller(args: tuple[str, int | None]) -> None:
    _helper("a", 1)
    _helper(*args)
""",
        ),
        (
            "kwargs_call_prevents_nt201_and_nt202",
            """
def _helper(mode: str, limit: int | None) -> int:
    return len(mode) + (limit or 0)

def caller(kw: dict[str, object]) -> None:
    _helper("a", 1)
    _helper(**kw)
""",
        ),
        (
            "bool_passed_to_int_not_treated_as_int_literal",
            """
def _count(n: int) -> int:
    return n + 1

def caller() -> None:
    _count(True)
    _count(False)
""",
        ),
        (
            "omitted_none_default_blocks_nt202",
            """
def _fetch(timeout: int | None = None) -> int:
    return timeout or 0

def caller() -> None:
    _fetch(10)
    _fetch()
""",
        ),
        (
            "reassigned_local_variable_blocks_nt201",
            """
def _use(mode: str) -> int:
    return len(mode)

def caller(flag: bool, dynamic: str) -> None:
    m = "fast"
    if flag:
        m = dynamic
    _use(m)
""",
        ),
        (
            "match_without_raising_wildcard_blocks_nt101",
            """
def handle(mode: str) -> int:
    match mode:
        case "a":
            return 1
        case "b":
            return 2
    return len(mode)
""",
        ),
        (
            "if_elif_without_else_raise_blocks_nt101",
            """
def handle(mode: str) -> int:
    if mode == "a":
        return 1
    elif mode == "b":
        return 2
    return len(mode)
""",
        ),
        (
            "non_none_return_with_possible_none_branch_blocks_nt102",
            """
def maybe(x: int | None) -> int | None:
    return x
""",
        ),
        (
            "try_except_with_fallthrough_blocks_nt102",
            """
def risky(flag: bool) -> str | None:
    try:
        if flag:
            return "ok"
    except ValueError:
        return "err"
""",
        ),
    ],
)
def test_hard_negatives_never_fire(tmp_path: Path, case_name: str, code: str) -> None:
    cands = _run_on_code(tmp_path, code)
    assert cands == [], f"Unexpected candidates for {case_name}: {cands}"


def test_max_literals_cap_suppresses_nt101_nt103_and_nt201(tmp_path: Path) -> None:
    code = """
def guarded(mode: str) -> int:
    if mode in ("a", "b", "c", "d"):
        return len(mode)
    raise ValueError(mode)

def ret_lit(x: int) -> str:
    if x == 1:
        return "a"
    if x == 2:
        return "b"
    if x == 3:
        return "c"
    return "d"

def _called(tag: str) -> int:
    return len(tag)

def run() -> None:
    _called("a")
    _called("b")
    _called("c")
    _called("d")
"""
    cands = _run_on_code(tmp_path, code, max_tier=2, max_literals=3)
    assert {c.code for c in cands} - {RuleCode.NT101, RuleCode.NT103, RuleCode.NT201} == set()
    assert cands == []


def test_hard_negative_nt101_reassigned_param_and_conditional_dict_subscript(
    tmp_path: Path,
) -> None:
    code = """DEFAULT_REPOS = {"pypi": "https://pypi.org", "testpypi": "https://test.pypi.org"}

def normalize_shell(shell: str = "AUTO") -> int:
    shell = shell.lower()
    match shell:
        case "bash" | "zsh":
            return 1
        case _:
            raise ValueError(shell)

def lookup_repo(name_or_url: str) -> str:
    if name_or_url in DEFAULT_REPOS:
        return DEFAULT_REPOS[name_or_url]
    return name_or_url

def lookup_with_early_return(name: str) -> str:
    if name.startswith("http"):
        return name
    return DEFAULT_REPOS[name]
"""
    cands = _run_on_code(tmp_path, code, max_tier=1)
    assert [c for c in cands if c.code == RuleCode.NT101] == []


def test_hard_negative_class_attr_does_not_leak_into_module_scope(tmp_path: Path) -> None:
    code = """from dataclasses import dataclass

@dataclass
class _Block:
    after: int = 0

def _compute() -> tuple[int, int]:
    return (1, 2)

def make_block() -> _Block:
    before, after = _compute()
    return _Block(after=after)
"""
    cands = _run_on_code(tmp_path, code, max_tier=2)
    assert [c for c in cands if c.code == RuleCode.NT203] == []


def test_hard_negative_no_type_check_skipped(tmp_path: Path) -> None:
    code = """from typing import no_type_check

@no_type_check
def _ignored_fn(flag: bool) -> str | None:
    if flag:
        return "a"
    return "b"

@no_type_check
class _IgnoredCls:
    def status(self, ok: bool) -> str:
        return "ok" if ok else "err"
"""
    cands = _run_on_code(tmp_path, code, max_tier=2)
    assert cands == []


def test_hard_negative_prose_and_overlong_string_literals_skipped(tmp_path: Path) -> None:
    code = """def _banner() -> str:
    return "Project plugins need to be moved from .pdm-plugins to the cache directory."

def _ask(question: str) -> str:
    return question

def _report(message: str) -> str:
    return message

def caller() -> None:
    _ask("Project name")
    _ask("Python requires('*' to allow any)")
    _report("matches a .gitignore file content")
    _report("matches the --exclude regular expression")
"""
    cands = _run_on_code(tmp_path, code, max_tier=2)
    assert [c for c in cands if c.code in {RuleCode.NT103, RuleCode.NT201}] == []


def test_hard_negative_overrides_and_external_base_subclasses(tmp_path: Path) -> None:
    code = """import ast
from abc import ABC, abstractmethod
from typing import IO

class Splitter(ABC):
    @abstractmethod
    def get_icon(self) -> str: ...

class RowSplitter(Splitter):
    def get_icon(self) -> str:
        return "row"

class NullFile(IO[str]):
    def read(self, n: int = 1) -> str:
        return ""

    def fileno(self) -> int:
        return -1

class MyVisitor(ast.NodeVisitor):
    def visit_Assign(self, node: ast.Assign | ast.AnnAssign) -> None:
        pass

    def _internal(self, mode: str) -> int:
        return len(mode)

def run(v: MyVisitor, ann: ast.AnnAssign) -> None:
    v.visit_Assign(ann)
    v._internal("fast")
    v._internal("slow")
"""
    cands = _run_on_code(tmp_path, code, max_tier=2)
    # Overrides (`RowSplitter.get_icon`) and external-base public methods (`NullFile.*`, `visit_Assign`) skipped.
    assert [c for c in cands if c.code == RuleCode.NT103] == []
    assert [c for c in cands if c.symbol.endswith("visit_Assign")] == []
    # Private helpers on external-base subclasses are still analyzed normally.
    assert any(c.symbol.endswith("_internal") and c.code == RuleCode.NT201 for c in cands)


def test_hard_negative_base_class_overridden_and_hook_returns_skipped(tmp_path: Path) -> None:
    code = """class BaseNode:
    def check_action(self) -> bool | None:
        return True

    def render(self) -> str:
        return ""

    def _gutter_width(self) -> int:
        return 0

    def leave_node(self, node: int) -> int | str:
        return node

    def _private_non_overridden(self, flag: bool) -> str | None:
        return "a" if flag else "b"

class SubNode(BaseNode):
    def check_action(self) -> bool | None:
        return None

    def render(self) -> str:
        return "sub"

    def _gutter_width(self) -> int:
        return 2
"""
    cands = _run_on_code(tmp_path, code, max_tier=1)
    by_sym = {(c.symbol.rsplit(".", maxsplit=1)[-1], c.code) for c in cands}
    # Overridden base methods and open-world public base hooks must not have return types narrowed.
    assert ("check_action", RuleCode.NT102) not in by_sym
    assert ("render", RuleCode.NT103) not in by_sym
    assert ("_gutter_width", RuleCode.NT103) not in by_sym
    assert ("leave_node", RuleCode.NT105) not in by_sym
    # Private non-overridden helper on the base class is still narrowed cleanly.
    assert ("_private_non_overridden", RuleCode.NT102) in by_sym


def test_hard_negative_getattr_dunders_skipped(tmp_path: Path) -> None:
    code = """def __getattr__(name: str) -> str:
    if name == "__version__":
        return "1.0"
    raise AttributeError(name)

class ReadOnlyList:
    def __getattr__(self, key: str) -> object:
        if key in ("append", "clear"):
            raise RuntimeError(key)
        raise AttributeError(key)
"""
    cands = _run_on_code(tmp_path, code, max_tier=2)
    assert cands == []


def test_hard_negative_generic_base_subclass_and_single_int_sentinel(tmp_path: Path) -> None:
    code = """from decimal import Decimal
from typing import Generic, TypeVar

T = TypeVar("T")

class ApproxScalar(Generic[T]):
    DEFAULT_TOL: float | Decimal = 1e-12

    def __init__(self, expected: T) -> None:
        self.expected = expected

class ApproxDecimal(ApproxScalar[Decimal]):
    DEFAULT_TOL = Decimal("1e-12")

class TreeNode:
    def __init__(self) -> None:
        self._line: int = -1
"""
    cands = _run_on_code(tmp_path, code, max_tier=1)
    assert [c for c in cands if c.code == RuleCode.NT104] == []


def test_indexer_skips_hidden_dirs_and_libcst_logic_errors(tmp_path: Path, monkeypatch) -> None:
    hidden = tmp_path / ".corpus" / "repo"
    hidden.mkdir(parents=True)
    (hidden / "bad.py").write_text("x = 1\n", encoding="utf-8")
    good = tmp_path / "good.py"
    good.write_text("def f() -> int:\n    return 1\n", encoding="utf-8")

    discovered = discover_python_files([tmp_path], exclude=[])
    assert discovered == [good.resolve()]

    # Simulate LibCST raising CSTLogicError on a malformed fixture file
    bad_file = tmp_path / "malformed.py"
    bad_file.write_text("x = 1\n", encoding="utf-8")
    real_parse = cst.parse_module

    def flaky_parse(source: str) -> cst.Module:
        if source == "x = 1\n":
            msg = "simulated logic error"
            raise cst.CSTLogicError(msg)
        return real_parse(source)

    monkeypatch.setattr(cst, "parse_module", flaky_parse)
    index = index_files([good, bad_file], project_root=tmp_path)
    assert good.resolve() in index.files
    assert bad_file.resolve() not in index.files


def test_hard_negative_variance_aware_override_and_external_subclass_guards(
    tmp_path: Path,
) -> None:
    code = """from collections.abc import Iterator, Mapping, Sequence
from typing import Any, override

from external_lib import ShellComplete

class BashComplete(ShellComplete):
    # Contravariant param narrowing (NT101) on public external-subclass method -> SKIP
    def handle(self, mode: str) -> int:
        if mode == "fast":
            return 1
        raise ValueError(mode)

    # Invariant mutable dict return (NT109) on public external-subclass method -> SKIP
    def source_vars(self) -> dict[str, Any]:
        return {"prog_name": "app", "shell": "bash"}

    # Invariant mutable list return (NT107) on public external-subclass method -> SKIP
    def items(self) -> list[int | None]:
        return [1, 2]

    # Covariant Mapping / Iterator / Sequence returns on external subclass -> SAFE!
    def readonly_vars(self) -> Mapping[str, Any]:
        return {"prog_name": "app"}

    def stream(self) -> Iterator[int | None]:
        yield 1

    def seq(self) -> Sequence[int | None]:
        return [1, 2]

    # Private helper on external subclass -> SAFE!
    def _internal_dict(self) -> dict[str, Any]:
        return {"k": 1}

    # Explicit @override even on a private method -> SKIP contravariant / invariant!
    @override
    def _hook(self, kind: str) -> dict[str, Any]:
        if kind == "a":
            return {"k": 1}
        raise ValueError(kind)
"""
    cands = _run_on_code(tmp_path, code, max_tier=1)
    found = {(c.symbol.rsplit(".", maxsplit=1)[-1], c.code) for c in cands}

    # Skipped because contravariant param or invariant mutable return on external/override method
    assert ("handle", RuleCode.NT101) not in found
    assert ("source_vars", RuleCode.NT109) not in found
    assert ("items", RuleCode.NT107) not in found
    assert ("_hook", RuleCode.NT101) not in found
    assert ("_hook", RuleCode.NT109) not in found

    # Allowed because covariant return or private non-override helper
    assert ("readonly_vars", RuleCode.NT109) in found
    assert ("stream", RuleCode.NT107) in found
    assert ("seq", RuleCode.NT107) in found
    assert ("_internal_dict", RuleCode.NT109) in found


def test_hard_negative_nt101_skips_when_indexed_caller_passes_dynamic_or_out_of_set_arg(
    tmp_path: Path,
) -> None:
    code = """from typing import Literal

def validator_dynamic(mode: str) -> int:
    if mode not in ("a", "b"):
        raise ValueError(mode)
    return 1

def validator_out_of_set(mode: str) -> int:
    if mode not in ("a", "b"):
        raise ValueError(mode)
    return 1

def validator_star_call(mode: str) -> int:
    if mode not in ("a", "b"):
        raise ValueError(mode)
    return 1

def safe_with_literal_callers(mode: str) -> int:
    if mode not in ("a", "b"):
        raise ValueError(mode)
    return 1

def safe_standalone(mode: str) -> int:
    if mode not in ("a", "b"):
        raise ValueError(mode)
    return 1

def callers(raw: str, narrow: Literal["a", "b"], args: tuple[str]) -> None:
    validator_dynamic(raw)
    validator_out_of_set("invalid")
    validator_star_call(*args)
    safe_with_literal_callers("a")
    safe_with_literal_callers(narrow)
"""
    cands = _run_on_code(tmp_path, code, max_tier=1)
    nt101_fns = {c.symbol.rsplit(".", maxsplit=1)[-1] for c in cands if c.code == RuleCode.NT101}

    assert "validator_dynamic" not in nt101_fns
    assert "validator_out_of_set" not in nt101_fns
    assert "validator_star_call" not in nt101_fns
    assert "safe_with_literal_callers" in nt101_fns
    assert "safe_standalone" in nt101_fns
