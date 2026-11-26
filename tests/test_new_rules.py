from pathlib import Path

from py_type_narrower.config import NarrowerConfig
from py_type_narrower.indexer import index_files
from py_type_narrower.models import RuleCode, TargetKind, Tier
from py_type_narrower.rewriter import rewrite_module_source
from py_type_narrower.rules import run_rules


def test_nt104_always_initialized_class_attributes_and_rewriter(tmp_path: Path) -> None:
    code = """class Client:
    pass

class Service:
    client: Client | None = None
    optional_cache: dict[str, int] | None = None

    def __init__(self, endpoint: str) -> None:
        self.client = Client()
        self.endpoint: str | None = endpoint
        self.optional_cache = None

    def reset(self) -> None:
        self.optional_cache = {}
"""
    mod = tmp_path / "svc.py"
    mod.write_text(code, encoding="utf-8")
    index = index_files([mod], project_root=tmp_path)
    cands = run_rules(index, NarrowerConfig(max_tier=1))
    nt104 = {c.target_name: c for c in cands if c.code == RuleCode.NT104}

    assert set(nt104.keys()) == {"client", "endpoint"}
    assert nt104["client"].suggested_type == "Client"
    assert nt104["client"].target_kind == TargetKind.ATTRIBUTE
    assert nt104["client"].drop_none_default is True
    assert nt104["endpoint"].suggested_type == "str"
    assert nt104["endpoint"].tier == Tier.LOCAL

    rewritten = rewrite_module_source(code, list(nt104.values()))
    assert "    client: Client\n" in rewritten
    assert "        self.endpoint: str = endpoint\n" in rewritten
    # Unchanged optional_cache
    assert "    optional_cache: dict[str, int] | None = None\n" in rewritten


def test_nt104_skips_if_set_to_none_in_another_method_or_deleted(tmp_path: Path) -> None:
    code = """class Runner:
    def __init__(self) -> None:
        self.a: int | None = 1
        self.b: int | None = 2

    def clear(self) -> None:
        self.a = None
        del self.b
"""
    mod = tmp_path / "runner.py"
    mod.write_text(code, encoding="utf-8")
    index = index_files([mod], project_root=tmp_path)
    cands = run_rules(index, NarrowerConfig(max_tier=1))
    assert [c for c in cands if c.code == RuleCode.NT104] == []


def test_nt105_unused_union_return_and_no_overlap_with_nt102(tmp_path: Path) -> None:
    code = """def parse_value(flag: bool) -> int | str | bytes:
    if flag:
        return 1
    return "one"

def parse_optional(flag: bool) -> int | str | None:
    if flag:
        return 1
    return None

def only_none_unused(flag: bool) -> int | None:
    if flag:
        return 1
    return 2
"""
    mod = tmp_path / "parser.py"
    mod.write_text(code, encoding="utf-8")
    index = index_files([mod], project_root=tmp_path)
    cands = run_rules(index, NarrowerConfig(max_tier=1))
    by_fn = {c.symbol.rsplit(".", maxsplit=1)[-1]: c for c in cands}

    assert by_fn["parse_value"].code == RuleCode.NT105
    assert by_fn["parse_value"].suggested_type == "int | str"

    assert by_fn["parse_optional"].code == RuleCode.NT105
    assert by_fn["parse_optional"].suggested_type == "int | None"

    # When ONLY None is unused, NT102 fires (not NT105)
    assert by_fn["only_none_unused"].code == RuleCode.NT102
    assert by_fn["only_none_unused"].suggested_type == "int"


def test_nt203_dataclass_and_namedtuple_field_narrowing(tmp_path: Path) -> None:
    code = """from dataclasses import dataclass
from typing import NamedTuple

@dataclass
class Violation:
    code: str
    lineno: int
    note: str | None = None

class Event(NamedTuple):
    kind: str
    payload: int

def produce() -> None:
    Violation("TDP001", 10, "msg")
    Violation(code="TDP001", lineno=20, note="other")
    Event("click", 1)
    Event("scroll", 2)
"""
    mod = tmp_path / "types_mod.py"
    mod.write_text(code, encoding="utf-8")
    index = index_files([mod], project_root=tmp_path)
    cands = run_rules(index, NarrowerConfig(max_tier=2, closed_world=True))
    nt203 = {
        (c.symbol.rsplit(".", maxsplit=1)[-1], c.target_name): c
        for c in cands
        if c.code == RuleCode.NT203
    }

    assert ("Violation", "code") in nt203
    assert nt203[("Violation", "code")].suggested_type == 'Literal["TDP001"]'
    assert nt203[("Violation", "code")].is_single_constant is True

    assert ("Violation", "note") in nt203
    assert nt203[("Violation", "note")].suggested_type == "str"
    assert nt203[("Violation", "note")].drop_none_default is True

    assert ("Event", "kind") in nt203
    assert nt203[("Event", "kind")].suggested_type == 'Literal["click", "scroll"]'

    rewritten = rewrite_module_source(code, list(nt203.values()))
    assert '    code: Literal["TDP001"]\n' in rewritten
    assert "    note: str\n" in rewritten
    assert '    kind: Literal["click", "scroll"]\n' in rewritten
    assert "from typing import NamedTuple, Literal" in rewritten


def test_nt203_respects_attribute_reassignment_and_subclasses(tmp_path: Path) -> None:
    code = """from dataclasses import dataclass

@dataclass
class Base:
    tag: str

@dataclass
class Child(Base):
    extra: int

@dataclass
class Mutable:
    mode: str

def run(dynamic: str) -> None:
    Base("a")
    Base("a")
    m = Mutable("fast")
    m.mode = dynamic
"""
    mod = tmp_path / "mut.py"
    mod.write_text(code, encoding="utf-8")
    index = index_files([mod], project_root=tmp_path)
    cands = run_rules(index, NarrowerConfig(max_tier=2, closed_world=True))
    assert [c for c in cands if c.code == RuleCode.NT203] == []


def test_nt204_unused_union_param(tmp_path: Path) -> None:
    code = """def _encode(data: int | str | bytes) -> int:
    return len(str(data))

def main(x: int) -> None:
    _encode(1)
    _encode("hello")
    _encode(x)
"""
    mod = tmp_path / "enc.py"
    mod.write_text(code, encoding="utf-8")
    index = index_files([mod], project_root=tmp_path)
    cands = run_rules(index, NarrowerConfig(max_tier=2))
    nt204 = [c for c in cands if c.code == RuleCode.NT204]

    assert len(nt204) == 1
    assert nt204[0].target_name == "data"
    assert nt204[0].suggested_type == "int | str"
    assert nt204[0].score >= 75


def test_nt105_and_nt204_handle_any_union_vars_and_default_none_hooks(tmp_path: Path) -> None:
    code = """from typing import Any

def default_hook() -> str | None:
    return None

def returns_any(val: Any, flag: bool) -> str | int:
    if flag:
        return "ok"
    return val

def _take_union(x: int | None) -> int:
    return x or 0

def caller(opt: int | None, dyn: Any) -> None:
    _take_union(None)
    _take_union(opt)
    returns_any(dyn, True)
"""
    mod = tmp_path / "edge.py"
    mod.write_text(code, encoding="utf-8")
    index = index_files([mod], project_root=tmp_path)
    cands = run_rules(index, NarrowerConfig(max_tier=2))
    assert [c for c in cands if c.code in {RuleCode.NT105, RuleCode.NT204}] == []


def test_nt205_constant_bool_param(tmp_path: Path) -> None:
    code = """def _deploy(dry_run: bool = False, verbose: bool = False) -> int:
    return int(dry_run) + int(verbose)

def main() -> None:
    _deploy(False, True)
    _deploy(dry_run=False, verbose=False)
    _deploy()
"""
    mod = tmp_path / "deploy.py"
    mod.write_text(code, encoding="utf-8")
    index = index_files([mod], project_root=tmp_path)
    cands = run_rules(index, NarrowerConfig(max_tier=2))
    nt205 = {c.target_name: c for c in cands if c.code == RuleCode.NT205}

    assert set(nt205.keys()) == {"dry_run"}
    assert nt205["dry_run"].suggested_type == "Literal[False]"
    assert nt205["dry_run"].is_single_constant is True

    rewritten = rewrite_module_source(code, list(nt205.values()))
    assert (
        "def _deploy(dry_run: Literal[False] = False, verbose: bool = False) -> int:" in rewritten
    )
    assert "from typing import Literal" in rewritten


def test_nt204_and_nt105_respect_type_aliases_numeric_tower_and_never_narrow_to_none(
    tmp_path: Path,
) -> None:
    code = """from typing import Literal

AlignMethod = Literal["top", "middle", "bottom"]

class Base:
    pass

class Sub(Base):
    pass

def _align(vertical: AlignMethod | None = None) -> str:
    return str(vertical)

def _progress(completed: float | None = None) -> float:
    return float(completed or 0)

def _only_none(opt: str | None = None) -> str:
    return opt or ""

def _ret_subclass(flag: bool) -> Base | int:
    if flag:
        return Sub()
    return 1

def caller() -> None:
    _align(None)
    _align("middle")
    _progress(None)
    _progress(42)
    _only_none(None)
    _only_none()
"""
    mod = tmp_path / "aliases.py"
    mod.write_text(code, encoding="utf-8")
    index = index_files([mod], project_root=tmp_path)
    cands = run_rules(index, NarrowerConfig(max_tier=2, closed_world=True))
    assert [c for c in cands if c.code in {RuleCode.NT105, RuleCode.NT204}] == []


def test_tier2_rules_always_include_non_none_default_values(tmp_path: Path) -> None:
    code = """def _emit(err: bool = False, mode: str = "fast", tag: str | int = "") -> str:
    return f"{err}:{mode}:{tag}"

def main() -> None:
    _emit(True, "slow", 1)
    _emit(True, "slow", 2)
"""
    mod = tmp_path / "defaults.py"
    mod.write_text(code, encoding="utf-8")
    index = index_files([mod], project_root=tmp_path)
    cands = run_rules(index, NarrowerConfig(max_tier=2))

    # `err: bool = False` called only with True cannot narrow to Literal[True].
    assert [c for c in cands if c.code == RuleCode.NT205] == []
    # `tag: str | int = ""` called only with int still needs `str` for its `""` default.
    assert [c for c in cands if c.code == RuleCode.NT204] == []
    # `mode: str = "fast"` called with "slow" must keep "fast" in the suggested Literal.
    nt201 = [c for c in cands if c.code == RuleCode.NT201]
    assert len(nt201) == 1
    assert nt201[0].suggested_type == 'Literal["fast", "slow"]'


def test_nt106_fixed_tuple_returns(tmp_path: Path) -> None:
    code = """from typing import Any

def coords(x: int, y: int) -> tuple[int, ...]:
    return (x, y)

def labeled(flag: bool) -> tuple[str | int, ...]:
    if flag:
        return ("a", 1)
    return ("b", 2)

def pair_any() -> tuple[Any, ...]:
    return ("ok", 42)

def variable_len(flag: bool) -> tuple[int, ...]:
    if flag:
        return (1, 2)
    return (1, 2, 3)

def starred(xs: list[int]) -> tuple[int, ...]:
    return (*xs, 1)

def empty() -> tuple[int, ...]:
    return ()
"""
    mod = tmp_path / "tuples.py"
    mod.write_text(code, encoding="utf-8")
    index = index_files([mod], project_root=tmp_path)
    cands = run_rules(index, NarrowerConfig(max_tier=1))
    nt106 = {c.symbol.rsplit(".", maxsplit=1)[-1]: c for c in cands if c.code == RuleCode.NT106}

    assert set(nt106.keys()) == {"coords", "labeled", "pair_any"}
    assert nt106["coords"].suggested_type == "tuple[int, int]"
    assert nt106["labeled"].suggested_type == "tuple[str, int]"
    assert nt106["pair_any"].suggested_type == "tuple[str, int]"

    rewritten = rewrite_module_source(code, list(nt106.values()))
    assert "def coords(x: int, y: int) -> tuple[int, int]:" in rewritten
    assert "def labeled(flag: bool) -> tuple[str, int]:" in rewritten
    assert "def pair_any() -> tuple[str, int]:" in rewritten
