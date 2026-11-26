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


def test_nt107_container_and_generator_non_none_returns(tmp_path: Path) -> None:
    code = """from collections.abc import Generator, Iterator

def gen_ints(n: int) -> Iterator[int | None]:
    for i in range(n):
        yield i + 1

def gen_with_send() -> Generator[str | None, None, None]:
    yield "ok"

def gen_yields_none(flag: bool) -> Iterator[int | None]:
    if flag:
        yield None
    yield 1

def gen_yield_from(xs: list[int | None]) -> Iterator[int | None]:
    yield from xs

def filtered_list(xs: list[str | None]) -> list[str | None]:
    return [x for x in xs if x is not None]

def list_with_none(flag: bool) -> list[int | None]:
    return [1, None] if flag else [2]

def dict_non_none(flag: bool) -> dict[str, int | None]:
    if flag:
        return {"a": 1, "b": 2}
    return {}

def empty_only() -> list[int | None]:
    return []
"""
    mod = tmp_path / "containers.py"
    mod.write_text(code, encoding="utf-8")
    index = index_files([mod], project_root=tmp_path)
    cands = run_rules(index, NarrowerConfig(max_tier=1))
    nt107 = {c.symbol.rsplit(".", maxsplit=1)[-1]: c for c in cands if c.code == RuleCode.NT107}

    assert set(nt107.keys()) == {"gen_ints", "gen_with_send", "filtered_list", "dict_non_none"}
    assert nt107["gen_ints"].suggested_type == "Iterator[int]"
    assert nt107["gen_with_send"].suggested_type == "Generator[str, None, None]"
    assert nt107["filtered_list"].suggested_type == "list[str]"
    assert nt107["dict_non_none"].suggested_type == "dict[str, int]"

    rewritten = rewrite_module_source(code, list(nt107.values()))
    assert "def gen_ints(n: int) -> Iterator[int]:" in rewritten
    assert "def gen_with_send() -> Generator[str, None, None]:" in rewritten
    assert "def filtered_list(xs: list[str | None]) -> list[str]:" in rewritten
    assert "def dict_non_none(flag: bool) -> dict[str, int]:" in rewritten


def test_nt108_concrete_any_and_object_returns(tmp_path: Path) -> None:
    code = """from typing import Any

class Widget:
    pass

def make_widget(flag: bool) -> Any:
    if flag:
        return Widget()
    return None

def compute_label(name: str) -> object:
    return f"label:{name}"

def returns_unknown(fn: Any) -> Any:
    return fn()

def returns_typed_list() -> Any:
    return [1, 2, 3]

def returns_empty_list() -> Any:
    return []

def returns_only_none() -> Any:
    return None
"""
    mod = tmp_path / "any_ret.py"
    mod.write_text(code, encoding="utf-8")
    index = index_files([mod], project_root=tmp_path)
    cands = run_rules(index, NarrowerConfig(max_tier=1))
    nt108 = {c.symbol.rsplit(".", maxsplit=1)[-1]: c for c in cands if c.code == RuleCode.NT108}

    assert set(nt108.keys()) == {"make_widget", "compute_label", "returns_typed_list"}
    assert nt108["make_widget"].suggested_type == "Widget | None"
    assert nt108["compute_label"].suggested_type == "str"
    assert nt108["returns_typed_list"].suggested_type == "list[int]"

    rewritten = rewrite_module_source(code, list(nt108.values()))
    assert "def make_widget(flag: bool) -> Widget | None:" in rewritten
    assert "def compute_label(name: str) -> str:" in rewritten
    assert "def returns_typed_list() -> list[int]:" in rewritten


def test_nt109_structured_dict_returns_typeddict_and_literal_keys(tmp_path: Path) -> None:
    code = """from typing import Any, NotRequired, TypedDict

class UserRecord(TypedDict):
    id: int
    name: str
    email: NotRequired[str]

def build_user(flag: bool) -> dict[str, Any]:
    if flag:
        return {"id": 1, "name": "Ada", "email": "ada@example.com"}
    return {"id": 2, "name": "Grace"}

def build_counts() -> dict[str, int]:
    return {"ok": 1, "err": 0}

def build_ad_hoc(flag: bool) -> dict[str, Any]:
    if flag:
        return {"host": "localhost", "port": 8080, "tls": True}
    return {"host": "localhost", "port": 80}

def dynamic_dict(k: str) -> dict[str, int]:
    return {k: 1}
"""
    mod = tmp_path / "records.py"
    mod.write_text(code, encoding="utf-8")
    index = index_files([mod], project_root=tmp_path)
    cands = run_rules(index, NarrowerConfig(max_tier=1))
    nt109 = {c.symbol.rsplit(".", maxsplit=1)[-1]: c for c in cands if c.code == RuleCode.NT109}

    assert set(nt109.keys()) == {"build_user", "build_counts", "build_ad_hoc"}

    # Matches existing in-scope TypedDict -> auto-fixable!
    assert nt109["build_user"].suggested_type == "UserRecord"
    assert nt109["build_user"].is_fixable is True

    # Homogeneous concrete dict[str, int] -> dict[Literal[...], int], auto-fixable!
    assert nt109["build_counts"].suggested_type == 'dict[Literal["err", "ok"], int]'
    assert nt109["build_counts"].is_fixable is True
    assert nt109["build_counts"].needs_literal_import is True

    # Broad dict[str, Any] with no matching TypedDict -> advisory schema (not auto-rewritten)
    assert (
        nt109["build_ad_hoc"].suggested_type
        == "TypedDict({'host': str, 'port': int, 'tls': NotRequired[bool]})"
    )
    assert nt109["build_ad_hoc"].is_fixable is False

    rewritten = rewrite_module_source(code, list(nt109.values()))
    assert "def build_user(flag: bool) -> UserRecord:" in rewritten
    assert 'def build_counts() -> dict[Literal["err", "ok"], int]:' in rewritten
    # Advisory candidate is preserved verbatim by the rewriter
    assert "def build_ad_hoc(flag: bool) -> dict[str, Any]:" in rewritten


def test_nt206_callsite_structured_dict_param(tmp_path: Path) -> None:
    code = """from typing import Any, TypedDict

class MetricEvent(TypedDict):
    name: str
    value: int

def _record_metric(payload: dict[str, Any]) -> None:
    _ = payload

def _track_scores(scores: dict[str, int]) -> None:
    _ = scores

def _log_extra(meta: dict[str, Any]) -> None:
    _ = meta

def caller() -> None:
    _record_metric({"name": "cpu", "value": 90})
    _record_metric({"name": "mem", "value": 45})
    _track_scores({"a": 1, "b": 2})
    _track_scores({"a": 3, "b": 4})
    _log_extra({"env": "prod", "retries": 2})
    _log_extra({"env": "dev", "retries": 0})
"""
    mod = tmp_path / "metrics.py"
    mod.write_text(code, encoding="utf-8")
    index = index_files([mod], project_root=tmp_path)
    cands = run_rules(index, NarrowerConfig(max_tier=2, closed_world=True))
    nt206 = {c.symbol.rsplit(".", maxsplit=1)[-1]: c for c in cands if c.code == RuleCode.NT206}

    assert set(nt206.keys()) == {"_record_metric", "_track_scores", "_log_extra"}
    assert nt206["_record_metric"].suggested_type == "MetricEvent"
    assert nt206["_record_metric"].is_fixable is True
    assert nt206["_track_scores"].suggested_type == 'dict[Literal["a", "b"], int]'
    assert nt206["_track_scores"].is_fixable is True
    assert nt206["_log_extra"].suggested_type == "TypedDict({'env': str, 'retries': int})"
    assert nt206["_log_extra"].is_fixable is False

    rewritten = rewrite_module_source(code, list(nt206.values()))
    assert "def _record_metric(payload: MetricEvent) -> None:" in rewritten
    assert 'def _track_scores(scores: dict[Literal["a", "b"], int]) -> None:' in rewritten
    assert "def _log_extra(meta: dict[str, Any]) -> None:" in rewritten


def test_strenum_and_intenum_narrowing_across_nt101_nt103_nt201(tmp_path: Path) -> None:
    code = """from enum import IntEnum, StrEnum

class Status(StrEnum):
    OK = "ok"
    ERR = "err"

class Level(IntEnum):
    LOW = 1
    HIGH = 2

def parse_status(raw: str) -> Status:
    return Status(raw)

def safe_parse_status(raw: str) -> Status | None:
    try:
        return Status(raw)
    except ValueError:
        return None

def pick_status(flag: bool) -> str:
    if flag:
        return Status.OK
    return Status.ERR

def pick_level(flag: bool) -> int:
    return Level.HIGH if flag else Level.LOW

def _record(status: str) -> int:
    return len(status)

def caller() -> None:
    _record(Status.OK)
    _record(Status.ERR)
"""
    mod = tmp_path / "enums.py"
    mod.write_text(code, encoding="utf-8")
    index = index_files([mod], project_root=tmp_path)
    cands = run_rules(index, NarrowerConfig(max_tier=2, closed_world=True))
    by_fn_code = {(c.symbol.rsplit(".", maxsplit=1)[-1], c.code): c for c in cands}

    assert ("parse_status", RuleCode.NT101) in by_fn_code
    assert by_fn_code[("parse_status", RuleCode.NT101)].suggested_type == 'Literal["err", "ok"]'
    assert ("safe_parse_status", RuleCode.NT101) not in by_fn_code

    assert ("pick_status", RuleCode.NT103) in by_fn_code
    assert by_fn_code[("pick_status", RuleCode.NT103)].suggested_type == "Status"
    assert by_fn_code[("pick_status", RuleCode.NT103)].needs_literal_import is False

    assert ("pick_level", RuleCode.NT103) in by_fn_code
    assert by_fn_code[("pick_level", RuleCode.NT103)].suggested_type == "Level"

    assert ("_record", RuleCode.NT201) in by_fn_code
    assert by_fn_code[("_record", RuleCode.NT201)].suggested_type == "Status"
    assert by_fn_code[("_record", RuleCode.NT201)].needs_literal_import is False

    rewritten = rewrite_module_source(code, cands)
    assert 'def parse_status(raw: Literal["err", "ok"]) -> Status:' in rewritten
    assert "def pick_status(flag: bool) -> Status:" in rewritten
    assert "def pick_level(flag: bool) -> Level:" in rewritten
    assert "def _record(status: Status) -> int:" in rewritten


def test_nt104_extends_to_literal_and_union_class_attributes(tmp_path: Path) -> None:
    code = """class Job:
    def __init__(self) -> None:
        self.state: str = "idle"
        self._kind: str = "batch"
        self.title: str = ""
        self.result: int | str | bytes = 0

    def start(self) -> None:
        self.state = "running"

    def finish(self) -> None:
        self.state = "done"
        self.result = "ok"
"""
    mod = tmp_path / "job.py"
    mod.write_text(code, encoding="utf-8")
    index = index_files([mod], project_root=tmp_path)
    cands = run_rules(index, NarrowerConfig(max_tier=1))
    nt104 = {c.target_name: c for c in cands if c.code == RuleCode.NT104}

    # Public multi-state FSM (`state`), private literal (`_kind`), and unused-branch union (`result`)
    # are narrowed; public single-sentinel `title: str = ""` is skipped in open-world mode.
    assert set(nt104.keys()) == {"state", "_kind", "result"}
    assert nt104["state"].suggested_type == 'Literal["done", "idle", "running"]'
    assert nt104["_kind"].suggested_type == 'Literal["batch"]'
    assert nt104["result"].suggested_type == "int | str"

    rewritten = rewrite_module_source(code, list(nt104.values()))
    assert '        self.state: Literal["done", "idle", "running"] = "idle"' in rewritten
    assert '        self._kind: Literal["batch"] = "batch"' in rewritten
    assert "        self.result: int | str = 0" in rewritten
    assert '        self.title: str = ""' in rewritten


def test_nt203_extends_to_attrs_pydantic_and_typeddict(tmp_path: Path) -> None:
    code = """from typing import NotRequired, TypedDict

import attrs
from pydantic import BaseModel, Field

@attrs.define
class JobSpec:
    kind: str
    note: str | None = None

class ApiConfig(BaseModel):
    env: str
    token: str | None = Field(default=None)

class Payload(TypedDict):
    action: str
    version: NotRequired[int]
    extra: NotRequired[str]

def make_payload() -> Payload:
    return {"action": "create", "version": 1, "extra": "x"}

def produce() -> None:
    JobSpec("sync", "ok")
    JobSpec("async", "done")
    ApiConfig(env="prod", token="abc")
    ApiConfig(env="staging", token="def")
    p: Payload = {"action": "update", "version": 2}
    _ = p
"""
    mod = tmp_path / "models_ext.py"
    mod.write_text(code, encoding="utf-8")
    index = index_files([mod], project_root=tmp_path)
    cands = run_rules(index, NarrowerConfig(max_tier=2, closed_world=True))
    nt203 = {
        (c.symbol.rsplit(".", maxsplit=1)[-1], c.target_name): c
        for c in cands
        if c.code == RuleCode.NT203
    }

    assert nt203[("JobSpec", "kind")].suggested_type == 'Literal["async", "sync"]'
    assert nt203[("JobSpec", "note")].suggested_type == "str"
    assert nt203[("ApiConfig", "env")].suggested_type == 'Literal["prod", "staging"]'
    assert nt203[("ApiConfig", "token")].suggested_type == "str"
    assert nt203[("ApiConfig", "token")].drop_none_default is True
    assert nt203[("Payload", "action")].suggested_type == 'Literal["create", "update"]'
    assert nt203[("Payload", "version")].suggested_type == "int"
    # `extra` was omitted in `p`, so it must stay NotRequired[str]
    assert ("Payload", "extra") not in nt203

    rewritten = rewrite_module_source(code, list(nt203.values()))
    assert '    kind: Literal["async", "sync"]\n' in rewritten
    assert "    note: str\n" in rewritten
    assert '    env: Literal["prod", "staging"]\n' in rewritten
    assert "    token: str\n" in rewritten
    assert '    action: Literal["create", "update"]\n' in rewritten
    assert "    version: int\n" in rewritten
    assert "    extra: NotRequired[str]\n" in rewritten


def test_nt109_infers_parameterized_list_and_self_attr_types(tmp_path: Path) -> None:
    code = """from typing import Any

class Completer:
    prog_name: str

    def __init__(self, prog_name: str, retries: int) -> None:
        self.prog_name = prog_name
        self.retries = retries

    @property
    def tag(self) -> str:
        return "v1"

    def source_vars(self) -> dict[str, Any]:
        return {
            "prog_name": self.prog_name,
            "retries": self.retries,
            "tag": self.tag,
            "formatter": ["black", "-"],
            "ports": {80, 443},
        }
"""
    mod = tmp_path / "comp.py"
    mod.write_text(code, encoding="utf-8")
    index = index_files([mod], project_root=tmp_path)
    cands = run_rules(index, NarrowerConfig(max_tier=1))
    nt109 = [c for c in cands if c.code == RuleCode.NT109]
    assert len(nt109) == 1
    assert (
        nt109[0].suggested_type
        == "TypedDict({'formatter': list[str], 'ports': set[int], 'prog_name': str, 'retries': int, 'tag': str})"
    )
