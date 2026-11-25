from pathlib import Path

from py_type_narrower.config import NarrowerConfig
from py_type_narrower.indexer import index_files
from py_type_narrower.models import RuleCode, TargetKind, Tier
from py_type_narrower.rules import run_rules


def _analyze_snippet(tmp_path: Path, code: str, config: NarrowerConfig | None = None):
    file_path = tmp_path / "sample.py"
    file_path.write_text(code, encoding="utf-8")
    cfg = config or NarrowerConfig(max_tier=1)
    index = index_files([file_path], project_root=tmp_path)
    return run_rules(index, cfg)


def test_nt101_if_elif_else_raise(tmp_path: Path) -> None:
    code = """
def choose(mode: str) -> int:
    if mode == "fast":
        return 1
    elif mode == "accurate":
        return 2
    else:
        raise ValueError(mode)
"""
    cands = _analyze_snippet(tmp_path, code)
    nt101 = [c for c in cands if c.code == RuleCode.NT101]
    assert len(nt101) == 1
    assert nt101[0].target_kind == TargetKind.PARAMETER
    assert nt101[0].target_name == "mode"
    assert nt101[0].suggested_type == 'Literal["accurate", "fast"]'
    assert nt101[0].tier == Tier.LOCAL
    assert nt101[0].score >= 90


def test_nt101_negative_guard_and_sequential_if(tmp_path: Path) -> None:
    code = """
def check_negative(env: str) -> str:
    if env not in ("dev", "prod"):
        raise ValueError("bad env")
    return env

def check_sequential(level: int) -> str:
    if level == 1:
        return "low"
    if level == 2:
        return "high"
    raise ValueError("bad level")
"""
    cands = _analyze_snippet(tmp_path, code)
    nt101 = {c.target_name: c for c in cands if c.code == RuleCode.NT101}
    assert nt101["env"].suggested_type == 'Literal["dev", "prod"]'
    assert nt101["level"].suggested_type == "Literal[1, 2]"


def test_nt101_match_statement_and_const_dict(tmp_path: Path) -> None:
    code = """
DISPATCH = {"json": 1, "yaml": 2}

def by_match(kind: str) -> int:
    match kind:
        case "a" | "b":
            return 1
        case "c":
            return 2
        case _:
            raise ValueError(kind)

def by_dict(fmt: str) -> int:
    return DISPATCH[fmt]

def by_inline_dict(code: int) -> str:
    return {200: "ok", 404: "missing"}[code]
"""
    cands = _analyze_snippet(tmp_path, code)
    nt101 = {c.target_name: c for c in cands if c.code == RuleCode.NT101}
    assert nt101["kind"].suggested_type == 'Literal["a", "b", "c"]'
    assert nt101["fmt"].suggested_type == 'Literal["json", "yaml"]'
    assert nt101["code"].suggested_type == "Literal[200, 404]"


def test_nt101_skips_override_and_protocol(tmp_path: Path) -> None:
    code = """
from typing import Protocol

class Base:
    def run(self, mode: str) -> int:
        return len(mode)

class Child(Base):
    def run(self, mode: str) -> int:
        if mode == "a":
            return 1
        else:
            raise ValueError(mode)

class Runner(Protocol):
    def go(self, mode: str) -> int:
        if mode == "a":
            return 1
        else:
            raise ValueError(mode)
"""
    cands = _analyze_snippet(tmp_path, code)
    nt101 = [c for c in cands if c.code == RuleCode.NT101]
    assert nt101 == []


def test_nt102_unused_optional_return(tmp_path: Path) -> None:
    code = """
from typing import Optional

def always_str(flag: bool) -> str | None:
    if flag:
        return "yes"
    return "no"

def always_int() -> Optional[int]:
    x = 42
    return x

def multi_union(flag: bool) -> int | str | None:
    if flag:
        return 1
    return "one"

def can_return_none(flag: bool) -> str | None:
    if flag:
        return "yes"
    return None

def implicit_fallthrough(flag: bool) -> str | None:
    if flag:
        return "yes"

def stub_func() -> str | None:
    ...
"""
    cands = _analyze_snippet(tmp_path, code)
    nt102 = {c.symbol.rsplit(".", maxsplit=1)[-1]: c for c in cands if c.code == RuleCode.NT102}
    assert set(nt102.keys()) == {"always_str", "always_int", "multi_union"}
    assert nt102["always_str"].suggested_type == "str"
    assert nt102["always_int"].suggested_type == "int"
    assert nt102["multi_union"].suggested_type == "int | str"


def test_nt103_literal_return(tmp_path: Path) -> None:
    code = """
def _status(ok: bool) -> str:
    if ok:
        return "ok"
    return "err"

def mixed(ok: bool, msg: str) -> str:
    if ok:
        return "ok"
    return msg

def exit_code(ok: bool) -> int:
    return 0 if ok else 1
"""
    cands = _analyze_snippet(tmp_path, code)
    nt103 = {c.symbol.rsplit(".", maxsplit=1)[-1]: c for c in cands if c.code == RuleCode.NT103}
    assert set(nt103.keys()) == {"_status"}
    assert nt103["_status"].suggested_type == 'Literal["err", "ok"]'
    assert nt103["_status"].score == 90
