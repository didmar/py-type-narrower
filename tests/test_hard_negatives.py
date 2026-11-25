from pathlib import Path

import pytest

from py_type_narrower.config import NarrowerConfig
from py_type_narrower.indexer import index_files
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
