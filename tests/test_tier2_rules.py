from pathlib import Path

from py_type_narrower.config import NarrowerConfig
from py_type_narrower.indexer import index_files
from py_type_narrower.models import RuleCode
from py_type_narrower.rules import run_rules


def test_nt201_cross_file_callsite_literal_and_scoring(tmp_path: Path) -> None:
    lib = tmp_path / "service.py"
    lib.write_text(
        """
def _fetch(env: str) -> str:
    if env == "prod":
        return "https://prod"
    return "https://dev"

def public_api(name: str) -> str:
    return name
""",
        encoding="utf-8",
    )
    caller = tmp_path / "app.py"
    caller.write_text(
        """
from service import _fetch, public_api

def main() -> None:
    _fetch("dev")
    _fetch("prod")
    _fetch("dev")
    _fetch("prod")
    public_api("alice")
    public_api("bob")
""",
        encoding="utf-8",
    )

    index = index_files([lib, caller], project_root=tmp_path)
    cands = run_rules(index, NarrowerConfig(max_tier=2))
    nt201 = {c.target_name: c for c in cands if c.code == RuleCode.NT201}

    assert "env" in nt201
    assert nt201["env"].suggested_type == 'Literal["dev", "prod"]'
    # Private + high repetition + body corroboration => high confidence
    assert nt201["env"].score >= 85

    # Public + open-domain param name `name` => very low confidence
    assert "name" in nt201
    assert nt201["name"].score < 40


def test_nt201_unknown_caller_prevents_narrowing(tmp_path: Path) -> None:
    mod = tmp_path / "mod.py"
    mod.write_text(
        """
def _helper(tag: str) -> int:
    return len(tag)

def _clamp(value: int) -> int:
    return max(0, value)

def caller(dynamic: str, flag: bool) -> None:
    _helper("a")
    _helper("b")
    _helper(dynamic)
    score = 50
    if flag:
        score += 10
    _clamp(score)
""",
        encoding="utf-8",
    )
    index = index_files([mod], project_root=tmp_path)
    cands = run_rules(index, NarrowerConfig(max_tier=2))
    assert [c for c in cands if c.code == RuleCode.NT201] == []


def test_nt201_single_constant_detection(tmp_path: Path) -> None:
    mod = tmp_path / "mod.py"
    mod.write_text(
        """
def _connect(version: str) -> str:
    return version

def run() -> None:
    _connect("v2")
    _connect("v2")
    _connect("v2")
""",
        encoding="utf-8",
    )
    index = index_files([mod], project_root=tmp_path)
    cands = run_rules(index, NarrowerConfig(max_tier=2))
    nt201 = [c for c in cands if c.code == RuleCode.NT201]
    assert len(nt201) == 1
    assert nt201[0].is_single_constant is True
    assert nt201[0].suggested_type == 'Literal["v2"]'
    assert "single constant" in nt201[0].message


def test_nt201_method_calls_on_self_and_typed_instance(tmp_path: Path) -> None:
    mod = tmp_path / "client.py"
    mod.write_text(
        """
class Client:
    def _request(self, method: str) -> int:
        return len(method)

    def get(self) -> int:
        return self._request("GET")

    def post(self) -> int:
        return self._request("POST")

def run(c: Client) -> None:
    c._request("GET")
""",
        encoding="utf-8",
    )
    index = index_files([mod], project_root=tmp_path)
    cands = run_rules(index, NarrowerConfig(max_tier=2))
    nt201 = [c for c in cands if c.code == RuleCode.NT201]
    assert len(nt201) == 1
    assert nt201[0].suggested_type == 'Literal["GET", "POST"]'


def test_nt202_unused_optional_param(tmp_path: Path) -> None:
    mod = tmp_path / "worker.py"
    mod.write_text(
        """
def _process(timeout: int | None, retries: int | None = None, tag: str | None = None) -> int:
    return (timeout or 0) + (retries or 0) + len(tag or "")

def main() -> None:
    _process(10, 3, "a")
    _process(20, 5, None)
    _process(30, retries=2, tag="b")
""",
        encoding="utf-8",
    )
    index = index_files([mod], project_root=tmp_path)
    cands = run_rules(index, NarrowerConfig(max_tier=2))
    nt202 = {c.target_name: c for c in cands if c.code == RuleCode.NT202}

    assert "timeout" in nt202
    assert nt202["timeout"].suggested_type == "int"
    assert nt202["timeout"].drop_none_default is False

    assert "retries" in nt202
    assert nt202["retries"].suggested_type == "int"
    assert nt202["retries"].drop_none_default is True

    # `tag` actually received None at line 7
    assert "tag" not in nt202


def test_nt201_test_only_caller_penalty(tmp_path: Path) -> None:
    prod = tmp_path / "core.py"
    prod.write_text(
        """
def _parse(fmt: str) -> int:
    return len(fmt)
""",
        encoding="utf-8",
    )
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_file = tests_dir / "test_core.py"
    test_file.write_text(
        """
from core import _parse

def test_it() -> None:
    _parse("a")
    _parse("b")
""",
        encoding="utf-8",
    )
    index = index_files([prod, test_file], project_root=tmp_path)
    cands = run_rules(index, NarrowerConfig(max_tier=2))
    nt201 = [c for c in cands if c.code == RuleCode.NT201]
    assert len(nt201) == 1
    assert any("test callers" in e.detail for e in nt201[0].evidence)
    assert nt201[0].score <= 55


def test_indexer_resolves_package_init_reexports(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    core = pkg / "core.py"
    core.write_text(
        """class Project:
    def cache(self, name: str) -> str:
        return name

    def fetch(self, *, exclude: list[str] | None = None) -> int:
        return len(exclude or [])
""",
        encoding="utf-8",
    )
    init = pkg / "__init__.py"
    init.write_text("from pkg.core import Project\n", encoding="utf-8")
    caller = tmp_path / "caller.py"
    caller.write_text(
        """from pkg import Project

def run(project: Project, dyn: str) -> None:
    project.cache("wheels")
    project.cache(dyn)
    project.fetch()
    project.fetch(exclude=["a"])
""",
        encoding="utf-8",
    )

    index = index_files([init, core, caller], project_root=tmp_path)
    cands = run_rules(index, NarrowerConfig(max_tier=2, closed_world=True))
    # Both `cache(dyn)` and `fetch()` (omitted exclude) are seen through `pkg.__init__` re-export!
    assert [c for c in cands if c.symbol.startswith("pkg.core.Project.")] == []
