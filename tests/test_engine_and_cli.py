import ast
import json
from pathlib import Path

from py_type_narrower.cli import main
from py_type_narrower.config import NarrowerConfig, load_config
from py_type_narrower.engine import RunOptions, analyze_paths
from py_type_narrower.flake8_plugin import TypeNarrowerChecker
from py_type_narrower.indexer import discover_python_files
from py_type_narrower.models import RuleCode


def test_multipass_fixpoint_propagates_literal_narrowing(tmp_path: Path) -> None:
    mod = tmp_path / "pipeline.py"
    mod.write_text(
        """
def _inner(mode: str) -> int:
    return len(mode)

def _outer(mode: str) -> int:
    return _inner(mode)

def main() -> None:
    _outer("fast")
    _outer("safe")
    _outer("fast")
""",
        encoding="utf-8",
    )
    cfg = NarrowerConfig(max_tier=2, max_passes=3, closed_world=True)
    result = analyze_paths(
        [mod],
        project_root=tmp_path,
        config=cfg,
        options=RunOptions(apply_fixes=True, unsafe_fixes=True),
    )
    assert result.passes_run >= 2
    rewritten = mod.read_text(encoding="utf-8")
    assert 'def _outer(mode: Literal["fast", "safe"]) -> int:' in rewritten
    assert 'def _inner(mode: Literal["fast", "safe"]) -> int:' in rewritten


def test_inline_ignores_suppress_specific_or_all_rules(tmp_path: Path) -> None:
    mod = tmp_path / "ignored.py"
    mod.write_text(
        """
def a(mode: str) -> int:  # type-narrow: ignore[NT101, NT103]
    if mode == "x":
        return 1
    else:
        raise ValueError(mode)

# type-narrow: ignore
def b(flag: bool) -> str | None:
    if flag:
        return "yes"
    return "no"
""",
        encoding="utf-8",
    )
    result = analyze_paths([mod], project_root=tmp_path, config=NarrowerConfig(max_tier=1))
    assert result.candidates == []


def test_load_config_from_pyproject(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """
[tool.py-type-narrower]
max-tier = 1
min-confidence = 80
closed-world = true
max-literals = 5
max-passes = 2
verifier = "ty"
select = ["NT101", "NT102"]
ignore = ["NT102"]
""",
        encoding="utf-8",
    )
    cfg = load_config(tmp_path)
    assert cfg.max_tier == 1
    assert cfg.min_confidence == 80
    assert cfg.closed_world is True
    assert cfg.max_literals == 5
    assert cfg.max_passes == 2
    assert cfg.verifier == "ty"
    assert cfg.active_rules() == {RuleCode.NT101}


def test_cli_check_diff_fix_and_formats(tmp_path: Path, capsys) -> None:
    mod = tmp_path / "demo.py"
    mod.write_text(
        """
def choose(mode: str) -> int:
    if mode == "a":
        return 1
    elif mode == "b":
        return 2
    else:
        raise ValueError(mode)
""",
        encoding="utf-8",
    )

    # check returns exit code 1 when candidates exist
    rc = main(["check", str(mod), "--tier", "1", "-v"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "NT101" in out
    assert "evidence" in out

    # JSON output
    rc_json = main(["check", str(mod), "--tier", "1", "--format", "json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc_json == 1
    assert payload["candidates"][0]["code"] == "NT101"

    # SARIF output
    rc_sarif = main(["check", str(mod), "--tier", "1", "--format", "sarif"])
    sarif = json.loads(capsys.readouterr().out)
    assert rc_sarif == 1
    assert sarif["version"] == "2.1.0"
    assert sarif["runs"][0]["results"][0]["ruleId"] == "NT101"

    # diff subcommand
    rc_diff = main(["diff", str(mod), "--tier", "1"])
    diff_out = capsys.readouterr().out
    assert rc_diff == 1
    assert '+def choose(mode: Literal["a", "b"]) -> Literal[1, 2]:' in diff_out

    # fix subcommand rewrites file and exits 0
    rc_fix = main(["fix", str(mod), "--tier", "1", "--verify", "--verifier", "ty"])
    fix_out = capsys.readouterr().out
    assert rc_fix == 0
    assert "applied 2" in fix_out
    assert 'def choose(mode: Literal["a", "b"]) -> Literal[1, 2]:' in mod.read_text(
        encoding="utf-8"
    )

    # Subsequent check is clean (exit code 0)
    rc_clean = main(["check", str(mod), "--tier", "1"])
    _ = capsys.readouterr()
    assert rc_clean == 0


def test_flake8_plugin_reports_tier1_rules(tmp_path: Path) -> None:
    source = """def choose(mode: str) -> int:
    if mode == "x":
        return 1
    else:
        raise ValueError(mode)
"""
    mod = tmp_path / "f8.py"
    mod.write_text(source, encoding="utf-8")
    checker = TypeNarrowerChecker(
        tree=ast.parse(source),
        filename=str(mod),
        lines=source.splitlines(keepends=True),
    )
    findings = list(checker.run())
    assert len(findings) == 2
    codes = {msg.split()[0] for _line, _col, msg, _cls in findings}
    assert codes == {"NT101", "NT103"}
    assert all(line == 1 and cls is TypeNarrowerChecker for line, _col, _msg, cls in findings)


def test_tier2_indexes_project_root_while_reporting_only_target_paths(tmp_path: Path) -> None:
    target = tmp_path / "target.py"
    target.write_text(
        """
def render(mode: str, label: str | None = None) -> int:
    return len(mode) + (len(label) if label else 0)
""",
        encoding="utf-8",
    )
    caller = tmp_path / "caller.py"
    caller.write_text(
        """
from target import render

def _caller_only_issue(x: str) -> str | None:
    return x

def run() -> None:
    render("fast", None)
    render("safe", "ok")
    render("fast", "ok")
""",
        encoding="utf-8",
    )

    cfg = NarrowerConfig(max_tier=2, closed_world=True)
    result = analyze_paths([target], project_root=tmp_path, config=cfg)

    # Only target.py candidates are reported (not caller.py's _caller_only_issue)
    assert all(c.file.resolve() == target.resolve() for c in result.candidates)
    # Cross-file callers in caller.py power NT201 for `mode`...
    nt201 = [c for c in result.candidates if c.code == RuleCode.NT201]
    assert len(nt201) == 1
    assert nt201[0].target_name == "mode"
    assert nt201[0].suggested_type == 'Literal["fast", "safe"]'
    # ...and prevent a false NT202 for `label` because caller.py passes None!
    assert not [c for c in result.candidates if c.code == RuleCode.NT202]


def test_reporting_exposes_advisory_is_fixable_flag(tmp_path: Path, capsys) -> None:
    mod = tmp_path / "advisory.py"
    mod.write_text(
        """from typing import Any

def build_cfg() -> dict[str, Any]:
    return {"host": "localhost", "port": 8080}
""",
        encoding="utf-8",
    )

    rc_text = main(["check", str(mod), "--tier", "1"])
    text_out = capsys.readouterr().out
    assert rc_text == 1
    assert "advisory]" in text_out

    rc_json = main(["check", str(mod), "--tier", "1", "--format", "json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc_json == 1
    assert payload["candidates"][0]["is_fixable"] is False

    rc_sarif = main(["check", str(mod), "--tier", "1", "--format", "sarif"])
    sarif = json.loads(capsys.readouterr().out)
    assert rc_sarif == 1
    res = sarif["runs"][0]["results"][0]
    assert res["level"] == "note"
    assert res["properties"]["fixable"] is False


def test_glob_excludes_and_no_project_index_cli(tmp_path: Path, capsys) -> None:
    (tmp_path / "src").mkdir()
    keep = tmp_path / "src" / "keep.py"
    keep.write_text("def f(mode: str) -> int:\n    return len(mode)\n", encoding="utf-8")
    gen = tmp_path / "src" / "foo_pb2.py"
    gen.write_text("x = 1\n", encoding="utf-8")
    bench_dir = tmp_path / "tests" / "benchmarks" / "corpus"
    bench_dir.mkdir(parents=True)
    (bench_dir / "heavy.py").write_text("y = 2\n", encoding="utf-8")
    caller = tmp_path / "caller.py"
    caller.write_text(
        'from src.keep import f\ndef run() -> None:\n    f("a")\n    f("b")\n',
        encoding="utf-8",
    )

    discovered = discover_python_files(
        [tmp_path],
        exclude=["tests/benchmarks/**", "*_pb2.py"],
    )
    assert discovered == sorted([caller.resolve(), keep.resolve()])

    # With --no-project-index, checking only src/keep.py ignores caller.py outside paths.
    rc = main(["check", str(keep), "--closed-world", "--no-project-index"])
    _ = capsys.readouterr()
    assert rc == 0
