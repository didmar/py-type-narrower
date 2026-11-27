"""Run py-type-narrower against a corpus of real-world repositories.

Clones repositories into a gitignored `.corpus/` directory, executes
`py-type-narrower check --format json`, enriches findings with source context
snippets, and produces both machine-readable (`results.json`) and triage-ready
Markdown (`report.md`) outputs for true-positive / false-positive analysis.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from py_type_narrower import __version__

logger = logging.getLogger("run_corpus")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "corpus.toml"
DEFAULT_CORPUS_DIR = REPO_ROOT / ".corpus"
RULE_ORDER = (
    "NT101",
    "NT102",
    "NT103",
    "NT104",
    "NT105",
    "NT106",
    "NT107",
    "NT108",
    "NT109",
    "NT201",
    "NT202",
    "NT203",
    "NT204",
    "NT205",
    "NT206",
)


@dataclass(frozen=True)
class RepoSpec:
    """Single repository entry loaded from `corpus.toml`."""

    name: str
    paths: tuple[str, ...]
    closed_world: bool
    groups: tuple[str, ...]
    url: str | None = None
    local: str | None = None
    ref: str | None = None
    notes: str = ""


@dataclass(frozen=True)
class SweepOptions:
    """CLI execution options for a corpus sweep."""

    manifest: Path
    corpus_dir: Path
    out_dir: Path | None
    group: str
    repos: tuple[str, ...]
    verify: bool
    verifier: str
    verify_command: str | None
    min_confidence: int
    tier: int | None
    select: str | None
    ignore: str | None
    timeout: int
    update: bool
    clean: bool
    max_per_rule: int
    compare: Path | None
    scaffold_typeddict: bool = False


@dataclass
class RepoRunResult:
    """Collected analysis results and metadata for one repository."""

    spec: RepoSpec
    repo_dir: Path | None = None
    commit: str = "unknown"
    elapsed_sec: float = 0.0
    file_count: int = 0
    verifier_used: str | None = None
    exit_code: int | None = None
    error: str | None = None
    candidates: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this repo result to a JSON-compatible dict."""
        return {
            "name": self.spec.name,
            "url": self.spec.url,
            "local": self.spec.local,
            "paths": list(self.spec.paths),
            "closed_world": self.spec.closed_world,
            "groups": list(self.spec.groups),
            "commit": self.commit,
            "elapsed_sec": round(self.elapsed_sec, 2),
            "file_count": self.file_count,
            "verifier_used": self.verifier_used,
            "exit_code": self.exit_code,
            "error": self.error,
            "candidates": self.candidates,
            "rejected": self.rejected,
        }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for corpus testing."""
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    opts = _parse_args(argv)

    all_specs = _load_manifest(opts.manifest)
    selected = _filter_specs(all_specs, opts.group, opts.repos)
    if not selected:
        logger.error("No repositories matched group=%r repos=%r", opts.group, opts.repos)
        return 2

    tool_exe = _resolve_tool_executable()
    git_exe = shutil.which("git")
    if git_exe is None:
        logger.error("git executable not found on PATH")
        return 2

    repos_cache_dir = opts.corpus_dir / "repos"
    repos_cache_dir.mkdir(parents=True, exist_ok=True)
    run_dir = _prepare_run_dir(opts)

    logger.info(
        "Running py-type-narrower v%s on %d repo(s) (verify=%s, verifier=%s, min_confidence=%d)",
        __version__,
        len(selected),
        opts.verify,
        opts.verifier if opts.verify else "off",
        opts.min_confidence,
    )

    results: list[RepoRunResult] = []
    cloned_dirs: list[Path] = []
    for spec in selected:
        result = _run_single_repo(spec, opts, repos_cache_dir, tool_exe, git_exe)
        results.append(result)
        if spec.local is None and result.repo_dir is not None:
            cloned_dirs.append(result.repo_dir)

    payload = _build_run_payload(opts, results, git_exe)
    results_json_path = run_dir / "results.json"
    report_md_path = run_dir / "report.md"

    results_json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report_text = _render_markdown_report(payload, max_per_rule=opts.max_per_rule)
    report_md_path.write_text(report_text, encoding="utf-8")

    _log_summary_table(results)
    logger.info("Saved JSON:   %s", results_json_path)
    logger.info("Saved report: %s", report_md_path)

    if opts.compare is not None:
        _print_comparison(opts.compare, payload)

    if opts.clean:
        for repo_dir in cloned_dirs:
            shutil.rmtree(repo_dir, ignore_errors=True)
        logger.info("Cleaned %d cached clone(s).", len(cloned_dirs))

    return 0


def _parse_args(argv: list[str] | None) -> SweepOptions:
    """Parse command-line options for the corpus runner."""
    parser = argparse.ArgumentParser(
        prog="run_corpus.py",
        description="Run py-type-narrower across real-world repos to inspect TPs and FPs.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to corpus TOML manifest (default: scripts/corpus.toml)",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=DEFAULT_CORPUS_DIR,
        help="Local cache and run output directory (default: .corpus/)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Explicit output directory for results.json and report.md",
    )
    parser.add_argument(
        "--group",
        default="starter",
        help="Manifest group to run: starter, modern, stress, or all (default: starter)",
    )
    parser.add_argument(
        "--repos",
        default=None,
        help="Comma-separated repo names to run (overrides --group)",
    )
    parser.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verify candidates with a type checker (default: enabled)",
    )
    parser.add_argument(
        "--verifier",
        choices=["auto", "ty", "mypy", "pyright", "basedpyright", "custom"],
        default="ty",
        help="Verifier backend when --verify is enabled (default: ty)",
    )
    parser.add_argument(
        "--verify-command",
        default=None,
        help="Custom command for `--verifier custom`",
    )
    parser.add_argument(
        "--min-confidence",
        type=int,
        default=50,
        help="Minimum confidence score to report (default: 50)",
    )
    parser.add_argument(
        "--tier",
        type=int,
        choices=[1, 2],
        default=None,
        help="Maximum rule tier to run (1 or 2)",
    )
    parser.add_argument("--select", default=None, help="Comma-separated rule codes to enable")
    parser.add_argument("--ignore", default=None, help="Comma-separated rule codes to ignore")
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Per-repo timeout in seconds (default: 180)",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Fetch and reset cached clones to latest remote commit before running",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove cloned repositories from cache after the run completes",
    )
    parser.add_argument(
        "--max-per-rule",
        type=int,
        default=40,
        help="Maximum candidates rendered per rule section in report.md (default: 40)",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="Compare this run against a previous results.json file",
    )
    parser.add_argument(
        "--scaffold-typeddict",
        action="store_true",
        help="Pass --scaffold-typeddict to py-type-narrower check",
    )

    args = parser.parse_args(argv)
    repos = tuple(r.strip() for r in args.repos.split(",") if r.strip()) if args.repos else ()
    return SweepOptions(
        manifest=args.manifest,
        corpus_dir=args.corpus_dir,
        out_dir=args.out,
        group=args.group,
        repos=repos,
        verify=args.verify,
        verifier=args.verifier,
        verify_command=args.verify_command,
        min_confidence=args.min_confidence,
        tier=args.tier,
        select=args.select,
        ignore=args.ignore,
        timeout=args.timeout,
        update=args.update,
        clean=args.clean,
        max_per_rule=args.max_per_rule,
        compare=args.compare,
        scaffold_typeddict=args.scaffold_typeddict,
    )


def _load_manifest(path: Path) -> list[RepoSpec]:
    """Load repository definitions from a TOML manifest."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return [
        RepoSpec(
            name=entry["name"],
            paths=tuple(entry.get("paths", ["."])),
            closed_world=bool(entry.get("closed_world", False)),
            groups=tuple(entry.get("groups", [])),
            url=entry.get("url"),
            local=entry.get("local"),
            ref=entry.get("ref"),
            notes=entry.get("notes", ""),
        )
        for entry in data.get("repos", [])
    ]


def _filter_specs(
    specs: list[RepoSpec],
    group: str,
    repos: tuple[str, ...],
) -> list[RepoSpec]:
    """Filter manifest entries by explicit repo names or group tag."""
    if repos:
        wanted = set(repos)
        return [s for s in specs if s.name in wanted]
    if group == "all":
        return specs
    return [s for s in specs if group in s.groups]


def _resolve_tool_executable() -> str:
    """Locate the `py-type-narrower` CLI executable in the current environment."""
    venv_bin = Path(sys.executable).parent / "py-type-narrower"
    if venv_bin.is_file():
        return str(venv_bin)
    found = shutil.which("py-type-narrower")
    if found is not None:
        return found
    msg = "py-type-narrower executable not found; run via `uv run python scripts/run_corpus.py`"
    raise RuntimeError(msg)


def _prepare_run_dir(opts: SweepOptions) -> Path:
    """Create and return the output directory for this run."""
    if opts.out_dir is not None:
        run_dir = opts.out_dir
    else:
        stamp = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%SZ")
        run_dir = opts.corpus_dir / "runs" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _run_single_repo(
    spec: RepoSpec,
    opts: SweepOptions,
    repos_cache_dir: Path,
    tool_exe: str,
    git_exe: str,
) -> RepoRunResult:
    """Ensure the repository is available locally, run analysis, and enrich findings."""
    result = RepoRunResult(spec=spec)
    logger.info("[%s] Preparing repository...", spec.name)

    try:
        repo_dir = _ensure_repo_checkout(spec, repos_cache_dir, git_exe, update=opts.update)
    except RuntimeError as exc:
        result.error = str(exc)
        logger.warning("[%s] Checkout failed: %s", spec.name, exc)
        return result

    result.repo_dir = repo_dir
    result.commit = _git_short_sha(repo_dir, git_exe)
    target_paths = [repo_dir / rel for rel in spec.paths]
    missing = [p for p in target_paths if not p.exists()]
    if missing:
        result.error = f"Target path(s) not found: {', '.join(str(p) for p in missing)}"
        logger.warning("[%s] %s", spec.name, result.error)
        return result

    result.file_count = _count_py_files(target_paths)
    cmd = _build_check_command(tool_exe, spec, target_paths, opts)

    logger.info(
        "[%s] Analyzing %d Python file(s) at %s...",
        spec.name,
        result.file_count,
        result.commit,
    )
    start = time.perf_counter()
    try:
        proc = subprocess.run(  # noqa: S603
            cmd,
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=opts.timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        result.elapsed_sec = time.perf_counter() - start
        result.error = f"Timed out after {opts.timeout}s"
        logger.warning("[%s] %s", spec.name, result.error)
        return result

    result.elapsed_sec = time.perf_counter() - start
    result.exit_code = proc.returncode

    if proc.returncode not in {0, 1} or (not proc.stdout.strip() and proc.stderr.strip()):
        stderr_lines = [ln for ln in proc.stderr.strip().splitlines() if ln.strip()]
        summary = " | ".join(stderr_lines[-3:]) if stderr_lines else proc.stdout.strip()[:400]
        result.error = f"Exited with code {proc.returncode}: {summary}"
        logger.warning("[%s] %s", spec.name, result.error)
        return result

    try:
        parsed = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        result.error = f"Invalid JSON output: {exc}"
        logger.warning("[%s] %s", spec.name, result.error)
        return result

    result.verifier_used = parsed.get("verifier")
    result.candidates = _enrich_items(parsed.get("candidates", []), repo_dir, spec.name)
    result.rejected = _enrich_items(parsed.get("rejected", []), repo_dir, spec.name)
    logger.info(
        "[%s] Done in %.2fs: %d candidate(s), %d rejected",
        spec.name,
        result.elapsed_sec,
        len(result.candidates),
        len(result.rejected),
    )
    return result


def _ensure_repo_checkout(
    spec: RepoSpec,
    repos_cache_dir: Path,
    git_exe: str,
    *,
    update: bool,
) -> Path:
    """Return local path for `spec`, cloning or updating in `.corpus/repos/` as needed."""
    if spec.local is not None:
        return (REPO_ROOT / spec.local).resolve()

    if spec.url is None:
        msg = f"Repo {spec.name!r} must specify either `url` or `local`"
        raise RuntimeError(msg)

    dest = repos_cache_dir / spec.name
    if not dest.exists():
        cmd = [git_exe, "clone", "--depth", "1"]
        if spec.ref is not None:
            cmd.extend(["--branch", spec.ref])
        cmd.extend([spec.url, str(dest)])
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
        if proc.returncode != 0:
            msg = f"git clone failed: {proc.stderr.strip()}"
            raise RuntimeError(msg)
        return dest

    if update:
        fetch_cmd = [git_exe, "fetch", "--depth", "1", "origin"]
        if spec.ref is not None:
            fetch_cmd.append(spec.ref)
        proc = subprocess.run(fetch_cmd, cwd=dest, capture_output=True, text=True, check=False)  # noqa: S603
        if proc.returncode != 0:
            msg = f"git fetch failed: {proc.stderr.strip()}"
            raise RuntimeError(msg)
        reset = subprocess.run(  # noqa: S603
            [git_exe, "reset", "--hard", "FETCH_HEAD"],
            cwd=dest,
            capture_output=True,
            text=True,
            check=False,
        )
        if reset.returncode != 0:
            msg = f"git reset failed: {reset.stderr.strip()}"
            raise RuntimeError(msg)
    return dest


def _git_short_sha(repo_dir: Path, git_exe: str) -> str:
    """Return the short HEAD commit SHA for a git repository."""
    proc = subprocess.run(  # noqa: S603
        [git_exe, "rev-parse", "--short", "HEAD"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def _count_py_files(paths: list[Path]) -> int:
    """Count Python source files under the given target paths."""
    total = 0
    for path in paths:
        if path.is_file() and path.suffix == ".py":
            total += 1
        elif path.is_dir():
            total += sum(1 for _ in path.rglob("*.py"))
    return total


def _build_check_command(
    tool_exe: str,
    spec: RepoSpec,
    target_paths: list[Path],
    opts: SweepOptions,
) -> list[str]:
    """Construct the `py-type-narrower check` command line for one repo."""
    cmd = [
        tool_exe,
        "check",
        *[str(p) for p in target_paths],
        "--format",
        "json",
        "--min-confidence",
        str(opts.min_confidence),
    ]
    if spec.closed_world:
        cmd.append("--closed-world")
    if opts.verify:
        cmd.extend(["--verify", "--verifier", opts.verifier])
        if opts.verify_command is not None:
            cmd.extend(["--verify-command", opts.verify_command])
    if opts.tier is not None:
        cmd.extend(["--tier", str(opts.tier)])
    if opts.select is not None:
        cmd.extend(["--select", opts.select])
    if opts.ignore is not None:
        cmd.extend(["--ignore", opts.ignore])
    if opts.scaffold_typeddict:
        cmd.append("--scaffold-typeddict")
    return cmd


def _enrich_items(
    items: list[dict[str, Any]],
    repo_dir: Path,
    repo_name: str,
) -> list[dict[str, Any]]:
    """Attach repository name and source code context snippet to each finding."""
    file_cache: dict[str, list[str]] = {}
    enriched: list[dict[str, Any]] = []
    for item in items:
        copy = dict(item)
        copy["repo"] = repo_name
        rel_file = str(item.get("file", ""))
        line_no = int(item.get("line", 1))
        copy["snippet"] = _extract_snippet(repo_dir, rel_file, line_no, file_cache)
        enriched.append(copy)
    return enriched


def _extract_snippet(
    repo_dir: Path,
    rel_file: str,
    line_no: int,
    file_cache: dict[str, list[str]],
    *,
    context: tuple[int, int] = (2, 7),
) -> str:
    """Extract a numbered source snippet around `line_no` for quick manual triage."""
    if rel_file not in file_cache:
        file_path = repo_dir / rel_file
        try:
            file_cache[rel_file] = file_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        except OSError:
            file_cache[rel_file] = []

    lines = file_cache[rel_file]
    if not lines:
        return ""

    before, after = context
    start = max(1, line_no - before)
    end = min(len(lines), line_no + after)
    width = len(str(end))
    rendered: list[str] = []
    for idx in range(start, end + 1):
        marker = ">" if idx == line_no else " "
        rendered.append(f"{marker} {idx:>{width}} | {lines[idx - 1]}")
    return "\n".join(rendered)


def _build_run_payload(
    opts: SweepOptions,
    results: list[RepoRunResult],
    git_exe: str,
) -> dict[str, Any]:
    """Assemble the top-level JSON payload persisted in `results.json`."""
    return {
        "timestamp_utc": datetime.now(tz=UTC).isoformat(),
        "tool_version": __version__,
        "tool_commit": _git_short_sha(REPO_ROOT, git_exe),
        "options": {
            "group": opts.group,
            "repos": list(opts.repos),
            "verify": opts.verify,
            "verifier": opts.verifier if opts.verify else None,
            "min_confidence": opts.min_confidence,
            "tier": opts.tier,
            "select": opts.select,
            "ignore": opts.ignore,
        },
        "repos": [r.to_dict() for r in results],
    }


def _render_markdown_report(payload: dict[str, Any], *, max_per_rule: int) -> str:
    """Render a triage-ready Markdown report grouped by overview, rejections, and rule."""
    repos = payload["repos"]
    opts = payload["options"]
    options_summary = (
        f"- **Options:** `verify={opts['verify']}`, `verifier={opts['verifier']}`, "
        f"`min_confidence={opts['min_confidence']}`, `group={opts['group']}`"
    )
    lines: list[str] = [
        "# py-type-narrower Corpus Report",
        "",
        f"- **Timestamp (UTC):** `{payload['timestamp_utc']}`",
        f"- **Tool:** `py-type-narrower {payload['tool_version']}` (`{payload['tool_commit']}`)",
        options_summary,
        "",
        "## Overview",
        "",
    ]
    lines.extend(_render_overview_table(repos))
    lines.extend(_render_errors_section(repos))
    lines.extend(_render_rejections_section(repos))
    lines.extend(_render_candidates_by_rule_section(repos, max_per_rule=max_per_rule))
    return "\n".join(lines).rstrip() + "\n"


def _render_overview_table(repos: list[dict[str, Any]]) -> list[str]:
    """Build the repo-by-rule summary Markdown table."""
    used_rules = _collect_present_rules(repos)
    header_cols = ["Repo", "Commit", "Mode", "Files", "Time", "Cands", "Rej", *used_rules, "Status"]
    sep_cols = ["---"] * len(header_cols)
    lines = [
        "| " + " | ".join(header_cols) + " |",
        "| " + " | ".join(sep_cols) + " |",
    ]
    total_cands = 0
    total_rej = 0
    total_counts: Counter[str] = Counter()

    for repo in repos:
        cands = repo["candidates"]
        rej = repo["rejected"]
        counts = Counter(c["code"] for c in cands)
        total_cands += len(cands)
        total_rej += len(rej)
        total_counts.update(counts)

        mode = "closed" if repo["closed_world"] else "open"
        status = "ERROR" if repo["error"] else "ok"
        row = [
            f"`{repo['name']}`",
            f"`{repo['commit']}`",
            mode,
            str(repo["file_count"]),
            f"{repo['elapsed_sec']:.1f}s",
            str(len(cands)),
            str(len(rej)),
            *[str(counts.get(code, 0)) for code in used_rules],
            status,
        ]
        lines.append("| " + " | ".join(row) + " |")

    total_row = [
        "**Total**",
        "",
        "",
        str(sum(r["file_count"] for r in repos)),
        f"{sum(r['elapsed_sec'] for r in repos):.1f}s",
        f"**{total_cands}**",
        f"**{total_rej}**",
        *[f"**{total_counts.get(code, 0)}**" for code in used_rules],
        "",
    ]
    lines.append("| " + " | ".join(total_row) + " |")
    lines.append("")
    return lines


def _collect_present_rules(repos: list[dict[str, Any]]) -> list[str]:
    """Return rule codes present across any candidate or rejection in canonical order."""
    seen = {item["code"] for repo in repos for item in (*repo["candidates"], *repo["rejected"])}
    ordered = [r for r in RULE_ORDER if r in seen]
    extras = sorted(seen - set(RULE_ORDER))
    return ordered + extras


def _render_errors_section(repos: list[dict[str, Any]]) -> list[str]:
    """Render any repository checkout, timeout, or execution failures."""
    failed = [r for r in repos if r["error"]]
    if not failed:
        return []
    lines = ["## Errors & Timeouts", ""]
    lines.extend(f"- **`{repo['name']}`**: {repo['error']}" for repo in failed)
    lines.append("")
    return lines


def _render_rejections_section(repos: list[dict[str, Any]]) -> list[str]:
    """Render verifier-rejected candidates grouped by rule code (FP goldmine)."""
    rejected = [item for repo in repos for item in repo["rejected"]]
    lines = [f"## Rejected by Soundness Verifier ({len(rejected)})", ""]
    if not rejected:
        lines.append("_None._")
        lines.append("")
        return lines

    by_rule = _group_by_rule(rejected)
    for code, items in by_rule.items():
        lines.append(f"### {code} ({len(items)} rejected)")
        lines.append("")
        for item in items:
            lines.extend(_render_finding_block(item, include_blocking=True))
    return lines


def _render_candidates_by_rule_section(
    repos: list[dict[str, Any]],
    *,
    max_per_rule: int,
) -> list[str]:
    """Render accepted/verified candidates grouped by rule code for pattern discovery."""
    candidates = [item for repo in repos for item in repo["candidates"]]
    lines = [f"## Candidates by Rule ({len(candidates)})", ""]
    if not candidates:
        lines.append("_None._")
        lines.append("")
        return lines

    by_rule = _group_by_rule(candidates)
    for code, items in by_rule.items():
        sorted_items = sorted(
            items,
            key=lambda x: (-int(x.get("score", 0)), str(x.get("repo", "")), str(x.get("file", ""))),
        )
        shown = sorted_items[:max_per_rule]
        suffix = (
            f" — showing top {len(shown)} of {len(sorted_items)}"
            if len(shown) < len(sorted_items)
            else ""
        )
        lines.append(f"### {code} ({len(sorted_items)}{suffix})")
        lines.append("")
        for item in shown:
            lines.extend(_render_finding_block(item, include_blocking=False))
    return lines


def _group_by_rule(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group findings by rule code in canonical NT101..NT205 order."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[str(item["code"])].append(item)
    ordered: dict[str, list[dict[str, Any]]] = {}
    for code in RULE_ORDER:
        if code in grouped:
            ordered[code] = grouped[code]
    for code in sorted(set(grouped) - set(RULE_ORDER)):
        ordered[code] = grouped[code]
    return ordered


def _render_finding_block(item: dict[str, Any], *, include_blocking: bool) -> list[str]:
    """Render a single candidate or rejection entry with evidence and code snippet."""
    ver = item.get("verification", "unverified")
    loc = f"`{item['repo']}` `{item['file']}:{item['line']}`"
    header = (
        f"- **{loc}** (`{item['symbol']}` · `{item['target_kind']}` · "
        f"score={item['score']} · {ver}): "
        f"`{item['current_type']}` -> `{item['suggested_type']}`"
    )
    lines = [header, f"  - {item['message']}"]
    lines.extend(f"  - evidence ({ev['kind']}): {ev['detail']}" for ev in item.get("evidence", []))
    if include_blocking:
        lines.extend(
            f"  - **blocked by:** `{diag}`" for diag in item.get("blocking_diagnostics", [])
        )
    snippet = item.get("snippet", "")
    if snippet:
        lines.append("  ```python")
        lines.extend(f"  {sn_line}" for sn_line in snippet.splitlines())
        lines.append("  ```")
    lines.append("")
    return lines


def _log_summary_table(results: list[RepoRunResult]) -> None:
    """Log a concise terminal summary after all repos finish."""
    logger.info("")
    logger.info("%-20s %6s %7s %7s %7s %s", "Repo", "Files", "Time", "Cands", "Rej", "Status")
    logger.info("-" * 64)
    for r in results:
        status = f"ERR: {r.error}" if r.error else "ok"
        logger.info(
            "%-20s %6d %6.1fs %7d %7d %s",
            r.spec.name,
            r.file_count,
            r.elapsed_sec,
            len(r.candidates),
            len(r.rejected),
            status,
        )
    logger.info("")


def _print_comparison(prev_path: Path, current: dict[str, Any]) -> None:
    """Diff candidates and rejections against a previous `results.json`."""
    prev = json.loads(prev_path.read_text(encoding="utf-8"))
    prev_cands = _index_findings(prev, "candidates")
    curr_cands = _index_findings(current, "candidates")
    prev_rej = _index_findings(prev, "rejected")
    curr_rej = _index_findings(current, "rejected")

    added = sorted(curr_cands.keys() - prev_cands.keys())
    removed = sorted(prev_cands.keys() - curr_cands.keys())
    newly_rejected = sorted(curr_rej.keys() - prev_rej.keys())
    fixed_rejections = sorted(prev_rej.keys() - curr_rej.keys())

    logger.info("=== Comparison vs %s ===", prev_path)
    logger.info(
        "Candidates: +%d added, -%d removed | Rejections: +%d new, -%d resolved",
        len(added),
        len(removed),
        len(newly_rejected),
        len(fixed_rejections),
    )
    for label, keys in (
        ("+ Added candidate", added[:20]),
        ("- Removed candidate", removed[:20]),
        ("+ New rejection", newly_rejected[:20]),
        ("- Resolved rejection", fixed_rejections[:20]),
    ):
        for key in keys:
            logger.info("  %s: %s", label, " | ".join(key))


def _index_findings(
    payload: dict[str, Any],
    bucket: str,
) -> dict[tuple[str, ...], dict[str, Any]]:
    """Index findings by stable identity key for run-to-run diffing."""
    out: dict[tuple[str, ...], dict[str, Any]] = {}
    for repo in payload.get("repos", []):
        repo_name = str(repo.get("name", ""))
        for item in repo.get(bucket, []):
            key = (
                repo_name,
                str(item.get("file", "")),
                str(item.get("symbol", "")),
                str(item.get("target_name", "")),
                str(item.get("code", "")),
                str(item.get("suggested_type", "")),
            )
            out[key] = item
    return out


if __name__ == "__main__":
    raise SystemExit(main())
