"""Command-line interface for py-type-narrower."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from py_type_narrower import __version__
from py_type_narrower.config import NarrowerConfig, load_config
from py_type_narrower.engine import RunOptions, analyze_paths
from py_type_narrower.models import RuleCode
from py_type_narrower.reporting import format_diffs, format_report


def main(argv: list[str] | None = None) -> int:
    """CLI entry point returning POSIX exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    paths = [Path(p) for p in args.paths] or [Path()]
    project_root = _infer_project_root(paths)
    config = load_config(project_root)
    _apply_cli_overrides(config, args)

    command = args.command
    apply_fixes = command == "fix"
    show_diff = command == "diff" or getattr(args, "diff", False)
    unsafe_fixes = getattr(args, "unsafe_fixes", False)

    try:
        result = analyze_paths(
            paths=paths,
            project_root=project_root,
            config=config,
            options=RunOptions(
                apply_fixes=apply_fixes,
                unsafe_fixes=unsafe_fixes,
                compute_diffs=show_diff,
            ),
        )
    except (ValueError, RuntimeError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2

    if show_diff:
        diff_output = format_diffs(result)
        if diff_output:
            sys.stdout.write(diff_output)
        if command == "diff":
            return 1 if result.candidates else 0

    report = format_report(
        result,
        project_root,
        output_format=getattr(args, "format", "text"),
        verbose=getattr(args, "verbose", False),
        show_rejected=getattr(args, "show_rejected", False),
    )
    sys.stdout.write(report + "\n")

    if command == "check":
        return 1 if result.candidates else 0
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse parser with check, diff, and fix subcommands."""
    parser = argparse.ArgumentParser(
        prog="py-type-narrower",
        description="Analyze and narrow overly broad Python type annotations.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)

    check_p = subparsers.add_parser("check", help="Report type-narrowing candidates")
    _add_common_options(check_p)
    check_p.add_argument("--diff", action="store_true", help="Also print unified diff")

    diff_p = subparsers.add_parser("diff", help="Show unified diff of proposed narrowings")
    _add_common_options(diff_p)
    diff_p.add_argument(
        "--unsafe-fixes",
        action="store_true",
        help="Include Tier 2 whole-repo candidates in the diff",
    )

    fix_p = subparsers.add_parser("fix", help="Apply narrowing fixes in place")
    _add_common_options(fix_p)
    fix_p.add_argument(
        "--unsafe-fixes",
        action="store_true",
        help="Also apply Tier 2 whole-repo candidates (default applies Tier 1 only)",
    )
    fix_p.add_argument("--diff", action="store_true", help="Print unified diff of applied changes")

    return parser


def _add_common_options(subparser: argparse.ArgumentParser) -> None:
    """Attach shared analysis, scoring, verification, and formatting flags."""
    subparser.add_argument("paths", nargs="*", help="Files or directories to analyze (default: .)")
    subparser.add_argument(
        "--tier",
        type=int,
        choices=[1, 2],
        default=None,
        help="Maximum rule tier to run (1=local only, 2=local + whole-repo)",
    )
    subparser.add_argument(
        "--min-confidence",
        type=int,
        default=None,
        help="Minimum confidence score (0-100) required to report a candidate",
    )
    subparser.add_argument(
        "--closed-world",
        action="store_true",
        default=None,
        help="Treat public functions as internal (application mode)",
    )
    subparser.add_argument(
        "--select",
        default=None,
        help="Comma-separated rule codes to enable (e.g. NT101,NT102)",
    )
    subparser.add_argument(
        "--ignore",
        default=None,
        help="Comma-separated rule codes to ignore (e.g. NT201)",
    )
    subparser.add_argument(
        "--exclude",
        action="append",
        default=None,
        help="Additional directory/file name or glob pattern to exclude (repeatable)",
    )
    subparser.add_argument(
        "--no-project-index",
        action="store_false",
        dest="index_project_root",
        default=None,
        help="Only index the explicitly passed paths instead of the full project root for Tier 2",
    )
    subparser.add_argument(
        "--max-literals",
        type=int,
        default=None,
        help="Maximum distinct literal values for Literal[...] narrowing",
    )
    subparser.add_argument(
        "--max-passes",
        type=int,
        default=None,
        help="Number of iterative fixpoint passes (default: 1)",
    )
    subparser.add_argument(
        "--verify",
        action="store_true",
        default=None,
        help="Verify candidates with a type checker before reporting/applying",
    )
    subparser.add_argument(
        "--verifier",
        choices=["auto", "ty", "mypy", "pyright", "basedpyright", "custom"],
        default=None,
        help="Type checker backend for soundness verification (default: auto)",
    )
    subparser.add_argument(
        "--verify-command",
        default=None,
        help="Custom command for `--verifier custom` (e.g. 'uv run mypy src')",
    )
    subparser.add_argument(
        "--format",
        choices=["text", "json", "sarif"],
        default="text",
        help="Output format (default: text)",
    )
    subparser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Include evidence provenance and score factors",
    )
    subparser.add_argument(
        "--show-rejected",
        action="store_true",
        help="Show candidates rejected by the soundness verifier and their blocking errors",
    )


def _infer_project_root(paths: list[Path]) -> Path:
    """Infer the repository root by walking up to pyproject.toml or .git."""
    start = paths[0].resolve()
    curr = start if start.is_dir() else start.parent
    for candidate in (curr, *curr.parents):
        if (candidate / "pyproject.toml").is_file() or (candidate / ".git").exists():
            return candidate
    return curr


def _apply_cli_overrides(config: NarrowerConfig, args: argparse.Namespace) -> None:
    """Apply non-None CLI flag values on top of pyproject.toml config."""
    direct_attrs = (
        ("tier", "max_tier"),
        ("min_confidence", "min_confidence"),
        ("max_literals", "max_literals"),
        ("max_passes", "max_passes"),
        ("verifier", "verifier"),
        ("verify_command", "verify_command"),
    )
    for arg_name, cfg_attr in direct_attrs:
        val = getattr(args, arg_name)
        if val is not None:
            setattr(config, cfg_attr, val)
    if args.closed_world is True:
        config.closed_world = True
    if args.verify is True:
        config.verify = True
    if args.select is not None:
        config.select = _parse_csv_codes(args.select)
    if args.ignore is not None:
        config.ignore = _parse_csv_codes(args.ignore)
    if args.exclude is not None:
        config.exclude.extend(args.exclude)
    if args.index_project_root is False:
        config.index_project_root = False


def _parse_csv_codes(raw: str) -> set[RuleCode]:
    """Parse a comma-separated list of rule codes."""
    codes: set[RuleCode] = set()
    for token in raw.split(","):
        cleaned = token.strip().upper()
        if cleaned:
            codes.add(RuleCode(cleaned))
    return codes
