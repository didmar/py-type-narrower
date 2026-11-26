"""Configuration loading from pyproject.toml and CLI overrides."""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from py_type_narrower.models import RuleCode

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_EXCLUDES: tuple[str, ...] = (
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
    ".tox",
    ".nox",
    "node_modules",
)

ALL_RULES: frozenset[RuleCode] = frozenset(RuleCode)


@dataclass
class NarrowerConfig:
    """Runtime configuration for py-type-narrower."""

    max_tier: int = 2
    min_confidence: int = 0
    closed_world: bool = False
    max_literals: int = 8
    min_call_sites: int = 2
    max_passes: int = 1
    select: set[RuleCode] = field(default_factory=lambda: set(ALL_RULES))
    ignore: set[RuleCode] = field(default_factory=set)
    exclude: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDES))
    verify: bool = False
    verifier: str = "auto"
    verify_command: str | None = None
    include_test_files: bool = False
    index_project_root: bool = True

    def active_rules(self) -> set[RuleCode]:
        """Return the set of enabled rule codes after applying ignores."""
        return self.select - self.ignore


def load_config(project_root: Path) -> NarrowerConfig:
    """Load configuration from pyproject.toml under project_root if present."""
    pyproject = project_root / "pyproject.toml"
    if not pyproject.is_file():
        return NarrowerConfig()

    try:
        with pyproject.open("rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        logger.warning("Failed to parse %s: %s", pyproject, exc)
        return NarrowerConfig()

    tool_section = data.get("tool", {})
    section = tool_section.get("py-type-narrower", {})
    if not isinstance(section, dict):
        return NarrowerConfig()

    return _parse_config_section(section)


def _parse_config_section(section: dict[str, object]) -> NarrowerConfig:
    """Build a NarrowerConfig from a parsed TOML dictionary."""
    cfg = NarrowerConfig()
    _apply_scalar_options(cfg, section)
    _apply_collection_options(cfg, section)
    return cfg


def _apply_scalar_options(cfg: NarrowerConfig, section: dict[str, object]) -> None:
    """Apply scalar fields from a TOML configuration table."""
    scalar_spec: tuple[tuple[str, str, type], ...] = (
        ("max-tier", "max_tier", int),
        ("min-confidence", "min_confidence", int),
        ("closed-world", "closed_world", bool),
        ("max-literals", "max_literals", int),
        ("min-call-sites", "min_call_sites", int),
        ("max-passes", "max_passes", int),
        ("verify", "verify", bool),
        ("verifier", "verifier", str),
        ("verify-command", "verify_command", str),
        ("include-test-files", "include_test_files", bool),
        ("index-project-root", "index_project_root", bool),
    )
    for toml_key, attr, expected_type in scalar_spec:
        val = section.get(toml_key)
        if expected_type is int and isinstance(val, bool):
            continue
        if isinstance(val, expected_type):
            setattr(cfg, attr, val)


def _apply_collection_options(cfg: NarrowerConfig, section: dict[str, object]) -> None:
    """Apply list/set fields from a TOML configuration table."""
    if isinstance(select_raw := section.get("select"), list):
        cfg.select = _parse_rule_codes(select_raw)
    if isinstance(ignore_raw := section.get("ignore"), list):
        cfg.ignore = _parse_rule_codes(ignore_raw)
    if isinstance(exclude_raw := section.get("exclude"), list):
        cfg.exclude = [item for item in exclude_raw if isinstance(item, str)]


def _parse_rule_codes(raw_codes: list[object]) -> set[RuleCode]:
    """Convert a list of raw rule code strings into valid RuleCode values."""
    codes: set[RuleCode] = set()
    for item in raw_codes:
        if not isinstance(item, str):
            continue
        upper = item.strip().upper()
        try:
            codes.add(RuleCode(upper))
        except ValueError:
            logger.warning("Ignoring unknown rule code in config: %s", item)
    return codes
