"""Core domain models for type narrowing analysis and verification."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class RuleCode(StrEnum):
    """Identifiers for type-narrowing rules."""

    NT101 = "NT101"  # Tier 1: Guarded str/int parameter -> Literal[...]
    NT102 = "NT102"  # Tier 1: Return T | None -> T when None is never returned
    NT103 = "NT103"  # Tier 1: Return str/int -> Literal[...] when all returns are literals
    NT201 = "NT201"  # Tier 2: Call-site str/int parameter -> Literal[...]
    NT202 = "NT202"  # Tier 2: Call-site T | None parameter -> T


class Tier(IntEnum):
    """Confidence tier of a narrowing rule."""

    LOCAL = 1
    WHOLE_REPO = 2


class TargetKind(StrEnum):
    """Kind of annotation target being narrowed."""

    PARAMETER = "parameter"
    RETURN = "return"


class VerificationStatus(StrEnum):
    """Soundness verification outcome for a candidate."""

    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    REJECTED = "rejected"


@dataclass(frozen=True)
class EvidenceItem:
    """Single piece of provenance explaining why a narrowing was suggested."""

    kind: str
    detail: str
    file: Path | None = None
    line: int | None = None


@dataclass(frozen=True)
class CheckerDiagnostic:
    """Normalized diagnostic emitted by an external type checker."""

    file: str
    line: int
    col: int
    severity: str
    message: str
    rule: str = ""

    def signature(self) -> tuple[str, int, str, str]:
        """Return a stable tuple for diffing diagnostics across runs."""
        return (self.file, self.line, self.rule, self.message)

    def format_short(self) -> str:
        """Render a concise single-line diagnostic summary."""
        rule_part = f" [{self.rule}]" if self.rule else ""
        return f"{self.file}:{self.line}:{self.col}: {self.message}{rule_part}"


@dataclass
class NarrowingCandidate:
    """Proposed narrowing of a single parameter or return type annotation."""

    code: RuleCode
    tier: Tier
    file: Path
    line: int
    col: int
    symbol: str
    target_kind: TargetKind
    target_name: str | None
    current_type: str
    suggested_type: str
    needs_literal_import: bool
    message: str
    evidence: list[EvidenceItem] = field(default_factory=list)
    score: int = 0
    is_public: bool = False
    is_single_constant: bool = False
    drop_none_default: bool = False
    verification: VerificationStatus = VerificationStatus.UNVERIFIED
    blocking_diagnostics: list[str] = field(default_factory=list)

    @property
    def candidate_key(self) -> tuple[Path, int, str, TargetKind, str | None, RuleCode]:
        """Return a deterministic unique key for this candidate."""
        return (
            self.file,
            self.line,
            self.symbol,
            self.target_kind,
            self.target_name,
            self.code,
        )


@dataclass
class AnalysisResult:
    """Aggregate output of an analysis or fix run."""

    candidates: list[NarrowingCandidate] = field(default_factory=list)
    rejected: list[NarrowingCandidate] = field(default_factory=list)
    applied: list[NarrowingCandidate] = field(default_factory=list)
    diffs: dict[Path, str] = field(default_factory=dict)
    passes_run: int = 1
    verifier_used: str | None = None
