"""Analyze and narrow overly broad Python type annotations."""

from py_type_narrower.config import NarrowerConfig, load_config
from py_type_narrower.engine import RunOptions, analyze_paths
from py_type_narrower.models import (
    AnalysisResult,
    NarrowingCandidate,
    RuleCode,
    TargetKind,
    Tier,
    VerificationStatus,
)
from py_type_narrower.scorer import ConfidenceBand, confidence_band

__version__ = "0.1.0"

__all__ = [
    "AnalysisResult",
    "ConfidenceBand",
    "NarrowerConfig",
    "NarrowingCandidate",
    "RuleCode",
    "RunOptions",
    "TargetKind",
    "Tier",
    "VerificationStatus",
    "__version__",
    "analyze_paths",
    "confidence_band",
    "load_config",
]
