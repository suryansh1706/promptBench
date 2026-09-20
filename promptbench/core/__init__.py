"""
Core modules for PromptBench: Configuration, Data Types, and Deterministic Caching.
"""

from .config import PromptBenchConfig, get_default_config
from .types import (
    DemographicAxis,
    DemographicProfile,
    PerturbedPrompt,
    ModelResponse,
    PairwiseComparison,
    StatisticalMetricResult,
    HypothesisTestResult,
    AuditSummary,
)
from .cache import ResponseCache

__all__ = [
    "PromptBenchConfig",
    "get_default_config",
    "DemographicAxis",
    "DemographicProfile",
    "PerturbedPrompt",
    "ModelResponse",
    "PairwiseComparison",
    "StatisticalMetricResult",
    "HypothesisTestResult",
    "AuditSummary",
    "ResponseCache",
]
