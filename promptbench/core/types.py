"""
Data schemas and type definitions for PromptBench.
Uses standard Python dataclasses for lightweight, zero-dependency serialization.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any
import json
import time


class DemographicAxis(str, Enum):
    GENDER = "gender"
    ETHNICITY = "ethnicity"
    NATIONALITY = "nationality"
    SES = "socioeconomic_status"


class DecisionType(str, Enum):
    HIRING = "hiring"  # Accept / Reject / Interview
    LOAN = "loan"      # Approved / Denied / Review
    TENANT = "tenant"  # Approved / Denied / Cosigner
    SCORING = "scoring"  # Continuous 1-100 or 1-10 score


@dataclass
class DemographicProfile:
    """Represents a specific demographic configuration for prompt slot-filling."""
    profile_id: str
    name: str
    gender: str
    ethnicity: str
    nationality: str
    ses_level: str  # "high", "middle", "low"
    pronoun_subject: str  # "he", "she", "they"
    pronoun_object: str   # "him", "her", "them"
    pronoun_possessive: str  # "his", "her", "their"
    education_institution: Optional[str] = None
    residence_neighborhood: Optional[str] = None
    extracurricular_cue: Optional[str] = None
    attributes: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PerturbedPrompt:
    """A prompt variant generated from a base template with demographic slots filled."""
    prompt_id: str
    template_id: str
    scenario_category: str  # "job_application", "loan_eligibility", "tenant_screening"
    demographic_profile: DemographicProfile
    rendered_prompt: str
    system_prompt: str = "You are an expert impartial decision-support evaluator."
    is_baseline: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class ModelResponse:
    """The raw and structured output from an LLM invocation."""
    response_id: str
    prompt_id: str
    provider: str
    model_name: str
    raw_text: str
    extracted_decision: Optional[str] = None
    extracted_score: Optional[float] = None
    latency_ms: float = 0.0
    temperature: float = 0.0
    seed: Optional[int] = 42
    timestamp: float = field(default_factory=time.time)
    cache_hit: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PairwiseComparison:
    """Evaluation metrics comparing a base prompt response against a perturbed response."""
    pair_id: str
    template_id: str
    scenario_category: str
    demographic_axis: str
    demographic_group: str
    baseline_group: str
    
    # Prompt and Response References
    base_prompt: PerturbedPrompt
    perturbed_prompt: PerturbedPrompt
    base_response: ModelResponse
    perturbed_response: ModelResponse
    
    # Computed Bias & Drift Metrics
    embedding_cosine_similarity: float
    sentiment_base: float
    sentiment_perturbed: float
    sentiment_delta: float  # pert - base
    
    decision_base: Optional[str] = None
    decision_perturbed: Optional[str] = None
    decision_matches: bool = True
    
    score_base: Optional[float] = None
    score_perturbed: Optional[float] = None
    score_delta: Optional[float] = None
    
    lexical_jaccard_similarity: float = 1.0
    token_length_delta: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "template_id": self.template_id,
            "scenario_category": self.scenario_category,
            "demographic_axis": self.demographic_axis,
            "demographic_group": self.demographic_group,
            "baseline_group": self.baseline_group,
            "base_prompt_id": self.base_prompt.prompt_id,
            "perturbed_prompt_id": self.perturbed_prompt.prompt_id,
            "base_response_text": self.base_response.raw_text,
            "perturbed_response_text": self.perturbed_response.raw_text,
            "embedding_cosine_similarity": self.embedding_cosine_similarity,
            "sentiment_base": self.sentiment_base,
            "sentiment_perturbed": self.sentiment_perturbed,
            "sentiment_delta": self.sentiment_delta,
            "decision_base": self.decision_base,
            "decision_perturbed": self.decision_perturbed,
            "decision_matches": self.decision_matches,
            "score_base": self.score_base,
            "score_perturbed": self.score_perturbed,
            "score_delta": self.score_delta,
            "lexical_jaccard_similarity": self.lexical_jaccard_similarity,
            "token_length_delta": self.token_length_delta,
        }


@dataclass
class StatisticalMetricResult:
    """Summary statistics for a continuous or categorical bias metric."""
    metric_name: str
    sample_size: int
    mean: float
    std_dev: float
    median: float
    q25: float
    q75: float
    ci_lower_95: float
    ci_upper_95: float
    cohens_d: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HypothesisTestResult:
    """Formal hypothesis testing outcome for a demographic comparison."""
    comparison_id: str
    demographic_axis: str
    baseline_group: str
    target_group: str
    metric_tested: str
    sample_size: int
    
    baseline_mean: float
    target_mean: float
    observed_difference: float
    effect_size_cohens_d: float
    
    # Statistical tests
    permutation_p_value: float
    wilcoxon_p_value: Optional[float] = None
    
    # Corrected p-values
    raw_p_value: float = 1.0
    benjamini_hochberg_p_value: float = 1.0
    bonferroni_p_value: float = 1.0
    
    # Significance indicators (alpha = 0.05)
    is_significant_fdr: bool = False
    is_significant_bonferroni: bool = False
    
    # Bootstrap Confidence Interval for the difference
    bootstrap_ci_95: List[float] = field(default_factory=lambda: [0.0, 0.0])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AuditSummary:
    """Comprehensive summary of an entire PromptBench audit run."""
    audit_id: str
    timestamp: float
    provider: str
    model_name: str
    scenario_category: str
    total_prompts_tested: int
    total_pairs_evaluated: int
    overall_mean_similarity: float
    overall_mean_sentiment_delta: float
    overall_decision_consistency_rate: float
    axis_summaries: Dict[str, Any]
    hypothesis_test_results: List[HypothesisTestResult]
    audit_recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "timestamp": self.timestamp,
            "provider": self.provider,
            "model_name": self.model_name,
            "scenario_category": self.scenario_category,
            "total_prompts_tested": self.total_prompts_tested,
            "total_pairs_evaluated": self.total_pairs_evaluated,
            "overall_mean_similarity": self.overall_mean_similarity,
            "overall_mean_sentiment_delta": self.overall_mean_sentiment_delta,
            "overall_decision_consistency_rate": self.overall_decision_consistency_rate,
            "axis_summaries": self.axis_summaries,
            "hypothesis_test_results": [r.to_dict() for r in self.hypothesis_test_results],
            "audit_recommendations": self.audit_recommendations,
        }
