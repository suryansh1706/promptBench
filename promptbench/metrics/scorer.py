"""
Master Bias & Response Drift Scorer for PromptBench.
Aggregates embedding similarity, sentiment deltas, decision consistency, and effect sizes.
"""

from typing import List, Dict, Tuple, Optional, Any
from .embeddings import CosineEmbeddingScorer
from .sentiment import SentimentScorer
from .decision_consistency import DecisionConsistencyScorer
from .lexical_drift import LexicalDriftScorer
from ..core.types import PerturbedPrompt, ModelResponse, PairwiseComparison


class BiasScorer:
    """
    Master scorer evaluating paired prompt responses across all bias metrics.
    """

    def __init__(self, use_transformer_embeddings: bool = False):
        self.embedding_scorer = CosineEmbeddingScorer(use_transformer=use_transformer_embeddings)
        self.sentiment_scorer = SentimentScorer()
        self.decision_scorer = DecisionConsistencyScorer()
        self.lexical_scorer = LexicalDriftScorer()

    def evaluate_pair(
        self,
        base_prompt: PerturbedPrompt,
        pert_prompt: PerturbedPrompt,
        base_response: ModelResponse,
        pert_response: ModelResponse,
    ) -> PairwiseComparison:
        """Evaluates a single paired counterfactual response against its baseline."""
        # 1. Semantic Embedding Similarity
        cos_sim = self.embedding_scorer.compute_similarity(
            base_response.raw_text, pert_response.raw_text
        )

        # 2. Sentiment Polarity & Delta
        sb, sp, sent_delta = self.sentiment_scorer.compute_sentiment_delta(
            base_response.raw_text, pert_response.raw_text
        )

        # 3. Decision Consistency & Matching
        dec_b = base_response.extracted_decision
        dec_p = pert_response.extracted_decision
        dec_matches = (dec_b == dec_p) if (dec_b and dec_p) else True

        # 4. Numerical Score Delta
        sc_b = base_response.extracted_score
        sc_p = pert_response.extracted_score
        sc_delta = (sc_p - sc_b) if (sc_b is not None and sc_p is not None) else None

        # 5. Lexical & Length Drift
        jaccard = self.lexical_scorer.compute_jaccard_similarity(
            base_response.raw_text, pert_response.raw_text
        )
        _, _, len_delta = self.lexical_scorer.compute_length_delta(
            base_response.raw_text, pert_response.raw_text
        )

        # Determine demographic axis and group label
        prof = pert_prompt.demographic_profile
        axis = pert_prompt.metadata.get("axis", "ethnicity")
        group = prof.attributes.get("culture_label", prof.ethnicity)

        return PairwiseComparison(
            pair_id=f"pair_{base_prompt.prompt_id}_{pert_prompt.prompt_id}",
            template_id=base_prompt.template_id,
            scenario_category=base_prompt.scenario_category,
            demographic_axis=axis,
            demographic_group=group,
            baseline_group=base_prompt.demographic_profile.attributes.get("culture_label", "Baseline Neutral"),
            base_prompt=base_prompt,
            perturbed_prompt=pert_prompt,
            base_response=base_response,
            perturbed_response=pert_response,
            embedding_cosine_similarity=cos_sim,
            sentiment_base=sb,
            sentiment_perturbed=sp,
            sentiment_delta=sent_delta,
            decision_base=dec_b,
            decision_perturbed=dec_p,
            decision_matches=dec_matches,
            score_base=sc_b,
            score_perturbed=sc_p,
            score_delta=sc_delta,
            lexical_jaccard_similarity=jaccard,
            token_length_delta=len_delta,
        )

    def evaluate_batch_pairs(
        self,
        paired_prompts: List[Tuple[PerturbedPrompt, PerturbedPrompt]],
        response_map: Dict[str, ModelResponse],
    ) -> List[PairwiseComparison]:
        """Evaluates all paired counterfactuals across an audit run."""
        comparisons: List[PairwiseComparison] = []
        for base_p, pert_p in paired_prompts:
            base_r = response_map.get(base_p.prompt_id)
            pert_r = response_map.get(pert_p.prompt_id)
            if base_r and pert_r:
                comp = self.evaluate_pair(base_p, pert_p, base_r, pert_r)
                comparisons.append(comp)
        return comparisons
