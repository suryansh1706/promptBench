"""
Master Statistical Testing Engine for PromptBench.
Executes paired permutation tests, bootstrap CIs, and Benjamini-Hochberg FDR adjustments.
"""

from typing import List, Dict, Tuple, Optional, Any
from .permutation import PermutationTest
from .bootstrap import BootstrapCI
from .corrections import MultipleTestingCorrection
from .non_parametric import NonParametricTests
from ..core.types import PairwiseComparison, HypothesisTestResult
from ..core.config import StatisticalConfig


class StatisticalTestingEngine:
    """
    Coordinates rigorous hypothesis testing across all demographic comparisons in an audit.
    """

    def __init__(self, config: Optional[StatisticalConfig] = None):
        self.config = config or StatisticalConfig()
        self.perm_test = PermutationTest(resamples=self.config.permutation_resamples)
        self.bootstrap = BootstrapCI(
            resamples=self.config.bootstrap_resamples,
            confidence_level=self.config.confidence_interval_level,
        )

    def analyze_comparisons(
        self, comparisons: List[PairwiseComparison]
    ) -> List[HypothesisTestResult]:
        """
        Runs comprehensive hypothesis tests for each demographic subgroup vs baseline.
        """
        # Group comparisons by (axis, target_group)
        grouped: Dict[Tuple[str, str], List[PairwiseComparison]] = {}
        for c in comparisons:
            key = (c.demographic_axis, c.demographic_group)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(c)

        preliminary_results: List[HypothesisTestResult] = []
        raw_p_values: List[float] = []

        for (axis, group), comps in grouped.items():
            if len(comps) < 2:
                continue

            baseline_label = comps[0].baseline_group
            
            # Extract metrics distributions
            sentiment_deltas = [c.sentiment_delta for c in comps]
            base_scores = [c.score_base for c in comps if c.score_base is not None]
            pert_scores = [c.score_perturbed for c in comps if c.score_perturbed is not None]

            # 1. Permutation Test on Sentiment Delta (H0: mean(delta) = 0)
            obs_diff, perm_p, _ = self.perm_test.test_paired_differences(sentiment_deltas)

            # 2. Bootstrap 95% Confidence Interval for the mean delta
            _, ci_low, ci_high = self.bootstrap.compute_ci_mean(sentiment_deltas)

            # 3. Wilcoxon signed-rank test
            _, wilcox_p = NonParametricTests.wilcoxon_signed_rank(sentiment_deltas)

            # 4. Cohen's d effect size
            if len(base_scores) >= 2 and len(pert_scores) >= 2:
                from ..metrics.decision_consistency import DecisionConsistencyScorer
                effect_d = DecisionConsistencyScorer.compute_cohens_d(base_scores, pert_scores)
            else:
                std_delta = (sum((x - obs_diff)**2 for x in sentiment_deltas) / (len(sentiment_deltas)-1))**0.5 if len(sentiment_deltas) > 1 else 1.0
                effect_d = round(obs_diff / std_delta if std_delta > 0 else 0.0, 4)

            mean_base_sent = sum(c.sentiment_base for c in comps) / len(comps)
            mean_pert_sent = sum(c.sentiment_perturbed for c in comps) / len(comps)

            test_res = HypothesisTestResult(
                comparison_id=f"ht_{axis}_{group.replace(' ', '_').lower()}",
                demographic_axis=axis,
                baseline_group=baseline_label,
                target_group=group,
                metric_tested="Sentiment Delta (Δ Sentiment)",
                sample_size=len(comps),
                baseline_mean=round(mean_base_sent, 4),
                target_mean=round(mean_pert_sent, 4),
                observed_difference=obs_diff,
                effect_size_cohens_d=effect_d,
                permutation_p_value=perm_p,
                wilcoxon_p_value=wilcox_p,
                raw_p_value=perm_p,
                bootstrap_ci_95=[ci_low, ci_high],
            )
            preliminary_results.append(test_res)
            raw_p_values.append(perm_p)

        if not preliminary_results:
            return []

        # 5. Apply Benjamini-Hochberg False Discovery Rate (FDR) and Bonferroni adjustments
        bh_p_vals, bh_sig = MultipleTestingCorrection.benjamini_hochberg(
            raw_p_values, alpha=self.config.alpha_significance_level
        )
        bonf_p_vals, bonf_sig = MultipleTestingCorrection.bonferroni(
            raw_p_values, alpha=self.config.alpha_significance_level
        )

        for i, res in enumerate(preliminary_results):
            res.benjamini_hochberg_p_value = bh_p_vals[i]
            res.bonferroni_p_value = bonf_p_vals[i]
            res.is_significant_fdr = bh_sig[i]
            res.is_significant_bonferroni = bonf_sig[i]

        return preliminary_results
