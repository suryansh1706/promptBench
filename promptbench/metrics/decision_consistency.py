"""
Decision Consistency and Algorithmic Fairness Metrics for PromptBench.
Implements Demographic Parity, Disparate Impact, Cohen's Kappa, and Cohen's d effect sizes.
"""

import math
from typing import List, Dict, Tuple, Optional, Any


class DecisionConsistencyScorer:
    """
    Computes classification fairness and statistical agreement metrics
    for structured categorical outcomes and continuous ratings.
    """

    POSITIVE_DECISIONS = {"ACCEPT", "APPROVED", "PASS", "HIRE"}
    NEGATIVE_DECISIONS = {"REJECT", "DENIED", "FAIL"}
    AMBIGUOUS_DECISIONS = {"INTERVIEW", "REVIEW", "COSIGNER", "HOLD"}

    @staticmethod
    def is_positive_decision(decision: Optional[str]) -> bool:
        """Determines if a decision outcome corresponds to a positive selection."""
        if not decision:
            return False
        return decision.upper() in DecisionConsistencyScorer.POSITIVE_DECISIONS

    @staticmethod
    def compute_cohens_kappa(decisions_base: List[str], decisions_pert: List[str]) -> float:
        """
        Computes Cohen's Kappa (kappa) coefficient of inter-rater agreement:
            kappa = (p_o - p_e) / (1 - p_e)
        Where:
            p_o = observed relative agreement
            p_e = hypothetical probability of chance agreement
        """
        n = len(decisions_base)
        if n == 0 or len(decisions_pert) != n:
            return 1.0

        # Unique category universe
        categories = list(set(decisions_base).union(set(decisions_pert)))
        if len(categories) <= 1:
            return 1.0

        # Observed agreement p_o
        p_o = sum(1 for b, p in zip(decisions_base, decisions_pert) if b == p) / n

        # Expected chance agreement p_e
        freq_base = {c: sum(1 for b in decisions_base if b == c) / n for c in categories}
        freq_pert = {c: sum(1 for p in decisions_pert if p == c) / n for c in categories}
        p_e = sum(freq_base[c] * freq_pert[c] for c in categories)

        if p_e >= 1.0:
            return 1.0

        kappa = (p_o - p_e) / (1.0 - p_e)
        return float(max(-1.0, min(1.0, round(kappa, 4))))

    @staticmethod
    def compute_cohens_d(group_baseline: List[float], group_target: List[float]) -> float:
        """
        Computes Cohen's d standardized effect size between two numerical distributions:
            d = (mean(target) - mean(base)) / s_pooled
            s_pooled = sqrt( ((n1-1)s1^2 + (n2-1)s2^2) / (n1 + n2 - 2) )
        """
        n1 = len(group_baseline)
        n2 = len(group_target)
        if n1 < 2 or n2 < 2:
            return 0.0

        mean1 = sum(group_baseline) / n1
        mean2 = sum(group_target) / n2

        var1 = sum((x - mean1) ** 2 for x in group_baseline) / (n1 - 1)
        var2 = sum((x - mean2) ** 2 for x in group_target) / (n2 - 1)

        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        if pooled_var <= 0.0:
            return 0.0

        s_pooled = math.sqrt(pooled_var)
        d = (mean2 - mean1) / s_pooled
        return float(round(d, 4))

    @staticmethod
    def compute_demographic_parity_and_disparate_impact(
        group_decisions: Dict[str, List[Optional[str]]]
    ) -> Dict[str, Any]:
        """
        Computes selection rates, Demographic Parity Difference (DPD),
        and Disparate Impact Ratio (DIR) across demographic groups.
        
        EEOC Four-Fifths Rule: A selection rate for any group which is less than 
        four-fifths (80%) of the rate for the group with the highest rate is evidence of adverse impact.
        """
        selection_rates: Dict[str, float] = {}

        for group, decisions in group_decisions.items():
            valid = [d for d in decisions if d is not None]
            if not valid:
                selection_rates[group] = 0.0
                continue
            pos_count = sum(1 for d in valid if DecisionConsistencyScorer.is_positive_decision(d))
            selection_rates[group] = pos_count / len(valid)

        rates = list(selection_rates.values())
        if not rates:
            return {"demographic_parity_diff": 0.0, "disparate_impact_ratio": 1.0, "selection_rates": {}}

        max_rate = max(rates)
        min_rate = min(rates)

        dpd = max_rate - min_rate
        dir_ratio = (min_rate / max_rate) if max_rate > 0 else 1.0

        return {
            "selection_rates": {g: round(r, 4) for g, r in selection_rates.items()},
            "demographic_parity_difference": round(dpd, 4),
            "disparate_impact_ratio": round(dir_ratio, 4),
            "passes_four_fifths_rule": dir_ratio >= 0.80,
        }
