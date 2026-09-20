"""
Non-Parametric Rank Tests for PromptBench (Wilcoxon Signed-Rank & Mann-Whitney U).
Provides distribution-free rank testing with pure-Python execution and SciPy integration.
"""

import math
from typing import List, Tuple, Optional


class NonParametricTests:
    """
    Implements rank-based non-parametric tests for matched pair and two-sample comparisons.
    """

    @staticmethod
    def wilcoxon_signed_rank(differences: List[float]) -> Tuple[float, float]:
        """
        Wilcoxon Signed-Rank Test for paired differences.
        Tests whether the median of paired differences is zero.
        """
        try:
            from scipy import stats
            # Filter zero differences
            nonzero = [d for d in differences if d != 0.0]
            if len(nonzero) < 5:
                return 0.0, 1.0
            stat, p_val = stats.wilcoxon(nonzero)
            return float(round(stat, 4)), float(round(p_val, 5))
        except (ImportError, ValueError):
            pass

        # Pure Python implementation
        nonzero = [d for d in differences if d != 0.0]
        n = len(nonzero)
        if n < 5:
            return 0.0, 1.0

        abs_diffs = sorted([(abs(d), d > 0) for d in nonzero], key=lambda x: x[0])
        
        # Rank assignment with average ties
        ranks = []
        i = 0
        while i < n:
            j = i
            while j < n and abs_diffs[j][0] == abs_diffs[i][0]:
                j += 1
            avg_rank = (i + 1 + j) / 2.0
            for _ in range(i, j):
                ranks.append(avg_rank)
            i = j

        w_pos = sum(r for r, (_, is_pos) in zip(ranks, abs_diffs) if is_pos)
        w_neg = sum(r for r, (_, is_pos) in zip(ranks, abs_diffs) if not is_pos)
        w_stat = min(w_pos, w_neg)

        # Normal approximation for large n
        mean_w = n * (n + 1) / 4.0
        var_w = n * (n + 1) * (2 * n + 1) / 24.0
        z = (w_stat - mean_w) / math.sqrt(var_w)

        # Two-sided standard normal p-value approximation
        p_val = 2.0 * 0.5 * math.erfc(abs(z) / math.sqrt(2.0))
        return float(round(w_stat, 4)), float(round(p_val, 5))

    @staticmethod
    def mann_whitney_u(sample1: List[float], sample2: List[float]) -> Tuple[float, float]:
        """
        Mann-Whitney U Test for two independent samples.
        """
        try:
            from scipy import stats
            if len(sample1) < 3 or len(sample2) < 3:
                return 0.0, 1.0
            stat, p_val = stats.mannwhitneyu(sample1, sample2, alternative="two-sided")
            return float(round(stat, 4)), float(round(p_val, 5))
        except (ImportError, ValueError):
            pass

        n1, n2 = len(sample1), len(sample2)
        if n1 < 3 or n2 < 3:
            return 0.0, 1.0

        combined = sorted([(x, 1) for x in sample1] + [(x, 2) for x in sample2], key=lambda x: x[0])
        rank_sum_1 = sum(i + 1 for i, (val, grp) in enumerate(combined) if grp == 1)
        u1 = rank_sum_1 - (n1 * (n1 + 1)) / 2.0
        u2 = n1 * n2 - u1
        u_stat = min(u1, u2)

        mean_u = (n1 * n2) / 2.0
        var_u = (n1 * n2 * (n1 + n2 + 1)) / 12.0
        z = (u_stat - mean_u) / math.sqrt(var_u)
        p_val = 2.0 * 0.5 * math.erfc(abs(z) / math.sqrt(2.0))

        return float(round(u_stat, 4)), float(round(p_val, 5))
