"""
Multiple Testing Corrections for PromptBench.
Implements Benjamini-Hochberg FDR, Bonferroni, and Holm-Bonferroni corrections.
"""

from typing import List, Tuple, Dict, Any


class MultipleTestingCorrection:
    """
    Adjusts raw p-values across multiple simultaneous demographic hypothesis tests
    to prevent false discovery inflation.
    """

    @staticmethod
    def benjamini_hochberg(p_values: List[float], alpha: float = 0.05) -> Tuple[List[float], List[bool]]:
        """
        Benjamini-Hochberg False Discovery Rate (FDR) Procedure.
        
        Algorithm:
            1. Sort p-values in ascending order: p_(1) <= p_(2) <= ... <= p_(m)
            2. Compute adjusted p-values with monotonicity enforcement:
                   p_adj(i) = min(1.0, min_{j >= i} ( (m / j) * p_(j) ))
            3. Reject H0 if p_adj(i) <= alpha.
        
        Returns:
            (adjusted_p_values, significance_flags) in original input order.
        """
        m = len(p_values)
        if m == 0:
            return [], []

        # Store indexed values: (original_index, raw_p)
        indexed = list(enumerate(p_values))
        indexed.sort(key=lambda x: x[1])  # Sort by p-value

        # Step-up adjustment
        adjusted_indexed: List[Tuple[int, float]] = [None] * m  # type: ignore

        # Backward cumulative minimum to enforce monotonicity
        min_so_far = 1.0
        for rank_idx in range(m - 1, -1, -1):
            orig_idx, p_val = indexed[rank_idx]
            rank = rank_idx + 1
            adj_p = (m / rank) * p_val
            min_so_far = min(min_so_far, adj_p)
            adjusted_indexed[rank_idx] = (orig_idx, min(1.0, max(0.0, min_so_far)))

        # Re-sort back to original index order
        adjusted_indexed.sort(key=lambda x: x[0])
        adj_p_values = [round(item[1], 5) for item in adjusted_indexed]
        significance = [p <= alpha for p in adj_p_values]

        return adj_p_values, significance

    @staticmethod
    def bonferroni(p_values: List[float], alpha: float = 0.05) -> Tuple[List[float], List[bool]]:
        """
        Single-step Bonferroni FWER correction:
            p_adj = min(1.0, m * p)
        """
        m = len(p_values)
        if m == 0:
            return [], []

        adj_p_values = [min(1.0, round(p * m, 5)) for p in p_values]
        significance = [p <= alpha for p in adj_p_values]
        return adj_p_values, significance

    @staticmethod
    def holm_bonferroni(p_values: List[float], alpha: float = 0.05) -> Tuple[List[float], List[bool]]:
        """
        Holm-Bonferroni Step-Down FWER correction.
        """
        m = len(p_values)
        if m == 0:
            return [], []

        indexed = list(enumerate(p_values))
        indexed.sort(key=lambda x: x[1])

        max_so_far = 0.0
        adjusted_indexed: List[Tuple[int, float]] = [None] * m  # type: ignore

        for rank_idx in range(m):
            orig_idx, p_val = indexed[rank_idx]
            multiplier = m - rank_idx
            adj_p = multiplier * p_val
            max_so_far = max(max_so_far, adj_p)
            adjusted_indexed[rank_idx] = (orig_idx, min(1.0, max(0.0, max_so_far)))

        adjusted_indexed.sort(key=lambda x: x[0])
        adj_p_values = [round(item[1], 5) for item in adjusted_indexed]
        significance = [p <= alpha for p in adj_p_values]
        return adj_p_values, significance
