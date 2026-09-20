"""
Lexical and Surface-Form Drift Metrics for PromptBench.
Measures token-level Jaccard overlap and response length disparity.
"""

import re
from typing import Set, Tuple


class LexicalDriftScorer:
    """Computes lexical surface similarity and output verbosity metrics."""

    @staticmethod
    def _get_token_set(text: str) -> Set[str]:
        """Extracts unique normalized alphanumeric tokens."""
        return set(re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower()))

    @classmethod
    def compute_jaccard_similarity(cls, text_base: str, text_pert: str) -> float:
        """
        Computes Jaccard index:
            J(A, B) = |A cap B| / |A cup B|
        Range: [0.0, 1.0].
        """
        tokens_a = cls._get_token_set(text_base)
        tokens_b = cls._get_token_set(text_pert)

        if not tokens_a and not tokens_b:
            return 1.0
        if not tokens_a or not tokens_b:
            return 0.0

        intersection = len(tokens_a.intersection(tokens_b))
        union = len(tokens_a.union(tokens_b))

        return float(round(intersection / union, 4))

    @staticmethod
    def compute_length_delta(text_base: str, text_pert: str) -> Tuple[int, int, int]:
        """
        Computes character length differences:
            (len_base, len_pert, delta) where delta = len_pert - len_base.
        """
        lb = len(text_base)
        lp = len(text_pert)
        return lb, lp, lp - lb
