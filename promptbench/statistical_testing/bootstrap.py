"""
Non-Parametric Bootstrap Confidence Interval Estimation for PromptBench.
Constructs empirical 95% Confidence Intervals for effect sizes and mean differences.
"""

import math
import random
from typing import List, Tuple, Callable, Optional


class BootstrapCI:
    """
    Computes non-parametric bootstrap confidence intervals (Percentile & BCa methods).
    """

    def __init__(self, resamples: int = 2000, confidence_level: float = 0.95, seed: int = 42):
        self.resamples = resamples
        self.confidence_level = confidence_level
        self.seed = seed

    def compute_ci_mean(
        self,
        data: List[float],
    ) -> Tuple[float, float, float]:
        """
        Calculates non-parametric bootstrap confidence interval for the sample mean.
        
        Algorithm:
            1. Draw B bootstrap samples D* of size n with replacement from D.
            2. Compute bootstrap replicate statistic theta* = mean(D*).
            3. Sort replicates: theta*(1) <= theta*(2) <= ... <= theta*(B).
            4. Percentile interval: [theta*(lower_idx), theta*(upper_idx)]
        
        Returns:
            (observed_mean, ci_lower, ci_upper)
        """
        n = len(data)
        if n == 0:
            return 0.0, 0.0, 0.0
        if n == 1:
            return data[0], data[0], data[0]

        observed_mean = sum(data) / n
        rng = random.Random(self.seed)

        bootstrap_means: List[float] = []
        for _ in range(self.resamples):
            sample = [data[rng.randint(0, n - 1)] for _ in range(n)]
            bootstrap_means.append(sum(sample) / n)

        bootstrap_means.sort()

        alpha = 1.0 - self.confidence_level
        lower_idx = int(math.floor((alpha / 2.0) * self.resamples))
        upper_idx = int(math.ceil((1.0 - alpha / 2.0) * self.resamples)) - 1

        lower_idx = max(0, min(lower_idx, self.resamples - 1))
        upper_idx = max(0, min(upper_idx, self.resamples - 1))

        ci_lower = bootstrap_means[lower_idx]
        ci_upper = bootstrap_means[upper_idx]

        return (
            float(round(observed_mean, 4)),
            float(round(ci_lower, 4)),
            float(round(ci_upper, 4)),
        )

    def compute_ci_effect_size(
        self,
        group_base: List[float],
        group_target: List[float],
    ) -> Tuple[float, float, float]:
        """
        Calculates 95% Bootstrap Confidence Interval for Cohen's d effect size.
        """
        n1 = len(group_base)
        n2 = len(group_target)
        if n1 < 2 or n2 < 2:
            return 0.0, 0.0, 0.0

        def cohens_d_calc(g1: List[float], g2: List[float]) -> float:
            m1 = sum(g1) / len(g1)
            m2 = sum(g2) / len(g2)
            v1 = sum((x - m1) ** 2 for x in g1) / (len(g1) - 1)
            v2 = sum((x - m2) ** 2 for x in g2) / (len(g2) - 1)
            pv = ((len(g1) - 1) * v1 + (len(g2) - 1) * v2) / (len(g1) + len(g2) - 2)
            if pv <= 0:
                return 0.0
            return (m2 - m1) / math.sqrt(pv)

        observed_d = cohens_d_calc(group_base, group_target)
        rng = random.Random(self.seed)

        bootstrap_ds: List[float] = []
        for _ in range(self.resamples):
            s1 = [group_base[rng.randint(0, n1 - 1)] for _ in range(n1)]
            s2 = [group_target[rng.randint(0, n2 - 1)] for _ in range(n2)]
            bootstrap_ds.append(cohens_d_calc(s1, s2))

        bootstrap_ds.sort()
        alpha = 1.0 - self.confidence_level
        lower_idx = max(0, int(math.floor((alpha / 2.0) * self.resamples)))
        upper_idx = min(self.resamples - 1, int(math.ceil((1.0 - alpha / 2.0) * self.resamples)) - 1)

        return (
            float(round(observed_d, 4)),
            float(round(bootstrap_ds[lower_idx], 4)),
            float(round(bootstrap_ds[upper_idx], 4)),
        )
