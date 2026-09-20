"""
Paired Two-Sided Permutation Testing for PromptBench.
Provides distribution-free statistical hypothesis testing for paired counterfactual prompt evaluations.
"""

import random
from typing import List, Tuple, Dict, Any


class PermutationTest:
    """
    Implements a Monte Carlo Paired Permutation Test comparing paired demographic differences.
    Null Hypothesis H0: The distribution of differences between base and perturbed responses
    is symmetric around zero (no demographic bias effect).
    """

    def __init__(self, resamples: int = 10000, seed: int = 42):
        self.resamples = resamples
        self.seed = seed

    def test_paired_differences(
        self,
        differences: List[float],
    ) -> Tuple[float, float, float]:
        """
        Runs Monte Carlo paired permutation test over a list of paired deltas:
            d_i = y_i - x_i
        
        Algorithm:
            1. Compute observed mean statistic: T_obs = mean(d)
            2. Under H0, the sign of each difference d_i is i.i.d. Rademacher (+1 or -1 with prob 0.5)
            3. In each resample b in 1..B:
                   T_b = (1/n) * sum( s_i * d_i ) where s_i in {-1, +1}
            4. Compute two-sided p-value:
                   p = (1 + sum( |T_b| >= |T_obs| )) / (B + 1)
        
        Returns:
            (observed_mean, p_value, effect_magnitude)
        """
        n = len(differences)
        if n == 0:
            return 0.0, 1.0, 0.0

        t_obs = sum(differences) / n
        abs_t_obs = abs(t_obs)

        if abs_t_obs == 0.0:
            return 0.0, 1.0, 0.0

        rng = random.Random(self.seed)
        extreme_count = 0

        for _ in range(self.resamples):
            # Fast Rademacher sign flipping
            perm_sum = 0.0
            for d in differences:
                sign = 1.0 if rng.random() >= 0.5 else -1.0
                perm_sum += sign * d
            
            t_b = perm_sum / n
            if abs(t_b) >= abs_t_obs - 1e-12:
                extreme_count += 1

        # Conservative pseudo-count p-value estimator
        p_value = (extreme_count + 1) / (self.resamples + 1)
        return float(round(t_obs, 4)), float(round(p_value, 5)), float(round(abs_t_obs, 4))
