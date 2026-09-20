# Stage 4: Statistical Testing & Inference Engine (`promptbench/statistical_testing`)

The **Statistical Testing Engine** is Stage 4 of PromptBench. It establishes whether observed differences across demographic prompt variants are statistically significant rather than stochastic noise, using distribution-free permutation tests, non-parametric bootstrap confidence intervals, and False Discovery Rate (FDR) control.

---

## 1. Research Motivation & Hypothesis Formulation

Standard benchmark audits frequently report raw metric deltas (e.g., a $+0.04$ shift in similarity) without assessing statistical significance. When evaluating across multiple demographic groups and templates, stochastic generation noise can produce spurious differences.

PromptBench formally evaluates:
- **Null Hypothesis ($H_0$)**: The distribution of paired differences between baseline and perturbed responses is symmetric around zero ($\mu_\Delta = 0$), implying no demographic bias.
- **Alternative Hypothesis ($H_1$)**: The distribution exhibits significant systematic drift ($\mu_\Delta \ne 0$).

---

## 2. Algorithms Used in this Module

### Algorithm 1: Monte Carlo Paired Permutation Test ($B \ge 10,000$)
For $n$ matched pairs with observed deltas $d_i = y_i - x_i$, the test statistic is the sample mean:
$$T_{\text{obs}} = \frac{1}{n}\sum_{i=1}^n d_i$$

Under $H_0$, each difference is equally likely to be positive or negative. The test generates $B$ permutations with Rademacher random sign flips $s_i \in \{-1, +1\}$:

$$T^{(b)} = \frac{1}{n}\sum_{i=1}^n s_i d_i, \quad b = 1, \dots, B$$

The exact two-sided empirical p-value is computed with conservative pseudo-count adjustment:
$$p = \frac{1 + \sum_{b=1}^B \mathbb{I}\big(|T^{(b)}| \ge |T_{\text{obs}}|\big)}{B + 1}$$

#### Complexity Analysis:
- **Time Complexity**: $\mathcal{O}(B \cdot n)$ where $B = 10,000$ and $n$ is sample size per demographic subgroup.
- **Properties**: Exact, non-parametric, requires zero normality assumptions.

---

### Algorithm 2: Non-Parametric Bootstrap Confidence Intervals ($K \ge 2,000$)
Estimates the empirical $95\%$ Confidence Interval for effect sizes without distributional assumptions:
1. Draw $K$ resamples $\mathcal{D}^{*(k)}$ of size $n$ with replacement from the observed sample $\mathcal{D}$.
2. Compute the replicate statistic $\theta^{*(k)} = \text{Statistic}(\mathcal{D}^{*(k)})$.
3. Sort replicates in ascending order: $\theta^{*(1)} \le \theta^{*(2)} \le \dots \le \theta^{*(K)}$.
4. Compute the percentile interval at $\alpha = 0.05$:
   $$\text{CI}_{95\%} = \left[ \theta^*_{\lfloor \frac{\alpha}{2} K \rfloor}, \; \theta^*_{\lceil (1 - \frac{\alpha}{2}) K \rceil} \right]$$

---

### Algorithm 3: Benjamini-Hochberg False Discovery Rate (FDR) Control
When testing $m$ demographic comparisons simultaneously, testing each at $\alpha = 0.05$ inflates the family-wise error rate:
$$P(\ge 1 \text{ false positive}) = 1 - (1 - \alpha)^m \xrightarrow[m=20]{} 64.2\%$$

The Benjamini-Hochberg procedure bounds the expected proportion of false discoveries:
$$\text{FDR} = \mathbb{E}\left[ \frac{V}{\max(R, 1)} \right] \le Q = 0.05$$

#### Algorithmic Steps:
1. Sort raw p-values: $p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$.
2. Compute adjusted p-values with monotonicity enforcement:
   $$p_{(i)}^{\text{BH}} = \min\left( 1.0, \, \min_{j \ge i} \left( \frac{m}{j} p_{(j)} \right) \right)$$
3. Declare significance if $p_{(i)}^{\text{BH}} \le 0.05$.

---

## 3. Usage Example

```python
from promptbench.statistical_testing import StatisticalTestingEngine, PermutationTest, BootstrapCI
from promptbench.core.config import StatisticalConfig

# Run standalone permutation test
perm = PermutationTest(resamples=10000)
observed_diff, p_value, magnitude = perm.test_paired_differences([-0.04, -0.06, -0.03, -0.05, -0.045])
print(f"Observed Delta: {observed_diff:.4f} | Permutation p-value: {p_value:.5f}")

# Run master engine on pairwise comparisons
engine = StatisticalTestingEngine()
test_results = engine.analyze_comparisons(comparisons)
for res in test_results:
    sig_badge = "[SIGNIFICANT BIAS]" if res.is_significant_fdr else "[NO DRIFT]"
    print(f"{res.target_group:20s} | p_raw={res.raw_p_value:.4f} | p_FDR={res.benjamini_hochberg_p_value:.4f} {sig_badge}")
```
