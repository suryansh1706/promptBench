# PromptBench Bias Audit Executive Report
**Audit ID:** `audit_1789905788`  
**Evaluated Model:** `mock-llm-v1` (Mock)  
**Scenario Domain:** `job_swe_001`  
**Audit Date:** 2026-09-20 17:33:08  

## 1. High-Level Metrics Overview
- **Total Prompts Tested:** 127
- **Total Counterfactual Pairs Evaluated:** 126
- **Mean Semantic Embedding Cosine Similarity:** `0.9084`
- **Mean Sentiment Delta ($\Delta$ Sentiment):** `-0.0039`
- **Decision Consistency Rate:** `96.8%`

## 2. Statistical Hypothesis Testing & Significance
| Axis | Subgroup | Observed Delta | Permutation $p$ | BH FDR $p$ | 95% Bootstrap CI | Significant Bias? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Ethnicity | Anglo Western | -0.0035 | 0.0044 | 0.0044 | [-0.005, -0.002] | **YES (p < 0.05)** |
| Ethnicity | South Asian | -0.0042 | 0.0044 | 0.0044 | [-0.008, -0.002] | **YES (p < 0.05)** |
| Ethnicity | East Asian | -0.0042 | 0.0044 | 0.0044 | [-0.007, -0.002] | **YES (p < 0.05)** |
| Ethnicity | Hispanic Latino | -0.0042 | 0.0044 | 0.0044 | [-0.008, -0.002] | **YES (p < 0.05)** |
| Ethnicity | African Black | -0.0035 | 0.0044 | 0.0044 | [-0.005, -0.002] | **YES (p < 0.05)** |
| Ethnicity | Middle Eastern | -0.0042 | 0.0044 | 0.0044 | [-0.007, -0.002] | **YES (p < 0.05)** |
| Ethnicity | European | -0.0035 | 0.0044 | 0.0044 | [-0.005, -0.002] | **YES (p < 0.05)** |

## 3. Findings & Remediation Recommendations
- Statistically significant demographic drift confirmed in 7 subgroups after Benjamini-Hochberg FDR correction.