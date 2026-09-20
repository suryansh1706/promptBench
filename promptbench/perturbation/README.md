# Stage 1: Perturbation Engine (`promptbench/perturbation`)

The **Perturbation Engine** is Stage 1 of the PromptBench pipeline. It is responsible for parsing base scenario prompt templates and systematically substituting surface-form demographic and socioeconomic attributes while guaranteeing that substantive qualifications remain $100\%$ invariant.

---

## 1. Problem & Research Rationale

In high-stakes algorithmic decision making (e.g., resume screening, commercial credit evaluation, tenant leasing), an AI model must satisfy **individual fairness**:
$$\mathcal{M}(P_i) \approx \mathcal{M}(P_j) \quad \text{whenever} \quad d_{\text{substantive}}(P_i, P_j) = 0$$

Where $P_i$ and $P_j$ differ *only* by surface-level demographic cues (candidate name, gender pronoun, nationality, socioeconomic indicator).

Stage 1 creates **counterfactual paired prompts** $(P_{\text{base}}, P_{\text{pert}})$ that isolate demographic variables across 4 principal axes:
1. **Gender**: Male (`he/him`), Female (`she/her`), Non-Binary / Neutral (`they/them`).
2. **Ethnicity / Cultural Origin**: Anglo-Western, South Asian, East Asian, Hispanic/Latino, African/Black, Middle Eastern, European.
3. **Socioeconomic Status (SES)**: High SES (Tier-1 Ivy/Elite University, affluent neighborhood, elite extracurriculars) vs. Middle SES vs. Low SES (Community College transfer, working-class transit corridor, working full-time shifts during college).
4. **Nationality**: Multi-national geographic origins.

---

## 2. Algorithms Used in this Module

### Algorithm 1: Combinatorial Slot-Filling & Counterfactual Pairing
The engine performs deterministic Cartesian product slot substitution over a base template $\mathcal{T}$:

$$\mathcal{D}_{\text{paired}} = \Big\{ \big( \mathcal{T}(\mathbf{x}_{\text{baseline}}), \, \mathcal{T}(\mathbf{x}_k) \big) \; \Big| \; \mathbf{x}_k \in \mathcal{E} \times \mathcal{G} \times \mathcal{S} \times \mathcal{N} \Big\}$$

```python
Input: Base Template T, Demographic Profiles {p_1, p_2, ..., p_K}, Baseline Profile p_base
Output: List of Prompts V, Paired Dataset Pairs

1. Render P_base = SubstituteSlots(T, p_base)
2. Add P_base to V
3. For each profile p_k in Profiles:
4.     Render P_variant = SubstituteSlots(T, p_k)
5.     Add P_variant to V
6.     Add (P_base, P_variant) to Pairs
7. Return V, Pairs
```

#### Complexity Analysis:
- **Time Complexity**: $\mathcal{O}(|\mathcal{T}_{\text{templates}}| \times |\mathcal{E}| \times |\mathcal{G}| \times |\mathcal{S}| \times |\mathcal{N}|)$ where slot substitution is linear in template length $\mathcal{O}(L)$.
- **Space Complexity**: $\mathcal{O}(N \cdot L)$ to store generated counterfactual prompt variations in memory.

---

## 3. Template Scenarios Provided

| Scenario Category | Template ID | Decision Domain | Substantive Qualifications Kept Constant |
| :--- | :--- | :--- | :--- |
| **Hiring** | `job_swe_001` | Senior Software Engineer | 6 yrs exp, Go/Python, p99 latency reduction, Tech Score 88/100, System Design 92/100 |
| **Hiring** | `job_ds_002` | Lead Data Scientist | 5 yrs exp, PyTorch, Causal Inference, 2 publications, ML Case Score 89/100 |
| **Credit / Lending** | `loan_sb_001` | Small Business Loan ($150k) | Credit Score 740, $520k Revenue, 1.45x DSCR, $187.5k Collateral |
| **Housing** | `tenant_res_001` | Apartment Rental Lease | 3.8x rent-to-income, 715 FICO score, 5-yr clean rental record |

---

## 4. Usage Example

```python
from promptbench.perturbation import PerturbationEngine, get_template_by_id

engine = PerturbationEngine()
all_prompts, paired_prompts = engine.generate_counterfactual_dataset(names_per_group=2)

print(f"Total Prompts Generated: {len(all_prompts)}")
print(f"Total Counterfactual Pairs: {len(paired_prompts)}")
print(f"Sample Variant Prompt:\n{all_prompts[1].rendered_prompt[:250]}...")
```
