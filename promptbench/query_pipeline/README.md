# Stage 2: LLM Query Pipeline (`promptbench/query_pipeline`)

The **LLM Query Pipeline** is Stage 2 of PromptBench. It manages the robust, rate-limited, and deterministic querying of commercial and open LLMs (Google Gemini, OpenAI GPT, and Mock LLMs) with enterprise resilience and zero duplicate API expenditure.

---

## 1. Architectural Overview

1. **Provider Abstraction (`providers/base.py`)**: Defines a uniform interface `generate_response()` and regular expression parsers for categorical decisions (`[DECISION: ACCEPT/REJECT/APPROVED/...]`) and continuous scores (`[SCORE: 0-100]`).
2. **Provider Implementations**:
   - `GeminiProvider`: Direct REST interface to Google's Gemini models (`gemini-1.5-flash`, `gemini-1.5-pro`, `gemini-2.0-flash`).
   - `OpenAIProvider`: Direct REST interface to OpenAI's models (`gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`).
   - `MockLLMProvider`: Deterministic local simulation engine with calibrated demographic drift models for zero-cost testing and automated verification.
3. **Query Orchestrator (`orchestrator.py`)**: Multi-threaded execution pool with SHA-256 cache verification, thread pooling, and progress callback telemetry.

---

## 2. Algorithms Used in this Module

### Algorithm 1: Exponential Backoff with Full Jitter
When querying commercial LLM APIs, rate-limits (HTTP 429) and transient network disconnects (HTTP 503) cause synchronized thundering herd spikes if naive backoff is used. PromptBench implements **Full Jitter Exponential Backoff**:

$$t_{\text{sleep}} \sim \text{Uniform}\Big(0, \, \min\big(t_{\text{max}}, \, t_{\text{base}} \times 2^{\text{attempt}}\big)\Big)$$

Where:
- $t_{\text{base}} = 1.0\text{s}$ (Initial retry delay)
- $t_{\text{max}} = 30.0\text{s}$ (Maximum delay ceiling)
- $\text{attempt} \in \{0, 1, \dots, \text{max\_retries}-1\}$

```python
Input: Max Retries M, Base Delay b, Max Delay c, Attempt i
Output: Sleep duration t

1. Cap = min(c, b * (2 ** i))
2. t = UniformRandom(0, Cap)
3. Sleep(t)
```

#### Why Full Jitter?
Compared to deterministic exponential backoff ($t = 2^i$), Full Jitter uniformly spreads competing retry requests across time intervals, minimizing collision probability and maximizing API throughput.

---

### Algorithm 2: Deterministic Structured Output Parsing
LLM responses can include explanatory reasoning text around the final decision. PromptBench uses a prioritized multi-stage regular expression grammar:

1. **Structured Token Capture**:
   $$\mathcal{R}_{\text{decision}} = \text{re.search}(\texttt{r"\[DECISION:\s*([A-Za-z]+)\]"}, \text{raw\_text})$$
   $$\mathcal{R}_{\text{score}} = \text{re.search}(\texttt{r"\[SCORE:\s*([0-9]+(?:\.[0-9]+)?)(?:/100)?\]"}, \text{raw\_text})$$
2. **Fallback Semantic Classifier**:
   If structured brackets are omitted by the model, semantic keyword classification maps sentiment to canonical tokens (`ACCEPT`, `REJECT`, `APPROVED`, `DENIED`, `REVIEW`).

---

## 3. Usage Example

```python
from promptbench.core import PromptBenchConfig
from promptbench.query_pipeline import QueryOrchestrator
from promptbench.perturbation import PerturbationEngine

# 1. Initialize config and generate prompts
config = PromptBenchConfig()
engine = PerturbationEngine()
prompts, pairs = engine.generate_counterfactual_dataset(names_per_group=2)

# 2. Initialize orchestrator (uses Mock provider by default for zero API cost)
orchestrator = QueryOrchestrator.from_config(config)

# 3. Execute batch with progress tracking
responses = orchestrator.execute_batch(
    prompts=prompts[:10],
    progress_callback=lambda curr, total, resp: print(f"[{curr}/{total}] Queried: {resp.prompt_id} -> {resp.extracted_decision}")
)
```
