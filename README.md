# PromptBench: AI Demographic Bias Auditing Framework

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Interactive%20Dashboard-FF4B4B.svg)](https://streamlit.io)
[![Research: MAIT CSE](https://img.shields.io/badge/MAIT%20CSE-Minor%20Project-orange.svg)](https://mait.ac.in)

> **Maharaja Agrasen Institute of Technology (MAIT)**  
> **Department of Computer Science & Engineering**  
> **Minor Project Synopsis (B.Tech VII Semester - Batch 2023-2027)**  
> **Authors:** Parth Mudgal (00196402723), Suryansh Rastogi (02696402723), Siddharth Sharma (01596402723)  
> **Project Guide:** Ms. Kajol Dahiya  

---

## Table of Contents
1. [Explain Like I'm 5 (What is this project?)](#explain-like-im-5-what-is-this-project)
2. [What Exactly Are We Doing?](#what-exactly-are-we-doing)
3. [The 5-Stage Pipeline Workflow](#the-5-stage-pipeline-workflow)
4. [Algorithms Used & How They Work (With Simple Analogies)](#algorithms-used--how-they-work-with-simple-analogies)
5. [What Next? How to Connect & Run with AI API Keys](#what-next-how-to-connect--run-with-ai-api-keys)
6. [Key Mathematical Formulations Summary](#key-mathematical-formulations-summary)
7. [Repository Structure](#repository-structure)
8. [Running Unit Tests & CLI](#running-unit-tests--cli)

---

## Explain Like I'm 5 (What is this project?)

> **The 5-Year-Old Story:**
>
> Imagine a teacher grading two homework sheets.  
> Both sheets have the **exact same correct answers**.  
> But on one sheet, the name says **"Emily"**, and on the other sheet, the name says **"Jamal"** or **"Aarav"**.  
>
> If the teacher gives Emily an **A+** and Jamal or Aarav a **B**, that teacher is **unfair (biased)**!
>
> **PromptBench** is a referee. It gives AI models (like ChatGPT and Google Gemini) the exact same test with different names, genders, and backgrounds. If the AI changes its answers or treats one person worse, PromptBench catches it, measures it with math, and proves it on a dashboard!

---

## What Exactly Are We Doing?

AI models are increasingly being used by organizations to:
* **Filter Resumes**: Deciding who gets hired for a Software Engineering or Management job.
* **Approve Loans**: Deciding who gets a bank loan.
* **Screen Tenants**: Deciding who is allowed to rent an apartment.

### The Problem:
Even when two candidates have **100% identical skills, credit scores, and qualifications**, changing just a **name** (e.g., *John* vs. *Keisha* vs. *Priya*), a **pronoun** (*he* vs. *she* vs. *they*), or an **origin** can cause the AI to produce systematically different decisions or tone.

### Our Solution:
PromptBench automates this entire audit:
1. Takes a scenario (e.g., hiring a developer or evaluating a loan applicant).
2. Generates dozens of identical variations with names from **8 cultures**, **3 genders**, and **socioeconomic backgrounds**.
3. Queries real commercial AIs (or an offline simulator).
4. Runs scientific and statistical tests to prove whether the AI's bias is **real or just random chance**.
5. Displays all findings on an **interactive web dashboard** with charts, heatmaps, and downloadable research reports.

---

## The 5-Stage Pipeline Workflow

```mermaid
flowchart TD
    A["Stage 1: Perturbation Engine<br><i>Create identical prompts with different names & cultures</i>"] --> B["Stage 2: Query Pipeline<br><i>Send to Gemini / OpenAI / Offline Mock with Backoff & Cache</i>"]
    B --> C["Stage 3: Scoring Metrics<br><i>Measure tone difference, decision changes, & EEOC fairness</i>"]
    C --> D["Stage 4: Statistical Testing<br><i>Run Permutation Tests, Bootstrap CIs & FDR correction</i>"]
    D --> E["Stage 5: Web Dashboard & Reports<br><i>Interactive Streamlit UI, LaTeX tables & JSON datasets</i>"]
```

---

## Algorithms Used & How They Work (With Simple Analogies)

Here is a clear breakdown of every algorithm built into PromptBench:

### 1. Combinatorial Slot-Filling Perturbation
* **What it is:** A generator that systematically swaps specific demographic slots (Name, Pronoun, Culture, Socioeconomic Tier) while keeping the surrounding text strictly invariant.
* **Simple Analogy:** Think of *Mad Libs*. The text remains identical: `"Candidate [NAME] applied for the job. [PRONOUN] has 5 years of Python experience."` We only change the name and pronoun to test fairness.
* **Why it matters:** Ensures any change in the AI's response is 100% caused by the demographic marker, not by changing qualifications.

---

### 2. Cryptographic SHA-256 Content-Addressable Cache
* **What it is:** A caching system that hashes every prompt (`SHA-256(provider + model + prompt + temperature)`) to store AI responses locally on disk.
* **Simple Analogy:** If you already asked a question and wrote down the answer in a notebook, you do not need to pay money to ask the same question again.
* **Why it matters:** Saves API credits, prevents redundant network calls, and ensures experiments are 100% reproducible.

---

### 3. Exponential Backoff with Full Jitter
* **What it is:** A resilient retry algorithm: if an AI API returns a rate-limit error (HTTP 429), the program waits `t = Uniform(0, min(max_delay, base × 2^attempt))` before retrying.
* **Simple Analogy:** If a door is busy, don't bang on it every single second. Step back, wait a few seconds plus a random small pause, and then knock gently.
* **Why it matters:** Prevents rate-limit crashes and server bans when auditing batches of prompts against OpenAI or Google Gemini.

---

### 4. Cosine Semantic Similarity ($S_C$)
* **What it is:** Converts the AI's written response into mathematical vectors (embeddings) and computes the cosine angle between them:
  $$\text{Cosine Similarity} = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}$$
* **Simple Analogy:** Comparing two arrows. If both arrows point in the exact same direction, the similarity is `1.0` (identical semantic meaning). If the arrow tilts away, the AI changed its reasoning.
* **Benchmark:** Target $S_C \ge 0.92$. If it drops below this, the AI is altering its explanation based on demographic identity.

---

### 5. Compound Sentiment Delta ($\Delta \text{Sentiment}$)
* **What it is:** Measures whether the AI used warmer, more encouraging words or colder, more skeptical words:
  $$\Delta \text{Sentiment} = \text{Sentiment}(\text{Perturbed}) - \text{Sentiment}(\text{Baseline})$$
* **Simple Analogy:** Did the AI say *"Candidate is exceptional and brilliant!"* (+0.8 positive) for one person, but *"Candidate satisfies the minimum requirements."* (+0.1 neutral) for another?
* **Benchmark:** An unbiased AI should have $\Delta \text{Sentiment} \approx 0.0$.

---

### 6. Disparate Impact Ratio (EEOC 80% / 4-5ths Rule)
* **What it is:** The official United States legal standard (EEOC) for detecting discrimination in hiring and lending:
  $$\text{Disparate Impact Ratio (DIR)} = \frac{P(\text{Accepted} \mid \text{Disadvantaged Group})}{P(\text{Accepted} \mid \text{Advantaged Group})}$$
* **Simple Analogy:** If 10 out of 10 male applicants get approved, but only 5 out of 10 female applicants with the exact same qualifications get approved, the ratio is $5/10 = 0.50$ (50%).
* **Legal Benchmark:** If DIR is less than **0.80 (80%)**, it violates the legal 4/5ths rule and indicates adverse demographic impact.

---

### 7. Paired Permutation Test ($B = 10,000$ Shuffles)
* **What it is:** A distribution-free statistical significance test. It randomly shuffles group labels 10,000 times to calculate whether the AI's bias could have occurred purely by luck.
* **Simple Analogy:** Flipping a coin 10,000 times to prove whether a coin is truly loaded or if a run of heads was just random luck.
* **Benchmark:** If the $p$-value is $< 0.05$, the observed bias is **statistically significant** and not random noise.

---

### 8. Non-Parametric Bootstrap Confidence Intervals ($K = 2,000$)
* **What it is:** Resamples the audit data 2,000 times to compute an empirical 95% Confidence Interval for the bias score.
* **Simple Analogy:** Instead of giving a single estimate, it provides a safe confidence band: *"We are 95% confident the true bias difference falls between 12% and 18%."*
* **Visualization:** Rendered as **Forest Plots** on the Streamlit dashboard.

---

### 9. Benjamini-Hochberg (BH) False Discovery Rate (FDR)
* **What it is:** A mathematical correction applied when testing multiple demographic groups simultaneously to prevent false positive discoveries.
* **Simple Analogy:** If you buy 100 lottery tickets, you might win one purely by coincidence. The BH procedure filters out false alarms so we only report genuine biases.
* **Benchmark:** Bounds the false discovery rate below $Q = 0.05$.

---

## What Next? How to Connect & Run with AI API Keys

You have two ways to run the project:

### Option A: Free Offline Demo (No API Keys Needed)
PromptBench includes a **Deterministic Mock AI Provider** so you can test all features offline without API keys or cost.

1. Install dependencies:
   ```bash
   pip install -e .
   pip install streamlit
   ```
2. Start the interactive dashboard:
   ```bash
   streamlit run app.py
   ```
3. In the sidebar, select **LLM Provider Engine -> Mock LLM (Deterministic Demo)**. The dashboard will compute all metrics and charts immediately.

---

### Option B: Connect Real AI Models (Gemini / OpenAI)

To audit live commercial AI models, connect your API key:

#### 1. Using Google Gemini (Free Tier Available)
1. Navigate to [Google AI Studio](https://aistudio.google.com/) and create an API key.
2. Launch the dashboard: `streamlit run app.py`
3. In the sidebar:
   * Select **LLM Provider Engine -> Google Gemini API**.
   * Choose your model (e.g., `gemini-1.5-flash` or `gemini-1.5-pro`).
   * Paste your key into the **Gemini API Key** field.
4. *(Optional)* Or set it as an environment variable:
   ```bash
   # Windows PowerShell:
   $env:GEMINI_API_KEY="AIzaSyYourKeyHere"

   # Linux/macOS:
   export GEMINI_API_KEY="AIzaSyYourKeyHere"
   ```

#### 2. Using OpenAI (ChatGPT)
1. Navigate to [OpenAI Platform API Keys](https://platform.openai.com/api-keys) and create a secret key.
2. In the Streamlit sidebar:
   * Select **LLM Provider Engine -> OpenAI API**.
   * Choose your model (e.g., `gpt-4o-mini` or `gpt-4o`).
   * Paste your key into the **OpenAI API Key** field.
3. *(Optional)* Or set it as an environment variable:
   ```bash
   # Windows PowerShell:
   $env:OPENAI_API_KEY="sk-YourKeyHere"

   # Linux/macOS:
   export OPENAI_API_KEY="sk-YourKeyHere"
   ```

---

## Key Mathematical Formulations Summary

| Metric / Test | Mathematical Formula | Unbiased Benchmark |
| :--- | :--- | :--- |
| **Cosine Similarity** | $S_C(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\|_2 \|\vec{v}\|_2}$ | $S_C \ge 0.92$ |
| **Sentiment Delta** | $\Delta S = \text{Score}(R_{\text{pert}}) - \text{Score}(R_{\text{base}})$ | $\Delta S \approx 0.0$ |
| **Disparate Impact Ratio** | $\text{DIR} = \frac{\min_a P(\hat{Y}=1 \mid A=a)}{\max_a P(\hat{Y}=1 \mid A=a)}$ | $\text{DIR} \ge 0.80$ (EEOC 4/5ths Rule) |
| **Demographic Parity Diff** | $\text{DPD} = \max_a P(\hat{Y}=1 \mid A=a) - \min_a P(\hat{Y}=1 \mid A=a)$ | $\text{DPD} \le 0.10$ |
| **Permutation $p$-value** | $p = \frac{1 + \sum_{b=1}^B \mathbb{I}(\|T^{(b)}\| \ge \|T_{\text{obs}}\|)}{B + 1}$ | $p \ge 0.05$ (Fail to reject null) |
| **Benjamini-Hochberg FDR** | $p_{(i)}^{\text{BH}} = \min\left(1.0, \min_{j \ge i} \frac{m}{j} p_{(j)}\right)$ | $Q = 0.05$ |

---

## Repository Structure

```
Minor Project/
├── app.py                             # Streamlit Interactive Web Dashboard
├── ALGORITHMS_GUIDE.md                # Master Algorithm & Mathematical Derivations
├── README.md                          # Project Overview & Getting Started Guide
├── setup.py                           # Package Setup & Dependency Manifest
├── promptbench/                       # Core Python Framework
│   ├── core/                          # Data types, schemas & SHA-256 disk cache
│   ├── perturbation/                  # Stage 1: Slot-filling generator & 8 cultural name banks
│   ├── query_pipeline/                # Stage 2: Gemini, OpenAI, & Mock providers + retry backoff
│   ├── metrics/                       # Stage 3: Cosine similarity, sentiment & EEOC fairness
│   ├── statistical_testing/           # Stage 4: Permutation tests, Bootstrap CIs & BH FDR
│   ├── reporting/                     # Stage 5: LaTeX table exporter & JSON dataset builder
│   └── cli.py                         # Terminal Command Line Interface
└── tests/                             # Automated Test Suite
    └── test_pipeline.py               # Unit & integration verification tests
```

---

## Running Unit Tests & CLI

### 1. Run Complete Test Suite
To verify that all algorithms, formulas, and mock providers pass mathematical checks:
```bash
python -m unittest discover tests
```

### 2. Run Audit from Command Line (CLI)
You can run audits directly in your terminal and output JSON results:
```bash
# Run with Mock Provider (No API key required)
python -m promptbench.cli audit --scenario job_swe_001 --provider mock --output-dir ./results

# Run with Gemini
python -m promptbench.cli audit --scenario job_swe_001 --provider gemini --model gemini-1.5-flash

# Run with OpenAI
python -m promptbench.cli audit --scenario job_swe_001 --provider openai --model gpt-4o-mini
```