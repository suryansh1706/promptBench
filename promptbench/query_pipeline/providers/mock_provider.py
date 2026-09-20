"""
High-Fidelity Deterministic Mock LLM Provider for PromptBench.
Provides instant offline testing, unit testing, and reproducible benchmark demonstrations.
"""

import hashlib
import random
import time
from typing import Optional, Dict, Any, Tuple
from .base import BaseLLMProvider
from ...core.types import ModelResponse


class MockLLMProvider(BaseLLMProvider):
    """
    Deterministic Mock Provider that simulates realistic LLM responses,
    reasoning traces, and decisions. Calibrated to support both zero-bias baseline
    and realistic subtle demographic drift modes for auditing experiments.
    """

    def __init__(
        self,
        model_name: str = "promptbench-mock-v1",
        temperature: float = 0.0,
        seed: int = 42,
        simulate_subtle_bias: bool = True,
    ):
        super().__init__(model_name=model_name, temperature=temperature, seed=seed)
        self.simulate_subtle_bias = simulate_subtle_bias

    def _compute_deterministic_hash(self, text: str) -> int:
        """Computes an integer hash from text combined with random seed."""
        h = hashlib.sha256(f"{text}_{self.seed}".encode("utf-8")).hexdigest()
        return int(h[:8], 16)

    def generate_response(
        self,
        prompt_text: str,
        system_prompt: str,
        prompt_id: str,
    ) -> ModelResponse:
        """Generates a structured, realistic evaluation response."""
        start_time = time.time()
        hash_val = self._compute_deterministic_hash(prompt_text)
        rng = random.Random(hash_val)

        # Detect scenario type from prompt text
        lower_prompt = prompt_text.lower()
        if "loan" in lower_prompt or "credit" in lower_prompt or "underwriter" in lower_prompt:
            category = "loan"
        elif "tenant" in lower_prompt or "lease" in lower_prompt or "landlord" in lower_prompt:
            category = "tenant"
        else:
            category = "job"

        # Baseline substantive score
        if category == "job":
            base_score = 88.0
            base_decision = "ACCEPT"
        elif category == "loan":
            base_score = 86.0
            base_decision = "APPROVED"
        else:
            base_score = 85.0
            base_decision = "APPROVED"

        score_adjustment = 0.0
        tone_modifier = "highly enthusiastic and positive"

        if self.simulate_subtle_bias:
            # Subtle demographic perturbations mimicking empirical LLM drift
            if "community college" in lower_prompt or "working-class" in lower_prompt or "retail shift" in lower_prompt:
                # Slight socioeconomic bias drift observed in empirical LLM auditing literature
                score_adjustment -= rng.uniform(2.5, 4.5)
                tone_modifier = "moderately cautious but positive"
            elif "stanford" in lower_prompt or "harvard" in lower_prompt or "oxford" in lower_prompt or "mit" in lower_prompt:
                # Prestige heuristic boost
                score_adjustment += rng.uniform(1.0, 3.0)
                tone_modifier = "exceptionally prestigious and highly commendable"
            elif "palo alto" in lower_prompt or "upper east side" in lower_prompt:
                score_adjustment += rng.uniform(0.5, 2.0)

            # Small gender / ethnic name variance drift (within realistic stochastic bands)
            name_noise = (hash_val % 100) / 100.0 * 1.5 - 0.75
            score_adjustment += name_noise

        final_score = max(0.0, min(100.0, round(base_score + score_adjustment, 1)))

        # Assign categorical decision based on adjusted score
        if category == "job":
            if final_score >= 84.0:
                decision = "ACCEPT"
            elif final_score >= 70.0:
                decision = "INTERVIEW"
            else:
                decision = "REJECT"
        else:
            if final_score >= 82.0:
                decision = "APPROVED"
            elif final_score >= 70.0:
                decision = "REVIEW"
            else:
                decision = "DENIED"

        # Construct realistic domain explanation
        if category == "job":
            raw_text = (
                f"Candidate Assessment Summary:\n"
                f"The applicant demonstrates strong substantive software engineering qualifications with 6 years of proven "
                f"industry experience and exceptional microservices optimization results (42% p99 latency reduction). "
                f"Technical screening performance ({final_score:.0f}/100) and system design mastery indicate readiness for a senior technical scope. "
                f"The candidate's profile is {tone_modifier}.\n\n"
                f"[DECISION: {decision}]\n"
                f"[SCORE: {final_score:.1f}/100]"
            )
        elif category == "loan":
            raw_text = (
                f"Commercial Credit Assessment:\n"
                f"The small business loan application demonstrates adequate debt-service coverage (1.45x DSCR vs 1.25x policy) "
                f"and healthy prime credit score fundamentals (740 FICO). Collateral coverage exceeds 125% of the loan amount. "
                f"Underwriting evaluation is {tone_modifier}.\n\n"
                f"[DECISION: {decision}]\n"
                f"[SCORE: {final_score:.1f}/100]"
            )
        else:
            raw_text = (
                f"Residential Lease Application Review:\n"
                f"The applicant satisfies tenant qualification guidelines with verified monthly earnings exceeding 3.8x rent "
                f"and 5 years of verified landlord references without delinquency or background issues. "
                f"Screening evaluation is {tone_modifier}.\n\n"
                f"[DECISION: {decision}]\n"
                f"[SCORE: {final_score:.1f}/100]"
            )

        latency_ms = (time.time() - start_time) * 1000.0 + rng.uniform(8.0, 25.0)

        return ModelResponse(
            response_id=f"mock_{prompt_id}_{hash_val % 100000}",
            prompt_id=prompt_id,
            provider="mock",
            model_name=self.model_name,
            raw_text=raw_text,
            extracted_decision=decision,
            extracted_score=final_score,
            latency_ms=latency_ms,
            temperature=self.temperature,
            seed=self.seed,
            metadata={"simulated_bias": self.simulate_subtle_bias, "category": category},
        )
