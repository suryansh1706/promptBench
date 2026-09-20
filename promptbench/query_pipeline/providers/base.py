"""
Abstract Base Class and Parser for LLM Query Providers in PromptBench.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Dict, Any
import re
import time
from ...core.types import ModelResponse, PerturbedPrompt


class BaseLLMProvider(ABC):
    """Abstract interface defining required methods for all LLM providers."""

    def __init__(self, model_name: str, temperature: float = 0.0, seed: int = 42):
        self.model_name = model_name
        self.temperature = temperature
        self.seed = seed

    @abstractmethod
    def generate_response(
        self,
        prompt_text: str,
        system_prompt: str,
        prompt_id: str,
    ) -> ModelResponse:
        """Executes a single prompt request against the underlying model."""
        pass

    @staticmethod
    def parse_decision_and_score(text: str) -> Tuple[Optional[str], Optional[float]]:
        """
        Extracts structured decision categories and numerical scores from raw LLM output.
        Supported decision tokens: ACCEPT, REJECT, INTERVIEW, APPROVED, DENIED, COSIGNER, REVIEW
        """
        decision: Optional[str] = None
        score: Optional[float] = None

        # Regex for structured decision tokens
        decision_pattern = r"\[DECISION:\s*([A-Za-z]+)\]"
        match_dec = re.search(decision_pattern, text, re.IGNORECASE)
        if match_dec:
            decision = match_dec.group(1).upper()
        else:
            # Fallback heuristic parsing
            lower = text.lower()
            if "accept" in lower or "hire" in lower or "approved" in lower:
                decision = "ACCEPT" if ("accept" in lower or "hire" in lower) else "APPROVED"
            elif "reject" in lower or "denied" in lower:
                decision = "REJECT" if "reject" in lower else "DENIED"
            elif "interview" in lower:
                decision = "INTERVIEW"
            elif "review" in lower or "cosigner" in lower:
                decision = "REVIEW"

        # Regex for structured score tokens [SCORE: 85] or [SCORE: 85/100]
        score_pattern = r"\[SCORE:\s*([0-9]+(?:\.[0-9]+)?)(?:/100)?\]"
        match_score = re.search(score_pattern, text, re.IGNORECASE)
        if match_score:
            try:
                score = float(match_score.group(1))
            except ValueError:
                score = None
        else:
            # Fallback numerical extraction (e.g. "Score: 88/100" or "Rating: 85")
            fb_match = re.search(r"(?:score|rating|evaluation)[:\s]+([0-9]{1,3}(?:\.[0-9]+)?)", text, re.IGNORECASE)
            if fb_match:
                try:
                    val = float(fb_match.group(1))
                    if 0 <= val <= 100:
                        score = val
                except ValueError:
                    score = None

        return decision, score
