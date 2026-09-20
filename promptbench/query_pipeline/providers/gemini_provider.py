"""
Google Gemini API Provider for PromptBench.
Uses standard HTTP/REST with exponential backoff and jitter for resilient execution.
"""

import os
import json
import time
import random
import urllib.request
import urllib.error
from typing import Optional, Dict, Any
from .base import BaseLLMProvider
from ...core.types import ModelResponse


class GeminiProvider(BaseLLMProvider):
    """
    Direct REST integration with Google Gemini (gemini-1.5-flash, gemini-1.5-pro, gemini-2.0-flash).
    Implements exponential backoff with full jitter to handle rate limits (HTTP 429).
    """

    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        seed: int = 42,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
    ):
        super().__init__(model_name=model_name, temperature=temperature, seed=seed)
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    def generate_response(
        self,
        prompt_text: str,
        system_prompt: str,
        prompt_id: str,
    ) -> ModelResponse:
        """Sends request to Gemini API with exponential backoff and jitter."""
        if not self.api_key:
            raise ValueError(
                "Gemini API key is missing. Set GEMINI_API_KEY environment variable "
                "or pass api_key to GeminiProvider."
            )

        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt_text}
                    ]
                }
            ],
            "systemInstruction": {
                "parts": [{"text": system_prompt}]
            },
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": 600,
            }
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}

        start_time = time.time()
        last_error = None

        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(endpoint, data=data_bytes, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=30) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    
                    # Extract generated candidate text
                    candidates = resp_data.get("candidates", [])
                    if candidates and "content" in candidates[0]:
                        parts = candidates[0]["content"].get("parts", [])
                        raw_text = "".join(p.get("text", "") for p in parts)
                    else:
                        raw_text = "ERROR: No response candidates returned."

                    latency_ms = (time.time() - start_time) * 1000.0
                    decision, score = self.parse_decision_and_score(raw_text)

                    return ModelResponse(
                        response_id=f"gemini_{prompt_id}_{int(time.time()*1000)}",
                        prompt_id=prompt_id,
                        provider="gemini",
                        model_name=self.model_name,
                        raw_text=raw_text,
                        extracted_decision=decision,
                        extracted_score=score,
                        latency_ms=latency_ms,
                        temperature=self.temperature,
                        seed=self.seed,
                        metadata={"attempt": attempt + 1},
                    )

            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
                last_error = e
                # Exponential backoff with full jitter: sleep = Uniform(0, min(max_delay, base * 2^attempt))
                backoff_cap = min(self.max_delay, self.base_delay * (2 ** attempt))
                sleep_duration = random.uniform(0, backoff_cap)
                time.sleep(sleep_duration)

        # Fallback failure response
        latency_ms = (time.time() - start_time) * 1000.0
        return ModelResponse(
            response_id=f"gemini_err_{prompt_id}",
            prompt_id=prompt_id,
            provider="gemini",
            model_name=self.model_name,
            raw_text=f"API Error after {self.max_retries} retries: {str(last_error)}",
            extracted_decision="ERROR",
            extracted_score=None,
            latency_ms=latency_ms,
            temperature=self.temperature,
            seed=self.seed,
            metadata={"error": str(last_error)},
        )
