"""
OpenAI API Provider for PromptBench (GPT-4o, GPT-4o-mini, GPT-3.5-turbo).
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


class OpenAIProvider(BaseLLMProvider):
    """
    Direct REST integration with OpenAI Chat Completions API.
    Implements exponential backoff with full jitter to handle HTTP 429 rate limits.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        seed: int = 42,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
    ):
        super().__init__(model_name=model_name, temperature=temperature, seed=seed)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    def generate_response(
        self,
        prompt_text: str,
        system_prompt: str,
        prompt_id: str,
    ) -> ModelResponse:
        """Sends request to OpenAI API with exponential backoff and jitter."""
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is missing. Set OPENAI_API_KEY environment variable "
                "or pass api_key to OpenAIProvider."
            )

        endpoint = "https://api.openai.com/v1/chat/completions"
        
        payload = {
            "model": self.model_name,
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": 500,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text},
            ],
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        start_time = time.time()
        last_error = None

        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(endpoint, data=data_bytes, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=30) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    choices = resp_data.get("choices", [])
                    raw_text = choices[0]["message"]["content"] if choices else "ERROR: No response choices."

                    latency_ms = (time.time() - start_time) * 1000.0
                    decision, score = self.parse_decision_and_score(raw_text)

                    return ModelResponse(
                        response_id=f"openai_{prompt_id}_{int(time.time()*1000)}",
                        prompt_id=prompt_id,
                        provider="openai",
                        model_name=self.model_name,
                        raw_text=raw_text,
                        extracted_decision=decision,
                        extracted_score=score,
                        latency_ms=latency_ms,
                        temperature=self.temperature,
                        seed=self.seed,
                        metadata={"attempt": attempt + 1, "usage": resp_data.get("usage", {})},
                    )

            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
                last_error = e
                backoff_cap = min(self.max_delay, self.base_delay * (2 ** attempt))
                sleep_duration = random.uniform(0, backoff_cap)
                time.sleep(sleep_duration)

        latency_ms = (time.time() - start_time) * 1000.0
        return ModelResponse(
            response_id=f"openai_err_{prompt_id}",
            prompt_id=prompt_id,
            provider="openai",
            model_name=self.model_name,
            raw_text=f"API Error after {self.max_retries} retries: {str(last_error)}",
            extracted_decision="ERROR",
            extracted_score=None,
            latency_ms=latency_ms,
            temperature=self.temperature,
            seed=self.seed,
            metadata={"error": str(last_error)},
        )
