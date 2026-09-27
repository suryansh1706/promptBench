"""
Groq API Provider for PromptBench (GPT-OSS models served on Groq's LPU inference).
Groq exposes an OpenAI-compatible Chat Completions endpoint.

NOTE: This provider shells out to the system's curl.exe binary instead of using
Python's urllib. Groq's Cloudflare front-end blocks requests with Python's default
TLS/HTTP client fingerprint (returns HTTP 403 even with a valid API key), but accepts
requests from curl. Requires curl to be installed and on PATH (built into Windows 10+,
macOS, and most Linux distros by default).
"""

import os
import json
import time
import random
import tempfile
import subprocess
from typing import Optional, Dict, Any
from .base import BaseLLMProvider
from ...core.types import ModelResponse


class GroqProvider(BaseLLMProvider):
    """
    REST integration with Groq's OpenAI-compatible Chat Completions API, using curl
    as the transport (see module docstring for why urllib doesn't work here).
    Implements exponential backoff with full jitter to handle rate limits and
    transient failures.
    """

    def __init__(
        self,
        model_name: str = "openai/gpt-oss-20b",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        seed: int = 42,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
    ):
        super().__init__(model_name=model_name, temperature=temperature, seed=seed)
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    def generate_response(
        self,
        prompt_text: str,
        system_prompt: str,
        prompt_id: str,
    ) -> ModelResponse:
        """Sends request to Groq's Chat Completions API via curl, with exponential backoff and jitter."""
        if not self.api_key:
            raise ValueError(
                "Groq API key is missing. Set GROQ_API_KEY environment variable "
                "or pass api_key to GroqProvider."
            )

        endpoint = "https://api.groq.com/openai/v1/chat/completions"

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

        start_time = time.time()
        last_error = None

        # Write the JSON payload to a temp file so curl reads it with -d @file,
        # avoiding shell quoting/escaping issues entirely.
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".json", text=True)
        try:
            with os.fdopen(tmp_fd, "w", encoding="ascii") as f:
                json.dump(payload, f)

            curl_cmd = [
                "curl.exe" if os.name == "nt" else "curl",
                endpoint,
                "-s",  # silent mode: suppress curl's own progress output
                "-H", "Content-Type: application/json",
                "-H", f"Authorization: Bearer {self.api_key}",
                "-H", "User-Agent: PromptBench-Research/1.0",
                "-d", f"@{tmp_path}",
                "--max-time", "30",
            ]

            for attempt in range(self.max_retries):
                try:
                    result = subprocess.run(
                        curl_cmd,
                        capture_output=True,
                        text=True,
                        timeout=35,
                    )

                    if result.returncode != 0:
                        raise RuntimeError(f"curl exited with code {result.returncode}: {result.stderr.strip()}")

                    resp_data = json.loads(result.stdout)

                    if "error" in resp_data:
                        raise RuntimeError(f"Groq API error: {resp_data['error'].get('message', resp_data['error'])}")

                    choices = resp_data.get("choices", [])
                    raw_text = choices[0]["message"]["content"] if choices else "ERROR: No response choices."

                    latency_ms = (time.time() - start_time) * 1000.0
                    decision, score = self.parse_decision_and_score(raw_text)

                    return ModelResponse(
                        response_id=f"groq_{prompt_id}_{int(time.time()*1000)}",
                        prompt_id=prompt_id,
                        provider="groq",
                        model_name=self.model_name,
                        raw_text=raw_text,
                        extracted_decision=decision,
                        extracted_score=score,
                        latency_ms=latency_ms,
                        temperature=self.temperature,
                        seed=self.seed,
                        metadata={"attempt": attempt + 1, "usage": resp_data.get("usage", {})},
                    )

                except (RuntimeError, json.JSONDecodeError, subprocess.TimeoutExpired) as e:
                    last_error = e
                    backoff_cap = min(self.max_delay, self.base_delay * (2 ** attempt))
                    sleep_duration = random.uniform(0, backoff_cap)
                    time.sleep(sleep_duration)
        finally:
            # Clean up the temp payload file regardless of success/failure.
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        latency_ms = (time.time() - start_time) * 1000.0
        return ModelResponse(
            response_id=f"groq_err_{prompt_id}",
            prompt_id=prompt_id,
            provider="groq",
            model_name=self.model_name,
            raw_text=f"API Error after {self.max_retries} retries: {str(last_error)}",
            extracted_decision="ERROR",
            extracted_score=None,
            latency_ms=latency_ms,
            temperature=self.temperature,
            seed=self.seed,
            metadata={"error": str(last_error)},
        )