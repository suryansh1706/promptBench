"""
Deterministic Content-Addressable Response Cache for PromptBench.
Guarantees experimental reproducibility and minimizes API consumption.
"""

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any
from .types import ModelResponse


class ResponseCache:
    """
    Persistent SQLite-backed cache keyed on the deterministic SHA-256 hash
    of the prompt payload, model version, system prompt, temperature, and random seed.
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "promptbench_cache.sqlite3"
        self._init_db()
        self.hits = 0
        self.misses = 0

    def _init_db(self):
        """Initializes the cache SQLite table."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS response_cache (
                    cache_key TEXT PRIMARY KEY,
                    prompt_id TEXT,
                    provider TEXT,
                    model_name TEXT,
                    raw_text TEXT,
                    extracted_decision TEXT,
                    extracted_score REAL,
                    latency_ms REAL,
                    temperature REAL,
                    seed INTEGER,
                    timestamp REAL,
                    metadata_json TEXT
                )
            """)
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def generate_key(
        rendered_prompt: str,
        system_prompt: str,
        provider: str,
        model_name: str,
        temperature: float,
        seed: Optional[int] = 42,
    ) -> str:
        """
        Computes a SHA-256 cryptographic digest of all query parameters.
        Algorithm:
            Key = SHA256( normalized_prompt || normalized_system || provider || model || temp || seed )
        """
        normalized_str = (
            f"PROMPT:{rendered_prompt.strip()}||"
            f"SYS:{system_prompt.strip()}||"
            f"PROV:{provider.lower()}||"
            f"MODEL:{model_name.lower()}||"
            f"TEMP:{temperature:.4f}||"
            f"SEED:{seed}"
        )
        return hashlib.sha256(normalized_str.encode("utf-8")).hexdigest()

    def get(self, cache_key: str) -> Optional[ModelResponse]:
        """Look up a cached response by key."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT prompt_id, provider, model_name, raw_text, 
                       extracted_decision, extracted_score, latency_ms, 
                       temperature, seed, timestamp, metadata_json
                FROM response_cache
                WHERE cache_key = ?
                """,
                (cache_key,),
            )
            row = cursor.fetchone()
            if row:
                self.hits += 1
                metadata = json.loads(row[10]) if row[10] else {}
                return ModelResponse(
                    response_id=f"cached_{cache_key[:12]}",
                    prompt_id=row[0],
                    provider=row[1],
                    model_name=row[2],
                    raw_text=row[3],
                    extracted_decision=row[4],
                    extracted_score=row[5],
                    latency_ms=row[6],
                    temperature=row[7],
                    seed=row[8],
                    timestamp=row[9],
                    cache_hit=True,
                    metadata=metadata,
                )
        finally:
            conn.close()
        self.misses += 1
        return None

    def set(self, cache_key: str, response: ModelResponse):
        """Store a model response in the persistent cache."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO response_cache
                (cache_key, prompt_id, provider, model_name, raw_text,
                 extracted_decision, extracted_score, latency_ms, temperature, seed, timestamp, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cache_key,
                    response.prompt_id,
                    response.provider,
                    response.model_name,
                    response.raw_text,
                    response.extracted_decision,
                    response.extracted_score,
                    response.latency_ms,
                    response.temperature,
                    response.seed,
                    response.timestamp,
                    json.dumps(response.metadata),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_stats(self) -> Dict[str, Any]:
        """Return cache hit rate and total entry count."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total) if total > 0 else 0.0
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM response_cache")
            count = cursor.fetchone()[0]
        finally:
            conn.close()
        return {
            "total_cached_entries": count,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_percentage": round(hit_rate * 100, 2),
        }
