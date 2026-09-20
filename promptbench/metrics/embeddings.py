"""
Embedding-Based Semantic Cosine Similarity Scorer for PromptBench.
Measures semantic vector representation drift between base and perturbed responses.
"""

import math
import re
from typing import List, Dict, Optional, Tuple


class CosineEmbeddingScorer:
    """
    Computes semantic cosine similarity between text responses.
    Features a dual-engine architecture:
    1. Fast vectorized n-gram TF-IDF embedding (zero external dependencies)
    2. Optional Transformer-based sentence embeddings (sentence-transformers / MiniLM)
    """

    def __init__(self, use_transformer: bool = False, model_name: str = "all-MiniLM-L6-v2"):
        self.use_transformer = use_transformer
        self.model_name = model_name
        self._st_model = None

        if self.use_transformer:
            try:
                from sentence_transformers import SentenceTransformer
                self._st_model = SentenceTransformer(model_name)
            except ImportError:
                self.use_transformer = False
                self._st_model = None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenizes text into normalized word tokens."""
        return re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())

    def _get_tfidf_vector(self, text1: str, text2: str) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Constructs joint TF-IDF / term frequency vectors across two comparison texts."""
        tokens1 = self._tokenize(text1)
        tokens2 = self._tokenize(text2)

        # Build vocabulary & term frequencies
        vocab = set(tokens1).union(set(tokens2))
        
        # Word bigrams for syntactic context
        bigrams1 = [f"{tokens1[i]}_{tokens1[i+1]}" for i in range(len(tokens1)-1)]
        bigrams2 = [f"{tokens2[i]}_{tokens2[i+1]}" for i in range(len(tokens2)-1)]
        vocab.update(bigrams1)
        vocab.update(bigrams2)

        tf1: Dict[str, float] = {}
        tf2: Dict[str, float] = {}

        for t in tokens1:
            tf1[t] = tf1.get(t, 0.0) + 1.0
        for bg in bigrams1:
            tf1[bg] = tf1.get(bg, 0.0) + 1.5

        for t in tokens2:
            tf2[t] = tf2.get(t, 0.0) + 1.0
        for bg in bigrams2:
            tf2[bg] = tf2.get(bg, 0.0) + 1.5

        # Sub-linear term frequency scaling: w = 1 + ln(tf) if tf > 0
        v1 = {w: (1.0 + math.log(tf1[w])) if w in tf1 else 0.0 for w in vocab}
        v2 = {w: (1.0 + math.log(tf2[w])) if w in tf2 else 0.0 for w in vocab}

        return v1, v2

    def compute_similarity(self, text_base: str, text_pert: str) -> float:
        """
        Computes cosine similarity:
            S_C(u, v) = (u . v) / (||u||_2 * ||v||_2)
        Range: [-1.0, 1.0], normalized to [0.0, 1.0] for text similarity.
        """
        if not text_base.strip() or not text_pert.strip():
            return 0.0

        if text_base.strip() == text_pert.strip():
            return 1.0

        if self.use_transformer and self._st_model is not None:
            # Transformer embedding cosine similarity
            emb1 = self._st_model.encode(text_base)
            emb2 = self._st_model.encode(text_pert)
            dot = sum(a * b for a, b in zip(emb1, emb2))
            norm1 = math.sqrt(sum(a * a for a in emb1))
            norm2 = math.sqrt(sum(b * b for b in emb2))
            if norm1 == 0 or norm2 == 0:
                return 0.0
            cos_sim = dot / (norm1 * norm2)
            return float(max(0.0, min(1.0, cos_sim)))

        # Fallback high-performance n-gram TF-IDF cosine similarity
        v1, v2 = self._get_tfidf_vector(text_base, text_pert)

        dot_product = sum(v1[w] * v2[w] for w in v1)
        norm1 = math.sqrt(sum(val * val for val in v1.values()))
        norm2 = math.sqrt(sum(val * val for val in v2.values()))

        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        return float(max(0.0, min(1.0, round(similarity, 4))))
