"""
Dataset Exporter for PromptBench.
Exports standardized, reusable, and anonymized benchmark datasets in JSON, JSONL, and CSV formats.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from ..core.types import PairwiseComparison, AuditSummary, HypothesisTestResult


class DatasetExporter:
    """
    Exports audit benchmark datasets for open-science AI safety research.
    """

    @staticmethod
    def export_to_json(
        summary: AuditSummary,
        comparisons: List[PairwiseComparison],
        output_path: Path,
    ):
        """Exports the full audit run and all pairwise comparisons to a structured JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "summary": summary.to_dict(),
            "comparisons": [c.to_dict() for c in comparisons],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def export_to_jsonl(comparisons: List[PairwiseComparison], output_path: Path):
        """Exports comparisons as streaming JSON Lines."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for c in comparisons:
                f.write(json.dumps(c.to_dict()) + "\n")

    @staticmethod
    def export_to_csv(comparisons: List[PairwiseComparison], output_path: Path):
        """Exports tabular metrics as a flat CSV file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not comparisons:
            return

        fieldnames = [
            "pair_id", "template_id", "scenario_category", "demographic_axis",
            "demographic_group", "baseline_group", "embedding_cosine_similarity",
            "sentiment_base", "sentiment_perturbed", "sentiment_delta",
            "decision_base", "decision_perturbed", "decision_matches",
            "score_base", "score_perturbed", "score_delta",
            "lexical_jaccard_similarity", "token_length_delta"
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for c in comparisons:
                writer.writerow(c.to_dict())
