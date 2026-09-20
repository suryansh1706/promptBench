"""
End-to-End Test Suite for PromptBench.
Validates perturbation generation, cache deduplication, scoring metrics, and statistical hypothesis testing.
"""

import unittest
import sys
from pathlib import Path
import tempfile

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from promptbench.core.config import PromptBenchConfig
from promptbench.core.types import DemographicAxis, DecisionType
from promptbench.core.cache import ResponseCache
from promptbench.perturbation.templates import SCENARIO_TEMPLATES, get_template_by_id
from promptbench.perturbation.engine import PerturbationEngine
from promptbench.query_pipeline.providers.mock_provider import MockLLMProvider
from promptbench.query_pipeline.orchestrator import QueryOrchestrator
from promptbench.metrics.embeddings import CosineEmbeddingScorer
from promptbench.metrics.sentiment import SentimentScorer
from promptbench.metrics.decision_consistency import DecisionConsistencyScorer
from promptbench.metrics.lexical_drift import LexicalDriftScorer
from promptbench.metrics.scorer import BiasScorer
from promptbench.statistical_testing.permutation import PermutationTest
from promptbench.statistical_testing.bootstrap import BootstrapCI
from promptbench.statistical_testing.corrections import MultipleTestingCorrection
from promptbench.statistical_testing.engine import StatisticalTestingEngine
from promptbench.reporting.report_generator import ReportGenerator
from promptbench.reporting.dataset_exporter import DatasetExporter


class TestPromptBenchPipeline(unittest.TestCase):
    """Unit and Integration Tests for PromptBench."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self.temp_dir.name) / "cache"
        self.cache = ResponseCache(cache_dir=self.cache_dir)

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_stage1_perturbation_engine(self):
        """Test combinatorial prompt perturbation and counterfactual pairing."""
        engine = PerturbationEngine()
        all_prompts, paired_prompts = engine.generate_counterfactual_dataset(
            template_ids=["job_swe_001"], names_per_group=1
        )
        self.assertGreater(len(all_prompts), 0)
        self.assertGreater(len(paired_prompts), 0)
        
        # Verify baseline exists
        baseline = all_prompts[0]
        self.assertTrue(baseline.is_baseline)
        
        # Verify slot replacement
        for p in all_prompts:
            self.assertNotIn("{name}", p.rendered_prompt)
            self.assertNotIn("{education}", p.rendered_prompt)
            self.assertIn("6 years of progressive", p.rendered_prompt)

    def test_stage2_cache_and_mock_provider(self):
        """Test deterministic caching and mock provider generation."""
        mock_provider = MockLLMProvider()
        prompt_text = "Evaluate applicant Alex Morgan for senior role. Tech score: 88."
        system_prompt = "You are an evaluator."

        # Compute key and set cache
        key = ResponseCache.generate_key(
            rendered_prompt=prompt_text,
            system_prompt=system_prompt,
            provider="mock",
            model_name="mock-llm-v1",
            temperature=0.0,
            seed=42,
        )
        resp1 = mock_provider.generate_response(prompt_text, system_prompt, "p1")
        self.cache.set(key, resp1)

        cached_resp = self.cache.get(key)
        self.assertIsNotNone(cached_resp)
        self.assertTrue(cached_resp.cache_hit)
        self.assertEqual(resp1.extracted_decision, cached_resp.extracted_decision)

    def test_stage3_bias_metrics(self):
        """Test semantic cosine similarity, sentiment, and fairness metrics."""
        emb_scorer = CosineEmbeddingScorer()
        t1 = "Candidate demonstrates outstanding system design and stellar engineering performance."
        t2 = "Candidate demonstrates solid system design and strong engineering performance."
        sim = emb_scorer.compute_similarity(t1, t2)
        self.assertGreaterEqual(sim, 0.50)
        self.assertLessEqual(sim, 1.0)

        # Test Sentiment Scorer
        sent_scorer = SentimentScorer()
        s_pos = sent_scorer.score_text("Candidate is exceptional and stellar.")
        s_neg = sent_scorer.score_text("Candidate is poor and inadequate.")
        self.assertGreater(s_pos, 0.0)
        self.assertLess(s_neg, 0.0)

        # Test Decision Consistency & Disparate Impact
        decisions = {
            "group_a": ["ACCEPT", "ACCEPT", "ACCEPT", "ACCEPT"],
            "group_b": ["ACCEPT", "REJECT", "ACCEPT", "REJECT"],
        }
        fairness = DecisionConsistencyScorer.compute_demographic_parity_and_disparate_impact(decisions)
        self.assertAlmostEqual(fairness["disparate_impact_ratio"], 0.50, places=2)
        self.assertFalse(fairness["passes_four_fifths_rule"])

    def test_stage4_statistical_testing(self):
        """Test paired permutation tests, bootstrap CI, and Benjamini-Hochberg FDR."""
        perm = PermutationTest(resamples=1000, seed=42)
        # Symmetrical zero mean differences
        zero_diffs = [0.01, -0.01, 0.02, -0.02, 0.00]
        obs, p_val, _ = perm.test_paired_differences(zero_diffs)
        self.assertGreater(p_val, 0.05)

        # Strong non-zero bias differences (n=8 ensures p < 0.05)
        strong_diffs = [-0.15, -0.18, -0.14, -0.16, -0.17, -0.15, -0.19, -0.16]
        obs2, p_val2, _ = perm.test_paired_differences(strong_diffs)
        self.assertLess(p_val2, 0.05)

        # Test Bootstrap CI
        boot = BootstrapCI(resamples=500, seed=42)
        m, low, high = boot.compute_ci_mean(strong_diffs)
        self.assertLess(low, high)
        self.assertLess(high, 0.0)

        # Test Benjamini-Hochberg
        p_raw = [0.001, 0.01, 0.04, 0.20, 0.80]
        bh_adj, sig = MultipleTestingCorrection.benjamini_hochberg(p_raw, alpha=0.05)
        self.assertTrue(sig[0])
        self.assertFalse(sig[4])

    def test_stage5_end_to_end_audit_and_export(self):
        """Test full 5-stage pipeline integration."""
        engine = PerturbationEngine()
        all_prompts, pairs = engine.generate_counterfactual_dataset(
            template_ids=["job_swe_001"], names_per_group=1
        )
        provider = MockLLMProvider()
        orchestrator = QueryOrchestrator(provider=provider, cache=self.cache)
        responses = orchestrator.execute_batch(all_prompts)

        scorer = BiasScorer()
        comparisons = scorer.evaluate_batch_pairs(pairs, responses)
        self.assertEqual(len(comparisons), len(pairs))

        stats_engine = StatisticalTestingEngine()
        results = stats_engine.analyze_comparisons(comparisons)
        self.assertGreater(len(results), 0)

        # Test LaTeX table export
        latex = ReportGenerator.generate_latex_table(results)
        self.assertIn(r"\begin{table*}", latex)
        self.assertIn(r"\end{tabular}", latex)


if __name__ == "__main__":
    unittest.main()
