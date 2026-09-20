"""
PromptBench Command-Line Interface (CLI).
Allows researchers to run end-to-end audits, perturbation generation, and statistical validations from the terminal.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

from .core.config import PromptBenchConfig
from .core.types import AuditSummary
from .perturbation.engine import PerturbationEngine
from .perturbation.templates import SCENARIO_TEMPLATES, get_template_by_id
from .query_pipeline.orchestrator import QueryOrchestrator
from .query_pipeline.providers.base import BaseLLMProvider
from .query_pipeline.providers.mock_provider import MockLLMProvider
from .query_pipeline.providers.gemini_provider import GeminiProvider
from .query_pipeline.providers.openai_provider import OpenAIProvider
from .metrics.scorer import BiasScorer
from .metrics.decision_consistency import DecisionConsistencyScorer
from .statistical_testing.engine import StatisticalTestingEngine
from .reporting.report_generator import ReportGenerator
from .reporting.dataset_exporter import DatasetExporter


def run_audit(args):
    """Executes an end-to-end bias audit."""
    print("=" * 70)
    print("PromptBench: LLM Demographic Bias Auditing Framework")
    print(f"Scenario: {args.scenario} | Provider: {args.provider} | Model: {args.model}")
    print("=" * 70)

    config = PromptBenchConfig()
    config.provider.provider_name = args.provider
    config.provider.model_name = args.model
    config.provider.temperature = args.temperature
    config.provider.seed = args.seed

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Perturbation Engine
    print("\n[Stage 1/5] Running Perturbation Engine...")
    engine = PerturbationEngine()
    template_ids = [args.scenario] if args.scenario != "all" else [t.template_id for t in SCENARIO_TEMPLATES]
    all_prompts, paired_prompts = engine.generate_counterfactual_dataset(
        template_ids=template_ids,
        names_per_group=args.samples_per_group,
    )
    print(f" -> Generated {len(all_prompts)} prompts across {len(paired_prompts)} counterfactual pairs.")

    # 2. LLM Query Pipeline
    print(f"\n[Stage 2/5] Querying Model ({args.provider}:{args.model})...")
    orchestrator = QueryOrchestrator.from_config(config)

    def on_progress(curr, total, resp):
        if curr % max(1, total // 10) == 0 or curr == total:
            print(f" -> Progress: {curr}/{total} prompts evaluated ({curr/total*100:.1f}%)")

    responses = orchestrator.execute_batch(all_prompts, progress_callback=on_progress)
    print(f" -> Completed {len(responses)} LLM queries.")

    # 3. Bias & Response Drift Scoring
    print("\n[Stage 3/5] Scoring Semantic, Sentiment, and Decision Disparities...")
    scorer = BiasScorer()
    comparisons = scorer.evaluate_batch_pairs(paired_prompts, responses)
    print(f" -> Evaluated {len(comparisons)} paired counterfactual comparisons.")

    # 4. Statistical Hypothesis Testing
    print("\n[Stage 4/5] Executing Paired Permutation Tests & Benjamini-Hochberg FDR...")
    stats_engine = StatisticalTestingEngine(config.stats)
    hyp_results = stats_engine.analyze_comparisons(comparisons)
    print(f" -> Completed statistical tests for {len(hyp_results)} demographic subgroups.")

    # Compute aggregate summary
    mean_sim = sum(c.embedding_cosine_similarity for c in comparisons) / len(comparisons) if comparisons else 1.0
    mean_sent_delta = sum(c.sentiment_delta for c in comparisons) / len(comparisons) if comparisons else 0.0
    match_rate = sum(1 for c in comparisons if c.decision_matches) / len(comparisons) if comparisons else 1.0

    # EEOC Disparate impact check
    group_decisions = {}
    for c in comparisons:
        grp = c.demographic_group
        if grp not in group_decisions:
            group_decisions[grp] = []
        group_decisions[grp].append(c.decision_perturbed)
    fairness_metrics = DecisionConsistencyScorer.compute_demographic_parity_and_disparate_impact(group_decisions)

    recommendations = []
    if not fairness_metrics.get("passes_four_fifths_rule", True):
        recommendations.append(
            f"ALERT: Disparate Impact Ratio ({fairness_metrics['disparate_impact_ratio']:.2f}) violates EEOC Four-Fifths rule (< 0.80)."
        )
    significant_biases = [r for r in hyp_results if r.is_significant_fdr]
    if significant_biases:
        recommendations.append(
            f"Statistically significant demographic drift confirmed in {len(significant_biases)} subgroups after Benjamini-Hochberg FDR correction."
        )
    else:
        recommendations.append("No statistically significant demographic drift detected under evaluated thresholds.")

    summary = AuditSummary(
        audit_id=f"audit_{int(time.time())}",
        timestamp=time.time(),
        provider=args.provider,
        model_name=args.model,
        scenario_category=args.scenario,
        total_prompts_tested=len(all_prompts),
        total_pairs_evaluated=len(comparisons),
        overall_mean_similarity=round(mean_sim, 4),
        overall_mean_sentiment_delta=round(mean_sent_delta, 4),
        overall_decision_consistency_rate=round(match_rate, 4),
        axis_summaries=fairness_metrics,
        hypothesis_test_results=hyp_results,
        audit_recommendations=recommendations,
    )

    # 5. Reporting & Dataset Export
    print("\n[Stage 5/5] Exporting Research Artifacts...")
    json_path = output_dir / "audit_dataset.json"
    csv_path = output_dir / "audit_comparisons.csv"
    report_md_path = output_dir / "audit_report.md"
    latex_path = output_dir / "table_results.tex"

    DatasetExporter.export_to_json(summary, comparisons, json_path)
    DatasetExporter.export_to_csv(comparisons, csv_path)
    
    report_md = ReportGenerator.generate_markdown_summary(summary)
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    latex_code = ReportGenerator.generate_latex_table(hyp_results)
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_code)

    print(f" -> JSON Dataset:  {json_path}")
    print(f" -> CSV Table:     {csv_path}")
    print(f" -> Markdown Summary: {report_md_path}")
    print(f" -> LaTeX Table:   {latex_path}")

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE - Summary Findings:")
    print(f"  * Semantic Cosine Similarity: {mean_sim:.4f}")
    print(f"  * Mean Sentiment Delta:       {mean_sent_delta:+.4f}")
    print(f"  * Decision Consistency Rate:  {match_rate * 100:.1f}%")
    print(f"  * Disparate Impact Ratio:     {fairness_metrics.get('disparate_impact_ratio', 1.0):.2f}")
    print(f"  * Significant Biases (FDR):   {len(significant_biases)}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="PromptBench: Demographic Bias Auditing CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Subcommand: audit
    audit_parser = subparsers.add_parser("audit", help="Run end-to-end bias audit")
    audit_parser.add_argument("--scenario", type=str, default="job_swe_001", help="Template ID or 'all'")
    audit_parser.add_argument("--provider", type=str, default="mock", choices=["mock", "gemini", "openai"])
    audit_parser.add_argument("--model", type=str, default="mock-llm-v1", help="Model name")
    audit_parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    audit_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    audit_parser.add_argument("--samples-per-group", type=int, default=2, help="Names per demographic category")
    audit_parser.add_argument("--output-dir", type=str, default="./results", help="Directory to save audit artifacts")

    args = parser.parse_args()

    if args.command == "audit" or args.command is None:
        if args.command is None:
            # Default run with sample settings
            args = parser.parse_args(["audit"])
        run_audit(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
