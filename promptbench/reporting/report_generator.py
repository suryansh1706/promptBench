"""
Automated Academic Research & LaTeX Report Generator for PromptBench.
Produces publication-ready LaTeX tables, Markdown executive summaries, and fairness scorecards.
"""

from typing import List, Dict, Any, Optional
import time
from ..core.types import AuditSummary, HypothesisTestResult, PairwiseComparison


class ReportGenerator:
    """
    Generates academic research reports and formatted LaTeX tables for Minor Project paper submissions.
    """

    @staticmethod
    def generate_latex_table(results: List[HypothesisTestResult]) -> str:
        """
        Formats hypothesis test results into a standard IEEE/NeurIPS LaTeX table.
        """
        latex = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Demographic Bias and Response Drift Audit Results with Multiple Comparisons Correction.}",
            r"\label{tab:promptbench_results}",
            r"\small",
            r"\begin{tabular}{llccccc}",
            r"\toprule",
            r"\textbf{Demographic Axis} & \textbf{Target Subgroup} & \textbf{Observed $\Delta$} & \textbf{Cohen's $d$} & \textbf{Raw $p$-val} & \textbf{BH FDR $p$-val} & \textbf{95\% Bootstrap CI} \\",
            r"\midrule",
        ]

        for r in results:
            ci_str = f"[{r.bootstrap_ci_95[0]:+.3f}, {r.bootstrap_ci_95[1]:+.3f}]"
            sig_marker = r"$^{*}$" if r.is_significant_fdr else ""
            latex.append(
                f"{r.demographic_axis.capitalize()} & {r.target_group} & {r.observed_difference:+.4f} & "
                f"{r.effect_size_cohens_d:+.2f} & {r.raw_p_value:.4f} & {r.benjamini_hochberg_p_value:.4f}{sig_marker} & {ci_str} \\\\"
            )

        latex.extend([
            r"\bottomrule",
            r"\multicolumn{7}{l}{\footnotesize $^{*}$Statistically significant after Benjamini-Hochberg False Discovery Rate control at $\alpha=0.05$.} \\",
            r"\end{tabular}",
            r"\end{table*}",
        ])

        return "\n".join(latex)

    @staticmethod
    def generate_markdown_summary(summary: AuditSummary) -> str:
        """
        Creates a clean, comprehensive executive summary in GitHub-Flavored Markdown.
        """
        lines = [
            f"# PromptBench Bias Audit Executive Report",
            f"**Audit ID:** `{summary.audit_id}`  ",
            f"**Evaluated Model:** `{summary.model_name}` ({summary.provider.capitalize()})  ",
            f"**Scenario Domain:** `{summary.scenario_category}`  ",
            f"**Audit Date:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(summary.timestamp))}  ",
            "",
            "## 1. High-Level Metrics Overview",
            f"- **Total Prompts Tested:** {summary.total_prompts_tested}",
            f"- **Total Counterfactual Pairs Evaluated:** {summary.total_pairs_evaluated}",
            f"- **Mean Semantic Embedding Cosine Similarity:** `{summary.overall_mean_similarity:.4f}`",
            f"- **Mean Sentiment Delta ($\Delta$ Sentiment):** `{summary.overall_mean_sentiment_delta:+.4f}`",
            f"- **Decision Consistency Rate:** `{summary.overall_decision_consistency_rate * 100:.1f}%`",
            "",
            "## 2. Statistical Hypothesis Testing & Significance",
            "| Axis | Subgroup | Observed Delta | Permutation $p$ | BH FDR $p$ | 95% Bootstrap CI | Significant Bias? |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]

        for r in summary.hypothesis_test_results:
            sig = "**YES (p < 0.05)**" if r.is_significant_fdr else "No (p >= 0.05)"
            ci = f"[{r.bootstrap_ci_95[0]:+.3f}, {r.bootstrap_ci_95[1]:+.3f}]"
            lines.append(
                f"| {r.demographic_axis.capitalize()} | {r.target_group} | {r.observed_difference:+.4f} | "
                f"{r.permutation_p_value:.4f} | {r.benjamini_hochberg_p_value:.4f} | {ci} | {sig} |"
            )

        lines.extend([
            "",
            "## 3. Findings & Remediation Recommendations",
        ])
        for rec in summary.audit_recommendations:
            lines.append(f"- {rec}")

        return "\n".join(lines)
