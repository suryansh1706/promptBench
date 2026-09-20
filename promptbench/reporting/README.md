# Stage 5: Reporting & Artifact Generation (`promptbench/reporting`)

The **Reporting Module** is Stage 5 of PromptBench. It converts raw pairwise comparisons and statistical hypothesis tests into standardized, reproducible research artifacts, including publication-ready LaTeX tables, Markdown executive summaries, and open-science benchmark datasets (JSON, JSONL, CSV).

---

## 1. Capabilities

1. **Academic LaTeX Formatter (`report_generator.py`)**:
   - Generates IEEE / ACM / NeurIPS compliant LaTeX tables with sample size, observed effect size ($d$), raw $p$-values, Benjamini-Hochberg FDR corrected $p$-values, and $95\%$ bootstrap confidence intervals.
2. **Executive Markdown Report**:
   - Formats complete audit metrics with EEOC disparate impact flags and remediation guidance.
3. **Structured Dataset Exporter (`dataset_exporter.py`)**:
   - Multi-format exporter (`.json`, `.jsonl`, `.csv`) for downstream analysis in Pandas, R, Python, and Excel.

---

## 2. Usage Example

```python
from promptbench.reporting import ReportGenerator, DatasetExporter
from pathlib import Path

# Generate LaTeX table for paper
latex_code = ReportGenerator.generate_latex_table(hypothesis_test_results)
print(latex_code)

# Export complete benchmark dataset to JSON and CSV
DatasetExporter.export_to_json(summary, comparisons, Path("results/audit_dataset.json"))
DatasetExporter.export_to_csv(comparisons, Path("results/audit_table.csv"))
```
