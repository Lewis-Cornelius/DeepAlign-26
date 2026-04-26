"""Analysis and visualization tools for DeepAlign experiment results."""

from dis_alignment.analysis.visualize import (
    create_success_criteria_table,
    create_summary_table,
    plot_error_vs_length,
    plot_error_histogram,
    plot_runtime_comparison,
    plot_alignment_path,
    plot_feature_comparison,
    plot_metric_boxplot,
    plot_success_criteria_summary,
)
from dis_alignment.analysis.statistics import (
    compute_significance,
    failure_analysis,
    friedman_nemenyi_analysis,
    summarize_results,
)

__all__ = [
    "create_success_criteria_table",
    "create_summary_table",
    "plot_error_histogram",
    "plot_error_vs_length",
    "plot_metric_boxplot",
    "plot_runtime_comparison",
    "plot_alignment_path",
    "plot_feature_comparison",
    "plot_success_criteria_summary",
    "summarize_results",
    "failure_analysis",
    "compute_significance",
    "friedman_nemenyi_analysis",
]
