"""Analysis and visualization tools for DeepAlign experiment results."""

from dis_alignment.analysis.visualize import (
    plot_error_vs_length,
    plot_runtime_comparison,
    plot_alignment_path,
    plot_feature_comparison,
)
from dis_alignment.analysis.statistics import (
    summarize_results,
    failure_analysis,
    compute_significance,
)

__all__ = [
    "plot_error_vs_length",
    "plot_runtime_comparison", 
    "plot_alignment_path",
    "plot_feature_comparison",
    "summarize_results",
    "failure_analysis",
    "compute_significance",
]
