"""Evaluation utilities for DeepAlign experiments and legacy baselines."""

from dis_alignment.evaluation.metrics import (
    mean_absolute_error,
    alignment_rate,
    coverage,
)
from dis_alignment.evaluation.swd import (
    check_success_criteria,
    evaluate_pair,
    evaluate_swd_dataset,
    save_evaluation_results,
    summarize_evaluation,
)

__all__ = [
    "alignment_rate",
    "check_success_criteria",
    "coverage",
    "evaluate_pair",
    "evaluate_swd_dataset",
    "mean_absolute_error",
    "save_evaluation_results",
    "summarize_evaluation",
]
