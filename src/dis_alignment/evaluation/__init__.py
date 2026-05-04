"""Evaluation utilities for DeepAlign experiments and legacy baselines."""

from dis_alignment.evaluation.common import (
    add_method_variant_column,
    check_success_criteria,
    merge_evaluation_results,
    parse_methods,
    save_evaluation_results,
    summarize_evaluation,
)
from dis_alignment.evaluation.mazurka import evaluate_mazurka_dataset
from dis_alignment.evaluation.metrics import (
    alignment_rate,
    coverage,
    mean_absolute_error,
)
from dis_alignment.evaluation.swd import (
    evaluate_pair,
    evaluate_swd_dataset,
)

__all__ = [
    "alignment_rate",
    "add_method_variant_column",
    "check_success_criteria",
    "coverage",
    "evaluate_mazurka_dataset",
    "evaluate_pair",
    "evaluate_swd_dataset",
    "mean_absolute_error",
    "merge_evaluation_results",
    "parse_methods",
    "save_evaluation_results",
    "summarize_evaluation",
]
