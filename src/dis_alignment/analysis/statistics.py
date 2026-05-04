"""Statistical analysis and failure diagnostics for experiment results."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from dis_alignment.evaluation import add_method_variant_column


def summarize_results(
    results: pd.DataFrame,
    group_by: str | None = None,
) -> pd.DataFrame:
    """
    Generate summary statistics for evaluation results.

    Args:
        results: Results DataFrame.
        group_by: Column to group results by.

    Returns:
        Summary DataFrame with mean, std, min, max for key metrics.
    """
    results = add_method_variant_column(results)
    method_col = group_by or _method_column(results)
    metrics = [
        column
        for column in ["mae", "median_ae", "ar_50ms", "ar_100ms", "runtime_s", "memory_mb"]
        if column in results.columns
    ]
    return results.groupby(method_col)[metrics].agg(["mean", "std", "min", "max", "count"])


def failure_analysis(
    results: pd.DataFrame,
    mae_threshold: float = 0.5,
    ar_threshold: float = 0.8,
) -> dict[str, Any]:
    """
    Analyze failure cases where alignment quality is poor.

    Args:
        results: Evaluation results DataFrame.
        mae_threshold: MAE above this value is considered a failure.
        ar_threshold: Alignment rate below this is considered a failure.

    Returns:
        Dict containing failure statistics and analysis.
    """
    results = add_method_variant_column(results)
    failures = results[(results["mae"] > mae_threshold) | (results["ar_50ms"] < ar_threshold)]
    successes = results[(results["mae"] <= mae_threshold) & (results["ar_50ms"] >= ar_threshold)]

    analysis = {
        "total_cases": len(results),
        "failure_count": len(failures),
        "failure_rate": len(failures) / len(results) if len(results) > 0 else 0,
        "failures_by_algorithm": failures.groupby(_method_column(results)).size().to_dict(),
        "mean_duration_failures": failures["duration_s"].mean() if len(failures) > 0 else 0,
        "mean_duration_successes": successes["duration_s"].mean() if len(successes) > 0 else 0,
    }

    if len(failures) > 0 and len(successes) > 0:
        _, p_value = stats.ttest_ind(failures["duration_s"], successes["duration_s"])
        analysis["duration_ttest_pvalue"] = float(p_value)
        analysis["longer_pieces_fail_more"] = (
            float(failures["duration_s"].mean()) > float(successes["duration_s"].mean())
        )

    id_col = _id_column(results)
    results = add_method_variant_column(results)
    method_col = _method_column(results)
    analysis["worst_pieces"] = failures.nlargest(10, "mae")[
        [id_col, method_col, "mae", "ar_50ms", "duration_s"]
    ].to_dict("records")
    return analysis


def compute_significance(
    results: pd.DataFrame,
    baseline: str = "chroma_dtw",
    candidate: str = "deepalign",
    metric: str = "mae",
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Compute paired significance between two methods.

    Uses the Wilcoxon signed-rank test for paired comparisons.
    """
    results = add_method_variant_column(results)
    method_col = _method_column(results)
    id_col = _id_column(results)

    baseline_results = results[results[method_col] == baseline].set_index(id_col)
    candidate_results = results[results[method_col] == candidate].set_index(id_col)
    common_items = baseline_results.index.intersection(candidate_results.index)

    if len(common_items) < 5:
        return {"error": "Not enough paired observations", "n_pairs": len(common_items)}

    baseline_vals = baseline_results.loc[common_items, metric].values
    candidate_vals = candidate_results.loc[common_items, metric].values
    stat, p_value = stats.wilcoxon(baseline_vals, candidate_vals, alternative="two-sided")
    _, p_value_less = stats.wilcoxon(candidate_vals, baseline_vals, alternative="less")

    n_pairs = len(common_items)
    effect_size = 1 - (2 * float(stat)) / (n_pairs * (n_pairs + 1))
    baseline_mean = float(np.mean(baseline_vals))
    candidate_mean = float(np.mean(candidate_vals))
    improvement = baseline_mean - candidate_mean if _smaller_is_better(metric) else candidate_mean - baseline_mean
    improvement_pct = improvement / baseline_mean * 100 if baseline_mean != 0 else np.nan

    return {
        "baseline": baseline,
        "candidate": candidate,
        "metric": metric,
        "n_pairs": n_pairs,
        "baseline_mean": baseline_mean,
        "candidate_mean": candidate_mean,
        "improvement": float(improvement),
        "improvement_pct": float(improvement_pct),
        "wilcoxon_statistic": float(stat),
        "p_value_twosided": float(p_value),
        "p_value_candidate_better": float(p_value_less),
        "is_significant": bool(p_value < alpha),
        "candidate_is_better": bool(p_value_less < alpha),
        "effect_size": float(effect_size),
        "effect_interpretation": _interpret_effect_size(abs(effect_size)),
    }


def friedman_nemenyi_analysis(
    results: pd.DataFrame,
    metric: str = "mae",
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Run a Friedman omnibus test with Nemenyi post-hoc comparisons.

    Returns a dict that is friendly to CLI/JSON style output.
    """
    results = add_method_variant_column(results)
    method_col = _method_column(results)
    id_col = _id_column(results)
    pivot = results.pivot_table(index=id_col, columns=method_col, values=metric, aggfunc="mean")
    pivot = pivot.dropna(axis=0, how="any")

    n_blocks, n_methods = pivot.shape
    if n_methods < 3:
        return {
            "available": False,
            "error": "Friedman/Nemenyi requires at least three methods.",
            "n_items": n_blocks,
            "n_methods": n_methods,
        }
    if n_blocks < 5:
        return {
            "available": False,
            "error": "Not enough paired observations for Friedman/Nemenyi.",
            "n_items": n_blocks,
            "n_methods": n_methods,
        }

    rank_ascending = _smaller_is_better(metric)
    ranks = pivot.rank(axis=1, method="average", ascending=rank_ascending)
    statistic, p_value = stats.friedmanchisquare(*(pivot[column].values for column in pivot.columns))

    average_ranks = ranks.mean(axis=0).sort_values()
    q_crit = stats.studentized_range.ppf(1 - alpha, n_methods, np.inf) / np.sqrt(2)
    cd = float(q_crit * np.sqrt(n_methods * (n_methods + 1) / (6 * n_blocks)))

    pairwise: list[dict[str, Any]] = []
    methods = list(average_ranks.index)
    scale = np.sqrt(n_methods * (n_methods + 1) / (6 * n_blocks))
    for index, method_a in enumerate(methods):
        for method_b in methods[index + 1 :]:
            diff = abs(float(average_ranks[method_a] - average_ranks[method_b]))
            q_stat = diff / scale
            p_val = float(stats.studentized_range.sf(q_stat * np.sqrt(2), n_methods, np.inf))
            pairwise.append(
                {
                    "method_a": method_a,
                    "method_b": method_b,
                    "rank_diff": diff,
                    "p_value": p_val,
                    "significant": bool(diff > cd),
                }
            )

    return {
        "available": True,
        "metric": metric,
        "n_items": n_blocks,
        "n_methods": n_methods,
        "friedman_statistic": float(statistic),
        "friedman_p_value": float(p_value),
        "friedman_significant": bool(p_value < alpha),
        "critical_difference": cd,
        "average_ranks": {method: float(rank) for method, rank in average_ranks.items()},
        "nemenyi": pairwise,
    }


def runtime_efficiency_analysis(results: pd.DataFrame) -> dict[str, Any]:
    """Analyze runtime efficiency and scaling behavior."""
    analysis = {}
    results = add_method_variant_column(results)
    method_col = _method_column(results)

    for algo in results[method_col].unique():
        data = results[results[method_col] == algo]
        if len(data) < 5:
            continue

        x = np.log(data["duration_s"].values)
        y = np.log(data["runtime_s"].values + 1e-6)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        analysis[algo] = {
            "complexity_exponent": float(slope),
            "complexity_estimate": _interpret_complexity(slope),
            "r_squared": float(r_value**2),
            "p_value": float(p_value),
            "mean_runtime": float(data["runtime_s"].mean()),
            "max_runtime": float(data["runtime_s"].max()),
            "pieces_per_second": float(len(data) / data["runtime_s"].sum()),
        }

    return analysis


def memory_analysis(results: pd.DataFrame) -> dict[str, Any]:
    """Analyze memory usage patterns."""
    analysis = {}
    results = add_method_variant_column(results)
    method_col = _method_column(results)

    for algo in results[method_col].unique():
        data = results[results[method_col] == algo]
        analysis[algo] = {
            "mean_memory_mb": float(data["memory_mb"].mean()),
            "max_memory_mb": float(data["memory_mb"].max()),
            "memory_per_minute": float(data["memory_mb"].sum() / (data["duration_s"].sum() / 60)),
        }

        valid = data[["duration_s", "memory_mb"]].dropna()
        if len(valid) >= 5:
            corr, p_val = stats.pearsonr(valid["duration_s"], valid["memory_mb"])
            analysis[algo]["memory_duration_correlation"] = float(corr)
            analysis[algo]["memory_scales_with_duration"] = bool(corr > 0.5 and p_val < 0.05)

    return analysis


def _interpret_effect_size(r: float) -> str:
    if r < 0.1:
        return "negligible"
    if r < 0.3:
        return "small"
    if r < 0.5:
        return "medium"
    return "large"


def _interpret_complexity(exponent: float) -> str:
    if exponent < 0.5:
        return "O(1) or sublinear"
    if exponent < 1.2:
        return "O(N) linear"
    if exponent < 1.7:
        return "O(N log N)"
    if exponent < 2.3:
        return "O(N^2) quadratic"
    return f"O(N^{exponent:.1f})"


def _smaller_is_better(metric: str) -> bool:
    return metric in {"mae", "median_ae", "runtime_s", "memory_mb"}


def _method_column(results: pd.DataFrame) -> str:
    if "method_variant" in results.columns:
        return "method_variant"
    if "algorithm" in results.columns:
        return "algorithm"
    if "method" in results.columns:
        return "method"
    raise KeyError("Expected an 'algorithm' or 'method' column")


def _id_column(results: pd.DataFrame) -> str:
    if "piece_id" in results.columns:
        return "piece_id"
    if "pair_id" in results.columns:
        return "pair_id"
    raise KeyError("Expected a 'piece_id' or 'pair_id' column")
