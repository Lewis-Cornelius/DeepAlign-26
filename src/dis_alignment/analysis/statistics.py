"""Statistical analysis and failure diagnostics for experiment results."""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def summarize_results(
    results: pd.DataFrame,
    group_by: str | None = None,
) -> pd.DataFrame:
    """
    Generate summary statistics for benchmark results.
    
    Args:
        results: Benchmark results DataFrame.
        group_by: Column to group results by.
        
    Returns:
        Summary DataFrame with mean, std, min, max for key metrics.
    """
    method_col = group_by or _method_column(results)
    metrics = [column for column in ["mae", "median_ae", "ar_50ms", "ar_100ms", "runtime_s", "memory_mb"] if column in results.columns]

    summary = results.groupby(method_col)[metrics].agg(
        ["mean", "std", "min", "max", "count"]
    )
    
    return summary


def failure_analysis(
    results: pd.DataFrame,
    mae_threshold: float = 0.5,
    ar_threshold: float = 0.8,
) -> dict[str, Any]:
    """
    Analyze failure cases where alignment quality is poor.
    
    Identifies pieces with high error rates and attempts to find
    common characteristics that predict failure.
    
    Args:
        results: Benchmark results DataFrame.
        mae_threshold: MAE above this value is considered a failure.
        ar_threshold: Alignment rate below this is considered a failure.
        
    Returns:
        Dict containing failure statistics and analysis.
    """
    # Identify failure cases
    failures = results[
        (results["mae"] > mae_threshold) | 
        (results["ar_50ms"] < ar_threshold)
    ]
    
    successes = results[
        (results["mae"] <= mae_threshold) & 
        (results["ar_50ms"] >= ar_threshold)
    ]
    
    analysis = {
        "total_cases": len(results),
        "failure_count": len(failures),
        "failure_rate": len(failures) / len(results) if len(results) > 0 else 0,
        "failures_by_algorithm": failures.groupby(_method_column(results)).size().to_dict(),
        "mean_duration_failures": failures["duration_s"].mean() if len(failures) > 0 else 0,
        "mean_duration_successes": successes["duration_s"].mean() if len(successes) > 0 else 0,
    }
    
    # Duration correlation with failure
    if len(failures) > 0 and len(successes) > 0:
        t_stat, p_value = stats.ttest_ind(
            failures["duration_s"],
            successes["duration_s"],
        )
        analysis["duration_ttest_pvalue"] = p_value
        analysis["longer_pieces_fail_more"] = failures["duration_s"].mean() > successes["duration_s"].mean()
    
    # Top failure pieces
    id_col = _id_column(results)
    method_col = _method_column(results)
    analysis["worst_pieces"] = failures.nlargest(10, "mae")[
        [id_col, method_col, "mae", "ar_50ms", "duration_s"]
    ].to_dict("records")
    
    return analysis


def compute_significance(
    results: pd.DataFrame,
    baseline: str = "global_dtw",
    candidate: str = "mrmsdtw",
    metric: str = "mae",
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Compute statistical significance between two algorithms.
    
    Uses paired Wilcoxon signed-rank test (non-parametric) for
    comparing alignment quality on the same pieces.
    
    Args:
        results: Benchmark results DataFrame.
        baseline: Name of baseline algorithm.
        candidate: Name of candidate algorithm to compare.
        metric: Metric to compare.
        alpha: Significance level.
        
    Returns:
        Dict with test statistics and interpretation.
    """
    # Get paired observations
    method_col = _method_column(results)
    id_col = _id_column(results)

    baseline_results = results[results[method_col] == baseline].set_index(id_col)
    candidate_results = results[results[method_col] == candidate].set_index(id_col)
    
    # Find common pieces
    common_pieces = baseline_results.index.intersection(candidate_results.index)
    
    if len(common_pieces) < 5:
        return {
            "error": "Not enough paired observations",
            "n_pairs": len(common_pieces),
        }
    
    baseline_vals = baseline_results.loc[common_pieces, metric].values
    candidate_vals = candidate_results.loc[common_pieces, metric].values
    
    # Wilcoxon signed-rank test
    stat, p_value = stats.wilcoxon(baseline_vals, candidate_vals, alternative="two-sided")
    
    # Effect size (rank-biserial correlation)
    n = len(common_pieces)
    effect_size = 1 - (2 * stat) / (n * (n + 1))
    
    # One-sided test: is candidate better (lower MAE)?
    _, p_value_less = stats.wilcoxon(candidate_vals, baseline_vals, alternative="less")
    
    return {
        "baseline": baseline,
        "candidate": candidate,
        "metric": metric,
        "n_pairs": len(common_pieces),
        "baseline_mean": float(np.mean(baseline_vals)),
        "candidate_mean": float(np.mean(candidate_vals)),
        "improvement": float(np.mean(baseline_vals) - np.mean(candidate_vals)),
        "improvement_pct": float(
            (np.mean(baseline_vals) - np.mean(candidate_vals)) / np.mean(baseline_vals) * 100
        ),
        "wilcoxon_statistic": float(stat),
        "p_value_twosided": float(p_value),
        "p_value_candidate_better": float(p_value_less),
        "is_significant": p_value < alpha,
        "candidate_is_better": p_value_less < alpha,
        "effect_size": float(effect_size),
        "effect_interpretation": _interpret_effect_size(abs(effect_size)),
    }


def _interpret_effect_size(r: float) -> str:
    """Interpret rank-biserial correlation magnitude."""
    if r < 0.1:
        return "negligible"
    elif r < 0.3:
        return "small"
    elif r < 0.5:
        return "medium"
    else:
        return "large"


def runtime_efficiency_analysis(
    results: pd.DataFrame,
) -> dict[str, Any]:
    """
    Analyze runtime efficiency and scaling behavior.
    
    Fits power-law models to runtime vs. duration to estimate
    algorithmic complexity.
    
    Args:
        results: Benchmark results DataFrame.
        
    Returns:
        Dict with scaling analysis per algorithm.
    """
    analysis = {}
    
    method_col = _method_column(results)

    for algo in results[method_col].unique():
        data = results[results[method_col] == algo]
        
        if len(data) < 5:
            continue
        
        x = np.log(data["duration_s"].values)
        y = np.log(data["runtime_s"].values + 1e-6)  # Avoid log(0)
        
        # Fit linear model to log-log data: log(t) = a * log(n) + b
        # Slope a is the complexity exponent
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        analysis[algo] = {
            "complexity_exponent": float(slope),
            "complexity_estimate": _interpret_complexity(slope),
            "r_squared": float(r_value ** 2),
            "p_value": float(p_value),
            "mean_runtime": float(data["runtime_s"].mean()),
            "max_runtime": float(data["runtime_s"].max()),
            "pieces_per_second": float(len(data) / data["runtime_s"].sum()),
        }
    
    return analysis


def _interpret_complexity(exponent: float) -> str:
    """Interpret complexity exponent as Big-O notation."""
    if exponent < 0.5:
        return "O(1) or sublinear"
    elif exponent < 1.2:
        return "O(N) linear"
    elif exponent < 1.7:
        return "O(N log N)"
    elif exponent < 2.3:
        return "O(N²) quadratic"
    else:
        return f"O(N^{exponent:.1f})"


def memory_analysis(results: pd.DataFrame) -> dict[str, Any]:
    """
    Analyze memory usage patterns.
    
    Args:
        results: Benchmark results.
        
    Returns:
        Memory usage statistics per algorithm.
    """
    analysis = {}
    
    method_col = _method_column(results)

    for algo in results[method_col].unique():
        data = results[results[method_col] == algo]
        
        analysis[algo] = {
            "mean_memory_mb": float(data["memory_mb"].mean()),
            "max_memory_mb": float(data["memory_mb"].max()),
            "memory_per_minute": float(
                data["memory_mb"].sum() / (data["duration_s"].sum() / 60)
            ),
        }
        
        # Memory scaling with duration
        valid = data[["duration_s", "memory_mb"]].dropna()
        if len(valid) >= 5:
            corr, p = stats.pearsonr(valid["duration_s"], valid["memory_mb"])
            analysis[algo]["memory_duration_correlation"] = float(corr)
            analysis[algo]["memory_scales_with_duration"] = corr > 0.5 and p < 0.05
    
    return analysis


def _method_column(results: pd.DataFrame) -> str:
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
