"""Visualization tools for benchmark analysis and dissertation figures."""

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray

# Set publication-quality defaults
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.figsize": (6, 4),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_error_vs_length(
    results: pd.DataFrame,
    metric: str = "mae",
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (8, 5),
) -> plt.Figure:
    """
    Plot alignment error as a function of piece duration.
    
    Creates a scatter plot with regression lines for each algorithm,
    showing how errors scale with piece length.
    
    Args:
        results: Benchmark results DataFrame.
        metric: Error metric column name ('mae', 'median_ae', etc.).
        output_path: Path to save figure. None = display only.
        figsize: Figure dimensions.
        
    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    method_col = _method_column(results)
    algorithms = results[method_col].unique()
    palette = sns.color_palette("husl", len(algorithms))
    
    for algo, color in zip(algorithms, palette):
        data = results[results[method_col] == algo]
        
        ax.scatter(
            data["duration_s"] / 60,  # Convert to minutes
            data[metric],
            alpha=0.6,
            label=algo,
            color=color,
            s=30,
        )
        
        # Add regression line
        if len(data) >= 2:
            z = np.polyfit(data["duration_s"] / 60, data[metric], 1)
            p = np.poly1d(z)
            x_line = np.linspace(data["duration_s"].min() / 60, data["duration_s"].max() / 60, 100)
            ax.plot(x_line, p(x_line), "--", color=color, alpha=0.8, linewidth=1.5)
    
    ax.set_xlabel("Piece Duration (minutes)")
    ax.set_ylabel(f"{metric.upper()} (seconds)")
    ax.set_title("Alignment Error vs. Piece Length")
    ax.legend(title="Algorithm", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path)
    
    return fig


def plot_runtime_comparison(
    results: pd.DataFrame,
    output_path: str | Path | None = None,
    log_scale: bool = True,
    figsize: tuple[float, float] = (8, 5),
) -> plt.Figure:
    """
    Plot runtime scaling comparison between algorithms.
    
    Shows how runtime grows with piece length for each algorithm,
    demonstrating O(N²) vs O(N) complexity.
    
    Args:
        results: Benchmark results DataFrame.
        output_path: Path to save figure.
        log_scale: Use log-log scale for clearer scaling visualization.
        figsize: Figure dimensions.
        
    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    method_col = _method_column(results)
    algorithms = results[method_col].unique()
    palette = sns.color_palette("husl", len(algorithms))
    
    for algo, color in zip(algorithms, palette):
        data = results[results[method_col] == algo].sort_values("duration_s")
        
        ax.scatter(
            data["duration_s"],
            data["runtime_s"],
            alpha=0.6,
            label=algo,
            color=color,
            s=30,
        )
        
        # Add trend line
        ax.plot(
            data["duration_s"],
            data["runtime_s"],
            "-",
            color=color,
            alpha=0.4,
            linewidth=1,
        )
    
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    
    ax.set_xlabel("Piece Duration (seconds)")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Algorithm Runtime Scaling")
    ax.legend(title="Algorithm", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.3, which="both")
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path)
    
    return fig


def plot_alignment_path(
    cost_matrix: NDArray[np.floating] | None,
    path: NDArray[np.intp],
    output_path: str | Path | None = None,
    title: str = "DTW Alignment Path",
    figsize: tuple[float, float] = (8, 6),
) -> plt.Figure:
    """
    Visualize DTW alignment path on cost matrix.
    
    Args:
        cost_matrix: Cost matrix of shape (n_query, n_ref). If None, 
            shows path only on empty background.
        path: Warping path of shape (2, path_length).
        output_path: Path to save figure.
        title: Plot title.
        figsize: Figure dimensions.
        
    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if cost_matrix is not None:
        im = ax.imshow(
            cost_matrix,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
        )
        plt.colorbar(im, ax=ax, label="Cost")
    
    # Plot alignment path
    ax.plot(
        path[1, :],  # Reference indices (x-axis)
        path[0, :],  # Query indices (y-axis)
        "r-",
        linewidth=2,
        label="Alignment Path",
    )
    
    # Add diagonal reference
    n_query = path[0, :].max() + 1
    n_ref = path[1, :].max() + 1
    ax.plot(
        [0, min(n_query, n_ref)],
        [0, min(n_query, n_ref)],
        "w--",
        alpha=0.5,
        linewidth=1,
        label="Diagonal",
    )
    
    ax.set_xlabel("Reference Frame")
    ax.set_ylabel("Query Frame")
    ax.set_title(title)
    ax.legend(loc="upper left")
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path)
    
    return fig


def plot_feature_comparison(
    chroma: NDArray[np.floating],
    dlnco: NDArray[np.floating],
    sr: int = 22050,
    hop_length: int = 512,
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (12, 6),
) -> plt.Figure:
    """
    Compare Chroma and DLNCO feature representations.
    
    Creates side-by-side chromagram visualizations showing the
    difference between standard chroma and onset-enhanced DLNCO.
    
    Args:
        chroma: Chroma features of shape (12, n_frames).
        dlnco: DLNCO features of shape (12, n_frames).
        sr: Sample rate for time axis.
        hop_length: Hop length for time axis.
        output_path: Path to save figure.
        figsize: Figure dimensions.
        
    Returns:
        Matplotlib Figure object.
    """
    import librosa.display
    
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Chroma
    librosa.display.specshow(
        chroma,
        y_axis="chroma",
        x_axis="time",
        sr=sr,
        hop_length=hop_length,
        ax=axes[0],
    )
    axes[0].set_title("CQT Chromagram")
    axes[0].set_ylabel("Pitch Class")
    
    # DLNCO
    librosa.display.specshow(
        dlnco,
        y_axis="chroma",
        x_axis="time",
        sr=sr,
        hop_length=hop_length,
        ax=axes[1],
    )
    axes[1].set_title("DLNCO Features (Onset-Enhanced)")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Pitch Class")
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path)
    
    return fig


def plot_metric_boxplot(
    results: pd.DataFrame,
    metric: str = "mae",
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (8, 5),
) -> plt.Figure:
    """
    Create boxplot comparison of metrics across algorithms.
    
    Args:
        results: Benchmark results DataFrame.
        metric: Metric column to plot.
        output_path: Path to save figure.
        figsize: Figure dimensions.
        
    Returns:
        Matplotlib Figure object.
    """
    method_col = _method_column(results)
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.boxplot(
        data=results,
        x=method_col,
        y=metric,
        palette="husl",
        ax=ax,
    )
    
    ax.set_xlabel("Algorithm")
    ax.set_ylabel(f"{metric.upper()} (seconds)")
    ax.set_title(f"Distribution of {metric.upper()} by Algorithm")
    plt.xticks(rotation=45, ha="right")
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path)
    
    return fig


def create_summary_table(
    results: pd.DataFrame,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Create summary statistics table for dissertation.
    
    Args:
        results: Benchmark results DataFrame.
        output_path: Path to save as CSV/LaTeX.
        
    Returns:
        Summary DataFrame.
    """
    method_col = _method_column(results)
    summary = results.groupby(method_col).agg({
        "mae": ["mean", "std", "median"],
        "ar_50ms": ["mean", "std"],
        "ar_100ms": ["mean", "std"],
        "runtime_s": ["mean", "std"],
        "memory_mb": ["mean", "max"],
    }).round(4)
    
    # Flatten column names
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    
    if output_path:
        path = Path(output_path)
        if path.suffix == ".tex":
            summary.to_latex(path, caption="Benchmark Results Summary")
        else:
            summary.to_csv(path)
    
    return summary


def _method_column(results: pd.DataFrame) -> str:
    if "algorithm" in results.columns:
        return "algorithm"
    if "method" in results.columns:
        return "method"
    raise KeyError("Expected an 'algorithm' or 'method' column")
