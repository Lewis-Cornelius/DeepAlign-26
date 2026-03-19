"""Evaluation metrics for alignment experiments.

Implements standard MIR evaluation metrics for alignment quality
assessment, compatible with mir_eval conventions.
"""

import numpy as np
from numpy.typing import NDArray


def mean_absolute_error(
    predicted_times: NDArray[np.floating],
    ground_truth_times: NDArray[np.floating],
) -> float:
    """
    Compute Mean Absolute Error between predicted and ground truth alignments.
    
    Args:
        predicted_times: Predicted time positions in seconds.
        ground_truth_times: Ground truth time positions in seconds.
            Must have the same length as predicted_times.
            
    Returns:
        Mean absolute error in seconds.
        
    Example:
        >>> pred = np.array([0.0, 0.5, 1.02, 1.5])
        >>> gt = np.array([0.0, 0.5, 1.0, 1.5])
        >>> mean_absolute_error(pred, gt)
        0.005
    """
    if len(predicted_times) != len(ground_truth_times):
        raise ValueError(
            f"Length mismatch: predicted={len(predicted_times)}, "
            f"ground_truth={len(ground_truth_times)}"
        )
    
    return float(np.mean(np.abs(predicted_times - ground_truth_times)))


def alignment_rate(
    predicted_times: NDArray[np.floating],
    ground_truth_times: NDArray[np.floating],
    tolerance: float = 0.05,
) -> float:
    """
    Compute alignment rate (percentage of frames within tolerance).
    
    Args:
        predicted_times: Predicted time positions in seconds.
        ground_truth_times: Ground truth time positions in seconds.
        tolerance: Maximum allowed deviation in seconds.
            Default 50ms is standard in MIR evaluation.
            
    Returns:
        Fraction of frames with error <= tolerance (0.0 to 1.0).
        
    Example:
        >>> pred = np.array([0.0, 0.52, 1.0, 1.5])
        >>> gt = np.array([0.0, 0.5, 1.0, 1.5])
        >>> alignment_rate(pred, gt, tolerance=0.05)
        0.75  # 3 out of 4 within 50ms
    """
    if len(predicted_times) != len(ground_truth_times):
        raise ValueError("Length mismatch between predicted and ground truth")
    
    errors = np.abs(predicted_times - ground_truth_times)
    return float(np.mean(errors <= tolerance))


def coverage(
    path: NDArray[np.intp],
    n_query_frames: int,
    n_reference_frames: int,
) -> tuple[float, float]:
    """
    Compute alignment coverage for query and reference sequences.
    
    Coverage measures what fraction of frames are included in the
    alignment path. Low coverage may indicate structural mismatches.
    
    Args:
        path: Warping path of shape (2, path_length).
        n_query_frames: Total number of query frames.
        n_reference_frames: Total number of reference frames.
        
    Returns:
        Tuple of (query_coverage, reference_coverage), each in [0, 1].
    """
    unique_query = len(np.unique(path[0, :]))
    unique_ref = len(np.unique(path[1, :]))
    
    query_cov = unique_query / n_query_frames if n_query_frames > 0 else 0.0
    ref_cov = unique_ref / n_reference_frames if n_reference_frames > 0 else 0.0
    
    return query_cov, ref_cov


def median_absolute_error(
    predicted_times: NDArray[np.floating],
    ground_truth_times: NDArray[np.floating],
) -> float:
    """
    Compute Median Absolute Error (more robust to outliers than MAE).
    
    Args:
        predicted_times: Predicted time positions in seconds.
        ground_truth_times: Ground truth time positions in seconds.
        
    Returns:
        Median absolute error in seconds.
    """
    errors = np.abs(predicted_times - ground_truth_times)
    return float(np.median(errors))


def percentile_error(
    predicted_times: NDArray[np.floating],
    ground_truth_times: NDArray[np.floating],
    percentiles: list[float] = [50, 90, 95, 99],
) -> dict[str, float]:
    """
    Compute error at various percentiles.
    
    Useful for understanding error distribution and worst-case behavior.
    
    Args:
        predicted_times: Predicted time positions.
        ground_truth_times: Ground truth time positions.
        percentiles: Percentiles to compute.
        
    Returns:
        Dict mapping percentile names (e.g., "p90") to error values.
    """
    errors = np.abs(predicted_times - ground_truth_times)
    return {
        f"p{int(p)}": float(np.percentile(errors, p))
        for p in percentiles
    }


def path_to_frame_alignment(
    path: NDArray[np.intp],
    n_query_frames: int,
) -> NDArray[np.intp]:
    """
    Convert DTW path to per-query-frame alignment.
    
    For each query frame, find the corresponding reference frame
    from the warping path.
    
    Args:
        path: Warping path of shape (2, path_length).
        n_query_frames: Number of query frames.
        
    Returns:
        Array of shape (n_query_frames,) with reference frame indices.
    """
    # For frames with multiple mappings, take the mean reference index
    alignment = np.zeros(n_query_frames, dtype=np.intp)
    counts = np.zeros(n_query_frames, dtype=np.intp)
    
    for i in range(path.shape[1]):
        q_idx = path[0, i]
        r_idx = path[1, i]
        alignment[q_idx] += r_idx
        counts[q_idx] += 1
    
    # Handle frames with multiple mappings
    mask = counts > 0
    alignment[mask] = alignment[mask] // counts[mask]
    
    # Interpolate any gaps (shouldn't happen with proper DTW)
    if not mask.all():
        valid_idx = np.where(mask)[0]
        alignment = np.interp(
            np.arange(n_query_frames),
            valid_idx,
            alignment[valid_idx],
        ).astype(np.intp)
    
    return alignment


def compute_all_metrics(
    predicted_times: NDArray[np.floating],
    ground_truth_times: NDArray[np.floating],
    path: NDArray[np.intp] | None = None,
    n_query_frames: int | None = None,
    n_reference_frames: int | None = None,
) -> dict[str, float]:
    """
    Compute all standard alignment metrics.
    
    Args:
        predicted_times: Predicted time alignment.
        ground_truth_times: Ground truth time alignment.
        path: Optional DTW path for coverage computation.
        n_query_frames: Total query frames (required if path provided).
        n_reference_frames: Total reference frames (required if path provided).
        
    Returns:
        Dict with all computed metrics.
    """
    metrics = {
        "mae": mean_absolute_error(predicted_times, ground_truth_times),
        "median_ae": median_absolute_error(predicted_times, ground_truth_times),
        "alignment_rate_50ms": alignment_rate(predicted_times, ground_truth_times, 0.05),
        "alignment_rate_100ms": alignment_rate(predicted_times, ground_truth_times, 0.10),
        "alignment_rate_200ms": alignment_rate(predicted_times, ground_truth_times, 0.20),
    }
    
    # Add percentile errors
    metrics.update(percentile_error(predicted_times, ground_truth_times))
    
    # Add coverage if path provided
    if path is not None and n_query_frames and n_reference_frames:
        q_cov, r_cov = coverage(path, n_query_frames, n_reference_frames)
        metrics["query_coverage"] = q_cov
        metrics["reference_coverage"] = r_cov
    
    return metrics
