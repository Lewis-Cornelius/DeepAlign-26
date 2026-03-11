"""Memory-restricted Multiscale DTW (MrMsDTW) wrapper.

Wraps the synctoolbox implementation of MrMsDTW for benchmarking.
MrMsDTW achieves O(N) complexity by computing alignment at multiple
resolutions with memory-bounded path constraints.

Reference:
    T. Prätzlich et al., "Memory-Restricted Multiscale Dynamic Time 
    Warping," ICASSP, 2016.
"""

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dis_alignment.alignment.baseline_dtw import AlignmentResult


def align_mrmsdtw(
    features_query: NDArray[np.floating],
    features_reference: NDArray[np.floating],
    memory_limit_mb: int = 500,
    num_scales: int = 3,
    step_weights: tuple[float, float, float] = (1.0, 1.0, 2.0),
    dtw_implementation: str = "synctoolbox",
) -> AlignmentResult:
    """
    Perform Memory-restricted Multiscale DTW alignment.
    
    MrMsDTW operates in multiple passes at increasing resolutions:
    1. Coarse alignment at low resolution (downsampled features)
    2. Constraint path derived from coarse alignment
    3. Fine alignment within constrained corridor
    
    This achieves linear O(N) complexity while maintaining accuracy
    comparable to full DTW.
    
    Args:
        features_query: Query features of shape (n_features, n_frames_query).
        features_reference: Reference features of shape (n_features, n_frames_ref).
        memory_limit_mb: Maximum memory budget for cost matrix in MB.
            Controls the width of the search corridor.
        num_scales: Number of resolution scales (typically 2-4).
            More scales = coarser initial pass = faster but less accurate.
        step_weights: Weights for (horizontal, vertical, diagonal) steps.
        dtw_implementation: Backend implementation to use.
            Currently only "synctoolbox" is supported.
            
    Returns:
        AlignmentResult with warping path, cost, and performance metrics.
        
    Example:
        >>> result = align_mrmsdtw(chroma_audio, chroma_score, memory_limit_mb=256)
        >>> print(f"Aligned in {result.runtime_seconds:.2f}s")
    """
    start_time = time.perf_counter()
    
    if dtw_implementation == "synctoolbox":
        path, cost = _align_synctoolbox(
            features_query,
            features_reference,
            memory_limit_mb=memory_limit_mb,
            num_scales=num_scales,
            step_weights=step_weights,
        )
    else:
        raise ValueError(f"Unknown DTW implementation: {dtw_implementation}")
    
    runtime = time.perf_counter() - start_time
    
    # Estimate memory used (corridor-based, not full matrix)
    n_query = features_query.shape[1]
    corridor_width = _estimate_corridor_width(memory_limit_mb, features_query.shape[0])
    memory_bytes = n_query * corridor_width * 8
    
    return AlignmentResult(
        path=path,
        cost=cost,
        cost_matrix=None,  # MrMsDTW doesn't retain full matrix
        runtime_seconds=runtime,
        memory_bytes=memory_bytes,
        algorithm=f"mrmsdtw_scales{num_scales}",
    )


def _align_synctoolbox(
    features_query: NDArray[np.floating],
    features_reference: NDArray[np.floating],
    memory_limit_mb: int,
    num_scales: int,
    step_weights: tuple[float, float, float],
) -> tuple[NDArray[np.intp], float]:
    """
    Perform alignment using synctoolbox's MrMsDTW implementation.
    
    Args:
        features_query: Query features.
        features_reference: Reference features.
        memory_limit_mb: Memory limit in MB.
        num_scales: Number of resolution scales.
        step_weights: DTW step weights.
        
    Returns:
        Tuple of (warping_path, total_cost).
    """
    try:
        from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw
        from synctoolbox.dtw.utils import compute_optimal_chroma_shift
    except ImportError as e:
        raise ImportError(
            "synctoolbox is required for MrMsDTW. "
            "Install with: pip install synctoolbox"
        ) from e
    
    # Compute optimal transposition for better alignment
    opt_shift = compute_optimal_chroma_shift(
        features_query, features_reference
    )
    
    # Apply shift to query features
    if opt_shift != 0:
        features_query = np.roll(features_query, opt_shift, axis=0)
    
    # Convert step weights to synctoolbox format
    step_sizes = np.array([[1, 0], [0, 1], [1, 1]])
    weights = np.array(step_weights)
    
    # Run MrMsDTW
    wp = sync_via_mrmsdtw(
        f_chroma1=features_query,
        f_chroma2=features_reference,
        input_feature_rate=1.0,  # Features already at target rate
        step_sizes=step_sizes,
        step_weights=weights,
        threshold_rec=memory_limit_mb * 1024 * 1024 // 8,  # Convert to cell count
        verbose=False,
    )
    
    # synctoolbox returns path as (N, 2), convert to (2, N)
    path = wp.T.astype(np.intp)
    
    # Compute path cost
    cost = _compute_path_cost(features_query, features_reference, path)
    
    return path, cost


def _compute_path_cost(
    features_query: NDArray[np.floating],
    features_reference: NDArray[np.floating],
    path: NDArray[np.intp],
) -> float:
    """Compute total cosine distance along the warping path."""
    total = 0.0
    for i in range(path.shape[1]):
        qi, ri = path[0, i], path[1, i]
        q_vec = features_query[:, qi]
        r_vec = features_reference[:, ri]
        
        # Cosine distance
        dot = np.dot(q_vec, r_vec)
        norm = np.linalg.norm(q_vec) * np.linalg.norm(r_vec)
        if norm > 0:
            total += 1 - dot / norm
    
    return total


def _estimate_corridor_width(memory_limit_mb: int, n_features: int) -> int:
    """Estimate DTW corridor width from memory budget."""
    bytes_per_cell = 8  # float64
    cells_available = (memory_limit_mb * 1024 * 1024) // bytes_per_cell
    # Assume square-ish aspect ratio for estimation
    return int(np.sqrt(cells_available))


def align_with_anchors(
    features_query: NDArray[np.floating],
    features_reference: NDArray[np.floating],
    anchors: list[tuple[int, int]],
    memory_limit_mb: int = 500,
) -> AlignmentResult:
    """
    Perform anchor-constrained MrMsDTW alignment.
    
    Anchors are known correspondences (e.g., from beat tracking or
    structural analysis) that constrain the warping path to pass
    through specific points.
    
    Args:
        features_query: Query features.
        features_reference: Reference features.
        anchors: List of (query_frame, reference_frame) anchor points.
        memory_limit_mb: Memory limit in MB.
        
    Returns:
        AlignmentResult with anchor-constrained path.
    """
    start_time = time.perf_counter()
    
    if len(anchors) < 2:
        # Fall back to unconstrained alignment
        return align_mrmsdtw(
            features_query, features_reference, memory_limit_mb=memory_limit_mb
        )
    
    # Sort anchors by query position
    anchors = sorted(anchors, key=lambda x: x[0])
    
    # Align segments between consecutive anchors
    segments: list[NDArray[np.intp]] = []
    
    for i in range(len(anchors) - 1):
        q_start, r_start = anchors[i]
        q_end, r_end = anchors[i + 1]
        
        # Extract segment features
        seg_query = features_query[:, q_start:q_end + 1]
        seg_ref = features_reference[:, r_start:r_end + 1]
        
        # Align segment
        result = align_mrmsdtw(
            seg_query, seg_ref, memory_limit_mb=memory_limit_mb // len(anchors)
        )
        
        # Offset path to global coordinates
        seg_path = result.path.copy()
        seg_path[0, :] += q_start
        seg_path[1, :] += r_start
        
        segments.append(seg_path)
    
    # Concatenate segment paths (removing duplicates at boundaries)
    full_path = _concatenate_paths(segments)
    
    runtime = time.perf_counter() - start_time
    cost = _compute_path_cost(features_query, features_reference, full_path)
    
    return AlignmentResult(
        path=full_path,
        cost=cost,
        cost_matrix=None,
        runtime_seconds=runtime,
        memory_bytes=memory_limit_mb * 1024 * 1024,
        algorithm="mrmsdtw_anchored",
    )


def _concatenate_paths(segments: list[NDArray[np.intp]]) -> NDArray[np.intp]:
    """Concatenate path segments, removing duplicate boundary points."""
    if not segments:
        return np.array([[0], [0]], dtype=np.intp)
    
    result = [segments[0]]
    for seg in segments[1:]:
        # Skip first point if it matches last point of previous segment
        if np.array_equal(seg[:, 0], result[-1][:, -1]):
            result.append(seg[:, 1:])
        else:
            result.append(seg)
    
    return np.hstack(result)
