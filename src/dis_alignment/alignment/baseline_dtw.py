"""Standard Global DTW implementation for audio-to-score alignment.

This serves as the baseline algorithm for benchmarking against
multiscale and memory-optimized variants.
"""

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist


@dataclass
class AlignmentResult:
    """Container for alignment results with metadata."""
    
    path: NDArray[np.intp]
    """Warping path of shape (2, path_length) with [query_indices, ref_indices]."""
    
    cost: float
    """Total accumulated cost of the alignment path."""
    
    cost_matrix: NDArray[np.floating] | None
    """Full cost matrix if retained, else None."""
    
    runtime_seconds: float
    """Wall-clock time for alignment computation."""
    
    memory_bytes: int
    """Peak memory usage estimate for the cost matrix."""
    
    algorithm: str
    """Name of the alignment algorithm used."""


def align_global_dtw(
    features_query: NDArray[np.floating],
    features_reference: NDArray[np.floating],
    distance: Literal["cosine", "euclidean", "manhattan"] = "cosine",
    sakoe_chiba_radius: int | None = None,
    step_pattern: Literal["symmetric1", "symmetric2"] = "symmetric2",
    retain_cost_matrix: bool = False,
) -> AlignmentResult:
    """
    Perform standard Global DTW alignment.
    
    Computes the full O(N*M) cost matrix and finds the optimal warping
    path using dynamic programming. This is the baseline algorithm for
    comparison with multiscale approaches.
    
    Args:
        features_query: Query features of shape (n_features, n_frames_query).
        features_reference: Reference features of shape (n_features, n_frames_ref).
        distance: Distance metric for cost computation.
        sakoe_chiba_radius: If set, constrains the warping path to stay
            within this many frames of the diagonal. Reduces complexity
            to O(N*R) where R is the radius.
        step_pattern: DTW step pattern:
            - "symmetric1": min(D[i-1,j], D[i,j-1], D[i-1,j-1]) + c[i,j]
            - "symmetric2": min(D[i-1,j]+c, D[i,j-1]+c, D[i-1,j-1]+2c)
        retain_cost_matrix: If True, include cost matrix in result.
            Warning: Can consume significant memory for long sequences.
            
    Returns:
        AlignmentResult with warping path (indices mapping query to reference),
        total cost, runtime, and memory usage.
        
    Raises:
        MemoryError: If cost matrix exceeds available memory.
        
    Example:
        >>> result = align_global_dtw(chroma_audio, chroma_score)
        >>> query_to_ref = result.path  # Shape: (2, path_length)
    """
    start_time = time.perf_counter()
    
    # Transpose to (n_frames, n_features) for cdist
    X = features_query.T
    Y = features_reference.T
    
    n_query, n_ref = len(X), len(Y)
    
    # Estimate memory requirement
    memory_bytes = n_query * n_ref * 8  # float64
    
    # Compute pairwise distance matrix
    if distance == "cosine":
        cost_local = cdist(X, Y, metric="cosine")
    elif distance == "euclidean":
        cost_local = cdist(X, Y, metric="euclidean")
    elif distance == "manhattan":
        cost_local = cdist(X, Y, metric="cityblock")
    else:
        raise ValueError(f"Unknown distance metric: {distance}")
    
    # Accumulated cost matrix
    D = np.full((n_query + 1, n_ref + 1), np.inf, dtype=np.float64)
    D[0, 0] = 0
    
    # Fill cost matrix with optional Sakoe-Chiba constraint
    for i in range(1, n_query + 1):
        if sakoe_chiba_radius is not None:
            j_start = max(1, i - sakoe_chiba_radius)
            j_end = min(n_ref + 1, i + sakoe_chiba_radius + 1)
        else:
            j_start, j_end = 1, n_ref + 1
            
        for j in range(j_start, j_end):
            c = cost_local[i - 1, j - 1]
            
            if step_pattern == "symmetric1":
                D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
            else:  # symmetric2
                D[i, j] = min(
                    D[i - 1, j] + c,
                    D[i, j - 1] + c,
                    D[i - 1, j - 1] + 2 * c,
                )
    
    # Backtrack to find optimal path
    path = _backtrack(D, step_pattern)
    
    runtime = time.perf_counter() - start_time
    
    return AlignmentResult(
        path=path,
        cost=D[n_query, n_ref],
        cost_matrix=D[1:, 1:] if retain_cost_matrix else None,
        runtime_seconds=runtime,
        memory_bytes=memory_bytes,
        algorithm="global_dtw",
    )


def _backtrack(
    D: NDArray[np.floating],
    step_pattern: str,
) -> NDArray[np.intp]:
    """
    Backtrack through accumulated cost matrix to find optimal path.
    
    Args:
        D: Accumulated cost matrix of shape (n_query+1, n_ref+1).
        step_pattern: Step pattern used during forward pass.
        
    Returns:
        Warping path of shape (2, path_length).
    """
    i, j = D.shape[0] - 1, D.shape[1] - 1
    path = [(i - 1, j - 1)]  # Convert to 0-indexed
    
    while i > 1 or j > 1:
        if i == 1:
            j -= 1
        elif j == 1:
            i -= 1
        else:
            # Find predecessor with minimum cost
            candidates = [
                (D[i - 1, j], i - 1, j),
                (D[i, j - 1], i, j - 1),
                (D[i - 1, j - 1], i - 1, j - 1),
            ]
            _, i, j = min(candidates, key=lambda x: x[0])
        
        path.append((i - 1, j - 1))
    
    path.reverse()
    return np.array(path, dtype=np.intp).T


def path_to_alignment(
    path: NDArray[np.intp],
    hop_length: int,
    sr: int,
) -> NDArray[np.floating]:
    """
    Convert DTW path to time-domain alignment.
    
    Args:
        path: Warping path of shape (2, path_length).
        hop_length: Feature extraction hop length.
        sr: Audio sample rate.
        
    Returns:
        Array of shape (2, path_length) with times in seconds.
    """
    frame_duration = hop_length / sr
    return path.astype(np.float64) * frame_duration
