"""Shared helpers for pairwise evaluation and report-ready summaries."""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
import tempfile
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from numba import njit

from dis_alignment.alignment.matchmaker import MatchmakerResult, run_matchmaker
from dis_alignment.alignment.multiscale_dtw import align_mrmsdtw
from dis_alignment.evaluation.metrics import alignment_rate, mean_absolute_error, median_absolute_error

PAIRWISE_METHODS = ("chroma_dtw", "mrmsdtw", "deepalign", "matchmaker")
DEEP_DECODE_MODES = (
    "unconstrained",
    "diagonal_band",
    "chroma_guided_band",
    "deepalign_transcription_fused",
    "deepalign_transcription_fused_refined",
    "deepalign_transcription_guided",
    "deepalign_score_guided_refined",
)
DEEP_VARIANT_COLUMNS = (
    "deep_decode",
    "band_radius_frames",
    "pool_size",
    "deep_distance",
    "fusion_deep_weight",
    "fusion_onset_weight",
    "fusion_note_weight",
    "fusion_dlnco_weight",
    "fusion_chroma_weight",
    "refine_window_sec",
    "score_refine_radius_sec",
)
METHOD_VARIANT_COLUMN = "method_variant"
MEMORY_EFFICIENT_DTW_MIN_BYTES = 4 * 1024 * 1024 * 1024
MEMORY_EFFICIENT_DTW_CHUNK_BYTES = 96 * 1024 * 1024
MEMORY_EFFICIENT_DTW_MEMMAP_CELLS = 512_000_000


def parse_methods(
    methods: str | list[str] | tuple[str, ...] | None,
    *,
    checkpoint_path: str | Path | None = None,
) -> list[str]:
    """Parse and validate a method selection."""
    if methods is None:
        resolved = ["chroma_dtw"]
        if checkpoint_path is not None:
            resolved.append("deepalign")
        return resolved

    if isinstance(methods, str):
        raw_methods = [item.strip() for item in methods.split(",")]
    else:
        raw_methods = [str(item).strip() for item in methods]

    resolved = [method for method in raw_methods if method]
    if not resolved:
        raise ValueError("No evaluation methods were provided.")

    unknown = [method for method in resolved if method not in PAIRWISE_METHODS]
    if unknown:
        raise ValueError(
            f"Unsupported methods: {', '.join(sorted(unknown))}. "
            f"Supported methods: {', '.join(PAIRWISE_METHODS)}."
        )

    if "deepalign" in resolved and checkpoint_path is None:
        raise ValueError("The deepalign method requires --checkpoint.")

    return resolved


def temporal_pool(features: np.ndarray, pool_size: int = 2) -> np.ndarray:
    """Average adjacent frames to reduce DTW memory cost during evaluation."""
    if pool_size <= 1:
        return features
    n_features, n_frames = features.shape
    pooled_frames = n_frames // pool_size
    if pooled_frames == 0:
        return features
    cropped = features[:, : pooled_frames * pool_size]
    return cropped.reshape(n_features, pooled_frames, pool_size).mean(axis=2)


def parse_deep_decode(mode: str | None) -> str:
    """Validate one DeepAlign decoding mode."""
    resolved = (mode or "unconstrained").strip().lower()
    if resolved not in DEEP_DECODE_MODES:
        raise ValueError(
            f"Unsupported deep decode mode: {resolved}. "
            f"Supported modes: {', '.join(DEEP_DECODE_MODES)}."
        )
    return resolved


def add_method_variant_column(results: pd.DataFrame) -> pd.DataFrame:
    """Return results with a reporting label that separates DeepAlign variants when needed."""
    if results.empty or METHOD_VARIANT_COLUMN in results.columns or "method" not in results.columns:
        return results
    if "deep_decode" not in results.columns or not _needs_method_variant_labels(results):
        return results

    labelled = results.copy()
    method = labelled["method"].astype(str)
    decode = labelled["deep_decode"].map(_normalize_variant_value)
    deep_mask = method.eq("deepalign") & decode.ne("")
    labelled[METHOD_VARIANT_COLUMN] = method
    labelled.loc[deep_mask, METHOD_VARIANT_COLUMN] = method[deep_mask] + ":" + decode[deep_mask]
    return labelled


def fast_dtw_align(
    features_a: np.ndarray,
    features_b: np.ndarray,
    *,
    distance: str = "cosine",
) -> tuple[np.ndarray, float, float]:
    """Align two feature sequences with librosa's DTW implementation."""
    import librosa

    start = time.perf_counter()
    n_frames_a = int(features_a.shape[1])
    n_frames_b = int(features_b.shape[1])
    if _should_use_memory_efficient_dtw(n_frames_a, n_frames_b, distance):
        return _memory_efficient_dtw_align(features_a, features_b, distance=distance)

    cost_matrix = cdist(features_a.T, features_b.T, metric=distance).astype(np.float64)
    cost_matrix = np.nan_to_num(cost_matrix, nan=1.0, posinf=1e6, neginf=1e6)
    accumulated_cost, warping_path = librosa.sequence.dtw(C=cost_matrix, backtrack=True)
    path = warping_path[::-1].T.astype(np.intp)
    runtime = time.perf_counter() - start
    return path, float(accumulated_cost[-1, -1]), runtime


def _should_use_memory_efficient_dtw(n_frames_a: int, n_frames_b: int, distance: str) -> bool:
    """Return True when librosa DTW would require an unsafe dense matrix budget."""
    if distance not in {"cosine", "sqeuclidean"}:
        return False
    estimated_bytes = int(n_frames_a) * int(n_frames_b) * 16
    return estimated_bytes >= MEMORY_EFFICIENT_DTW_MIN_BYTES


def _memory_efficient_dtw_align(
    features_a: np.ndarray,
    features_b: np.ndarray,
    *,
    distance: str = "cosine",
) -> tuple[np.ndarray, float, float]:
    """Full unconstrained DTW with rolling cost rows and compact backpointers."""
    if distance not in {"cosine", "sqeuclidean"}:
        raise ValueError(f"Memory-efficient DTW does not support distance={distance!r}")

    start = time.perf_counter()
    x = np.asarray(features_a.T, dtype=np.float32, order="C")
    y = np.asarray(features_b.T, dtype=np.float32, order="C")
    if distance == "cosine":
        x = _row_normalize(x)
        y = _row_normalize(y)
        y_norm_sq = None
    else:
        y_norm_sq = np.einsum("ij,ij->i", y, y, dtype=np.float64).astype(np.float32)

    n_rows, n_cols = x.shape[0], y.shape[0]
    directions, direction_path = _allocate_direction_matrix(n_rows, n_cols)
    prev = np.full(n_cols + 1, np.inf, dtype=np.float64)
    curr = np.full(n_cols + 1, np.inf, dtype=np.float64)
    prev[0] = 0.0

    rows_per_chunk = max(1, int(MEMORY_EFFICIENT_DTW_CHUNK_BYTES // max(n_cols * 4, 1)))
    try:
        for start_row in range(0, n_rows, rows_per_chunk):
            end_row = min(start_row + rows_per_chunk, n_rows)
            costs = _pairwise_cost_block(
                x[start_row:end_row],
                y,
                distance=distance,
                y_norm_sq=y_norm_sq,
            )
            np.nan_to_num(costs, copy=False, nan=1.0, posinf=1e6, neginf=1e6)
            _accumulate_dtw_chunk(costs, prev, curr, directions[start_row:end_row])

        path = _backtrack_direction_matrix(directions, n_rows, n_cols)
        runtime = time.perf_counter() - start
        return path, float(prev[n_cols]), runtime
    finally:
        if direction_path is not None:
            del directions
            try:
                os.remove(direction_path)
            except OSError:
                pass


def _allocate_direction_matrix(
    n_rows: int,
    n_cols: int,
) -> tuple[np.ndarray, str | None]:
    cells = int(n_rows) * int(n_cols)
    if cells >= MEMORY_EFFICIENT_DTW_MEMMAP_CELLS:
        handle = tempfile.NamedTemporaryFile(prefix="deepalign_dtw_steps_", suffix=".dat", delete=False)
        path = handle.name
        handle.close()
        return np.memmap(path, mode="w+", dtype=np.uint8, shape=(n_rows, n_cols)), path
    return np.empty((n_rows, n_cols), dtype=np.uint8), None


def _row_normalize(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, eps)


def _pairwise_cost_block(
    x_block: np.ndarray,
    y: np.ndarray,
    *,
    distance: str,
    y_norm_sq: np.ndarray | None,
) -> np.ndarray:
    dots = x_block @ y.T
    if distance == "cosine":
        costs = 1.0 - dots
    else:
        x_norm_sq = np.einsum("ij,ij->i", x_block, x_block, dtype=np.float64).astype(np.float32)
        costs = x_norm_sq[:, None] + y_norm_sq[None, :] - (2.0 * dots)
    return np.maximum(costs, 0.0).astype(np.float32, copy=False)


@njit(cache=True)
def _accumulate_dtw_chunk(
    costs: np.ndarray,
    prev: np.ndarray,
    curr: np.ndarray,
    directions: np.ndarray,
) -> None:
    n_rows, n_cols = costs.shape
    inf = np.inf
    for i in range(n_rows):
        curr[0] = inf
        for j in range(1, n_cols + 1):
            diag = prev[j - 1]
            up = prev[j]
            left = curr[j - 1]
            best = diag
            direction = 0
            if up < best:
                best = up
                direction = 1
            if left < best:
                best = left
                direction = 2
            curr[j] = costs[i, j - 1] + best
            directions[i, j - 1] = direction
        for j in range(n_cols + 1):
            prev[j] = curr[j]


def _backtrack_direction_matrix(
    directions: np.ndarray,
    n_rows: int,
    n_cols: int,
) -> np.ndarray:
    i = n_rows - 1
    j = n_cols - 1
    path: list[tuple[int, int]] = []
    while True:
        path.append((i, j))
        if i == 0 and j == 0:
            break
        direction = int(directions[i, j])
        if direction == 0:
            i -= 1
            j -= 1
        elif direction == 1:
            i -= 1
        else:
            j -= 1
        if i < 0:
            i = 0
        if j < 0:
            j = 0
    path.reverse()
    return np.asarray(path, dtype=np.intp).T


def banded_dtw_align(
    features_a: np.ndarray,
    features_b: np.ndarray,
    *,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    distance: str = "cosine",
) -> tuple[np.ndarray, float, float]:
    """Align with DTW constrained to a row-wise band."""
    start = time.perf_counter()
    cost_matrix = cdist(features_a.T, features_b.T, metric=distance).astype(np.float64)
    cost_matrix = np.nan_to_num(cost_matrix, nan=1.0, posinf=1e6, neginf=1e6)
    path, cost = _dtw_with_bounds(cost_matrix, lower_bounds=lower_bounds, upper_bounds=upper_bounds)
    runtime = time.perf_counter() - start
    return path, cost, runtime


def evaluate_pairwise_methods(
    *,
    methods: list[str],
    pair_id: str,
    group_id: str,
    dataset_name: str,
    piece_a_id: str,
    piece_b_id: str,
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    gt_a: np.ndarray,
    gt_b: np.ndarray,
    chroma_hop: int,
    sr: int,
    encoder: Any | None = None,
    deep_hop: int = 220,
    pool_size: int = 2,
    device: str | None = None,
    mrmsdtw_memory_limit_mb: int = 500,
    audio_a_path: str | Path | None = None,
    audio_b_path: str | Path | None = None,
    cache_root: str | Path | None = None,
    deep_decode: str = "unconstrained",
    deep_distance: str = "sqeuclidean",
    band_radius_frames: int | None = None,
    transcription_cache_root: str | Path | None = None,
    fusion_deep_weight: float = 1.0,
    fusion_onset_weight: float = 1.0,
    fusion_note_weight: float = 0.75,
    fusion_dlnco_weight: float = 0.5,
    fusion_chroma_weight: float = 0.25,
    refine_window_sec: float = 8.0,
    score_refine_radius_sec: float = 0.5,
    score_path: str | Path | None = None,
    event_ids: list[str] | tuple[str, ...] | np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Evaluate audio-to-audio methods that produce a direct warping path."""
    from dis_alignment.features.chroma import extract_chroma_cqt

    results: list[dict[str, Any]] = []
    duration_s = max(len(audio_a), len(audio_b)) / sr

    if "chroma_dtw" in methods or "mrmsdtw" in methods:
        chroma_a = extract_chroma_cqt(audio_a, sr=sr, hop_length=chroma_hop)
        chroma_b = extract_chroma_cqt(audio_b, sr=sr, hop_length=chroma_hop)
    else:
        chroma_a = chroma_b = None

    if "chroma_dtw" in methods and chroma_a is not None and chroma_b is not None:
        path, _, runtime = fast_dtw_align(chroma_a, chroma_b, distance="cosine")
        frame_duration = chroma_hop / sr
        pred_b = _interp_monotonic(gt_a, path[0] * frame_duration, path[1] * frame_duration)
        results.append(
            _build_result_row(
                dataset=dataset_name,
                pair_id=pair_id,
                group_id=group_id,
                piece_a_id=piece_a_id,
                piece_b_id=piece_b_id,
                method="chroma_dtw",
                duration_s=duration_s,
                runtime_s=runtime,
                memory_mb=np.nan,
                gt_a=gt_a,
                gt_b=gt_b,
                pred_b=pred_b,
            )
        )

    if "mrmsdtw" in methods and chroma_a is not None and chroma_b is not None:
        result = align_mrmsdtw(chroma_a, chroma_b, memory_limit_mb=mrmsdtw_memory_limit_mb, feature_rate=sr / chroma_hop)
        frame_duration = chroma_hop / sr
        pred_b = _interp_monotonic(gt_a, result.path[0] * frame_duration, result.path[1] * frame_duration)
        results.append(
            _build_result_row(
                dataset=dataset_name,
                pair_id=pair_id,
                group_id=group_id,
                piece_a_id=piece_a_id,
                piece_b_id=piece_b_id,
                method="mrmsdtw",
                duration_s=duration_s,
                runtime_s=result.runtime_seconds,
                memory_mb=result.memory_bytes / (1024 * 1024),
                gt_a=gt_a,
                gt_b=gt_b,
                pred_b=pred_b,
            )
        )

    resolved_deep_decode = parse_deep_decode(deep_decode)

    if "deepalign" in methods:
        if encoder is None:
            raise ValueError("The deepalign method was requested without a loaded encoder.")
        from dis_alignment.model.inference import extract_deep_features

        feat_a = _extract_deep_features_compat(
            extract_deep_features,
            audio_a=audio_a,
            audio_path=audio_a_path,
            encoder=encoder,
            sr=sr,
            hop_length=deep_hop,
            device=device,
            cache_root=cache_root,
        )
        feat_b = _extract_deep_features_compat(
            extract_deep_features,
            audio_a=audio_b,
            audio_path=audio_b_path,
            encoder=encoder,
            sr=sr,
            hop_length=deep_hop,
            device=device,
            cache_root=cache_root,
        )
        pooled_a = temporal_pool(feat_a, pool_size=pool_size)
        pooled_b = temporal_pool(feat_b, pool_size=pool_size)
        refinement_features: tuple[np.ndarray, np.ndarray] | None = None
        if resolved_deep_decode == "unconstrained":
            path_d, _, runtime_d = fast_dtw_align(pooled_a, pooled_b, distance=deep_distance)
        elif resolved_deep_decode == "diagonal_band":
            resolved_band = band_radius_frames if band_radius_frames is not None else 150
            lower, upper = _diagonal_band_bounds(pooled_a.shape[1], pooled_b.shape[1], resolved_band)
            path_d, _, runtime_d = banded_dtw_align(
                pooled_a,
                pooled_b,
                lower_bounds=lower,
                upper_bounds=upper,
                distance=deep_distance,
            )
        elif resolved_deep_decode == "chroma_guided_band":
            if chroma_a is None or chroma_b is None:
                chroma_a = extract_chroma_cqt(audio_a, sr=sr, hop_length=chroma_hop)
                chroma_b = extract_chroma_cqt(audio_b, sr=sr, hop_length=chroma_hop)
            coarse_path, _, coarse_runtime = fast_dtw_align(chroma_a, chroma_b, distance="cosine")
            resolved_band = band_radius_frames if band_radius_frames is not None else 150
            lower, upper = _guided_band_bounds(
                coarse_path=coarse_path,
                n_query=pooled_a.shape[1],
                n_reference=pooled_b.shape[1],
                coarse_frame_duration=chroma_hop / sr,
                deep_frame_duration=(deep_hop * pool_size) / sr,
                band_radius_frames=resolved_band,
            )
            path_d, _, runtime_d = banded_dtw_align(
                pooled_a,
                pooled_b,
                lower_bounds=lower,
                upper_bounds=upper,
                distance=deep_distance,
            )
            runtime_d += coarse_runtime
        elif resolved_deep_decode in {"deepalign_transcription_fused", "deepalign_transcription_fused_refined"}:
            fused_a = _build_transcription_fused_features(
                audio=audio_a,
                audio_path=audio_a_path,
                deep_features=pooled_a,
                sr=sr,
                frame_hop=deep_hop * pool_size,
                transcription_cache_root=transcription_cache_root,
                fusion_deep_weight=fusion_deep_weight,
                fusion_onset_weight=fusion_onset_weight,
                fusion_note_weight=fusion_note_weight,
                fusion_dlnco_weight=fusion_dlnco_weight,
                fusion_chroma_weight=fusion_chroma_weight,
            )
            fused_b = _build_transcription_fused_features(
                audio=audio_b,
                audio_path=audio_b_path,
                deep_features=pooled_b,
                sr=sr,
                frame_hop=deep_hop * pool_size,
                transcription_cache_root=transcription_cache_root,
                fusion_deep_weight=fusion_deep_weight,
                fusion_onset_weight=fusion_onset_weight,
                fusion_note_weight=fusion_note_weight,
                fusion_dlnco_weight=fusion_dlnco_weight,
                fusion_chroma_weight=fusion_chroma_weight,
            )
            refinement_features = (fused_a, fused_b)
            path_d, _, runtime_d = fast_dtw_align(fused_a, fused_b, distance=deep_distance)
        elif resolved_deep_decode == "deepalign_transcription_guided":
            fused_a = _build_transcription_fused_features(
                audio=audio_a,
                audio_path=audio_a_path,
                deep_features=pooled_a,
                sr=sr,
                frame_hop=deep_hop * pool_size,
                transcription_cache_root=transcription_cache_root,
                fusion_deep_weight=fusion_deep_weight,
                fusion_onset_weight=fusion_onset_weight,
                fusion_note_weight=fusion_note_weight,
                fusion_dlnco_weight=fusion_dlnco_weight,
                fusion_chroma_weight=fusion_chroma_weight,
            )
            fused_b = _build_transcription_fused_features(
                audio=audio_b,
                audio_path=audio_b_path,
                deep_features=pooled_b,
                sr=sr,
                frame_hop=deep_hop * pool_size,
                transcription_cache_root=transcription_cache_root,
                fusion_deep_weight=fusion_deep_weight,
                fusion_onset_weight=fusion_onset_weight,
                fusion_note_weight=fusion_note_weight,
                fusion_dlnco_weight=fusion_dlnco_weight,
                fusion_chroma_weight=fusion_chroma_weight,
            )
            coarse_a = _build_transcription_coarse_features(
                audio=audio_a,
                audio_path=audio_a_path,
                target_frames=fused_a.shape[1],
                sr=sr,
                frame_hop=deep_hop * pool_size,
                transcription_cache_root=transcription_cache_root,
            )
            coarse_b = _build_transcription_coarse_features(
                audio=audio_b,
                audio_path=audio_b_path,
                target_frames=fused_b.shape[1],
                sr=sr,
                frame_hop=deep_hop * pool_size,
                transcription_cache_root=transcription_cache_root,
            )
            coarse_path, _, coarse_runtime = fast_dtw_align(coarse_a, coarse_b, distance=deep_distance)
            frame_duration_d = (deep_hop * pool_size) / sr
            resolved_band = (
                band_radius_frames
                if band_radius_frames is not None
                else max(1, int(np.ceil(refine_window_sec / frame_duration_d)))
            )
            lower, upper = _guided_band_bounds(
                coarse_path=coarse_path,
                n_query=fused_a.shape[1],
                n_reference=fused_b.shape[1],
                coarse_frame_duration=frame_duration_d,
                deep_frame_duration=frame_duration_d,
                band_radius_frames=resolved_band,
            )
            try:
                path_d, _, runtime_d = banded_dtw_align(
                    fused_a,
                    fused_b,
                    lower_bounds=lower,
                    upper_bounds=upper,
                    distance=deep_distance,
                )
                runtime_d += coarse_runtime
            except ValueError:
                path_d, _, runtime_d = fast_dtw_align(fused_a, fused_b, distance=deep_distance)
                runtime_d += coarse_runtime
        else:
            frame_duration_d = (deep_hop * pool_size) / sr
            pred_b_d, runtime_d = _predict_score_guided_pairwise_anchors(
                audio_a=audio_a,
                audio_b=audio_b,
                audio_a_path=audio_a_path,
                audio_b_path=audio_b_path,
                fused_a=_build_transcription_fused_features(
                    audio=audio_a,
                    audio_path=audio_a_path,
                    deep_features=pooled_a,
                    sr=sr,
                    frame_hop=deep_hop * pool_size,
                    transcription_cache_root=transcription_cache_root,
                    fusion_deep_weight=fusion_deep_weight,
                    fusion_onset_weight=fusion_onset_weight,
                    fusion_note_weight=fusion_note_weight,
                    fusion_dlnco_weight=fusion_dlnco_weight,
                    fusion_chroma_weight=fusion_chroma_weight,
                ),
                fused_b=_build_transcription_fused_features(
                    audio=audio_b,
                    audio_path=audio_b_path,
                    deep_features=pooled_b,
                    sr=sr,
                    frame_hop=deep_hop * pool_size,
                    transcription_cache_root=transcription_cache_root,
                    fusion_deep_weight=fusion_deep_weight,
                    fusion_onset_weight=fusion_onset_weight,
                    fusion_note_weight=fusion_note_weight,
                    fusion_dlnco_weight=fusion_dlnco_weight,
                    fusion_chroma_weight=fusion_chroma_weight,
                ),
                gt_a=gt_a,
                event_ids=event_ids,
                score_path=score_path,
                sr=sr,
                frame_hop=deep_hop * pool_size,
                transcription_cache_root=transcription_cache_root,
                frame_duration=frame_duration_d,
                score_refine_radius_sec=score_refine_radius_sec,
            )
        if resolved_deep_decode != "deepalign_score_guided_refined":
            frame_duration_d = (deep_hop * pool_size) / sr
            pred_b_d = _interp_monotonic(gt_a, path_d[0] * frame_duration_d, path_d[1] * frame_duration_d)
            if resolved_deep_decode == "deepalign_transcription_fused_refined":
                if refinement_features is None:
                    raise ValueError("Local refinement requires fused transcription features.")
                start = time.perf_counter()
                pred_b_d = _refine_predicted_times_from_query(
                    query_times=gt_a,
                    predicted_reference_times=pred_b_d,
                    features_a=refinement_features[0],
                    features_b=refinement_features[1],
                    frame_duration=frame_duration_d,
                    radius_sec=score_refine_radius_sec,
                )
                runtime_d += time.perf_counter() - start
        results.append(
            _build_result_row(
                dataset=dataset_name,
                pair_id=pair_id,
                group_id=group_id,
                piece_a_id=piece_a_id,
                piece_b_id=piece_b_id,
                method="deepalign",
                duration_s=duration_s,
                runtime_s=runtime_d,
                memory_mb=np.nan,
                gt_a=gt_a,
                gt_b=gt_b,
                pred_b=pred_b_d,
                extra_fields={
                    "deep_decode": resolved_deep_decode,
                    "band_radius_frames": float(band_radius_frames) if band_radius_frames is not None else np.nan,
                    "pool_size": float(pool_size),
                    "deep_distance": deep_distance,
                    "fusion_deep_weight": float(fusion_deep_weight),
                    "fusion_onset_weight": float(fusion_onset_weight),
                    "fusion_note_weight": float(fusion_note_weight),
                    "fusion_dlnco_weight": float(fusion_dlnco_weight),
                    "fusion_chroma_weight": float(fusion_chroma_weight),
                    "refine_window_sec": float(refine_window_sec),
                    "score_refine_radius_sec": float(score_refine_radius_sec),
                },
            )
        )

    return results


def build_matchmaker_row(
    *,
    dataset_name: str,
    pair_id: str,
    group_id: str,
    piece_a_id: str,
    piece_b_id: str,
    gt_a: np.ndarray,
    gt_b: np.ndarray,
    pred_a_anchor: np.ndarray,
    pred_b_anchor: np.ndarray,
    result_a: MatchmakerResult,
    result_b: MatchmakerResult,
) -> dict[str, Any]:
    """Compose two score-following runs into a pairwise alignment row."""
    pred_a_anchor, pred_b_anchor = _filter_monotonic_pair(pred_a_anchor, pred_b_anchor)
    pred_b = np.interp(gt_a, pred_a_anchor, pred_b_anchor)
    duration_s = max(result_a.times_s[-1], result_b.times_s[-1])
    return _build_result_row(
        dataset=dataset_name,
        pair_id=pair_id,
        group_id=group_id,
        piece_a_id=piece_a_id,
        piece_b_id=piece_b_id,
        method="matchmaker",
        duration_s=float(duration_s),
        runtime_s=float(result_a.runtime_seconds + result_b.runtime_seconds),
        memory_mb=np.nan,
        gt_a=gt_a,
        gt_b=gt_b,
        pred_b=pred_b,
    )


def map_score_positions_to_events(
    result: MatchmakerResult,
    event_positions: pd.DataFrame,
) -> pd.DataFrame:
    """Map score positions to estimated event times in the performance."""
    positions = np.maximum.accumulate(result.score_positions.astype(np.float64))
    times = result.times_s.astype(np.float64)
    if len(positions) != len(times):
        raise ValueError("Matchmaker time and position arrays must have the same length.")
    return pd.DataFrame(
        {
            "event_id": event_positions["event_id"].astype(str).to_numpy(),
            "time_s": np.interp(
                event_positions["score_position"].to_numpy(dtype=float),
                positions,
                times,
            ),
        }
    )


def run_matchmaker_for_piece(
    score_file: str | Path,
    performance_file: str | Path,
    *,
    method: str = "arzt",
    feature_type: str = "chroma",
    sample_rate: int = 22050,
    frame_rate: int = 100,
) -> MatchmakerResult:
    """Thin wrapper kept here for test-friendly monkeypatching."""
    return run_matchmaker(
        score_file,
        performance_file,
        method=method,
        feature_type=feature_type,
        sample_rate=sample_rate,
        frame_rate=frame_rate,
    )


def extract_musicxml_measure_positions(path: str | Path) -> pd.DataFrame:
    """Extract measure-start score positions from a MusicXML score."""
    root = ET.parse(path).getroot()
    ns_prefix = ""
    if root.tag.startswith("{"):
        ns_prefix = root.tag.split("}", 1)[0] + "}"

    part = root.find(f"{ns_prefix}part")
    if part is None:
        raise ValueError(f"Could not find a MusicXML <part> in {path}")

    positions: list[tuple[str, float]] = []
    current_position = 0.0
    divisions = 1.0

    for measure in part.findall(f"{ns_prefix}measure"):
        measure_number = measure.attrib.get("number")
        if measure_number:
            positions.append((str(measure_number), current_position))

        attributes = measure.find(f"{ns_prefix}attributes")
        if attributes is not None:
            divisions_el = attributes.find(f"{ns_prefix}divisions")
            if divisions_el is not None and divisions_el.text:
                divisions = float(divisions_el.text)

        measure_duration = _measure_duration_in_beats(measure, ns_prefix, divisions)
        current_position += measure_duration

    if not positions:
        raise ValueError(f"No measures found in MusicXML score {path}")
    return pd.DataFrame(positions, columns=["event_id", "score_position"])


def extract_score_event_positions_from_annotations(
    annotations: pd.DataFrame,
) -> pd.DataFrame:
    """Use annotation ids as score positions when they already encode monotonic order."""
    return pd.DataFrame(
        {
            "event_id": annotations["event_id"].astype(str).to_numpy(),
            "score_position": np.arange(len(annotations), dtype=np.float64),
        }
    )


def save_evaluation_results(results: pd.DataFrame, path: str | Path) -> Path:
    """Persist evaluation results to CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    return output_path


def merge_evaluation_results(
    paths: list[str | Path] | tuple[str | Path, ...],
    *,
    dedupe_on: list[str] | tuple[str, ...] = (
        "dataset",
        "pair_id",
        "group_id",
        "piece_a_id",
        "piece_b_id",
        "method",
        *DEEP_VARIANT_COLUMNS,
    ),
) -> pd.DataFrame:
    """Merge one or more evaluation CSVs into a single de-duplicated table."""
    resolved_paths = [Path(path) for path in paths]
    if not resolved_paths:
        raise ValueError("At least one results path is required.")

    frames: list[pd.DataFrame] = []
    for path in resolved_paths:
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame = frame.copy()
        for column in DEEP_VARIANT_COLUMNS:
            if column not in frame.columns:
                frame[column] = np.nan
        frame["_source_path"] = str(path)
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True, sort=False)
    missing = [column for column in dedupe_on if column not in merged.columns]
    if missing:
        raise ValueError(
            "Cannot merge evaluation results because required columns are missing: "
            + ", ".join(missing)
        )

    merged = merged.drop_duplicates(subset=list(dedupe_on), keep="last")
    merged = merged.drop(columns="_source_path", errors="ignore")

    sort_columns = [
        column
        for column in ("dataset", "group_id", "pair_id", "method", "deep_decode", "piece_a_id", "piece_b_id")
        if column in merged.columns
    ]
    if sort_columns:
        merged = merged.sort_values(sort_columns, kind="stable").reset_index(drop=True)

    return merged


def summarize_evaluation(results: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Aggregate evaluation metrics per method."""
    summary: dict[str, dict[str, float]] = {}
    if results.empty:
        return summary

    labelled = add_method_variant_column(results)
    method_col = METHOD_VARIANT_COLUMN if METHOD_VARIANT_COLUMN in labelled.columns else "method"
    for method, method_frame in labelled.groupby(method_col):
        summary[method] = {
            "pairs": float(len(method_frame)),
            "mae_ms": float(method_frame["mae"].mean() * 1000),
            "median_ae_ms": float(method_frame["median_ae"].mean() * 1000),
            "ar_50ms_pct": float(method_frame["ar_50ms"].mean() * 100),
            "ar_100ms_pct": float(method_frame["ar_100ms"].mean() * 100),
            "ar_200ms_pct": float(method_frame["ar_200ms"].mean() * 100),
            "runtime_s": float(method_frame["runtime_s"].mean()),
        }
    return summary


def check_success_criteria(
    results: pd.DataFrame,
    candidate_method: str = "deepalign",
    candidate_deep_decode: str | None = None,
) -> dict[str, Any]:
    """Evaluate the dissertation success criteria against one candidate method."""
    candidate_rows = results[results["method"] == candidate_method]
    if candidate_deep_decode is not None:
        if "deep_decode" not in candidate_rows.columns:
            candidate_rows = candidate_rows.iloc[0:0]
        else:
            requested_decode = _normalize_variant_value(candidate_deep_decode)
            candidate_rows = candidate_rows[
                candidate_rows["deep_decode"].map(_normalize_variant_value) == requested_decode
            ]

    if candidate_deep_decode is None and candidate_method == "deepalign" and _needs_method_variant_labels(results):
        variants = sorted(
            {
                _normalize_variant_value(value)
                for value in results.loc[results["method"].astype(str).eq(candidate_method), "deep_decode"]
                if _normalize_variant_value(value)
            }
        )
        return {
            "available": False,
            "criterion_1_pass": False,
            "criterion_2_mae_pass": False,
            "criterion_2_ar_pass": False,
            "candidate_method": candidate_method,
            "error": (
                "Multiple DeepAlign variants are present; choose one with "
                "`candidate_deep_decode`. Available variants: " + ", ".join(variants)
            ),
        }

    if candidate_rows.empty:
        return {
            "available": False,
            "criterion_1_pass": False,
            "criterion_2_mae_pass": False,
            "criterion_2_ar_pass": False,
            "candidate_method": candidate_method,
        }

    mae = float(candidate_rows["mae"].mean())
    ar_50 = float(candidate_rows["ar_50ms"].mean())
    return {
        "available": True,
        "candidate_method": candidate_method,
        "mae_seconds": mae,
        "ar_50ms": ar_50,
        "criterion_1_pass": mae < 0.05,
        "criterion_2_mae_pass": mae < 0.02,
        "criterion_2_ar_pass": ar_50 > 0.98,
    }


def _needs_method_variant_labels(results: pd.DataFrame) -> bool:
    if results.empty or "method" not in results.columns or "deep_decode" not in results.columns:
        return False
    deep_rows = results[results["method"].astype(str).eq("deepalign")]
    if deep_rows.empty:
        return False
    variants = {_normalize_variant_value(value) for value in deep_rows["deep_decode"]}
    variants.discard("")
    has_unlabelled_deep_rows = any(_normalize_variant_value(value) == "" for value in deep_rows["deep_decode"])
    return len(variants) > 1 or (bool(variants) and has_unlabelled_deep_rows)


def _normalize_variant_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none"} else text


def _build_result_row(
    *,
    dataset: str,
    pair_id: str,
    group_id: str,
    piece_a_id: str,
    piece_b_id: str,
    method: str,
    duration_s: float,
    runtime_s: float,
    memory_mb: float,
    gt_a: np.ndarray,
    gt_b: np.ndarray,
    pred_b: np.ndarray,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "dataset": dataset,
        "pair_id": pair_id,
        "group_id": group_id,
        "piece_a_id": piece_a_id,
        "piece_b_id": piece_b_id,
        "method": method,
        "duration_s": float(duration_s),
        "memory_mb": float(memory_mb) if np.isfinite(memory_mb) else np.nan,
        "mae": mean_absolute_error(pred_b, gt_b),
        "median_ae": median_absolute_error(pred_b, gt_b),
        "ar_50ms": alignment_rate(pred_b, gt_b, 0.05),
        "ar_100ms": alignment_rate(pred_b, gt_b, 0.10),
        "ar_200ms": alignment_rate(pred_b, gt_b, 0.20),
        "runtime_s": float(runtime_s),
        "n_gt_points": int(len(gt_a)),
    }
    if extra_fields:
        row.update(extra_fields)
    return row


def _build_transcription_fused_features(
    *,
    audio: np.ndarray,
    audio_path: str | Path | None,
    deep_features: np.ndarray,
    sr: int,
    frame_hop: int,
    transcription_cache_root: str | Path | None,
    fusion_deep_weight: float,
    fusion_onset_weight: float,
    fusion_note_weight: float,
    fusion_dlnco_weight: float,
    fusion_chroma_weight: float,
) -> np.ndarray:
    from dis_alignment.features.chroma import extract_chroma_cqt
    from dis_alignment.features.dlnco import extract_dlnco
    from dis_alignment.features.transcription import extract_basic_pitch_features

    target_frames = deep_features.shape[1]
    if audio_path is None:
        raise ValueError("Transcription-backed DeepAlign decoding requires audio file paths.")

    transcription = extract_basic_pitch_features(
        audio_path,
        cache_root=transcription_cache_root or ".cache/transcription/basic_pitch",
        target_frames=target_frames,
    )
    chroma = extract_chroma_cqt(audio, sr=sr, hop_length=frame_hop)
    dlnco = extract_dlnco(audio, sr=sr, hop_length=frame_hop)
    frame_features = np.vstack([transcription["note"], transcription["contour"]])
    return _weighted_feature_stack(
        (deep_features, fusion_deep_weight),
        (transcription["onset"], fusion_onset_weight),
        (frame_features, fusion_note_weight),
        (dlnco, fusion_dlnco_weight),
        (chroma, fusion_chroma_weight),
        target_frames=target_frames,
    )


def _build_transcription_coarse_features(
    *,
    audio: np.ndarray,
    audio_path: str | Path | None,
    target_frames: int,
    sr: int,
    frame_hop: int,
    transcription_cache_root: str | Path | None,
) -> np.ndarray:
    from dis_alignment.features.chroma import extract_chroma_cqt
    from dis_alignment.features.transcription import extract_basic_pitch_features

    if audio_path is None:
        raise ValueError("Transcription-backed DeepAlign decoding requires audio file paths.")

    transcription = extract_basic_pitch_features(
        audio_path,
        cache_root=transcription_cache_root or ".cache/transcription/basic_pitch",
        target_frames=target_frames,
    )
    chroma = extract_chroma_cqt(audio, sr=sr, hop_length=frame_hop)
    return _weighted_feature_stack(
        (chroma, 0.5),
        (transcription["pitch_class_onset"], 1.0),
        (transcription["pitch_class_note"], 0.75),
        target_frames=target_frames,
    )


def _predict_score_guided_pairwise_anchors(
    *,
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    audio_a_path: str | Path | None,
    audio_b_path: str | Path | None,
    fused_a: np.ndarray,
    fused_b: np.ndarray,
    gt_a: np.ndarray,
    event_ids: list[str] | tuple[str, ...] | np.ndarray | None,
    score_path: str | Path | None,
    sr: int,
    frame_hop: int,
    transcription_cache_root: str | Path | None,
    frame_duration: float,
    score_refine_radius_sec: float,
) -> tuple[np.ndarray, float]:
    if audio_a_path is None or audio_b_path is None:
        raise ValueError("Score-guided DeepAlign decoding requires audio file paths.")
    if score_path is None:
        raise ValueError("Score-guided DeepAlign decoding requires a score path.")
    if event_ids is None:
        raise ValueError("Score-guided DeepAlign decoding requires event ids for the target anchors.")

    score_features, score_event_times, score_frame_duration = _build_score_reference_features(
        score_path=score_path,
        event_ids=[str(event_id) for event_id in event_ids],
        frame_rate=1.0 / frame_duration,
    )
    audio_score_a = _build_audio_score_features(
        audio=audio_a,
        audio_path=audio_a_path,
        sr=sr,
        frame_hop=frame_hop,
        target_frames=fused_a.shape[1],
        transcription_cache_root=transcription_cache_root,
    )
    audio_score_b = _build_audio_score_features(
        audio=audio_b,
        audio_path=audio_b_path,
        sr=sr,
        frame_hop=frame_hop,
        target_frames=fused_b.shape[1],
        transcription_cache_root=transcription_cache_root,
    )

    pred_anchor_a, runtime_a = _predict_audio_times_from_score_path(
        audio_score_a,
        score_features,
        score_event_times=score_event_times,
        audio_frame_duration=frame_duration,
        score_frame_duration=score_frame_duration,
    )
    pred_anchor_b, runtime_b = _predict_audio_times_from_score_path(
        audio_score_b,
        score_features,
        score_event_times=score_event_times,
        audio_frame_duration=frame_duration,
        score_frame_duration=score_frame_duration,
    )

    refine_runtime = 0.0
    if score_refine_radius_sec > 0:
        start = time.perf_counter()
        pred_anchor_a, pred_anchor_b = _refine_pairwise_anchors(
            pred_anchor_a=pred_anchor_a,
            pred_anchor_b=pred_anchor_b,
            features_a=fused_a,
            features_b=fused_b,
            frame_duration=frame_duration,
            radius_sec=score_refine_radius_sec,
        )
        refine_runtime = time.perf_counter() - start

    pred_anchor_a, pred_anchor_b = _filter_monotonic_pair(pred_anchor_a, pred_anchor_b)
    pred_b = np.interp(gt_a, pred_anchor_a, pred_anchor_b)
    return pred_b, runtime_a + runtime_b + refine_runtime


def _build_score_reference_features(
    *,
    score_path: str | Path,
    event_ids: list[str],
    frame_rate: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    from dis_alignment.features.score import extract_midi_score_features, score_position_to_time

    xml_path = Path(score_path)
    midi_path = _resolve_score_midi_path(xml_path)
    score_grid = extract_midi_score_features(midi_path, frame_rate=frame_rate)
    measure_positions = extract_musicxml_measure_positions(xml_path)
    position_lookup = dict(
        zip(measure_positions["event_id"].astype(str), measure_positions["score_position"], strict=False)
    )
    numeric_events, numeric_positions = _numeric_score_position_index(measure_positions)

    event_times = np.asarray(
        [
            np.clip(
                score_position_to_time(
                    score_grid.midi,
                    _resolve_score_position(event_id, position_lookup, numeric_events, numeric_positions),
                ),
                0.0,
                score_grid.duration_s,
            )
            for event_id in event_ids
        ],
        dtype=np.float64,
    )
    score_features = _weighted_feature_stack(
        (score_grid.onset, 1.0),
        (score_grid.note, 0.75),
        (score_grid.onset_decay, 0.5),
        (score_grid.note, 0.25),
        target_frames=score_grid.note.shape[1],
    )
    return score_features, event_times, 1.0 / score_grid.frame_rate


def _numeric_score_position_index(measure_positions: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    events: list[float] = []
    positions: list[float] = []
    for event_id, score_position in measure_positions[["event_id", "score_position"]].itertuples(index=False):
        try:
            events.append(float(event_id))
            positions.append(float(score_position))
        except (TypeError, ValueError):
            continue
    if len(events) < 2:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    order = np.argsort(events)
    return np.asarray(events, dtype=np.float64)[order], np.asarray(positions, dtype=np.float64)[order]


def _resolve_score_position(
    event_id: str,
    position_lookup: dict[str, float],
    numeric_events: np.ndarray,
    numeric_positions: np.ndarray,
) -> float:
    if event_id in position_lookup:
        return float(position_lookup[event_id])
    try:
        numeric_id = float(event_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Score-guided DeepAlign could not resolve score position for event id {event_id}.") from exc

    if numeric_events.size < 2:
        raise ValueError(f"Score-guided DeepAlign could not resolve score position for event id {event_id}.")
    if numeric_id < numeric_events[0]:
        slope = (numeric_positions[1] - numeric_positions[0]) / max(numeric_events[1] - numeric_events[0], 1e-12)
        return float(numeric_positions[0] + (numeric_id - numeric_events[0]) * slope)
    if numeric_id > numeric_events[-1]:
        slope = (numeric_positions[-1] - numeric_positions[-2]) / max(numeric_events[-1] - numeric_events[-2], 1e-12)
        return float(numeric_positions[-1] + (numeric_id - numeric_events[-1]) * slope)
    return float(np.interp(numeric_id, numeric_events, numeric_positions))


def _build_audio_score_features(
    *,
    audio: np.ndarray,
    audio_path: str | Path,
    sr: int,
    frame_hop: int,
    target_frames: int,
    transcription_cache_root: str | Path | None,
) -> np.ndarray:
    from dis_alignment.features.chroma import extract_chroma_cqt
    from dis_alignment.features.dlnco import extract_dlnco
    from dis_alignment.features.transcription import extract_basic_pitch_features

    transcription = extract_basic_pitch_features(
        audio_path,
        cache_root=transcription_cache_root or ".cache/transcription/basic_pitch",
        target_frames=target_frames,
    )
    chroma = extract_chroma_cqt(audio, sr=sr, hop_length=frame_hop)
    dlnco = extract_dlnco(audio, sr=sr, hop_length=frame_hop)
    return _weighted_feature_stack(
        (transcription["pitch_class_onset"], 1.0),
        (transcription["pitch_class_note"], 0.75),
        (dlnco, 0.5),
        (chroma, 0.25),
        target_frames=target_frames,
    )


def _predict_audio_times_from_score_path(
    audio_features: np.ndarray,
    score_features: np.ndarray,
    *,
    score_event_times: np.ndarray,
    audio_frame_duration: float,
    score_frame_duration: float,
) -> tuple[np.ndarray, float]:
    path, _, runtime = fast_dtw_align(audio_features, score_features, distance="sqeuclidean")
    audio_times = path[0] * audio_frame_duration
    score_times = path[1] * score_frame_duration
    pred_audio_times = _interp_monotonic(score_event_times, score_times, audio_times)
    return np.maximum.accumulate(pred_audio_times), runtime


def _refine_pairwise_anchors(
    *,
    pred_anchor_a: np.ndarray,
    pred_anchor_b: np.ndarray,
    features_a: np.ndarray,
    features_b: np.ndarray,
    frame_duration: float,
    radius_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    radius_frames = max(1, int(round(radius_sec / frame_duration)))
    refined_a: list[float] = []
    refined_b: list[float] = []
    for time_a, time_b in zip(pred_anchor_a, pred_anchor_b, strict=False):
        center_a = int(np.clip(round(time_a / frame_duration), 0, features_a.shape[1] - 1))
        center_b = int(np.clip(round(time_b / frame_duration), 0, features_b.shape[1] - 1))
        best_a = _best_matching_frame(
            query=_local_context(features_b, center_b),
            candidates=features_a,
            center=center_a,
            radius_frames=radius_frames,
        )
        best_b = _best_matching_frame(
            query=_local_context(features_a, best_a),
            candidates=features_b,
            center=center_b,
            radius_frames=radius_frames,
        )
        refined_a.append(best_a * frame_duration)
        refined_b.append(best_b * frame_duration)
    return np.asarray(refined_a, dtype=np.float64), np.asarray(refined_b, dtype=np.float64)


def _refine_predicted_times_from_query(
    *,
    query_times: np.ndarray,
    predicted_reference_times: np.ndarray,
    features_a: np.ndarray,
    features_b: np.ndarray,
    frame_duration: float,
    radius_sec: float,
) -> np.ndarray:
    """Snap predicted reference times to the nearest local feature match."""
    if radius_sec <= 0:
        return predicted_reference_times
    radius_frames = max(1, int(round(radius_sec / frame_duration)))
    refined: list[float] = []
    for query_time, predicted_time in zip(query_times, predicted_reference_times, strict=False):
        query_frame = int(np.clip(round(query_time / frame_duration), 0, features_a.shape[1] - 1))
        center_b = int(np.clip(round(predicted_time / frame_duration), 0, features_b.shape[1] - 1))
        best_b = _best_matching_frame(
            query=_local_context(features_a, query_frame),
            candidates=features_b,
            center=center_b,
            radius_frames=radius_frames,
        )
        refined.append(best_b * frame_duration)
    return np.maximum.accumulate(np.asarray(refined, dtype=np.float64))


def _best_matching_frame(
    *,
    query: np.ndarray,
    candidates: np.ndarray,
    center: int,
    radius_frames: int,
) -> int:
    lower = max(0, center - radius_frames)
    upper = min(candidates.shape[1] - 1, center + radius_frames)
    window = candidates[:, lower : upper + 1]
    costs = np.sum((window - query[:, None]) ** 2, axis=0)
    if radius_frames > 0:
        offsets = np.arange(lower, upper + 1, dtype=np.float64) - float(center)
        costs = costs + 0.05 * (offsets / float(radius_frames)) ** 2
    return int(lower + np.argmin(costs))


def _local_context(features: np.ndarray, center: int, half_width: int = 2) -> np.ndarray:
    lower = max(0, center - half_width)
    upper = min(features.shape[1], center + half_width + 1)
    return np.mean(features[:, lower:upper], axis=1)


def _interp_monotonic(query_x: np.ndarray, path_x: np.ndarray, path_y: np.ndarray) -> np.ndarray:
    order = np.argsort(path_x, kind="stable")
    sorted_x = np.asarray(path_x, dtype=np.float64)[order]
    sorted_y = np.asarray(path_y, dtype=np.float64)[order]
    unique_x: list[float] = []
    unique_y: list[float] = []
    start = 0
    while start < len(sorted_x):
        end = start + 1
        while end < len(sorted_x) and sorted_x[end] == sorted_x[start]:
            end += 1
        unique_x.append(float(sorted_x[start]))
        unique_y.append(float(np.median(sorted_y[start:end])))
        start = end
    if len(unique_x) < 2:
        raise ValueError("Need at least two score path points for interpolation.")
    y = np.maximum.accumulate(np.asarray(unique_y, dtype=np.float64))
    return np.interp(query_x, np.asarray(unique_x, dtype=np.float64), y)


def _resolve_score_midi_path(score_path: Path) -> Path:
    if score_path.suffix.lower() in {".mid", ".midi"} and score_path.exists():
        return score_path

    candidates = [
        score_path.with_suffix(".mid"),
        score_path.with_suffix(".midi"),
        score_path.parent.parent / "score_midi" / f"{score_path.stem}.mid",
        score_path.parent.parent / "score_midi" / f"{score_path.stem}.midi",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve a MIDI score for {score_path}.")


def _weighted_feature_stack(
    *blocks: tuple[np.ndarray, float],
    target_frames: int,
) -> np.ndarray:
    prepared: list[np.ndarray] = []
    for block, weight in blocks:
        if weight <= 0:
            continue
        trimmed = _match_frame_count(np.asarray(block, dtype=np.float64), target_frames)
        prepared.append(_normalize_columns(trimmed) * float(weight))
    if not prepared:
        raise ValueError("At least one positive-weight feature block is required.")
    return np.vstack(prepared)


def _match_frame_count(features: np.ndarray, target_frames: int) -> np.ndarray:
    from dis_alignment.features.transcription import resample_feature_frames

    if features.shape[1] == target_frames:
        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return resample_feature_frames(features, target_frames)


def _normalize_columns(features: np.ndarray) -> np.ndarray:
    matrix = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    norms = np.linalg.norm(matrix, axis=0, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _filter_monotonic_pair(times_a: np.ndarray, times_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    filtered_a: list[float] = []
    filtered_b: list[float] = []
    last_a = -np.inf
    last_b = -np.inf
    for time_a, time_b in zip(times_a, times_b, strict=False):
        if time_a >= last_a and time_b >= last_b:
            filtered_a.append(float(time_a))
            filtered_b.append(float(time_b))
            last_a = float(time_a)
            last_b = float(time_b)
    if len(filtered_a) < 2:
        raise ValueError("Need at least two monotonic anchor points to compose a pairwise mapping.")
    return np.asarray(filtered_a, dtype=np.float64), np.asarray(filtered_b, dtype=np.float64)


def _diagonal_band_bounds(
    n_query: int,
    n_reference: int,
    band_radius_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    query_idx = np.arange(n_query, dtype=np.float64)
    if n_query <= 1:
        center = np.zeros(n_query, dtype=np.intp)
    else:
        center = np.rint(query_idx * max(n_reference - 1, 0) / max(n_query - 1, 1)).astype(np.intp)
    lower = np.clip(center - band_radius_frames, 0, max(n_reference - 1, 0))
    upper = np.clip(center + band_radius_frames, 0, max(n_reference - 1, 0))
    if n_query:
        lower[0] = 0
        upper[-1] = max(n_reference - 1, 0)
    return lower, upper


def _guided_band_bounds(
    *,
    coarse_path: np.ndarray,
    n_query: int,
    n_reference: int,
    coarse_frame_duration: float,
    deep_frame_duration: float,
    band_radius_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    coarse_times_q = coarse_path[0] * coarse_frame_duration
    coarse_times_r = coarse_path[1] * coarse_frame_duration
    deep_times_q = np.arange(n_query, dtype=np.float64) * deep_frame_duration
    guided_reference = np.interp(deep_times_q, coarse_times_q, coarse_times_r)
    center = np.rint(guided_reference / deep_frame_duration).astype(np.intp)
    lower = np.clip(center - band_radius_frames, 0, max(n_reference - 1, 0))
    upper = np.clip(center + band_radius_frames, 0, max(n_reference - 1, 0))
    if n_query:
        lower[0] = 0
        upper[-1] = max(n_reference - 1, 0)
    return lower, upper


def _dtw_with_bounds(
    cost_matrix: np.ndarray,
    *,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Run symmetric2 DTW while restricting each row to an allowed column interval."""
    n_query, n_reference = cost_matrix.shape
    D = np.full((n_query + 1, n_reference + 1), np.inf, dtype=np.float64)
    backptr = np.full((n_query + 1, n_reference + 1), -1, dtype=np.int8)
    D[0, 0] = 0.0

    for i in range(1, n_query + 1):
        j_start = max(1, int(lower_bounds[i - 1]) + 1)
        j_end = min(n_reference, int(upper_bounds[i - 1]) + 1)
        for j in range(j_start, j_end + 1):
            c = float(cost_matrix[i - 1, j - 1])
            candidates = (
                D[i - 1, j] + c,
                D[i, j - 1] + c,
                D[i - 1, j - 1] + 2.0 * c,
            )
            move = int(np.argmin(candidates))
            D[i, j] = candidates[move]
            backptr[i, j] = move

    if not np.isfinite(D[n_query, n_reference]):
        raise ValueError("Band-constrained DTW could not reach the end point with the requested bounds.")

    path: list[tuple[int, int]] = []
    i, j = n_query, n_reference
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        move = backptr[i, j]
        if move == 0:
            i -= 1
        elif move == 1:
            j -= 1
        elif move == 2:
            i -= 1
            j -= 1
        else:
            raise ValueError("Encountered an invalid backpointer during constrained DTW backtracking.")

    while i > 0:
        i -= 1
        path.append((i, 0))
    while j > 0:
        j -= 1
        path.append((0, j))

    path.reverse()
    return np.asarray(path, dtype=np.intp).T, float(D[n_query, n_reference])


def _measure_duration_in_beats(measure: ET.Element, ns_prefix: str, divisions: float) -> float:
    max_cursor = 0.0
    cursor = 0.0
    for child in measure:
        if child.tag == f"{ns_prefix}backup":
            duration_el = child.find(f"{ns_prefix}duration")
            if duration_el is not None and duration_el.text:
                cursor -= float(duration_el.text) / divisions
        elif child.tag == f"{ns_prefix}forward":
            duration_el = child.find(f"{ns_prefix}duration")
            if duration_el is not None and duration_el.text:
                cursor += float(duration_el.text) / divisions
                max_cursor = max(max_cursor, cursor)
        elif child.tag == f"{ns_prefix}note":
            if child.find(f"{ns_prefix}grace") is not None:
                continue
            duration_el = child.find(f"{ns_prefix}duration")
            if duration_el is None or duration_el.text is None:
                continue
            cursor += float(duration_el.text) / divisions
            max_cursor = max(max_cursor, cursor)
    return max_cursor


def _extract_deep_features_compat(
    fn,
    *,
    audio_a: np.ndarray,
    audio_path: str | Path | None,
    encoder: Any,
    sr: int,
    hop_length: int,
    device: str | None,
    cache_root: str | Path | None,
) -> np.ndarray:
    try:
        return fn(
            audio_a,
            encoder,
            sr=sr,
            hop_length=hop_length,
            device=device,
            audio_path=audio_path,
            cache_root=cache_root,
        )
    except TypeError:
        return fn(audio_a, encoder, sr=sr, hop_length=hop_length, device=device)
