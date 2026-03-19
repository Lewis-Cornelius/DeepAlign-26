"""SWD evaluation utilities for DeepAlign-26 and its baseline features."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from tqdm import tqdm

from dis_alignment.data.swd import SWDDataset, SWDPair, compute_ground_truth_alignment, load_swd_audio
from dis_alignment.evaluation.metrics import alignment_rate, mean_absolute_error, median_absolute_error


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


def fast_dtw_align(
    features_a: np.ndarray,
    features_b: np.ndarray,
    *,
    distance: str = "cosine",
) -> tuple[np.ndarray, float, float]:
    """Align two feature sequences with librosa's DTW implementation."""
    import librosa

    start = time.perf_counter()
    cost_matrix = cdist(features_a.T, features_b.T, metric=distance).astype(np.float64)
    accumulated_cost, warping_path = librosa.sequence.dtw(C=cost_matrix, backtrack=True)
    path = warping_path[::-1].T.astype(np.intp)
    runtime = time.perf_counter() - start
    return path, float(accumulated_cost[-1, -1]), runtime


def evaluate_pair(
    pair: SWDPair,
    *,
    encoder: Any | None = None,
    sr: int = 22050,
    chroma_hop: int = 440,
    deep_hop: int = 220,
    pool_size: int = 2,
    device: str | None = None,
) -> list[dict[str, Any]]:
    """Evaluate baseline chroma DTW and optional DeepAlign features on one SWD pair."""
    from dis_alignment.features.chroma import extract_chroma_cqt

    results: list[dict[str, Any]] = []
    ground_truth = compute_ground_truth_alignment(pair)
    if ground_truth is None:
        return results

    gt_times_a, gt_times_b = ground_truth
    audio_a, _ = load_swd_audio(pair.piece_a, sr=sr)
    audio_b, _ = load_swd_audio(pair.piece_b, sr=sr)
    duration_s = max(len(audio_a), len(audio_b)) / sr

    chroma_a = extract_chroma_cqt(audio_a, sr=sr, hop_length=chroma_hop)
    chroma_b = extract_chroma_cqt(audio_b, sr=sr, hop_length=chroma_hop)
    path, _, runtime = fast_dtw_align(chroma_a, chroma_b, distance="cosine")
    frame_duration = chroma_hop / sr
    pred_b = np.interp(gt_times_a, path[0] * frame_duration, path[1] * frame_duration)

    results.append({
        "pair_id": pair.pair_id,
        "lied_id": pair.lied_id,
        "method": "chroma_dtw",
        "duration_s": duration_s,
        "memory_mb": np.nan,
        "mae": mean_absolute_error(pred_b, gt_times_b),
        "median_ae": median_absolute_error(pred_b, gt_times_b),
        "ar_50ms": alignment_rate(pred_b, gt_times_b, 0.05),
        "ar_100ms": alignment_rate(pred_b, gt_times_b, 0.10),
        "ar_200ms": alignment_rate(pred_b, gt_times_b, 0.20),
        "runtime_s": runtime,
        "n_gt_points": len(gt_times_a),
    })

    if encoder is not None:
        from dis_alignment.model.inference import extract_deep_features

        feat_a = extract_deep_features(audio_a, encoder, sr=sr, hop_length=deep_hop, device=device)
        feat_b = extract_deep_features(audio_b, encoder, sr=sr, hop_length=deep_hop, device=device)
        pooled_a = temporal_pool(feat_a, pool_size=pool_size)
        pooled_b = temporal_pool(feat_b, pool_size=pool_size)
        path_d, _, runtime_d = fast_dtw_align(pooled_a, pooled_b, distance="sqeuclidean")
        frame_duration_d = (deep_hop * pool_size) / sr
        pred_b_d = np.interp(gt_times_a, path_d[0] * frame_duration_d, path_d[1] * frame_duration_d)

        results.append({
            "pair_id": pair.pair_id,
            "lied_id": pair.lied_id,
            "method": "deepalign",
            "duration_s": duration_s,
            "memory_mb": np.nan,
            "mae": mean_absolute_error(pred_b_d, gt_times_b),
            "median_ae": median_absolute_error(pred_b_d, gt_times_b),
            "ar_50ms": alignment_rate(pred_b_d, gt_times_b, 0.05),
            "ar_100ms": alignment_rate(pred_b_d, gt_times_b, 0.10),
            "ar_200ms": alignment_rate(pred_b_d, gt_times_b, 0.20),
            "runtime_s": runtime_d,
            "n_gt_points": len(gt_times_a),
        })

    return results


def evaluate_swd_dataset(
    dataset: SWDDataset,
    *,
    checkpoint_path: str | Path | None = None,
    device: str | None = None,
    performances: list[str] | None = None,
    lieder: list[str] | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Evaluate SWD pairs with the baseline and optional DeepAlign checkpoint."""
    encoder = None
    if checkpoint_path is not None:
        from dis_alignment.model.inference import load_trained_encoder

        encoder, _ = load_trained_encoder(checkpoint_path, device=device)

    pair_iter = list(dataset.iter_pairs(performances=performances, lieder=lieder))
    iterator = tqdm(pair_iter, desc="Evaluating SWD", disable=not show_progress)
    rows: list[dict[str, Any]] = []

    for pair in iterator:
        rows.extend(evaluate_pair(pair, encoder=encoder, device=device))

    return pd.DataFrame(rows)


def save_evaluation_results(results: pd.DataFrame, path: str | Path) -> Path:
    """Persist SWD evaluation results to CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    return output_path


def summarize_evaluation(results: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Aggregate evaluation metrics per method."""
    summary: dict[str, dict[str, float]] = {}
    if results.empty:
        return summary

    for method, method_frame in results.groupby("method"):
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


def check_success_criteria(results: pd.DataFrame) -> dict[str, Any]:
    """Evaluate the dissertation success criteria against DeepAlign results."""
    deepalign_rows = results[results["method"] == "deepalign"]
    if deepalign_rows.empty:
        return {
            "available": False,
            "criterion_1_pass": False,
            "criterion_2_mae_pass": False,
            "criterion_2_ar_pass": False,
        }

    mae = float(deepalign_rows["mae"].mean())
    ar_50 = float(deepalign_rows["ar_50ms"].mean())
    return {
        "available": True,
        "mae_seconds": mae,
        "ar_50ms": ar_50,
        "criterion_1_pass": mae < 0.05,
        "criterion_2_mae_pass": mae < 0.02,
        "criterion_2_ar_pass": ar_50 > 0.98,
    }
