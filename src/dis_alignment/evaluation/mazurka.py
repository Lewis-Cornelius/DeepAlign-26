"""MazurkaBL evaluation helpers for DeepAlign robustness checks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from dis_alignment.data.mazurka import (
    MazurkaDataset,
    MazurkaPair,
    compute_ground_truth_alignment,
    load_mazurka_audio,
    load_mazurka_annotations,
)
from dis_alignment.evaluation.common import (
    build_matchmaker_row,
    evaluate_pairwise_methods,
    extract_score_event_positions_from_annotations,
    map_score_positions_to_events,
    parse_methods,
    run_matchmaker_for_piece,
)

LOGGER = logging.getLogger(__name__)


def evaluate_pair(
    pair: MazurkaPair,
    *,
    methods: list[str],
    encoder: Any | None = None,
    sr: int = 22050,
    chroma_hop: int = 440,
    deep_hop: int = 220,
    pool_size: int = 2,
    device: str | None = None,
    mrmsdtw_memory_limit_mb: int = 500,
    cache_root: str | Path | None = None,
    deep_decode: str = "unconstrained",
    band_radius_frames: int | None = None,
    transcription_cache_root: str | Path | None = None,
    fusion_deep_weight: float = 1.0,
    fusion_onset_weight: float = 1.0,
    fusion_note_weight: float = 0.75,
    fusion_dlnco_weight: float = 0.5,
    fusion_chroma_weight: float = 0.25,
    refine_window_sec: float = 8.0,
    score_refine_radius_sec: float = 0.5,
    matchmaker_cache: dict[tuple[str, str, str], tuple[pd.DataFrame, Any]] | None = None,
    matchmaker_method: str = "arzt",
    matchmaker_feature_type: str = "chroma",
    matchmaker_frame_rate: int = 100,
) -> list[dict[str, Any]]:
    """Evaluate one Mazurka pair across the requested methods."""
    ground_truth = compute_ground_truth_alignment(pair)
    if ground_truth is None:
        return []

    annotations_a, annotations_b = ground_truth
    gt_a = annotations_a["time_s"].to_numpy(dtype=float)
    gt_b = annotations_b["time_s"].to_numpy(dtype=float)
    audio_a, _ = load_mazurka_audio(pair.piece_a, sr=sr)
    audio_b, _ = load_mazurka_audio(pair.piece_b, sr=sr)

    results = evaluate_pairwise_methods(
        methods=[method for method in methods if method != "matchmaker"],
        pair_id=pair.pair_id,
        group_id=pair.work_id,
        dataset_name="mazurka",
        piece_a_id=pair.piece_a.performance_id,
        piece_b_id=pair.piece_b.performance_id,
        audio_a=audio_a,
        audio_b=audio_b,
        gt_a=gt_a,
        gt_b=gt_b,
        chroma_hop=chroma_hop,
        sr=sr,
        encoder=encoder,
        deep_hop=deep_hop,
        pool_size=pool_size,
        device=device,
        mrmsdtw_memory_limit_mb=mrmsdtw_memory_limit_mb,
        audio_a_path=pair.piece_a.audio_path,
        audio_b_path=pair.piece_b.audio_path,
        cache_root=cache_root,
        deep_decode=deep_decode,
        band_radius_frames=band_radius_frames,
        transcription_cache_root=transcription_cache_root,
        fusion_deep_weight=fusion_deep_weight,
        fusion_onset_weight=fusion_onset_weight,
        fusion_note_weight=fusion_note_weight,
        fusion_dlnco_weight=fusion_dlnco_weight,
        fusion_chroma_weight=fusion_chroma_weight,
        refine_window_sec=refine_window_sec,
        score_refine_radius_sec=score_refine_radius_sec,
        score_path=pair.piece_a.score_path,
        event_ids=annotations_a["event_id"].astype(str).tolist(),
    )

    if "matchmaker" not in methods:
        return results

    if pair.piece_a.score_path is None or pair.piece_b.score_path is None:
        raise FileNotFoundError(
            f"Mazurka Matchmaker evaluation requires score files for {pair.piece_a.performance_id} "
            f"and {pair.piece_b.performance_id}."
        )

    if matchmaker_cache is None:
        matchmaker_cache = {}

    try:
        predicted_a, result_a = _get_matchmaker_predictions(
            piece=pair.piece_a,
            method=matchmaker_method,
            feature_type=matchmaker_feature_type,
            frame_rate=matchmaker_frame_rate,
            cache=matchmaker_cache,
            fallback_annotations=annotations_a,
        )
        predicted_b, result_b = _get_matchmaker_predictions(
            piece=pair.piece_b,
            method=matchmaker_method,
            feature_type=matchmaker_feature_type,
            frame_rate=matchmaker_frame_rate,
            cache=matchmaker_cache,
            fallback_annotations=annotations_b,
        )
    except Exception as exc:
        LOGGER.warning("Skipping Matchmaker for Mazurka pair %s: %s", pair.pair_id, exc)
        return results

    audio_a_ids = annotations_a["event_id"].astype(str).tolist()
    audio_b_ids = set(annotations_b["event_id"].astype(str))
    lookup_a = dict(zip(predicted_a["event_id"], predicted_a["time_s"], strict=False))
    lookup_b = dict(zip(predicted_b["event_id"], predicted_b["time_s"], strict=False))
    common_ids = [
        event_id
        for event_id in audio_a_ids
        if event_id in audio_b_ids and event_id in lookup_a and event_id in lookup_b
    ]
    if len(common_ids) < 2:
        return results

    pred_anchor_a = np.asarray([lookup_a[event_id] for event_id in common_ids], dtype=float)
    pred_anchor_b = np.asarray([lookup_b[event_id] for event_id in common_ids], dtype=float)
    gt_lookup_a = dict(zip(annotations_a["event_id"], annotations_a["time_s"], strict=False))
    gt_lookup_b = dict(zip(annotations_b["event_id"], annotations_b["time_s"], strict=False))
    gt_a_common = np.asarray([gt_lookup_a[event_id] for event_id in common_ids], dtype=float)
    gt_b_common = np.asarray([gt_lookup_b[event_id] for event_id in common_ids], dtype=float)

    results.append(
        build_matchmaker_row(
            dataset_name="mazurka",
            pair_id=pair.pair_id,
            group_id=pair.work_id,
            piece_a_id=pair.piece_a.performance_id,
            piece_b_id=pair.piece_b.performance_id,
            gt_a=gt_a_common,
            gt_b=gt_b_common,
            pred_a_anchor=pred_anchor_a,
            pred_b_anchor=pred_anchor_b,
            result_a=result_a,
            result_b=result_b,
        )
    )
    return results


def evaluate_mazurka_dataset(
    dataset: MazurkaDataset,
    *,
    checkpoint_path: str | Path | None = None,
    device: str | None = None,
    methods: str | list[str] | tuple[str, ...] | None = None,
    works: list[str] | None = None,
    show_progress: bool = True,
    matchmaker_method: str = "arzt",
    matchmaker_feature_type: str = "chroma",
    matchmaker_frame_rate: int = 100,
    sr: int = 22050,
    chroma_hop: int = 440,
    deep_hop: int = 220,
    pool_size: int = 2,
    mrmsdtw_memory_limit_mb: int = 500,
    cache_root: str | Path | None = None,
    deep_decode: str = "unconstrained",
    band_radius_frames: int | None = None,
    transcription_cache_root: str | Path | None = None,
    fusion_deep_weight: float = 1.0,
    fusion_onset_weight: float = 1.0,
    fusion_note_weight: float = 0.75,
    fusion_dlnco_weight: float = 0.5,
    fusion_chroma_weight: float = 0.25,
    refine_window_sec: float = 8.0,
    score_refine_radius_sec: float = 0.5,
) -> pd.DataFrame:
    """Evaluate Mazurka pairs with the requested methods."""
    resolved_methods = parse_methods(methods, checkpoint_path=checkpoint_path)
    encoder = None
    if "deepalign" in resolved_methods:
        from dis_alignment.model.inference import load_trained_encoder

        encoder, _ = load_trained_encoder(checkpoint_path, device=device)

    pair_iter = list(dataset.iter_pairs(works=works))
    iterator = tqdm(pair_iter, desc="Evaluating Mazurka", disable=not show_progress)
    rows: list[dict[str, Any]] = []
    matchmaker_cache: dict[tuple[str, str, str], tuple[pd.DataFrame, Any]] = {}

    for pair in iterator:
        try:
            rows.extend(
                evaluate_pair(
                    pair,
                    methods=resolved_methods,
                    encoder=encoder,
                    sr=sr,
                    chroma_hop=chroma_hop,
                    deep_hop=deep_hop,
                    pool_size=pool_size,
                    device=device,
                    mrmsdtw_memory_limit_mb=mrmsdtw_memory_limit_mb,
                    cache_root=cache_root,
                    deep_decode=deep_decode,
                    band_radius_frames=band_radius_frames,
                    transcription_cache_root=transcription_cache_root,
                    fusion_deep_weight=fusion_deep_weight,
                    fusion_onset_weight=fusion_onset_weight,
                    fusion_note_weight=fusion_note_weight,
                    fusion_dlnco_weight=fusion_dlnco_weight,
                    fusion_chroma_weight=fusion_chroma_weight,
                    refine_window_sec=refine_window_sec,
                    score_refine_radius_sec=score_refine_radius_sec,
                    matchmaker_cache=matchmaker_cache,
                    matchmaker_method=matchmaker_method,
                    matchmaker_feature_type=matchmaker_feature_type,
                    matchmaker_frame_rate=matchmaker_frame_rate,
                )
            )
        except Exception as exc:
            LOGGER.warning("Skipping Mazurka pair %s due to evaluation error: %s", pair.pair_id, exc)

    return pd.DataFrame(rows)


def _get_matchmaker_predictions(
    *,
    piece,
    method: str,
    feature_type: str,
    frame_rate: int,
    cache: dict[tuple[str, str, str], tuple[pd.DataFrame, Any]],
    fallback_annotations: pd.DataFrame,
) -> tuple[pd.DataFrame, Any]:
    cache_key = (piece.performance_id, method, feature_type)
    if cache_key in cache:
        return cache[cache_key]

    result = run_matchmaker_for_piece(
        piece.score_path,
        piece.audio_path,
        method=method,
        feature_type=feature_type,
        frame_rate=frame_rate,
    )
    event_positions = extract_score_event_positions_from_annotations(fallback_annotations)
    predicted = map_score_positions_to_events(result, event_positions)
    cache[cache_key] = (predicted, result)
    return predicted, result
