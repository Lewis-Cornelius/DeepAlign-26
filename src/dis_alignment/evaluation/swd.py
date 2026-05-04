"""SWD evaluation utilities for DeepAlign-26 and its baselines."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from dis_alignment.data.swd import (
    SWDDataset,
    SWDPair,
    compute_ground_truth_measure_alignment,
    load_swd_audio,
)
from dis_alignment.evaluation.common import (
    build_matchmaker_row,
    check_success_criteria,
    evaluate_pairwise_methods,
    extract_musicxml_measure_positions,
    map_score_positions_to_events,
    parse_methods,
    run_matchmaker_for_piece,
    save_evaluation_results,
    summarize_evaluation,
    temporal_pool,
    fast_dtw_align,
)

LOGGER = logging.getLogger(__name__)


def evaluate_pair(
    pair: SWDPair,
    *,
    dataset: SWDDataset | None = None,
    methods: list[str] | None = None,
    encoder: Any | None = None,
    sr: int = 22050,
    chroma_hop: int = 440,
    deep_hop: int = 220,
    pool_size: int = 2,
    device: str | None = None,
    mrmsdtw_memory_limit_mb: int = 500,
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
    matchmaker_cache: dict[tuple[str, str, str], tuple[pd.DataFrame, Any]] | None = None,
    matchmaker_method: str = "arzt",
    matchmaker_feature_type: str = "chroma",
    matchmaker_frame_rate: int = 100,
) -> list[dict[str, Any]]:
    """Evaluate one SWD pair across the requested methods."""
    resolved_methods = methods or ["chroma_dtw"]
    ground_truth = compute_ground_truth_measure_alignment(pair)
    if ground_truth is None:
        return []

    annotations_a, annotations_b = ground_truth
    gt_a = annotations_a["time_s"].to_numpy(dtype=float)
    gt_b = annotations_b["time_s"].to_numpy(dtype=float)
    audio_a, _ = load_swd_audio(pair.piece_a, sr=sr)
    audio_b, _ = load_swd_audio(pair.piece_b, sr=sr)
    score_path = dataset.get_score_path(pair.lied_id) if dataset is not None else None

    results = evaluate_pairwise_methods(
        methods=[method for method in resolved_methods if method != "matchmaker"],
        pair_id=pair.pair_id,
        group_id=pair.lied_id,
        dataset_name="swd",
        piece_a_id=pair.piece_a.piece_id,
        piece_b_id=pair.piece_b.piece_id,
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
        deep_distance=deep_distance,
        band_radius_frames=band_radius_frames,
        transcription_cache_root=transcription_cache_root,
        fusion_deep_weight=fusion_deep_weight,
        fusion_onset_weight=fusion_onset_weight,
        fusion_note_weight=fusion_note_weight,
        fusion_dlnco_weight=fusion_dlnco_weight,
        fusion_chroma_weight=fusion_chroma_weight,
        refine_window_sec=refine_window_sec,
        score_refine_radius_sec=score_refine_radius_sec,
        score_path=score_path,
        event_ids=annotations_a["event_id"].astype(str).tolist(),
    )

    if "matchmaker" not in resolved_methods:
        return results

    if dataset is None:
        raise ValueError("SWD Matchmaker evaluation requires the owning SWDDataset instance.")

    if matchmaker_cache is None:
        matchmaker_cache = {}

    try:
        predicted_a, result_a = _get_matchmaker_predictions(
            dataset=dataset,
            piece=pair.piece_a,
            lied_id=pair.lied_id,
            method=matchmaker_method,
            feature_type=matchmaker_feature_type,
            frame_rate=matchmaker_frame_rate,
            cache=matchmaker_cache,
        )
        predicted_b, result_b = _get_matchmaker_predictions(
            dataset=dataset,
            piece=pair.piece_b,
            lied_id=pair.lied_id,
            method=matchmaker_method,
            feature_type=matchmaker_feature_type,
            frame_rate=matchmaker_frame_rate,
            cache=matchmaker_cache,
        )
    except Exception as exc:
        LOGGER.warning("Skipping Matchmaker for SWD pair %s: %s", pair.pair_id, exc)
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
            dataset_name="swd",
            pair_id=pair.pair_id,
            group_id=pair.lied_id,
            piece_a_id=pair.piece_a.piece_id,
            piece_b_id=pair.piece_b.piece_id,
            gt_a=gt_a_common,
            gt_b=gt_b_common,
            pred_a_anchor=pred_anchor_a,
            pred_b_anchor=pred_anchor_b,
            result_a=result_a,
            result_b=result_b,
        )
    )
    return results


def evaluate_swd_dataset(
    dataset: SWDDataset,
    *,
    checkpoint_path: str | Path | None = None,
    device: str | None = None,
    methods: str | list[str] | tuple[str, ...] | None = None,
    performances: list[str] | None = None,
    lieder: list[str] | None = None,
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
) -> pd.DataFrame:
    """Evaluate SWD pairs with the requested methods."""
    resolved_methods = parse_methods(methods, checkpoint_path=checkpoint_path)
    encoder = None
    if "deepalign" in resolved_methods:
        from dis_alignment.model.inference import load_trained_encoder

        encoder, _ = load_trained_encoder(checkpoint_path, device=device)

    pair_iter = list(dataset.iter_pairs(performances=performances, lieder=lieder))
    iterator = tqdm(pair_iter, desc="Evaluating SWD", disable=not show_progress)
    rows: list[dict[str, Any]] = []
    matchmaker_cache: dict[tuple[str, str, str], tuple[pd.DataFrame, Any]] = {}

    for pair in iterator:
        try:
            rows.extend(
                evaluate_pair(
                    pair,
                    dataset=dataset,
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
                    deep_distance=deep_distance,
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
            LOGGER.warning("Skipping SWD pair %s due to evaluation error: %s", pair.pair_id, exc)

    return pd.DataFrame(rows)


def _get_matchmaker_predictions(
    *,
    dataset: SWDDataset,
    piece,
    lied_id: str,
    method: str,
    feature_type: str,
    frame_rate: int,
    cache: dict[tuple[str, str, str], tuple[pd.DataFrame, Any]],
) -> tuple[pd.DataFrame, Any]:
    cache_key = (piece.piece_id, method, feature_type)
    if cache_key in cache:
        return cache[cache_key]

    score_path = dataset.get_score_path(lied_id)
    if score_path is None:
        raise FileNotFoundError(f"Could not resolve a score for SWD lied {lied_id}.")

    result = run_matchmaker_for_piece(
        score_path,
        piece.audio_path,
        method=method,
        feature_type=feature_type,
        frame_rate=frame_rate,
    )
    measure_positions = extract_musicxml_measure_positions(score_path)
    predicted = map_score_positions_to_events(result, measure_positions)
    cache[cache_key] = (predicted, result)
    return predicted, result
