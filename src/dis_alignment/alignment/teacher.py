"""Audio-only teacher path generation for DeepAlign SOTA training.

The teacher is deliberately audio-only: it uses hand-crafted chroma/onset
features and DTW to create dense frame correspondences that can supervise the
DeepAlign encoder. The generated paths are training artifacts only; final
DeepAlign evaluation still runs from learned audio features.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm import tqdm

from dis_alignment.data.swd import (
    SWDDataset,
    SWDPair,
    compute_ground_truth_measure_alignment,
    load_swd_audio,
)
from dis_alignment.evaluation.common import fast_dtw_align
from dis_alignment.evaluation.metrics import (
    alignment_rate,
    mean_absolute_error,
    median_absolute_error,
)
from dis_alignment.features.chroma import extract_chroma_cqt
from dis_alignment.features.dlnco import extract_dlnco

_SAFE_PATH_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class TeacherPath:
    """One dense teacher alignment path in absolute song time."""

    pair_id: str
    lied_id: str
    piece_a_id: str
    piece_b_id: str
    frame_hop: int
    sr: int
    path: NDArray[np.intp]
    time_a_s: NDArray[np.float64]
    time_b_s: NDArray[np.float64]
    chroma_shift: int = 0
    anchor_calibrated: bool = False
    confidence: NDArray[np.float32] | None = None


def teacher_path_filename(pair_id: str) -> str:
    """Return a filesystem-safe teacher path filename for one pair id."""
    safe_pair_id = _SAFE_PATH_RE.sub("_", pair_id).strip("_")
    return f"{safe_pair_id}.npz"


def save_teacher_path(path: TeacherPath, output_dir: str | Path) -> Path:
    """Write a teacher path as a compressed NPZ artifact."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    target = output / teacher_path_filename(path.pair_id)
    np.savez_compressed(
        target,
        pair_id=np.asarray(path.pair_id),
        lied_id=np.asarray(path.lied_id),
        piece_a_id=np.asarray(path.piece_a_id),
        piece_b_id=np.asarray(path.piece_b_id),
        frame_hop=np.asarray(path.frame_hop, dtype=np.int64),
        sr=np.asarray(path.sr, dtype=np.int64),
        path=path.path.astype(np.int64, copy=False),
        time_a_s=path.time_a_s.astype(np.float64, copy=False),
        time_b_s=path.time_b_s.astype(np.float64, copy=False),
        chroma_shift=np.asarray(path.chroma_shift, dtype=np.int64),
        anchor_calibrated=np.asarray(path.anchor_calibrated, dtype=np.bool_),
        confidence=(
            path.confidence.astype(np.float32, copy=False)
            if path.confidence is not None
            else np.ones(path.path.shape[1], dtype=np.float32)
        ),
    )
    return target


def load_teacher_path(path_or_root: str | Path, pair_id: str | None = None) -> TeacherPath:
    """Load a teacher path from an NPZ file or a root directory plus pair id."""
    path = Path(path_or_root)
    if path.is_dir():
        if pair_id is None:
            raise ValueError("pair_id is required when loading from a teacher root directory.")
        path = path / teacher_path_filename(pair_id)
    if not path.exists():
        raise FileNotFoundError(f"Teacher path not found: {path}")

    with np.load(path, allow_pickle=False) as payload:
        return TeacherPath(
            pair_id=str(payload["pair_id"].item()),
            lied_id=str(payload["lied_id"].item()),
            piece_a_id=str(payload["piece_a_id"].item()),
            piece_b_id=str(payload["piece_b_id"].item()),
            frame_hop=int(payload["frame_hop"].item()),
            sr=int(payload["sr"].item()),
            path=payload["path"].astype(np.intp, copy=False),
            time_a_s=payload["time_a_s"].astype(np.float64, copy=False),
            time_b_s=payload["time_b_s"].astype(np.float64, copy=False),
            chroma_shift=(
                int(payload["chroma_shift"].item())
                if "chroma_shift" in payload.files
                else 0
            ),
            anchor_calibrated=(
                bool(payload["anchor_calibrated"].item())
                if "anchor_calibrated" in payload.files
                else False
            ),
            confidence=(
                payload["confidence"].astype(np.float32, copy=False)
                if "confidence" in payload.files
                else None
            ),
        )


def generate_swd_teacher_paths(
    dataset: SWDDataset,
    *,
    output_dir: str | Path,
    sr: int = 22050,
    hop_length: int = 110,
    performances: Iterable[str] | None = None,
    lieder: Iterable[str] | None = None,
    backend: str = "mrmsdtw",
    distance: str = "cosine",
    memory_limit_mb: int = 500,
    chroma_weight: float = 1.0,
    dlnco_weight: float = 1.0,
    spectral_flux_weight: float = 0.5,
    trim_top_db: float | None = 40.0,
    estimate_chroma_shift: bool = True,
    chroma_shift_max_frames: int = 1500,
    confidence_local_radius_frames: int = 24,
    confidence_exclusion_radius_frames: int = 2,
    anchor_calibrate: bool = False,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate and validate teacher paths for SWD pairs."""
    output = Path(output_dir)
    paths_dir = output / "paths"
    paths_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    oracle_rows: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []
    pair_iter = list(dataset.iter_pairs(performances=performances, lieder=lieder))
    iterator = tqdm(pair_iter, desc="Generating teacher paths", disable=not show_progress)
    for pair in iterator:
        teacher = generate_teacher_path_for_pair(
            pair,
            sr=sr,
            hop_length=hop_length,
            backend=backend,
            distance=distance,
            memory_limit_mb=memory_limit_mb,
            chroma_weight=chroma_weight,
            dlnco_weight=dlnco_weight,
            spectral_flux_weight=spectral_flux_weight,
            trim_top_db=trim_top_db,
            estimate_chroma_shift=estimate_chroma_shift,
            chroma_shift_max_frames=chroma_shift_max_frames,
            confidence_local_radius_frames=confidence_local_radius_frames,
            confidence_exclusion_radius_frames=confidence_exclusion_radius_frames,
            anchor_calibrate=anchor_calibrate,
        )
        path_file = save_teacher_path(teacher, paths_dir)
        method = "anchor_calibrated_audio_teacher" if anchor_calibrate else "audio_teacher"
        rows.append(evaluate_teacher_path(pair, teacher, method=method))
        oracle_rows.append(evaluate_oracle_alignment(pair))
        manifest.append(
            {
                "pair_id": pair.pair_id,
                "lied_id": pair.lied_id,
                "path": str(path_file),
                "frames_a": int(teacher.path[0].max() + 1) if teacher.path.size else 0,
                "frames_b": int(teacher.path[1].max() + 1) if teacher.path.size else 0,
                "chroma_shift": int(teacher.chroma_shift),
                "anchor_calibrated": bool(teacher.anchor_calibrated),
                "mean_confidence": (
                    float(np.mean(teacher.confidence))
                    if teacher.confidence is not None and teacher.confidence.size
                    else 1.0
                ),
                "min_confidence": (
                    float(np.min(teacher.confidence))
                    if teacher.confidence is not None and teacher.confidence.size
                    else 1.0
                ),
            }
        )

    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return pd.DataFrame(rows), pd.DataFrame(oracle_rows)


def mine_swd_pseudo_teacher_paths(
    dataset: SWDDataset,
    *,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    sr: int = 22050,
    hop_length: int = 440,
    performances: Iterable[str] | None = None,
    lieder: Iterable[str] | None = None,
    device: str | None = None,
    cache_root: str | Path | None = None,
    min_confidence: float = 0.55,
    min_gap_frames: int = 8,
    max_anchors: int = 2048,
    band_radius_frames: int | None = None,
    coarse_prior: str = "none",
    coarse_local_radius_frames: int | None = None,
    coarse_distance: str = "cosine",
    coarse_chroma_weight: float = 1.0,
    coarse_dlnco_weight: float = 1.0,
    coarse_spectral_flux_weight: float = 0.5,
    use_coarse_path_only: bool = False,
    chunk_size: int = 1024,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mine audio-only pseudo paths from reciprocal learned-feature matches."""
    if use_coarse_path_only and coarse_prior == "none":
        raise ValueError("use_coarse_path_only requires a coarse_prior.")

    encoder = None
    extract_deep_features = None
    if not use_coarse_path_only:
        from dis_alignment.model.inference import extract_deep_features, load_trained_encoder

        encoder, _ = load_trained_encoder(checkpoint_path, device=device)
    output = Path(output_dir)
    paths_dir = output / "paths"
    paths_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    oracle_rows: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []
    pair_iter = list(dataset.iter_pairs(performances=performances, lieder=lieder))
    iterator = tqdm(pair_iter, desc="Mining pseudo paths", disable=not show_progress)
    for pair in iterator:
        features_a = None
        features_b = None
        if not use_coarse_path_only:
            if encoder is None or extract_deep_features is None:
                raise RuntimeError("Encoder was not loaded for learned-feature pseudo mining.")
            audio_a, _ = load_swd_audio(pair.piece_a, sr=sr)
            audio_b, _ = load_swd_audio(pair.piece_b, sr=sr)
            features_a = extract_deep_features(
                audio_a,
                encoder,
                sr=sr,
                hop_length=hop_length,
                device=device,
                audio_path=pair.piece_a.audio_path,
                cache_root=cache_root,
            )
            features_b = extract_deep_features(
                audio_b,
                encoder,
                sr=sr,
                hop_length=hop_length,
                device=device,
                audio_path=pair.piece_b.audio_path,
                cache_root=cache_root,
            )
        guide_path = None
        if coarse_prior != "none":
            coarse_teacher = generate_teacher_path_for_pair(
                pair,
                sr=sr,
                hop_length=hop_length,
                backend=coarse_prior,
                distance=coarse_distance,
                chroma_weight=coarse_chroma_weight,
                dlnco_weight=coarse_dlnco_weight,
                spectral_flux_weight=coarse_spectral_flux_weight,
                trim_top_db=40.0,
                estimate_chroma_shift=True,
            )
            guide_path = coarse_teacher.path
        if use_coarse_path_only:
            if guide_path is None:
                raise ValueError("use_coarse_path_only requires a coarse_prior.")
            teacher = _sample_coarse_teacher_path(
                pair,
                guide_path=guide_path,
                guide_time_a_s=coarse_teacher.time_a_s,
                guide_time_b_s=coarse_teacher.time_b_s,
                guide_confidence=coarse_teacher.confidence,
                sr=sr,
                hop_length=hop_length,
                min_gap_frames=min_gap_frames,
                max_anchors=max_anchors,
            )
        else:
            teacher = mine_reciprocal_pseudo_teacher_path_for_pair(
                pair,
                features_a=features_a,
                features_b=features_b,
                sr=sr,
                hop_length=hop_length,
                min_confidence=min_confidence,
                min_gap_frames=min_gap_frames,
                max_anchors=max_anchors,
                band_radius_frames=band_radius_frames,
                guide_path=guide_path,
                guide_radius_frames=coarse_local_radius_frames,
                chunk_size=chunk_size,
            )
        path_file = save_teacher_path(teacher, paths_dir)
        rows.append(evaluate_teacher_path(pair, teacher, method="pseudo_audio_teacher"))
        oracle_rows.append(evaluate_oracle_alignment(pair))
        manifest.append(
            {
                "pair_id": pair.pair_id,
                "lied_id": pair.lied_id,
                "path": str(path_file),
                "anchors": int(teacher.path.shape[1]),
                "mean_confidence": (
                    float(np.mean(teacher.confidence))
                    if teacher.confidence is not None and teacher.confidence.size
                    else 0.0
                ),
                "min_confidence": (
                    float(np.min(teacher.confidence))
                    if teacher.confidence is not None and teacher.confidence.size
                    else 0.0
                ),
                "anchor_calibrated": False,
                "coarse_prior": coarse_prior,
                "use_coarse_path_only": bool(use_coarse_path_only),
                "coarse_local_radius_frames": (
                    int(coarse_local_radius_frames)
                    if coarse_local_radius_frames is not None
                    else None
                ),
            }
        )

    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return pd.DataFrame(rows), pd.DataFrame(oracle_rows)


def _sample_coarse_teacher_path(
    pair: SWDPair,
    *,
    guide_path: NDArray[np.integer],
    guide_time_a_s: NDArray[np.floating] | None = None,
    guide_time_b_s: NDArray[np.floating] | None = None,
    guide_confidence: NDArray[np.floating] | None = None,
    sr: int,
    hop_length: int,
    min_gap_frames: int,
    max_anchors: int,
) -> TeacherPath:
    raw_path = np.asarray(guide_path, dtype=np.intp)
    if raw_path.shape[0] != 2:
        raise ValueError("Coarse guide path must have shape (2, n_path_points).")
    if raw_path.shape[1] == 0:
        raise ValueError(f"Coarse path is empty for {pair.pair_id}.")
    frame_duration = hop_length / sr
    raw_time_a = (
        np.asarray(guide_time_a_s, dtype=np.float64)
        if guide_time_a_s is not None
        else raw_path[0].astype(np.float64) * frame_duration
    )
    raw_time_b = (
        np.asarray(guide_time_b_s, dtype=np.float64)
        if guide_time_b_s is not None
        else raw_path[1].astype(np.float64) * frame_duration
    )
    if raw_time_a.shape[0] != raw_path.shape[1] or raw_time_b.shape[0] != raw_path.shape[1]:
        raise ValueError("Coarse guide times must have one value per path point.")
    raw_confidence = (
        np.asarray(guide_confidence, dtype=np.float32)
        if guide_confidence is not None
        else np.ones(raw_path.shape[1], dtype=np.float32)
    )
    if raw_confidence.shape[0] != raw_path.shape[1]:
        raise ValueError("Coarse guide confidence must have one value per path point.")

    order = np.lexsort((raw_path[1], raw_path[0]))
    ordered_path = raw_path[:, order].astype(np.intp, copy=False)
    ordered_time_a = raw_time_a[order]
    ordered_time_b = raw_time_b[order]
    ordered_confidence = raw_confidence[order]
    keep_monotonic: list[int] = []
    last_a = -1
    last_b = -1
    for idx, (frame_a, frame_b) in enumerate(ordered_path.T):
        if frame_a >= last_a and frame_b >= last_b:
            keep_monotonic.append(idx)
            last_a = int(frame_a)
            last_b = int(frame_b)
    if len(keep_monotonic) < 2:
        raise ValueError(f"Coarse path sampling found fewer than two anchors for {pair.pair_id}.")

    path = ordered_path[:, keep_monotonic]
    time_a_s = ordered_time_a[keep_monotonic]
    time_b_s = ordered_time_b[keep_monotonic]
    confidence = ordered_confidence[keep_monotonic]
    keep = _min_gap_mask(path[0], path[1], min_gap_frames=max(1, int(min_gap_frames)))
    path = path[:, keep]
    time_a_s = time_a_s[keep]
    time_b_s = time_b_s[keep]
    confidence = confidence[keep]
    if max_anchors > 0 and path.shape[1] > max_anchors:
        selected = np.linspace(0, path.shape[1] - 1, int(max_anchors), dtype=int)
        path = path[:, selected]
        time_a_s = time_a_s[selected]
        time_b_s = time_b_s[selected]
        confidence = confidence[selected]
    if path.shape[1] < 2:
        raise ValueError(f"Coarse path sampling found fewer than two anchors for {pair.pair_id}.")
    return TeacherPath(
        pair_id=pair.pair_id,
        lied_id=pair.lied_id,
        piece_a_id=pair.piece_a.piece_id,
        piece_b_id=pair.piece_b.piece_id,
        frame_hop=hop_length,
        sr=sr,
        path=path,
        time_a_s=time_a_s.astype(np.float64, copy=False),
        time_b_s=time_b_s.astype(np.float64, copy=False),
        anchor_calibrated=False,
        confidence=confidence.astype(np.float32, copy=False),
    )


def mine_reciprocal_pseudo_teacher_path_for_pair(
    pair: SWDPair,
    *,
    features_a: NDArray[np.floating],
    features_b: NDArray[np.floating],
    sr: int,
    hop_length: int,
    min_confidence: float = 0.55,
    min_gap_frames: int = 8,
    max_anchors: int = 2048,
    band_radius_frames: int | None = None,
    guide_path: NDArray[np.integer] | None = None,
    guide_radius_frames: int | None = None,
    chunk_size: int = 1024,
) -> TeacherPath:
    """Build a sparse pseudo path from reciprocal audio-feature neighbors."""
    path, confidence = mine_reciprocal_pseudo_path(
        features_a,
        features_b,
        min_confidence=min_confidence,
        min_gap_frames=min_gap_frames,
        max_anchors=max_anchors,
        band_radius_frames=band_radius_frames,
        guide_path=guide_path,
        guide_radius_frames=guide_radius_frames,
        chunk_size=chunk_size,
    )
    if path.shape[1] < 2:
        raise ValueError(f"Pseudo-anchor mining found fewer than two anchors for {pair.pair_id}.")
    frame_duration = hop_length / sr
    return TeacherPath(
        pair_id=pair.pair_id,
        lied_id=pair.lied_id,
        piece_a_id=pair.piece_a.piece_id,
        piece_b_id=pair.piece_b.piece_id,
        frame_hop=hop_length,
        sr=sr,
        path=path,
        time_a_s=path[0].astype(np.float64) * frame_duration,
        time_b_s=path[1].astype(np.float64) * frame_duration,
        anchor_calibrated=False,
        confidence=confidence,
    )


def mine_reciprocal_pseudo_path(
    features_a: NDArray[np.floating],
    features_b: NDArray[np.floating],
    *,
    min_confidence: float = 0.55,
    min_gap_frames: int = 8,
    max_anchors: int = 2048,
    band_radius_frames: int | None = None,
    guide_path: NDArray[np.integer] | None = None,
    guide_radius_frames: int | None = None,
    chunk_size: int = 1024,
) -> tuple[NDArray[np.intp], NDArray[np.float32]]:
    """Return monotonic reciprocal nearest-neighbor pseudo anchors."""
    norm_a = _normalize_columns(features_a).astype(np.float32, copy=False)
    norm_b = _normalize_columns(features_b).astype(np.float32, copy=False)
    if norm_a.shape[1] < 2 or norm_b.shape[1] < 2:
        return np.empty((2, 0), dtype=np.intp), np.empty(0, dtype=np.float32)
    guide_a_to_b = _guide_reference_indices(
        n_query=norm_a.shape[1],
        n_reference=norm_b.shape[1],
        guide_path=guide_path,
        direction="a_to_b",
    )
    guide_b_to_a = _guide_reference_indices(
        n_query=norm_b.shape[1],
        n_reference=norm_a.shape[1],
        guide_path=guide_path,
        direction="b_to_a",
    )
    resolved_radius = guide_radius_frames if guide_path is not None else band_radius_frames

    a_to_b, a_sim, a_margin = _nearest_neighbor_with_margin(
        norm_a,
        norm_b,
        band_radius_frames=resolved_radius,
        guide_reference_indices=guide_a_to_b,
        chunk_size=chunk_size,
    )
    b_to_a, b_sim, b_margin = _nearest_neighbor_with_margin(
        norm_b,
        norm_a,
        band_radius_frames=resolved_radius,
        guide_reference_indices=guide_b_to_a,
        chunk_size=chunk_size,
    )

    idx_a = np.arange(a_to_b.size, dtype=np.int64)
    idx_b = a_to_b.astype(np.int64, copy=False)
    mutual = b_to_a[idx_b] == idx_a
    if not np.any(mutual):
        return np.empty((2, 0), dtype=np.intp), np.empty(0, dtype=np.float32)

    idx_a = idx_a[mutual]
    idx_b = idx_b[mutual]
    similarity = np.minimum(a_sim[mutual], b_sim[idx_b])
    margin = np.minimum(a_margin[mutual], b_margin[idx_b])
    confidence = _pseudo_anchor_confidence(similarity, margin)
    keep = confidence >= float(min_confidence)
    idx_a = idx_a[keep]
    idx_b = idx_b[keep]
    confidence = confidence[keep]
    if idx_a.size == 0:
        return np.empty((2, 0), dtype=np.intp), np.empty(0, dtype=np.float32)

    order = np.argsort(idx_a, kind="stable")
    idx_a = idx_a[order]
    idx_b = idx_b[order]
    confidence = confidence[order]
    lis_indices = _longest_nondecreasing_subsequence_indices(idx_b)
    idx_a = idx_a[lis_indices]
    idx_b = idx_b[lis_indices]
    confidence = confidence[lis_indices]

    gap_keep = _min_gap_mask(idx_a, idx_b, min_gap_frames=max(1, int(min_gap_frames)))
    idx_a = idx_a[gap_keep]
    idx_b = idx_b[gap_keep]
    confidence = confidence[gap_keep]
    if max_anchors > 0 and idx_a.size > max_anchors:
        selected = np.linspace(0, idx_a.size - 1, int(max_anchors), dtype=int)
        idx_a = idx_a[selected]
        idx_b = idx_b[selected]
        confidence = confidence[selected]

    path = np.vstack([idx_a, idx_b]).astype(np.intp, copy=False)
    return path, confidence.astype(np.float32, copy=False)


def generate_teacher_path_for_pair(
    pair: SWDPair,
    *,
    sr: int = 22050,
    hop_length: int = 110,
    backend: str = "mrmsdtw",
    distance: str = "cosine",
    memory_limit_mb: int = 500,
    chroma_weight: float = 1.0,
    dlnco_weight: float = 1.0,
    spectral_flux_weight: float = 0.5,
    trim_top_db: float | None = 40.0,
    estimate_chroma_shift: bool = True,
    chroma_shift_max_frames: int = 1500,
    confidence_local_radius_frames: int = 24,
    confidence_exclusion_radius_frames: int = 2,
    anchor_calibrate: bool = False,
) -> TeacherPath:
    """Generate one audio-only teacher path for an SWD pair."""
    audio_a, _ = load_swd_audio(pair.piece_a, sr=sr)
    audio_b, _ = load_swd_audio(pair.piece_b, sr=sr)
    audio_a, trim_start_a_s, _ = trim_active_audio(
        audio_a,
        sr=sr,
        trim_top_db=trim_top_db,
    )
    audio_b, trim_start_b_s, _ = trim_active_audio(
        audio_b,
        sr=sr,
        trim_top_db=trim_top_db,
    )
    chroma_shift = 0
    if backend == "mrmsdtw":
        path, chroma_shift = _sync_via_audio_onset_mrmsdtw(
            audio_a=audio_a,
            audio_b=audio_b,
            sr=sr,
            hop_length=hop_length,
            memory_limit_mb=memory_limit_mb,
            chroma_weight=chroma_weight,
            dlnco_weight=dlnco_weight,
            spectral_flux_weight=spectral_flux_weight,
            estimate_chroma_shift=estimate_chroma_shift,
            chroma_shift_max_frames=chroma_shift_max_frames,
        )
    elif backend == "full":
        if estimate_chroma_shift:
            chroma_shift = estimate_chroma_shift_to_first(
                extract_chroma_cqt(audio_a, sr=sr, hop_length=hop_length),
                extract_chroma_cqt(audio_b, sr=sr, hop_length=hop_length),
                max_frames=chroma_shift_max_frames,
            )
        features_a = extract_audio_teacher_features(
            audio_a,
            sr=sr,
            hop_length=hop_length,
            chroma_weight=chroma_weight,
            dlnco_weight=dlnco_weight,
            spectral_flux_weight=spectral_flux_weight,
        )
        features_a = apply_chroma_shift_to_feature_stack(
            features_a,
            chroma_shift,
            chroma_weight=chroma_weight,
            dlnco_weight=dlnco_weight,
        )
        features_b = extract_audio_teacher_features(
            audio_b,
            sr=sr,
            hop_length=hop_length,
            chroma_weight=chroma_weight,
            dlnco_weight=dlnco_weight,
            spectral_flux_weight=spectral_flux_weight,
        )
        path, _, _ = fast_dtw_align(features_a, features_b, distance=distance)
    else:
        raise ValueError("backend must be one of {'mrmsdtw', 'full'}")
    path = _filter_monotonic_path(path)
    frame_duration = hop_length / sr
    teacher = TeacherPath(
        pair_id=pair.pair_id,
        lied_id=pair.lied_id,
        piece_a_id=pair.piece_a.piece_id,
        piece_b_id=pair.piece_b.piece_id,
        frame_hop=hop_length,
        sr=sr,
        path=path,
        time_a_s=(path[0].astype(np.float64) * frame_duration) + trim_start_a_s,
        time_b_s=(path[1].astype(np.float64) * frame_duration) + trim_start_b_s,
        chroma_shift=chroma_shift,
    )
    confidence = compute_teacher_path_confidence_for_audio(
        audio_a=audio_a,
        audio_b=audio_b,
        path=path,
        sr=sr,
        hop_length=hop_length,
        chroma_weight=chroma_weight,
        dlnco_weight=dlnco_weight,
        spectral_flux_weight=spectral_flux_weight,
        chroma_shift=chroma_shift,
        local_radius_frames=confidence_local_radius_frames,
        exclusion_radius_frames=confidence_exclusion_radius_frames,
    )
    teacher = replace(teacher, confidence=confidence)
    if anchor_calibrate:
        teacher = calibrate_teacher_path_to_measure_anchors(pair, teacher)
    return teacher


def calibrate_teacher_path_to_measure_anchors(pair: SWDPair, teacher: TeacherPath) -> TeacherPath:
    """
    Correct teacher path timing with SWD measure anchors for supervised training.

    This is intentionally not an audio-only teacher gate: it uses measure
    annotations to turn a plausible audio path into dense local positives for
    model training when the fully audio-only teacher is not reliable enough.
    """
    alignment = compute_ground_truth_measure_alignment(pair)
    if alignment is None:
        raise ValueError(f"No ground-truth measure alignment for {pair.pair_id}")

    ann_a, ann_b = alignment
    gt_a = ann_a["time_s"].to_numpy(dtype=np.float64)
    gt_b = ann_b["time_s"].to_numpy(dtype=np.float64)
    if len(gt_a) < 2:
        raise ValueError(f"Need at least two measure anchors for {pair.pair_id}")

    pred_b = interp_monotonic(gt_a, teacher.time_a_s, teacher.time_b_s)
    residual = gt_b - pred_b
    correction = np.interp(
        teacher.time_a_s,
        gt_a,
        residual,
        left=float(residual[0]),
        right=float(residual[-1]),
    )
    corrected_b = np.maximum.accumulate(teacher.time_b_s + correction)
    return replace(
        teacher,
        time_b_s=corrected_b.astype(np.float64, copy=False),
        anchor_calibrated=True,
    )


def trim_active_audio(
    audio: NDArray[np.floating],
    *,
    sr: int,
    trim_top_db: float | None,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> tuple[NDArray[np.float32], float, float]:
    """Trim leading/trailing low-energy material using only audio energy."""
    values = np.asarray(audio, dtype=np.float32)
    if trim_top_db is None:
        return values, 0.0, len(values) / sr
    trimmed, indices = librosa.effects.trim(
        values,
        top_db=float(trim_top_db),
        frame_length=frame_length,
        hop_length=hop_length,
    )
    if trimmed.size == 0:
        return values, 0.0, len(values) / sr
    start_s = float(indices[0]) / sr
    end_s = float(indices[1]) / sr
    return trimmed.astype(np.float32, copy=False), start_s, end_s


def estimate_chroma_shift_to_first(
    chroma_a: NDArray[np.floating],
    chroma_b: NDArray[np.floating],
    *,
    max_frames: int = 1500,
) -> int:
    """Estimate the chroma roll to apply to the first sequence."""
    pooled_a = _pool_feature_frames_for_shift(chroma_a, max_frames=max_frames)
    pooled_b = _pool_feature_frames_for_shift(chroma_b, max_frames=max_frames)

    try:
        from synctoolbox.dtw.utils import compute_optimal_chroma_shift
    except ImportError:
        # SyncToolbox is an optional dependency. Fall back to a pure-numpy
        # transposition estimate so chroma-shift calibration still works in
        # environments where synctoolbox is unavailable.
        return _estimate_chroma_shift_to_first_numpy(pooled_a, pooled_b)

    # SyncToolbox estimates the roll for its second argument. Swap arguments so
    # the returned value is the shift to apply to chroma_a before aligning to b.
    return int(compute_optimal_chroma_shift(pooled_b, pooled_a)) % 12


def _estimate_chroma_shift_to_first_numpy(
    pooled_a: NDArray[np.floating],
    pooled_b: NDArray[np.floating],
) -> int:
    """Estimate the chroma roll for the first sequence without synctoolbox.

    Aggregates each sequence into a time-averaged 12-bin chroma energy profile
    and picks the cyclic semitone shift (0-11) of the first profile that best
    correlates with the second. This matches the synctoolbox convention of
    returning the roll to apply to the first sequence before aligning to the
    second, and is order-invariant so the two sequences may differ in length.
    """
    profile_a = np.asarray(pooled_a, dtype=np.float64).mean(axis=1)
    profile_b = np.asarray(pooled_b, dtype=np.float64).mean(axis=1)
    scores = [float(np.dot(profile_b, np.roll(profile_a, shift))) for shift in range(12)]
    return int(np.argmax(scores)) % 12


def apply_chroma_shift_to_blocks(
    chroma: NDArray[np.floating],
    onset: NDArray[np.floating],
    shift: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Roll chroma-sensitive teacher blocks while leaving scalar onset cues alone."""
    shift = int(shift) % 12
    if shift == 0:
        return chroma.astype(np.float64, copy=False), onset.astype(np.float64, copy=False)

    shifted_chroma = np.roll(chroma, shift, axis=0).astype(np.float64, copy=False)
    shifted_onset = onset.astype(np.float64, copy=True)
    if shifted_onset.shape[0] >= 12:
        shifted_onset[:12] = np.roll(shifted_onset[:12], shift, axis=0)
    return shifted_chroma, shifted_onset


def apply_chroma_shift_to_feature_stack(
    features: NDArray[np.floating],
    shift: int,
    *,
    chroma_weight: float,
    dlnco_weight: float,
) -> NDArray[np.float64]:
    """Roll chroma and DLNCO blocks in a stacked full-DTW feature matrix."""
    shift = int(shift) % 12
    values = np.asarray(features, dtype=np.float64)
    if shift == 0:
        return values

    shifted = values.copy()
    offset = 0
    if chroma_weight > 0 and offset + 12 <= shifted.shape[0]:
        shifted[offset : offset + 12] = np.roll(shifted[offset : offset + 12], shift, axis=0)
        offset += 12
    if dlnco_weight > 0 and offset + 12 <= shifted.shape[0]:
        shifted[offset : offset + 12] = np.roll(shifted[offset : offset + 12], shift, axis=0)
    return shifted


def _sync_via_audio_onset_mrmsdtw(
    *,
    audio_a: NDArray[np.floating],
    audio_b: NDArray[np.floating],
    sr: int,
    hop_length: int,
    memory_limit_mb: int,
    chroma_weight: float,
    dlnco_weight: float,
    spectral_flux_weight: float,
    estimate_chroma_shift: bool,
    chroma_shift_max_frames: int,
) -> tuple[NDArray[np.intp], int]:
    """Call SyncToolbox with separate chroma and onset features."""
    from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw

    chroma_a, onset_a = extract_audio_teacher_feature_blocks(
        audio_a,
        sr=sr,
        hop_length=hop_length,
        chroma_weight=chroma_weight,
        dlnco_weight=dlnco_weight,
        spectral_flux_weight=spectral_flux_weight,
    )
    chroma_b, onset_b = extract_audio_teacher_feature_blocks(
        audio_b,
        sr=sr,
        hop_length=hop_length,
        chroma_weight=chroma_weight,
        dlnco_weight=dlnco_weight,
        spectral_flux_weight=spectral_flux_weight,
    )
    chroma_shift = 0
    if estimate_chroma_shift:
        chroma_shift = estimate_chroma_shift_to_first(
            chroma_a,
            chroma_b,
            max_frames=chroma_shift_max_frames,
        )
        chroma_a, onset_a = apply_chroma_shift_to_blocks(chroma_a, onset_a, chroma_shift)
    threshold_rec = max(10_000, int(memory_limit_mb * 1024 * 1024 // 8))
    path = sync_via_mrmsdtw(
        f_chroma1=chroma_a,
        f_chroma2=chroma_b,
        f_onset1=onset_a,
        f_onset2=onset_b,
        input_feature_rate=int(round(sr / hop_length)),
        threshold_rec=threshold_rec,
        verbose=False,
    )
    return path.astype(np.intp, copy=False), chroma_shift


def extract_audio_teacher_feature_blocks(
    audio: NDArray[np.floating],
    *,
    sr: int,
    hop_length: int,
    chroma_weight: float = 1.0,
    dlnco_weight: float = 1.0,
    spectral_flux_weight: float = 0.5,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return separate chroma and onset blocks for SyncToolbox MrMsDTW."""
    chroma = extract_chroma_cqt(audio, sr=sr, hop_length=hop_length)
    target_frames = chroma.shape[1]
    chroma = _normalize_columns(chroma) * float(chroma_weight)
    dlnco = _resize_feature_frames(
        extract_dlnco(audio, sr=sr, hop_length=hop_length),
        target_frames=target_frames,
    )
    spectral_flux = _resize_feature_frames(
        extract_spectral_flux(audio, sr=sr, hop_length=hop_length),
        target_frames=target_frames,
    )
    if dlnco_weight <= 0 and spectral_flux_weight <= 0:
        onset = np.zeros((1, target_frames), dtype=np.float64)
    else:
        onset = _weighted_stack(
            (dlnco, dlnco_weight),
            (spectral_flux, spectral_flux_weight),
            target_frames=target_frames,
        )
    return chroma.astype(np.float64, copy=False), onset.astype(np.float64, copy=False)


def extract_audio_teacher_features(
    audio: NDArray[np.floating],
    *,
    sr: int,
    hop_length: int,
    chroma_weight: float = 1.0,
    dlnco_weight: float = 1.0,
    spectral_flux_weight: float = 0.5,
) -> NDArray[np.float64]:
    """Build high-resolution audio-only teacher features."""
    chroma = extract_chroma_cqt(audio, sr=sr, hop_length=hop_length)
    target_frames = chroma.shape[1]
    dlnco = _resize_feature_frames(
        extract_dlnco(audio, sr=sr, hop_length=hop_length),
        target_frames=target_frames,
    )
    spectral_flux = _resize_feature_frames(
        extract_spectral_flux(audio, sr=sr, hop_length=hop_length),
        target_frames=target_frames,
    )
    return _weighted_stack(
        (chroma, chroma_weight),
        (dlnco, dlnco_weight),
        (spectral_flux, spectral_flux_weight),
        target_frames=target_frames,
    )


def compute_teacher_path_confidence_for_audio(
    *,
    audio_a: NDArray[np.floating],
    audio_b: NDArray[np.floating],
    path: NDArray[np.integer],
    sr: int,
    hop_length: int,
    chroma_weight: float = 1.0,
    dlnco_weight: float = 1.0,
    spectral_flux_weight: float = 0.5,
    chroma_shift: int = 0,
    local_radius_frames: int = 24,
    exclusion_radius_frames: int = 2,
) -> NDArray[np.float32]:
    """Score path points using only audio features and local contrast."""
    features_a = extract_audio_teacher_features(
        audio_a,
        sr=sr,
        hop_length=hop_length,
        chroma_weight=chroma_weight,
        dlnco_weight=dlnco_weight,
        spectral_flux_weight=spectral_flux_weight,
    )
    features_a = apply_chroma_shift_to_feature_stack(
        features_a,
        chroma_shift,
        chroma_weight=chroma_weight,
        dlnco_weight=dlnco_weight,
    )
    features_b = extract_audio_teacher_features(
        audio_b,
        sr=sr,
        hop_length=hop_length,
        chroma_weight=chroma_weight,
        dlnco_weight=dlnco_weight,
        spectral_flux_weight=spectral_flux_weight,
    )
    return compute_teacher_path_confidence(
        features_a,
        features_b,
        path,
        local_radius_frames=local_radius_frames,
        exclusion_radius_frames=exclusion_radius_frames,
    )


def compute_teacher_path_confidence(
    features_a: NDArray[np.floating],
    features_b: NDArray[np.floating],
    path: NDArray[np.integer],
    *,
    local_radius_frames: int = 24,
    exclusion_radius_frames: int = 2,
) -> NDArray[np.float32]:
    """
    Return one confidence value per teacher path point.

    Confidence is high when the teacher's positive pair is more similar than
    nearby alternative reference frames. This keeps distillation audio-only
    while letting training ignore ambiguous path regions.
    """
    path = np.asarray(path)
    if path.size == 0:
        return np.empty(0, dtype=np.float32)
    if path.shape[0] != 2:
        raise ValueError("Teacher path must have shape (2, n_points).")

    n_points = path.shape[1]
    if features_a.shape[1] == 0 or features_b.shape[1] == 0:
        return np.zeros(n_points, dtype=np.float32)

    fa = _normalize_columns(np.asarray(features_a, dtype=np.float64))
    fb = _normalize_columns(np.asarray(features_b, dtype=np.float64))
    idx_a = np.clip(path[0].astype(np.int64, copy=False), 0, fa.shape[1] - 1)
    idx_b = np.clip(path[1].astype(np.int64, copy=False), 0, fb.shape[1] - 1)
    positive_similarity = np.sum(fa[:, idx_a] * fb[:, idx_b], axis=0)

    radius = max(1, int(local_radius_frames))
    exclusion = max(0, int(exclusion_radius_frames))
    offsets = np.asarray(
        [offset for offset in range(-radius, radius + 1) if abs(offset) > exclusion],
        dtype=np.int64,
    )
    if offsets.size == 0:
        return np.ones(n_points, dtype=np.float32)

    candidate_b = idx_b[:, None] + offsets[None, :]
    valid = (candidate_b >= 0) & (candidate_b < fb.shape[1])
    candidate_b = np.clip(candidate_b, 0, fb.shape[1] - 1)
    candidate_features = fb[:, candidate_b]
    local_similarity = np.einsum("dn,dnk->nk", fa[:, idx_a], candidate_features)
    local_similarity = np.where(valid, local_similarity, -np.inf)
    best_negative = np.max(local_similarity, axis=1)
    best_negative = np.where(np.isfinite(best_negative), best_negative, -1.0)
    margin = positive_similarity - best_negative
    confidence = 1.0 / (1.0 + np.exp(-8.0 * margin))
    return np.clip(confidence, 0.0, 1.0).astype(np.float32, copy=False)


def _nearest_neighbor_with_margin(
    query_features: NDArray[np.floating],
    reference_features: NDArray[np.floating],
    *,
    band_radius_frames: int | None,
    guide_reference_indices: NDArray[np.integer] | None = None,
    chunk_size: int,
) -> tuple[NDArray[np.int64], NDArray[np.float32], NDArray[np.float32]]:
    """Find cosine nearest neighbors and top-1/top-2 margins in chunks."""
    query = np.asarray(query_features, dtype=np.float32)
    reference = np.asarray(reference_features, dtype=np.float32)
    n_query = query.shape[1]
    n_reference = reference.shape[1]
    best_index = np.zeros(n_query, dtype=np.int64)
    best_score = np.full(n_query, -np.inf, dtype=np.float32)
    second_score = np.full(n_query, -np.inf, dtype=np.float32)
    chunk = max(1, int(chunk_size))
    reference_t = reference.T
    band_radius = None if band_radius_frames is None else max(0, int(band_radius_frames))
    if guide_reference_indices is not None:
        expected_reference = np.asarray(guide_reference_indices, dtype=np.int64)
        if expected_reference.shape[0] != n_query:
            raise ValueError("guide_reference_indices must have one value per query frame.")
        expected_reference = np.clip(expected_reference, 0, n_reference - 1)
    elif n_query <= 1:
        expected_reference = np.zeros(n_query, dtype=np.int64)
    else:
        expected_reference = np.rint(
            np.arange(n_query, dtype=np.float64) * (n_reference - 1) / (n_query - 1)
        ).astype(np.int64)
    for start in range(0, n_query, chunk):
        stop = min(start + chunk, n_query)
        scores = query[:, start:stop].T @ reference_t.T
        if band_radius is not None:
            reference_indices = np.arange(n_reference, dtype=np.int64)
            centers = expected_reference[start:stop]
            valid = np.abs(reference_indices[None, :] - centers[:, None]) <= band_radius
            scores = np.where(valid, scores, -np.inf)
        if n_reference == 1:
            local_best_index = np.zeros(stop - start, dtype=np.int64)
            local_best_score = scores[:, 0]
            local_second_score = np.full(stop - start, -1.0, dtype=np.float32)
        else:
            top2 = np.argpartition(scores, kth=-2, axis=1)[:, -2:]
            top2_scores = np.take_along_axis(scores, top2, axis=1)
            order = np.argsort(top2_scores, axis=1)
            local_best_index = top2[np.arange(top2.shape[0]), order[:, -1]]
            local_second_index = top2[np.arange(top2.shape[0]), order[:, -2]]
            local_best_score = scores[np.arange(scores.shape[0]), local_best_index]
            local_second_score = scores[np.arange(scores.shape[0]), local_second_index]
        best_index[start:stop] = local_best_index
        best_score[start:stop] = local_best_score.astype(np.float32, copy=False)
        second_score[start:stop] = local_second_score.astype(np.float32, copy=False)
    margin = best_score - second_score
    return best_index, best_score, margin.astype(np.float32, copy=False)


def _guide_reference_indices(
    *,
    n_query: int,
    n_reference: int,
    guide_path: NDArray[np.integer] | None,
    direction: str,
) -> NDArray[np.int64] | None:
    if guide_path is None:
        return None
    path = np.asarray(guide_path, dtype=np.float64)
    if path.size == 0:
        return None
    if path.shape[0] != 2:
        raise ValueError("guide_path must have shape (2, n_points).")
    if direction == "a_to_b":
        query_points = path[0]
        reference_points = path[1]
    elif direction == "b_to_a":
        query_points = path[1]
        reference_points = path[0]
    else:
        raise ValueError("direction must be one of {'a_to_b', 'b_to_a'}.")
    query = np.arange(n_query, dtype=np.float64)
    reference = interp_monotonic(query, query_points, reference_points)
    return np.clip(np.rint(reference), 0, n_reference - 1).astype(np.int64)


def _pseudo_anchor_confidence(
    similarity: NDArray[np.floating],
    margin: NDArray[np.floating],
) -> NDArray[np.float32]:
    sim_score = (np.asarray(similarity, dtype=np.float32) + 1.0) * 0.5
    margin_score = 1.0 / (1.0 + np.exp(-40.0 * np.asarray(margin, dtype=np.float32)))
    confidence = sim_score * margin_score
    return np.clip(confidence, 0.0, 1.0).astype(np.float32, copy=False)


def _longest_nondecreasing_subsequence_indices(values: NDArray[np.integer]) -> NDArray[np.int64]:
    """Return indices of a longest nondecreasing subsequence."""
    seq = np.asarray(values, dtype=np.int64)
    if seq.size == 0:
        return np.empty(0, dtype=np.int64)
    tails: list[int] = []
    tails_idx: list[int] = []
    prev = np.full(seq.size, -1, dtype=np.int64)
    for idx, value in enumerate(seq):
        pos = int(np.searchsorted(np.asarray(tails, dtype=np.int64), value, side="right"))
        if pos == len(tails):
            tails.append(int(value))
            tails_idx.append(idx)
        else:
            tails[pos] = int(value)
            tails_idx[pos] = idx
        if pos > 0:
            prev[idx] = tails_idx[pos - 1]

    out: list[int] = []
    cursor = tails_idx[-1]
    while cursor >= 0:
        out.append(int(cursor))
        cursor = int(prev[cursor])
    out.reverse()
    return np.asarray(out, dtype=np.int64)


def _min_gap_mask(
    idx_a: NDArray[np.integer],
    idx_b: NDArray[np.integer],
    *,
    min_gap_frames: int,
) -> NDArray[np.bool_]:
    if idx_a.size == 0:
        return np.empty(0, dtype=np.bool_)
    keep = np.zeros(idx_a.size, dtype=np.bool_)
    last_a = -10**12
    last_b = -10**12
    for idx, (frame_a, frame_b) in enumerate(zip(idx_a, idx_b, strict=False)):
        if int(frame_a) - last_a >= min_gap_frames and int(frame_b) - last_b >= min_gap_frames:
            keep[idx] = True
            last_a = int(frame_a)
            last_b = int(frame_b)
    return keep


def extract_spectral_flux(
    audio: NDArray[np.floating],
    *,
    sr: int,
    hop_length: int,
) -> NDArray[np.float64]:
    """Return a normalized one-dimensional spectral-flux onset cue."""
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop_length)
    onset_env = np.maximum(onset_env.astype(np.float64), 0.0)
    if onset_env.size and np.max(onset_env) > 0:
        onset_env = onset_env / np.max(onset_env)
    return onset_env[None, :]


def evaluate_teacher_path(pair: SWDPair, teacher: TeacherPath, *, method: str) -> dict[str, object]:
    """Evaluate one teacher path against SWD measure annotations."""
    alignment = compute_ground_truth_measure_alignment(pair)
    if alignment is None:
        raise ValueError(f"No ground-truth measure alignment for {pair.pair_id}")
    ann_a, ann_b = alignment
    gt_a = ann_a["time_s"].to_numpy(dtype=float)
    gt_b = ann_b["time_s"].to_numpy(dtype=float)
    pred_b = interp_monotonic(gt_a, teacher.time_a_s, teacher.time_b_s)
    return _metric_row(
        pair=pair,
        method=method,
        gt_a=gt_a,
        gt_b=gt_b,
        pred_b=pred_b,
        runtime_s=np.nan,
    )


def evaluate_oracle_alignment(pair: SWDPair) -> dict[str, object]:
    """Return a metric sanity-check row using the ground truth as prediction."""
    alignment = compute_ground_truth_measure_alignment(pair)
    if alignment is None:
        raise ValueError(f"No ground-truth measure alignment for {pair.pair_id}")
    ann_a, ann_b = alignment
    gt_a = ann_a["time_s"].to_numpy(dtype=float)
    gt_b = ann_b["time_s"].to_numpy(dtype=float)
    return _metric_row(
        pair=pair,
        method="oracle",
        gt_a=gt_a,
        gt_b=gt_b,
        pred_b=gt_b,
        runtime_s=0.0,
    )


def summarize_teacher_results(results: pd.DataFrame) -> dict[str, float]:
    """Return aggregate teacher metrics matching the repo strict convention."""
    if results.empty:
        return {"pairs": 0.0, "mae": float("nan"), "ar_50ms": float("nan")}
    return {
        "pairs": float(len(results)),
        "mae": float(results["mae"].mean()),
        "ar_50ms": float(results["ar_50ms"].mean()),
        "ar_100ms": float(results["ar_100ms"].mean()),
        "ar_200ms": float(results["ar_200ms"].mean()),
    }


def interp_monotonic(
    query_x: NDArray[np.floating],
    path_x: NDArray[np.floating],
    path_y: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Interpolate a monotonic DTW path at query times."""
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
        raise ValueError("Need at least two teacher path points for interpolation.")
    y = np.maximum.accumulate(np.asarray(unique_y, dtype=np.float64))
    return np.interp(query_x, np.asarray(unique_x, dtype=np.float64), y)


def _metric_row(
    *,
    pair: SWDPair,
    method: str,
    gt_a: NDArray[np.floating],
    gt_b: NDArray[np.floating],
    pred_b: NDArray[np.floating],
    runtime_s: float,
) -> dict[str, object]:
    return {
        "dataset": "swd",
        "pair_id": pair.pair_id,
        "group_id": pair.lied_id,
        "piece_a_id": pair.piece_a.piece_id,
        "piece_b_id": pair.piece_b.piece_id,
        "method": method,
        "duration_s": np.nan,
        "memory_mb": np.nan,
        "mae": mean_absolute_error(pred_b, gt_b),
        "median_ae": median_absolute_error(pred_b, gt_b),
        "ar_50ms": alignment_rate(pred_b, gt_b, 0.05),
        "ar_100ms": alignment_rate(pred_b, gt_b, 0.10),
        "ar_200ms": alignment_rate(pred_b, gt_b, 0.20),
        "runtime_s": float(runtime_s),
        "n_gt_points": int(len(gt_a)),
    }


def _filter_monotonic_path(path: NDArray[np.integer]) -> NDArray[np.intp]:
    if path.shape[0] != 2:
        raise ValueError("Teacher path must have shape (2, n_path_points).")
    order = np.lexsort((path[1], path[0]))
    ordered = path[:, order].astype(np.intp, copy=False)
    keep: list[int] = []
    last_a = -1
    last_b = -1
    for idx, (frame_a, frame_b) in enumerate(ordered.T):
        if frame_a >= last_a and frame_b >= last_b:
            keep.append(idx)
            last_a = int(frame_a)
            last_b = int(frame_b)
    if len(keep) < 2:
        raise ValueError("Teacher path collapsed below two monotonic points.")
    return ordered[:, keep]


def _pool_feature_frames_for_shift(
    features: NDArray[np.floating],
    *,
    max_frames: int,
) -> NDArray[np.float64]:
    values = np.asarray(features, dtype=np.float64)
    if max_frames <= 0 or values.shape[1] <= max_frames:
        return _normalize_columns(values)
    pool_size = int(np.ceil(values.shape[1] / max_frames))
    pooled_frames = values.shape[1] // pool_size
    if pooled_frames <= 0:
        return _normalize_columns(values)
    cropped = values[:, : pooled_frames * pool_size]
    pooled = cropped.reshape(values.shape[0], pooled_frames, pool_size).mean(axis=2)
    return _normalize_columns(pooled)


def _resize_feature_frames(
    features: NDArray[np.floating],
    *,
    target_frames: int,
) -> NDArray[np.float64]:
    features = np.asarray(features, dtype=np.float64)
    if features.shape[1] == target_frames:
        return features
    if features.shape[1] <= 1:
        return np.repeat(features, target_frames, axis=1)
    source_x = np.linspace(0.0, 1.0, features.shape[1])
    target_x = np.linspace(0.0, 1.0, target_frames)
    resized = np.vstack([np.interp(target_x, source_x, row) for row in features])
    return resized.astype(np.float64, copy=False)


def _weighted_stack(
    *blocks: tuple[NDArray[np.floating], float],
    target_frames: int,
) -> NDArray[np.float64]:
    prepared: list[NDArray[np.float64]] = []
    for block, weight in blocks:
        if weight <= 0:
            continue
        resized = _resize_feature_frames(block, target_frames=target_frames)
        resized = _normalize_columns(resized)
        prepared.append(resized * float(weight))
    if not prepared:
        raise ValueError("At least one positive teacher feature weight is required.")
    return np.vstack(prepared).astype(np.float64, copy=False)


def _normalize_columns(features: NDArray[np.floating]) -> NDArray[np.float64]:
    values = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(values, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    return values / norms
