"""Optional transcription-model features for alignment refinement.

The adapter is intentionally thin: Basic Pitch is an optional dependency, and
tests can exercise the reshaping/resampling logic without downloading a model.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


BASIC_PITCH_KEYS = ("note", "onset", "contour")


def extract_basic_pitch_features(
    audio_path: str | Path,
    *,
    cache_root: str | Path = ".cache/transcription/basic_pitch",
    target_frames: int | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Load or compute Basic Pitch raw outputs for one audio file."""
    path = Path(audio_path)
    cache_path = _cache_path(path, cache_root)
    if cache_path.exists():
        with np.load(cache_path) as data:
            features = {key: np.asarray(data[key], dtype=np.float64) for key in BASIC_PITCH_KEYS if key in data}
    else:
        features = _predict_basic_pitch(path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **features)

    return build_transcription_feature_blocks(features, target_frames=target_frames)


def build_transcription_feature_blocks(
    model_output: dict[str, Any],
    *,
    target_frames: int | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Convert raw transcription outputs to note and pitch-class feature blocks."""
    note = _orient_pitch_time(_get_output_array(model_output, "note"))
    onset = _orient_pitch_time(_get_output_array(model_output, "onset"))
    contour = _orient_pitch_time(_get_output_array(model_output, "contour"))

    if target_frames is not None:
        note = resample_feature_frames(note, target_frames)
        onset = resample_feature_frames(onset, target_frames)
        contour = resample_feature_frames(contour, target_frames)

    return {
        "note": _sanitize(note),
        "onset": _sanitize(onset),
        "contour": _sanitize(contour),
        "pitch_class_note": fold_pitch_to_pitch_class(note),
        "pitch_class_onset": fold_pitch_to_pitch_class(onset),
        "pitch_class_contour": fold_pitch_to_pitch_class(contour),
    }


def fold_pitch_to_pitch_class(
    features: NDArray[np.floating],
    *,
    n_pitch_classes: int = 12,
) -> NDArray[np.float64]:
    """Fold an arbitrary pitch-axis feature matrix to pitch classes."""
    matrix = _sanitize(features)
    folded = np.zeros((n_pitch_classes, matrix.shape[1]), dtype=np.float64)
    for pitch_idx in range(matrix.shape[0]):
        folded[pitch_idx % n_pitch_classes] += matrix[pitch_idx]
    return _sanitize(folded)


def resample_feature_frames(
    features: NDArray[np.floating],
    target_frames: int,
) -> NDArray[np.float64]:
    """Linearly resample feature columns to a target frame count."""
    matrix = _sanitize(features)
    if target_frames <= 0:
        raise ValueError("target_frames must be positive.")
    if matrix.shape[1] == target_frames:
        return matrix
    if matrix.shape[1] == 0:
        return np.zeros((matrix.shape[0], target_frames), dtype=np.float64)
    if matrix.shape[1] == 1:
        return np.repeat(matrix, target_frames, axis=1)

    source_x = np.linspace(0.0, 1.0, matrix.shape[1])
    target_x = np.linspace(0.0, 1.0, target_frames)
    return np.vstack([np.interp(target_x, source_x, row) for row in matrix]).astype(np.float64)


def _predict_basic_pitch(audio_path: Path) -> dict[str, NDArray[np.float64]]:
    try:
        from basic_pitch.inference import predict
    except ImportError as exc:
        raise ImportError(
            "Basic Pitch is required for transcription-backed DeepAlign modes. "
            'Install the runtime with: pip install -e ".[transcription]" && '
            "pip install basic-pitch --no-deps"
        ) from exc

    model_output, _, _ = predict(str(audio_path))
    return {
        key: _orient_pitch_time(_get_output_array(model_output, key))
        for key in BASIC_PITCH_KEYS
    }


def _get_output_array(model_output: dict[str, Any], key: str) -> NDArray[np.floating]:
    if key not in model_output:
        raise KeyError(f"Basic Pitch output is missing the '{key}' array.")
    array = np.asarray(model_output[key], dtype=np.float64)
    if array.ndim == 3:
        # Basic Pitch outputs are commonly batched as (batch, frames, pitches).
        array = array.reshape(-1, array.shape[-1])
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D transcription output for '{key}', got shape {array.shape}.")
    return array


def _orient_pitch_time(array: NDArray[np.floating]) -> NDArray[np.float64]:
    matrix = np.asarray(array, dtype=np.float64)
    # Basic Pitch usually returns (frames, pitches). Alignment code expects
    # (features, frames), so transpose when the pitch axis appears last.
    known_pitch_bins = {88, 264}
    if matrix.shape[1] in known_pitch_bins and matrix.shape[0] not in known_pitch_bins:
        matrix = matrix.T
    elif matrix.shape[0] not in known_pitch_bins and matrix.shape[0] >= matrix.shape[1]:
        matrix = matrix.T
    return _sanitize(matrix)


def _sanitize(features: NDArray[np.floating]) -> NDArray[np.float64]:
    return np.nan_to_num(np.asarray(features, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def _cache_path(audio_path: Path, cache_root: str | Path) -> Path:
    resolved = audio_path.resolve()
    stat = resolved.stat()
    digest = hashlib.sha1(f"{resolved}|{stat.st_mtime_ns}|{stat.st_size}".encode("utf-8")).hexdigest()[:16]
    return Path(cache_root) / f"{resolved.stem}_{digest}.npz"
