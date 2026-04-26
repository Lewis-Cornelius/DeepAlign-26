"""Cached CQT helpers for faster DeepAlign training and evaluation."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import librosa
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def compute_log_cqt(
    audio: NDArray[np.floating],
    *,
    sr: int,
    hop_length: int,
    n_bins: int,
    bins_per_octave: int,
) -> NDArray[np.float32]:
    """Compute a log-magnitude CQT without window-level standardization."""
    cqt = librosa.cqt(
        y=audio.astype(np.float32),
        sr=sr,
        hop_length=hop_length,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
    )
    cqt_mag = np.abs(cqt)
    cqt_log = librosa.amplitude_to_db(cqt_mag, ref=np.max).astype(np.float32)
    return cqt_log


def standardize_log_cqt(log_cqt: NDArray[np.floating]) -> NDArray[np.float32]:
    """Standardize one log-CQT window to zero mean and unit variance."""
    if log_cqt.size == 0:
        return log_cqt.astype(np.float32, copy=False)
    standardized = (log_cqt - log_cqt.mean()) / (log_cqt.std() + 1e-8)
    return standardized.astype(np.float32, copy=False)


def seconds_to_frame(seconds: float, *, sr: int, hop_length: int) -> int:
    """Convert seconds to a CQT frame index."""
    return int(round(seconds * sr / hop_length))


def slice_log_cqt(
    log_cqt: NDArray[np.floating],
    *,
    start_s: float,
    end_s: float,
    sr: int,
    hop_length: int,
) -> NDArray[np.float32]:
    """Slice a cached full-song log CQT using time boundaries."""
    start_frame = max(0, seconds_to_frame(start_s, sr=sr, hop_length=hop_length))
    end_frame = max(start_frame + 1, seconds_to_frame(end_s, sr=sr, hop_length=hop_length))
    start_frame = min(start_frame, log_cqt.shape[1] - 1)
    end_frame = min(end_frame, log_cqt.shape[1])
    return log_cqt[:, start_frame:end_frame].astype(np.float32, copy=False)


class SpectrogramCache:
    """Disk-backed cache for full-song log CQTs keyed by file content metadata."""

    def __init__(
        self,
        cache_root: str | Path,
        *,
        sr: int,
        hop_length: int,
        n_bins: int,
        bins_per_octave: int,
    ):
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.sr = sr
        self.hop_length = hop_length
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self._memory: dict[str, NDArray[np.float32]] = {}

    def load_or_compute(self, audio_path: str | Path) -> NDArray[np.float32]:
        """Load one cached full-song log CQT or compute and persist it."""
        path = Path(audio_path).resolve()
        cache_key = self._build_cache_key(path)
        if cache_key in self._memory:
            return self._memory[cache_key]

        cache_path = self.cache_root / f"{cache_key}.npy"
        if cache_path.exists():
            log_cqt = np.load(cache_path)
        else:
            audio, _ = librosa.load(path, sr=self.sr, mono=True)
            log_cqt = compute_log_cqt(
                audio,
                sr=self.sr,
                hop_length=self.hop_length,
                n_bins=self.n_bins,
                bins_per_octave=self.bins_per_octave,
            )
            np.save(cache_path, log_cqt.astype(np.float32, copy=False))
            logger.info("Cached CQT for %s at %s", path.name, cache_path)

        self._memory[cache_key] = log_cqt.astype(np.float32, copy=False)
        return self._memory[cache_key]

    def _build_cache_key(self, audio_path: Path) -> str:
        stat = audio_path.stat()
        digest = hashlib.sha1()
        digest.update(str(audio_path).encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
        digest.update(
            f"{self.sr}:{self.hop_length}:{self.n_bins}:{self.bins_per_octave}".encode("utf-8")
        )
        return digest.hexdigest()
