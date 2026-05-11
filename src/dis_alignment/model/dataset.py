"""PyTorch Dataset for SWD audio pairs with aligned-window sampling."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from dis_alignment.alignment.teacher import TeacherPath, load_teacher_path
from dis_alignment.data.swd import (
    SWDDataset,
    SWDPair,
    compute_ground_truth_measure_alignment,
    load_swd_audio,
)
from dis_alignment.model.cqt_cache import SpectrogramCache, slice_log_cqt, standardize_log_cqt
from dis_alignment.model.augmentation import AudioAugmentor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlignedMeasureWindow:
    """One shared aligned segment anchored on SWD measure annotations."""

    start_event_id: str
    end_boundary_event_id: str | None
    start_a_s: float
    end_a_s: float
    start_b_s: float
    end_b_s: float
    num_measures: int
    anchor_event_ids: tuple[str, ...]
    anchor_a_s: tuple[float, ...]
    anchor_b_s: tuple[float, ...]


@dataclass(frozen=True)
class TeacherPathWindow:
    """One audio-only crop window selected from a teacher path."""

    start_a_s: float
    end_a_s: float
    start_b_s: float
    end_b_s: float
    center_a_s: float
    center_b_s: float


class SWDPairDataset(Dataset):
    """
    PyTorch dataset for pairs of SWD recordings.

    Each item returns CQT spectrograms for a pair of recordings of the same lied.
    In the default ``aligned_measures`` mode, the two waveforms are cropped to the
    same shared measure window before feature extraction. This keeps the training
    target musically aligned instead of using independent random crops.
    """

    def __init__(
        self,
        swd: SWDDataset | None = None,
        sr: int = 22050,
        hop_length: int = 220,
        n_bins: int = 84,
        bins_per_octave: int = 12,
        max_length_sec: float = 60.0,
        augmentor: AudioAugmentor | None = None,
        performances: list[str] | None = None,
        lieder: list[str] | None = None,
        *,
        pairs: list[SWDPair] | None = None,
        segment_sampling: str = "aligned_measures",
        samples_per_epoch: int | None = None,
        deterministic: bool = False,
        random_seed: int = 42,
        cache_spectrograms: bool = False,
        cache_root: str | Path | None = None,
        teacher_path_root: str | Path | None = None,
        num_teacher_samples: int = 64,
        teacher_min_confidence: float = 0.0,
    ):
        if pairs is None:
            if swd is None:
                raise ValueError("Provide either an SWDDataset or an explicit pair list.")
            pairs = list(swd.iter_pairs(performances=performances, lieder=lieder))

        self.sr = sr
        self.hop_length = hop_length
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.max_length_sec = max_length_sec
        self.augmentor = augmentor
        self.max_samples = int(max_length_sec * sr)
        self.segment_sampling = segment_sampling
        self.samples_per_epoch = samples_per_epoch
        self.deterministic = deterministic
        self._rng = np.random.default_rng(random_seed)
        self.cache_spectrograms = cache_spectrograms
        self.cache_root = Path(cache_root) if cache_root is not None else None
        self.teacher_path_root = Path(teacher_path_root) if teacher_path_root is not None else None
        self.num_teacher_samples = max(0, int(num_teacher_samples))
        self.teacher_min_confidence = float(teacher_min_confidence)
        self._teacher_cache: dict[str, TeacherPath | None] = {}

        if self.segment_sampling not in {
            "aligned_measures",
            "independent_random",
            "teacher_path",
            "self_audio",
        }:
            raise ValueError(
                "segment_sampling must be one of "
                "{'aligned_measures', 'independent_random', 'teacher_path', 'self_audio'}"
            )
        if self.segment_sampling == "teacher_path" and self.teacher_path_root is None:
            raise ValueError("teacher_path sampling requires teacher_path_root.")

        self.pairs = list(pairs)
        self._windows_by_pair_id: dict[str, list[AlignedMeasureWindow]] = {}
        if self.segment_sampling == "aligned_measures":
            self._windows_by_pair_id = {
                pair.pair_id: self._build_aligned_windows(pair) for pair in self.pairs
            }
        self._spectrogram_cache: SpectrogramCache | None = None
        if self.cache_spectrograms and self.cache_root is not None and self.augmentor is None:
            self._spectrogram_cache = SpectrogramCache(
                self.cache_root,
                sr=self.sr,
                hop_length=self.hop_length,
                n_bins=self.n_bins,
                bins_per_octave=self.bins_per_octave,
            )
        elif self.cache_spectrograms and self.augmentor is not None:
            logger.warning(
                "Spectrogram caching is disabled because waveform augmentation is enabled."
            )

        logger.info(
            "Created dataset with %s pairs, sampling=%s, samples_per_epoch=%s, cache=%s",
            len(self.pairs),
            self.segment_sampling,
            self.samples_per_epoch,
            bool(self._spectrogram_cache),
        )

    def __len__(self) -> int:
        if self.samples_per_epoch is not None:
            return self.samples_per_epoch
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get a training sample.

        Returns a dict with spectrogram tensors and lightweight metadata useful for
        debugging aligned-window selection.
        """
        if not self.pairs:
            raise IndexError("SWDPairDataset is empty")

        pair = self.pairs[idx % len(self.pairs)]

        window = None
        teacher_window = None
        spec_a: NDArray[np.float32] | None = None
        spec_b: NDArray[np.float32] | None = None
        if self.segment_sampling == "aligned_measures":
            window = self._select_aligned_window(pair)
            if window is not None and self._spectrogram_cache is not None:
                log_cqt_a = self._spectrogram_cache.load_or_compute(pair.piece_a.audio_path)
                log_cqt_b = self._spectrogram_cache.load_or_compute(pair.piece_b.audio_path)
                spec_a = standardize_log_cqt(
                    slice_log_cqt(
                        log_cqt_a,
                        start_s=window.start_a_s,
                        end_s=window.end_a_s,
                        sr=self.sr,
                        hop_length=self.hop_length,
                    )
                )
                spec_b = standardize_log_cqt(
                    slice_log_cqt(
                        log_cqt_b,
                        start_s=window.start_b_s,
                        end_s=window.end_b_s,
                        sr=self.sr,
                        hop_length=self.hop_length,
                    )
                )
            elif window is not None:
                audio_a, _ = load_swd_audio(pair.piece_a, sr=self.sr)
                audio_b, _ = load_swd_audio(pair.piece_b, sr=self.sr)
                audio_a = self._crop_audio_window(audio_a, window.start_a_s, window.end_a_s)
                audio_b = self._crop_audio_window(audio_b, window.start_b_s, window.end_b_s)
            else:
                logger.warning(
                    "Falling back to independent crops for %s because no aligned windows were found.",
                    pair.pair_id,
                )
                audio_a, _ = load_swd_audio(pair.piece_a, sr=self.sr)
                audio_b, _ = load_swd_audio(pair.piece_b, sr=self.sr)
                audio_a = self._maybe_crop(audio_a)
                audio_b = self._maybe_crop(audio_b)
        elif self.segment_sampling == "teacher_path":
            teacher_window = self._select_teacher_path_window(pair)
            if teacher_window is not None and self._spectrogram_cache is not None:
                log_cqt_a = self._spectrogram_cache.load_or_compute(pair.piece_a.audio_path)
                log_cqt_b = self._spectrogram_cache.load_or_compute(pair.piece_b.audio_path)
                spec_a = standardize_log_cqt(
                    slice_log_cqt(
                        log_cqt_a,
                        start_s=teacher_window.start_a_s,
                        end_s=teacher_window.end_a_s,
                        sr=self.sr,
                        hop_length=self.hop_length,
                    )
                )
                spec_b = standardize_log_cqt(
                    slice_log_cqt(
                        log_cqt_b,
                        start_s=teacher_window.start_b_s,
                        end_s=teacher_window.end_b_s,
                        sr=self.sr,
                        hop_length=self.hop_length,
                    )
                )
            elif teacher_window is not None:
                audio_a, _ = load_swd_audio(pair.piece_a, sr=self.sr)
                audio_b, _ = load_swd_audio(pair.piece_b, sr=self.sr)
                audio_a = self._crop_audio_window(
                    audio_a,
                    teacher_window.start_a_s,
                    teacher_window.end_a_s,
                )
                audio_b = self._crop_audio_window(
                    audio_b,
                    teacher_window.start_b_s,
                    teacher_window.end_b_s,
                )
            else:
                logger.warning(
                    "Falling back to independent crops for %s because no teacher window was found.",
                    pair.pair_id,
                )
                audio_a, _ = load_swd_audio(pair.piece_a, sr=self.sr)
                audio_b, _ = load_swd_audio(pair.piece_b, sr=self.sr)
                audio_a = self._maybe_crop(audio_a)
                audio_b = self._maybe_crop(audio_b)
        elif self.segment_sampling == "self_audio":
            piece = pair.piece_a
            if not self.deterministic and self._rng.random() >= 0.5:
                piece = pair.piece_b
            audio, _ = load_swd_audio(piece, sr=self.sr)
            audio = self._maybe_crop(audio)
            audio_a = np.asarray(audio, dtype=np.float32).copy()
            audio_b = np.asarray(audio, dtype=np.float32).copy()
        else:
            audio_a, _ = load_swd_audio(pair.piece_a, sr=self.sr)
            audio_b, _ = load_swd_audio(pair.piece_b, sr=self.sr)
            audio_a = self._maybe_crop(audio_a)
            audio_b = self._maybe_crop(audio_b)

        if spec_a is None or spec_b is None:
            if self.augmentor is not None:
                audio_a = self._augment_audio(audio_a)
                audio_b = self._augment_audio(audio_b)

            spec_a = self._compute_cqt(audio_a)
            spec_b = self._compute_cqt(audio_b)

        sample: dict[str, Any] = {
            "spec_a": torch.from_numpy(spec_a).float().unsqueeze(0),
            "spec_b": torch.from_numpy(spec_b).float().unsqueeze(0),
            "pair_id": pair.pair_id,
            "segment_sampling": self.segment_sampling,
        }
        if window is not None:
            anchor_frames_a, anchor_frames_b = self._anchor_frames_for_window(window, spec_a.shape[1], spec_b.shape[1])
            teacher_frames_a, teacher_frames_b = self._teacher_frames_for_bounds(
                pair,
                start_a_s=window.start_a_s,
                end_a_s=window.end_a_s,
                start_b_s=window.start_b_s,
                end_b_s=window.end_b_s,
                n_frames_a=spec_a.shape[1],
                n_frames_b=spec_b.shape[1],
            )
            sample.update(
                {
                    "window_start_event_id": window.start_event_id,
                    "window_end_boundary_event_id": window.end_boundary_event_id,
                    "window_duration_a_s": (window.end_a_s - window.start_a_s),
                    "window_duration_b_s": (window.end_b_s - window.start_b_s),
                    "window_num_measures": window.num_measures,
                    "anchor_event_ids": list(window.anchor_event_ids),
                    "anchor_frame_indices_a": anchor_frames_a,
                    "anchor_frame_indices_b": anchor_frames_b,
                    "teacher_frame_indices_a": teacher_frames_a,
                    "teacher_frame_indices_b": teacher_frames_b,
                }
            )
        elif teacher_window is not None:
            teacher_frames_a, teacher_frames_b = self._teacher_frames_for_bounds(
                pair,
                start_a_s=teacher_window.start_a_s,
                end_a_s=teacher_window.end_a_s,
                start_b_s=teacher_window.start_b_s,
                end_b_s=teacher_window.end_b_s,
                n_frames_a=spec_a.shape[1],
                n_frames_b=spec_b.shape[1],
            )
            sample.update(
                {
                    "teacher_window_start_a_s": teacher_window.start_a_s,
                    "teacher_window_start_b_s": teacher_window.start_b_s,
                    "teacher_window_duration_a_s": teacher_window.end_a_s - teacher_window.start_a_s,
                    "teacher_window_duration_b_s": teacher_window.end_b_s - teacher_window.start_b_s,
                    "teacher_frame_indices_a": teacher_frames_a,
                    "teacher_frame_indices_b": teacher_frames_b,
                    "anchor_frame_indices_a": [],
                    "anchor_frame_indices_b": [],
                }
            )
        elif self.segment_sampling == "self_audio":
            frames_a, frames_b = self._identity_frames_for_specs(spec_a.shape[1], spec_b.shape[1])
            sample.update(
                {
                    "teacher_frame_indices_a": frames_a,
                    "teacher_frame_indices_b": frames_b,
                    "anchor_frame_indices_a": [],
                    "anchor_frame_indices_b": [],
                }
            )

        return sample

    def _anchor_frames_for_window(
        self,
        window: AlignedMeasureWindow,
        n_frames_a: int,
        n_frames_b: int,
    ) -> tuple[list[int], list[int]]:
        if not window.anchor_event_ids:
            return [], []
        anchor_frames_a = [
            int(np.clip(round((time_s - window.start_a_s) * self.sr / self.hop_length), 0, n_frames_a - 1))
            for time_s in window.anchor_a_s
        ]
        anchor_frames_b = [
            int(np.clip(round((time_s - window.start_b_s) * self.sr / self.hop_length), 0, n_frames_b - 1))
            for time_s in window.anchor_b_s
        ]
        return anchor_frames_a, anchor_frames_b

    def _teacher_frames_for_bounds(
        self,
        pair: SWDPair,
        *,
        start_a_s: float,
        end_a_s: float,
        start_b_s: float,
        end_b_s: float,
        n_frames_a: int,
        n_frames_b: int,
    ) -> tuple[list[int], list[int]]:
        if self.teacher_path_root is None or self.num_teacher_samples <= 0:
            return [], []
        teacher = self._load_teacher_path(pair)
        if teacher is None:
            return [], []

        mask = (
            (teacher.time_a_s >= start_a_s)
            & (teacher.time_a_s <= end_a_s)
            & (teacher.time_b_s >= start_b_s)
            & (teacher.time_b_s <= end_b_s)
        )
        if teacher.confidence is not None:
            confidence = np.asarray(teacher.confidence, dtype=np.float32)
            if confidence.shape[0] == mask.shape[0]:
                mask &= confidence >= self.teacher_min_confidence
        available = np.flatnonzero(mask)
        if available.size < 2:
            return [], []

        if self.deterministic or available.size <= self.num_teacher_samples:
            positions = np.linspace(0, available.size - 1, min(available.size, self.num_teacher_samples), dtype=int)
            selected = available[positions]
        else:
            selected = np.sort(
                self._rng.choice(available, size=self.num_teacher_samples, replace=False)
            )

        frames_a = np.rint((teacher.time_a_s[selected] - start_a_s) * self.sr / self.hop_length)
        frames_b = np.rint((teacher.time_b_s[selected] - start_b_s) * self.sr / self.hop_length)
        frames_a = np.clip(frames_a, 0, n_frames_a - 1).astype(int)
        frames_b = np.clip(frames_b, 0, n_frames_b - 1).astype(int)
        frame_pairs = sorted(set(zip(frames_a.tolist(), frames_b.tolist(), strict=False)))
        if not frame_pairs:
            return [], []
        unique_a, unique_b = zip(*frame_pairs, strict=False)
        return list(unique_a), list(unique_b)

    def _load_teacher_path(self, pair: SWDPair) -> TeacherPath | None:
        if pair.pair_id in self._teacher_cache:
            return self._teacher_cache[pair.pair_id]
        if self.teacher_path_root is None:
            return None
        try:
            teacher = load_teacher_path(self.teacher_path_root, pair.pair_id)
        except FileNotFoundError:
            logger.warning("No teacher path found for %s under %s", pair.pair_id, self.teacher_path_root)
            teacher = None
        self._teacher_cache[pair.pair_id] = teacher
        return teacher

    def _identity_frames_for_specs(self, n_frames_a: int, n_frames_b: int) -> tuple[list[int], list[int]]:
        n_frames = min(int(n_frames_a), int(n_frames_b))
        if n_frames <= 0 or self.num_teacher_samples <= 0:
            return [], []
        n_samples = min(n_frames, self.num_teacher_samples)
        if self.deterministic or n_frames <= n_samples:
            frames = np.linspace(0, n_frames - 1, n_samples, dtype=int)
        else:
            frames = np.sort(self._rng.choice(n_frames, size=n_samples, replace=False))
        values = frames.astype(int).tolist()
        return values, list(values)

    def _select_aligned_window(self, pair: SWDPair) -> AlignedMeasureWindow | None:
        windows = self._windows_by_pair_id.get(pair.pair_id, [])
        if not windows:
            return None
        if self.deterministic:
            return windows[len(windows) // 2]
        return windows[int(self._rng.integers(0, len(windows)))]

    def _select_teacher_path_window(self, pair: SWDPair) -> TeacherPathWindow | None:
        teacher = self._load_teacher_path(pair)
        if teacher is None or teacher.time_a_s.size == 0 or teacher.time_b_s.size == 0:
            return None

        mask = np.isfinite(teacher.time_a_s) & np.isfinite(teacher.time_b_s)
        if teacher.confidence is not None:
            confidence = np.asarray(teacher.confidence, dtype=np.float32)
            if confidence.shape[0] == mask.shape[0]:
                mask &= confidence >= self.teacher_min_confidence
        available = np.flatnonzero(mask)
        if available.size == 0:
            return None

        selected = available[available.size // 2] if self.deterministic else int(self._rng.choice(available))
        center_a = float(teacher.time_a_s[selected])
        center_b = float(teacher.time_b_s[selected])
        duration_a = max(float(np.nanmax(teacher.time_a_s)), center_a, 1.0)
        duration_b = max(float(np.nanmax(teacher.time_b_s)), center_b, 1.0)
        start_a, end_a = self._centered_time_window(center_a, duration_a)
        start_b, end_b = self._centered_time_window(center_b, duration_b)
        return TeacherPathWindow(
            start_a_s=start_a,
            end_a_s=end_a,
            start_b_s=start_b,
            end_b_s=end_b,
            center_a_s=center_a,
            center_b_s=center_b,
        )

    def _centered_time_window(self, center_s: float, duration_s: float) -> tuple[float, float]:
        window_s = min(float(self.max_length_sec), max(float(duration_s), 1.0))
        if duration_s <= window_s:
            return 0.0, max(float(duration_s), 1.0 / self.sr)
        start_s = center_s - (window_s / 2.0)
        start_s = min(max(0.0, start_s), max(0.0, duration_s - window_s))
        return start_s, start_s + window_s

    def _build_aligned_windows(self, pair: SWDPair) -> list[AlignedMeasureWindow]:
        alignment = compute_ground_truth_measure_alignment(pair)
        if alignment is None:
            return []

        ann_a, ann_b = alignment
        event_ids = ann_a["event_id"].astype(str).tolist()
        times_a = ann_a["time_s"].astype(float).to_numpy()
        times_b = ann_b["time_s"].astype(float).to_numpy()

        if len(event_ids) < 2:
            return []

        windows: list[AlignedMeasureWindow] = []
        for start_idx in range(len(event_ids) - 1):
            best_end_idx = None
            for end_idx in range(start_idx + 1, len(event_ids)):
                duration_a = float(times_a[end_idx] - times_a[start_idx])
                duration_b = float(times_b[end_idx] - times_b[start_idx])
                if max(duration_a, duration_b) <= self.max_length_sec:
                    best_end_idx = end_idx
                else:
                    break

            if best_end_idx is None:
                fallback_end_idx = start_idx + 1
                anchor_ids = tuple(event_ids[start_idx : fallback_end_idx + 1])
                anchor_a = tuple(float(value) for value in times_a[start_idx : fallback_end_idx + 1])
                anchor_b = tuple(float(value) for value in times_b[start_idx : fallback_end_idx + 1])
                windows.append(
                    AlignedMeasureWindow(
                        start_event_id=event_ids[start_idx],
                        end_boundary_event_id=event_ids[fallback_end_idx],
                        start_a_s=float(times_a[start_idx]),
                        end_a_s=float(times_a[start_idx] + self.max_length_sec),
                        start_b_s=float(times_b[start_idx]),
                        end_b_s=float(times_b[start_idx] + self.max_length_sec),
                        num_measures=1,
                        anchor_event_ids=anchor_ids,
                        anchor_a_s=anchor_a,
                        anchor_b_s=anchor_b,
                    )
                )
                continue

            anchor_ids = tuple(event_ids[start_idx : best_end_idx + 1])
            anchor_a = tuple(float(value) for value in times_a[start_idx : best_end_idx + 1])
            anchor_b = tuple(float(value) for value in times_b[start_idx : best_end_idx + 1])
            windows.append(
                AlignedMeasureWindow(
                    start_event_id=event_ids[start_idx],
                    end_boundary_event_id=event_ids[best_end_idx],
                    start_a_s=float(times_a[start_idx]),
                    end_a_s=float(times_a[best_end_idx]),
                    start_b_s=float(times_b[start_idx]),
                    end_b_s=float(times_b[best_end_idx]),
                    num_measures=best_end_idx - start_idx,
                    anchor_event_ids=anchor_ids,
                    anchor_a_s=anchor_a,
                    anchor_b_s=anchor_b,
                )
            )

        return windows

    def _augment_audio(self, audio: NDArray[np.floating]) -> NDArray[np.float32]:
        """Apply the configured waveform augmentor to one side of a pair."""
        if self.augmentor is None:
            return audio.astype(np.float32, copy=False)
        audio_tensor = torch.from_numpy(np.asarray(audio, dtype=np.float32)).float()
        return self.augmentor(audio_tensor).detach().cpu().numpy().astype(np.float32, copy=False)

    def _maybe_crop(self, audio: NDArray[np.floating]) -> NDArray[np.floating]:
        """Crop audio to max_length_sec, randomly for training and centrally for validation."""
        if len(audio) <= self.max_samples:
            return audio

        if self.deterministic:
            start = (len(audio) - self.max_samples) // 2
        else:
            start = int(self._rng.integers(0, len(audio) - self.max_samples + 1))
        return audio[start : start + self.max_samples]

    def _crop_audio_window(
        self,
        audio: NDArray[np.floating],
        start_s: float,
        end_s: float,
    ) -> NDArray[np.floating]:
        """Crop one waveform to the requested aligned time window."""
        if len(audio) == 0:
            return audio

        start_sample = max(0, int(round(start_s * self.sr)))
        end_sample = max(start_sample + 1, int(round(end_s * self.sr)))
        start_sample = min(start_sample, len(audio) - 1)
        end_sample = min(end_sample, len(audio))

        cropped = audio[start_sample:end_sample]
        if len(cropped) > self.max_samples:
            cropped = cropped[: self.max_samples]
        return cropped

    def _compute_cqt(self, audio: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute a standardized log-magnitude CQT."""
        cqt = librosa.cqt(
            y=audio.astype(np.float32),
            sr=self.sr,
            hop_length=self.hop_length,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave,
        )

        cqt_mag = np.abs(cqt)
        cqt_log = librosa.amplitude_to_db(cqt_mag, ref=np.max)
        cqt_log = (cqt_log - cqt_log.mean()) / (cqt_log.std() + 1e-8)
        return cqt_log


def collate_variable_length(batch: list[dict]) -> dict[str, Any]:
    """
    Collate variable-length spectrograms by padding on the time axis.

    Any extra metadata fields are preserved as simple lists.
    """
    max_len_a = max(item["spec_a"].shape[-1] for item in batch)
    max_len_b = max(item["spec_b"].shape[-1] for item in batch)

    specs_a = []
    specs_b = []
    lengths_a = []
    lengths_b = []
    pair_ids = []

    for item in batch:
        sa = item["spec_a"]
        sb = item["spec_b"]

        pad_a = max_len_a - sa.shape[-1]
        pad_b = max_len_b - sb.shape[-1]

        if pad_a > 0:
            sa = torch.nn.functional.pad(sa, (0, pad_a))
        if pad_b > 0:
            sb = torch.nn.functional.pad(sb, (0, pad_b))

        specs_a.append(sa)
        specs_b.append(sb)
        lengths_a.append(item["spec_a"].shape[-1])
        lengths_b.append(item["spec_b"].shape[-1])
        pair_ids.append(item["pair_id"])

    collated: dict[str, Any] = {
        "spec_a": torch.stack(specs_a),
        "spec_b": torch.stack(specs_b),
        "lengths_a": torch.tensor(lengths_a),
        "lengths_b": torch.tensor(lengths_b),
        "pair_ids": pair_ids,
    }

    extra_keys = {
        key
        for item in batch
        for key in item
        if key not in {"spec_a", "spec_b", "pair_id"}
    }
    for key in sorted(extra_keys):
        values = [item.get(key) for item in batch]
        collated[key] = values

    return collated
