"""PyTorch Dataset for SWD audio pairs with CQT spectrogram extraction.

Handles loading audio pairs, computing CQT spectrograms, and
applying augmentations for training the DeepAlign-26 model.
"""

import logging
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset

from dis_alignment.data.swd import SWDDataset, SWDPair, load_swd_audio
from dis_alignment.model.augmentation import AudioAugmentor

logger = logging.getLogger(__name__)


class SWDPairDataset(Dataset):
    """
    PyTorch Dataset for pairs of SWD recordings.

    Each item returns CQT spectrograms for a pair of recordings
    of the same lied, suitable for training with Soft-DTW loss.

    Args:
        swd: SWDDataset instance.
        sr: Sample rate for audio loading.
        hop_length: Hop length for CQT computation.
            Default 220 samples at 22050 Hz = ~10ms (as per plan).
        n_bins: Number of CQT frequency bins.
            Default 84 = 7 octaves × 12 bins per octave.
        bins_per_octave: CQT resolution.
        max_length_sec: Maximum audio length in seconds.
            Longer pieces are randomly cropped during training.
        augmentor: Optional AudioAugmentor for data augmentation.
        performances: Limit to specific performance IDs.
        lieder: Limit to specific lied numbers.
    """

    def __init__(
        self,
        swd: SWDDataset,
        sr: int = 22050,
        hop_length: int = 220,
        n_bins: int = 84,
        bins_per_octave: int = 12,
        max_length_sec: float = 60.0,
        augmentor: AudioAugmentor | None = None,
        performances: list[str] | None = None,
        lieder: list[str] | None = None,
    ):
        self.sr = sr
        self.hop_length = hop_length
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.max_length_sec = max_length_sec
        self.augmentor = augmentor
        self.max_samples = int(max_length_sec * sr)

        # Collect all pairs
        self.pairs: list[SWDPair] = list(
            swd.iter_pairs(performances=performances, lieder=lieder)
        )

        logger.info(f"Created dataset with {len(self.pairs)} pairs")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get a training sample.

        Returns:
            Dict with keys:
                - spec_a: CQT spectrogram A, shape (1, freq_bins, time)
                - spec_b: CQT spectrogram B, shape (1, freq_bins, time)
                - pair_id: String identifier for the pair
        """
        pair = self.pairs[idx]

        # Load audio
        audio_a, _ = load_swd_audio(pair.piece_a, sr=self.sr)
        audio_b, _ = load_swd_audio(pair.piece_b, sr=self.sr)

        # Random crop if too long
        audio_a = self._maybe_crop(audio_a)
        audio_b = self._maybe_crop(audio_b)

        # Apply augmentation (only to one stream for asymmetry)
        if self.augmentor is not None:
            audio_tensor_a = torch.from_numpy(audio_a).float()
            audio_a = self.augmentor(audio_tensor_a).numpy()

        # Compute CQT spectrograms
        spec_a = self._compute_cqt(audio_a)
        spec_b = self._compute_cqt(audio_b)

        return {
            "spec_a": torch.from_numpy(spec_a).float().unsqueeze(0),  # (1, F, T)
            "spec_b": torch.from_numpy(spec_b).float().unsqueeze(0),  # (1, F, T)
            "pair_id": pair.pair_id,
        }

    def _maybe_crop(self, audio: NDArray) -> NDArray:
        """Randomly crop audio if longer than max_length_sec."""
        if len(audio) > self.max_samples:
            start = np.random.randint(0, len(audio) - self.max_samples)
            return audio[start : start + self.max_samples]
        return audio

    def _compute_cqt(self, audio: NDArray) -> NDArray:
        """
        Compute CQT spectrogram.

        Returns:
            Log-magnitude CQT of shape (n_bins, n_frames).
        """
        cqt = librosa.cqt(
            y=audio.astype(np.float32),
            sr=self.sr,
            hop_length=self.hop_length,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave,
        )

        # Convert to log magnitude
        cqt_mag = np.abs(cqt)
        cqt_log = librosa.amplitude_to_db(cqt_mag, ref=np.max)

        # Standardize to zero-mean, unit-variance
        cqt_log = (cqt_log - cqt_log.mean()) / (cqt_log.std() + 1e-8)

        return cqt_log


def collate_variable_length(batch: list[dict]) -> dict[str, Any]:
    """
    Custom collate function for variable-length spectrograms.

    Pads all spectrograms in the batch to the same length along the
    time axis.

    Args:
        batch: List of dataset items.

    Returns:
        Batched dict with padded tensors and length info.
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

        # Pad time dimension
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

    return {
        "spec_a": torch.stack(specs_a),
        "spec_b": torch.stack(specs_b),
        "lengths_a": torch.tensor(lengths_a),
        "lengths_b": torch.tensor(lengths_b),
        "pair_ids": pair_ids,
    }
