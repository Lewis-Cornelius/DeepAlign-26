"""On-the-fly data augmentation for audio alignment training.

Augmentations make the learned features robust to recording
conditions, tempo variation, and pitch differences.

Each augmentation operates on raw audio waveforms (1D tensors)
and is applied randomly during training.
"""

import random

import torch
import torchaudio
from torch import Tensor


class AudioAugmentor:
    """
    Composes multiple audio augmentations for training.

    Args:
        time_stretch_range: Min/max stretch factors (1.0 = no change).
        pitch_shift_range: Min/max semitones for pitch shifting.
        noise_snr_range: Min/max SNR in dB for additive noise.
        prob: Probability of applying each augmentation.
        sr: Sample rate of input audio.
    """

    def __init__(
        self,
        time_stretch_range: tuple[float, float] = (0.8, 1.2),
        pitch_shift_range: tuple[int, int] = (-2, 2),
        noise_snr_range: tuple[float, float] = (20.0, 40.0),
        prob: float = 0.5,
        sr: int = 22050,
    ):
        self.time_stretch_range = time_stretch_range
        self.pitch_shift_range = pitch_shift_range
        self.noise_snr_range = noise_snr_range
        self.prob = prob
        self.sr = sr

    def __call__(self, audio: Tensor) -> Tensor:
        """
        Apply random augmentations to audio.

        Args:
            audio: Audio tensor of shape (samples,) or (1, samples).

        Returns:
            Augmented audio tensor.
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        if random.random() < self.prob:
            audio = self._time_stretch(audio)

        if random.random() < self.prob:
            audio = self._pitch_shift(audio)

        if random.random() < self.prob:
            audio = self._add_noise(audio)

        return audio.squeeze(0)

    def _time_stretch(self, audio: Tensor) -> Tensor:
        """Apply random time stretching."""
        rate = random.uniform(*self.time_stretch_range)
        if abs(rate - 1.0) < 0.01:
            return audio

        # Use torchaudio's stretch
        effects = [["tempo", str(rate)]]
        try:
            augmented, _ = torchaudio.sox_effects.apply_effects_tensor(
                audio, self.sr, effects, channels_first=True
            )
            return augmented
        except Exception:
            # Fallback: simple resampling-based stretch
            new_len = int(audio.shape[-1] / rate)
            return torch.nn.functional.interpolate(
                audio.unsqueeze(0), size=new_len, mode="linear",
                align_corners=False
            ).squeeze(0)

    def _pitch_shift(self, audio: Tensor) -> Tensor:
        """Apply random pitch shifting."""
        semitones = random.randint(*self.pitch_shift_range)
        if semitones == 0:
            return audio

        effects = [["pitch", str(semitones * 100)], ["rate", str(self.sr)]]
        try:
            augmented, _ = torchaudio.sox_effects.apply_effects_tensor(
                audio, self.sr, effects, channels_first=True
            )
            return augmented
        except Exception:
            # Skip if sox not available
            return audio

    def _add_noise(self, audio: Tensor) -> Tensor:
        """Add Gaussian noise at a random SNR."""
        snr_db = random.uniform(*self.noise_snr_range)
        signal_power = audio.pow(2).mean()
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(audio) * noise_power.sqrt()
        return audio + noise


class SpecAugment:
    """
    SpecAugment-style augmentation on spectrograms.

    Applies frequency and time masking to spectrograms for
    additional robustness during training.

    Args:
        freq_mask_param: Maximum number of frequency bins to mask.
        time_mask_param: Maximum number of time frames to mask.
        num_freq_masks: Number of frequency masks to apply.
        num_time_masks: Number of time masks to apply.
    """

    def __init__(
        self,
        freq_mask_param: int = 10,
        time_mask_param: int = 20,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
    ):
        self.freq_masker = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=freq_mask_param
        )
        self.time_masker = torchaudio.transforms.TimeMasking(
            time_mask_param=time_mask_param
        )
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def __call__(self, spectrogram: Tensor) -> Tensor:
        """
        Apply SpecAugment masking to spectrogram.

        Args:
            spectrogram: Shape (freq_bins, time_frames) or
                (batch, freq_bins, time_frames).

        Returns:
            Masked spectrogram.
        """
        for _ in range(self.num_freq_masks):
            spectrogram = self.freq_masker(spectrogram)
        for _ in range(self.num_time_masks):
            spectrogram = self.time_masker(spectrogram)
        return spectrogram
