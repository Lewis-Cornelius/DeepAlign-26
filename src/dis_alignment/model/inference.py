"""Inference helpers for extracting and aligning DeepAlign-26 features."""

import logging
from pathlib import Path

import librosa
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from dis_alignment.alignment.baseline_dtw import AlignmentResult, align_global_dtw
from dis_alignment.model.encoder import CRNNEncoder

logger = logging.getLogger(__name__)


def load_trained_encoder(
    checkpoint_path: str | Path,
    device: str | None = None,
) -> tuple[CRNNEncoder, dict]:
    """
    Load a trained CRNN encoder from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        device: Device to load model to.

    Returns:
        Tuple of (encoder, config_dict).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    encoder = CRNNEncoder(
        n_freq_bins=config.get("n_freq_bins", 84),
        embed_dim=config.get("embed_dim", 64),
        num_conv_channels=config.get("num_conv_channels"),
        gru_hidden_size=config.get("gru_hidden_size", 128),
        num_gru_layers=config.get("num_gru_layers", 2),
        dropout=config.get("dropout", 0.1),
    )

    # Load weights
    if "encoder_state_dict" in checkpoint:
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
    elif "model_state_dict" in checkpoint:
        # Extract encoder weights from full model
        state = {
            k.replace("encoder.", ""): v
            for k, v in checkpoint["model_state_dict"].items()
            if k.startswith("encoder.")
        }
        encoder.load_state_dict(state)

    encoder.to(device)
    encoder.eval()

    logger.info(f"Loaded encoder from {checkpoint_path} (epoch {checkpoint.get('epoch', '?')})")
    return encoder, config


def extract_deep_features(
    audio: NDArray[np.floating],
    encoder: CRNNEncoder,
    sr: int = 22050,
    hop_length: int = 220,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    device: str | None = None,
) -> NDArray[np.floating]:
    """
    Extract learned features from audio using a trained encoder.

    Args:
        audio: Audio time series (mono).
        encoder: Trained CRNNEncoder.
        sr: Sample rate.
        hop_length: CQT hop length (~10ms at 22050 Hz).
        n_bins: Number of CQT bins.
        bins_per_octave: CQT resolution.
        device: Compute device.

    Returns:
        Learned feature embeddings of shape (embed_dim, n_frames).
    """
    if device is None:
        device = next(encoder.parameters()).device

    # Compute CQT spectrogram
    cqt = librosa.cqt(
        y=audio.astype(np.float32),
        sr=sr,
        hop_length=hop_length,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
    )
    cqt_mag = np.abs(cqt)
    cqt_log = librosa.amplitude_to_db(cqt_mag, ref=np.max)
    
    # Standardize to zero-mean, unit-variance (matches InstanceNorm in training)
    cqt_log = (cqt_log - cqt_log.mean()) / (cqt_log.std() + 1e-8)

    # Convert to tensor: (1, 1, freq, time)
    spec = torch.from_numpy(cqt_log).float().unsqueeze(0).unsqueeze(0).to(device)

    # Extract features
    with torch.no_grad():
        embeddings = encoder.extract_features(spec)  # (1, time, embed_dim)

    # Convert to (embed_dim, time) format for compatibility with DTW code
    features = embeddings.squeeze(0).T.cpu().numpy()  # (embed_dim, time)

    return features


def align_with_deep_features(
    audio_a: NDArray[np.floating],
    audio_b: NDArray[np.floating],
    encoder: CRNNEncoder,
    sr: int = 22050,
    hop_length: int = 220,
    distance: str = "cosine",
    device: str | None = None,
) -> AlignmentResult:
    """
    Align two audio recordings using learned features + classical DTW.

    This is the full DeepAlign inference pipeline:
    1. Extract deep features from both recordings
    2. Run classical DTW on the learned features

    Args:
        audio_a: First audio recording.
        audio_b: Second audio recording.
        encoder: Trained CRNNEncoder.
        sr: Sample rate.
        hop_length: CQT hop length.
        distance: Distance metric for DTW.
        device: Compute device.

    Returns:
        AlignmentResult with warping path and metrics.
    """
    import time as time_mod

    start = time_mod.perf_counter()

    # Extract deep features
    features_a = extract_deep_features(
        audio_a, encoder, sr=sr, hop_length=hop_length, device=device
    )
    features_b = extract_deep_features(
        audio_b, encoder, sr=sr, hop_length=hop_length, device=device
    )

    # Run classical DTW on learned features
    result = align_global_dtw(features_a, features_b, distance=distance)

    # Override runtime to include feature extraction
    total_time = time_mod.perf_counter() - start
    result.runtime_seconds = total_time
    result.algorithm = "deepalign"

    return result
