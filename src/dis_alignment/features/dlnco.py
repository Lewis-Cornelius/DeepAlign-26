"""Decay-Locked Note Cue Onset (DLNCO) feature extraction.

DLNCO features capture onset information with exponential decay,
providing temporal precision for alignment tasks.

Reference:
    S. Ewert and M. Müller, "High Resolution Audio Synchronization 
    using Chroma Onset Features," ICASSP, 2009.
"""

import numpy as np
from numpy.typing import NDArray
import librosa


def extract_dlnco(
    audio: NDArray[np.floating],
    sr: int = 22050,
    hop_length: int = 512,
    n_chroma: int = 12,
    decay_rate: float = 0.8,
    onset_threshold: float = 0.05,
) -> NDArray[np.floating]:
    """
    Extract Decay-Locked Note Cue Onset (DLNCO) features.
    
    DLNCO combines chroma information with onset detection, applying
    an exponential decay envelope to create time-localized pitch cues.
    This improves alignment precision at note boundaries.
    
    Args:
        audio: Audio time series (mono).
        sr: Sample rate of the audio.
        hop_length: Number of samples between successive frames.
        n_chroma: Number of chroma bins.
        decay_rate: Exponential decay factor per frame (0 < decay_rate < 1).
            Higher values = slower decay = more temporal smearing.
        onset_threshold: Minimum onset strength to trigger a cue.
            
    Returns:
        DLNCO features of shape (n_chroma, n_frames).
        
    Example:
        >>> audio, sr = librosa.load("piano.wav", sr=22050)
        >>> dlnco = extract_dlnco(audio, sr=sr, decay_rate=0.8)
        >>> dlnco.shape
        (12, 4321)
    """
    # Extract base chromagram
    chroma = librosa.feature.chroma_cqt(
        y=audio,
        sr=sr,
        hop_length=hop_length,
        n_chroma=n_chroma,
    )
    
    # Compute onset strength envelope
    onset_env = librosa.onset.onset_strength(
        y=audio,
        sr=sr,
        hop_length=hop_length,
    )
    
    # Compute spectral flux in each chroma band for band-specific onsets
    chroma_diff = np.diff(chroma, axis=1, prepend=0)
    chroma_onset = np.maximum(chroma_diff, 0)  # Half-wave rectification
    
    # Apply onset gating with threshold
    onset_mask = onset_env > onset_threshold
    chroma_onset[:, ~onset_mask] *= 0.1  # Reduce non-onset frames
    
    # Apply exponential decay envelope
    dlnco = _apply_decay_envelope(chroma_onset, decay_rate)
    
    # Normalize
    # Normalize using L2 norm (2 = L2)
    dlnco = librosa.util.normalize(dlnco, norm=2, axis=0)
    
    return dlnco


def _apply_decay_envelope(
    features: NDArray[np.floating],
    decay_rate: float,
) -> NDArray[np.floating]:
    """
    Apply exponential decay envelope to features.
    
    At each onset, the feature value is set, then decays exponentially
    until the next onset (which resets the envelope).
    
    Args:
        features: Input features of shape (n_features, n_frames).
        decay_rate: Decay factor per frame.
        
    Returns:
        Features with decay envelope applied.
    """
    n_features, n_frames = features.shape
    output = np.zeros_like(features)
    envelope = np.zeros(n_features)
    
    for t in range(n_frames):
        # Decay existing envelope
        envelope *= decay_rate
        
        # Update with new onsets (max operation for overlapping decays)
        envelope = np.maximum(envelope, features[:, t])
        
        output[:, t] = envelope
    
    return output


def extract_combined_features(
    audio: NDArray[np.floating],
    sr: int = 22050,
    hop_length: int = 512,
    chroma_weight: float = 0.5,
    dlnco_weight: float = 0.5,
) -> NDArray[np.floating]:
    """
    Extract combined Chroma + DLNCO features.
    
    Concatenates normalized chroma and DLNCO features for robust
    alignment that captures both harmonic content and onset precision.
    
    Args:
        audio: Audio time series.
        sr: Sample rate.
        hop_length: Hop length for feature extraction.
        chroma_weight: Weight for chroma features in combination.
        dlnco_weight: Weight for DLNCO features in combination.
        
    Returns:
        Combined features of shape (24, n_frames) - 12 chroma + 12 DLNCO.
    """
    chroma = librosa.feature.chroma_cqt(
        y=audio, sr=sr, hop_length=hop_length
    )
    chroma = librosa.util.normalize(chroma, norm=2, axis=0)
    
    dlnco = extract_dlnco(audio, sr=sr, hop_length=hop_length)
    
    # Weight and concatenate
    combined = np.vstack([
        chroma * chroma_weight,
        dlnco * dlnco_weight,
    ])
    
    return combined
