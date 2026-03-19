"""CQT-based chroma features for DeepAlign baselines and inspection."""

from typing import Literal

import librosa
import numpy as np
from numpy.typing import NDArray


def extract_chroma_cqt(
    audio: NDArray[np.floating],
    sr: int = 22050,
    hop_length: int = 512,
    n_chroma: int = 12,
    bins_per_octave: int = 36,
    fmin: float | None = None,
    norm: Literal["l1", "l2", "max"] | None = "l2",
    tuning: float | None = None,
) -> NDArray[np.floating]:
    """
    Extract CQT-based chromagram features from audio.
    
    Uses the Constant-Q Transform for improved harmonic resolution,
    which is more robust to timbral variations than STFT-based chroma.
    
    Args:
        audio: Audio time series (mono).
        sr: Sample rate of the audio.
        hop_length: Number of samples between successive chroma frames.
        n_chroma: Number of chroma bins (typically 12 for Western music).
        bins_per_octave: CQT resolution (36 = 3 bins per semitone).
        fmin: Minimum frequency. Defaults to C1 (~32.7 Hz).
        norm: Normalization type for each frame. None for no normalization.
        tuning: Deviation from A440 tuning in fractions of a CQT bin.
            If None, tuning is estimated automatically.

    Returns:
        Chromagram of shape (n_chroma, n_frames).
        
    Example:
        >>> audio, sr = librosa.load("piano.wav", sr=22050)
        >>> chroma = extract_chroma_cqt(audio, sr=sr)
        >>> chroma.shape
        (12, 4321)
    """
    if fmin is None:
        fmin = librosa.note_to_hz("C1")
    
    # Estimate tuning if not provided
    if tuning is None:
        tuning = librosa.estimate_tuning(y=audio, sr=sr)
    
    # Extract CQT-based chromagram
    chroma = librosa.feature.chroma_cqt(
        y=audio,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        n_chroma=n_chroma,
        bins_per_octave=bins_per_octave,
        tuning=tuning,
        norm=None,  # Apply normalization separately for flexibility
    )
    
    # Apply normalization if requested
    if norm is not None:
        # librosa.util.normalize expects int (1, 2) or np.inf, not strings
        norm_map = {"l1": 1, "l2": 2, "max": np.inf}
        norm_val = norm_map.get(norm, 2)
        chroma = librosa.util.normalize(chroma, norm=norm_val, axis=0)
    
    return chroma


def extract_chroma_from_midi(
    midi_path: str,
    sr: int = 22050,
    hop_length: int = 512,
    n_chroma: int = 12,
) -> NDArray[np.floating]:
    """
    Extract chromagram features from a MIDI file.
    
    Synthesizes the MIDI to audio using a simple sinusoidal model,
    then extracts chroma features for alignment with recorded audio.
    
    Args:
        midi_path: Path to the MIDI file.
        sr: Target sample rate for synthesis.
        hop_length: Hop length for chroma extraction.
        n_chroma: Number of chroma bins.
        
    Returns:
        Chromagram of shape (n_chroma, n_frames).
    """
    import pretty_midi
    
    # Load MIDI and synthesize to audio
    midi = pretty_midi.PrettyMIDI(midi_path)
    audio = midi.synthesize(fs=sr)
    
    # Extract chroma from synthesized audio
    return extract_chroma_cqt(
        audio=audio,
        sr=sr,
        hop_length=hop_length,
        n_chroma=n_chroma,
    )
