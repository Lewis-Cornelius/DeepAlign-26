"""MIDI-derived score features for score-guided synchronization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ScoreFeatureGrid:
    """Pitch-class activity and onset grids sampled in score time."""

    note: NDArray[np.float64]
    onset: NDArray[np.float64]
    onset_decay: NDArray[np.float64]
    frame_rate: float
    duration_s: float
    midi: object


def extract_midi_score_features(
    midi_path: str | Path,
    *,
    frame_rate: float,
    onset_decay_sec: float = 0.12,
) -> ScoreFeatureGrid:
    """Extract lightweight 12-bin score features from a MIDI score."""
    try:
        import pretty_midi
    except ImportError as exc:
        raise ImportError(
            "pretty_midi is required for score-guided DeepAlign decoding. "
            'Install the transcription extras with: pip install -e ".[transcription]"'
        ) from exc

    midi = pretty_midi.PrettyMIDI(str(midi_path))
    duration_s = max(float(midi.get_end_time()), 1.0 / frame_rate)
    n_frames = max(2, int(np.ceil(duration_s * frame_rate)) + 1)
    note = np.zeros((12, n_frames), dtype=np.float64)
    onset = np.zeros((12, n_frames), dtype=np.float64)

    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for midi_note in instrument.notes:
            velocity = max(float(midi_note.velocity) / 127.0, 1e-3)
            pitch_class = int(midi_note.pitch) % 12
            start = int(np.clip(np.floor(midi_note.start * frame_rate), 0, n_frames - 1))
            end = int(np.clip(np.ceil(midi_note.end * frame_rate), start + 1, n_frames))
            note[pitch_class, start:end] = np.maximum(note[pitch_class, start:end], velocity)
            onset[pitch_class, start] = max(onset[pitch_class, start], velocity)

    onset_decay = _decay_onsets(onset, max(1, int(round(onset_decay_sec * frame_rate))))
    return ScoreFeatureGrid(
        note=np.nan_to_num(note, nan=0.0, posinf=0.0, neginf=0.0),
        onset=np.nan_to_num(onset, nan=0.0, posinf=0.0, neginf=0.0),
        onset_decay=onset_decay,
        frame_rate=float(frame_rate),
        duration_s=float(duration_s),
        midi=midi,
    )


def score_position_to_time(midi: object, score_position_beats: float) -> float:
    """Convert a MusicXML quarter-beat position to MIDI seconds."""
    resolution = getattr(midi, "resolution", None)
    tick_to_time = getattr(midi, "tick_to_time", None)
    if resolution is None or tick_to_time is None:
        raise ValueError("The MIDI object does not expose resolution and tick_to_time().")
    tick = int(round(float(score_position_beats) * float(resolution)))
    return float(tick_to_time(max(0, tick)))


def _decay_onsets(onsets: NDArray[np.float64], decay_frames: int) -> NDArray[np.float64]:
    if decay_frames <= 1:
        return onsets.copy()
    weights = np.exp(-np.arange(decay_frames, dtype=np.float64) / max(decay_frames / 3.0, 1.0))
    decayed = np.zeros_like(onsets, dtype=np.float64)
    for offset, weight in enumerate(weights):
        if offset == 0:
            decayed += onsets * weight
        else:
            decayed[:, offset:] = np.maximum(decayed[:, offset:], onsets[:, :-offset] * weight)
    return decayed
