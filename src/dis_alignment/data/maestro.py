"""Legacy MAESTRO dataset helpers kept for baseline benchmarking."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from numpy.typing import NDArray


@dataclass(frozen=True)
class MAESTROPiece:
    """One MAESTRO recording / score pair."""

    piece_id: str
    split: str
    audio_path: Path
    midi_path: Path
    duration_seconds: float
    metadata: dict[str, Any]


class MAESTRODataset:
    """Lightweight metadata wrapper for legacy MAESTRO experiments."""

    def __init__(self, root: str | Path):
        self.root = _resolve_maestro_root(root)
        self.metadata_path = self.root / "maestro-v3.0.0.json"
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Could not find MAESTRO metadata at {self.metadata_path}. "
                "Point the dataset at the extracted MAESTRO root."
            )
        self.metadata = _load_maestro_metadata(self.metadata_path)

    def iter_split(self, split: str) -> Iterable[MAESTROPiece]:
        """Yield MAESTRO pieces for one split."""
        split_frame = self.metadata[self.metadata["split"] == split].reset_index(drop=True)
        for _, row in split_frame.iterrows():
            audio_path = self.root / row["audio_filename"]
            midi_path = self.root / row["midi_filename"]
            yield MAESTROPiece(
                piece_id=audio_path.stem,
                split=split,
                audio_path=audio_path,
                midi_path=midi_path,
                duration_seconds=float(row.get("duration", row.get("duration_seconds", 0.0))),
                metadata=row.to_dict(),
            )


def load_audio(
    piece_or_path: MAESTROPiece | str | Path,
    *,
    sr: int = 22050,
    mono: bool = True,
) -> tuple[NDArray[Any], int]:
    """Load a MAESTRO recording with librosa."""
    import librosa

    audio_path = piece_or_path.audio_path if isinstance(piece_or_path, MAESTROPiece) else Path(piece_or_path)
    audio, sample_rate = librosa.load(audio_path, sr=sr, mono=mono)
    return audio, sample_rate


def _resolve_maestro_root(root: str | Path) -> Path:
    candidate = Path(root)
    search_roots = [candidate, candidate / "maestro-v3.0.0"]
    for path in search_roots:
        if (path / "maestro-v3.0.0.json").exists():
            return path
    raise FileNotFoundError(
        f"Could not resolve MAESTRO root from {candidate}. "
        "Expected an extracted MAESTRO directory containing maestro-v3.0.0.json."
    )


def _load_maestro_metadata(path: Path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)

    if isinstance(raw, list):
        return pd.DataFrame(raw)

    if isinstance(raw, dict):
        return pd.DataFrame(raw)

    raise TypeError(f"Unsupported MAESTRO metadata structure in {path}")
