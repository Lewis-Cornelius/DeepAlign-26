"""Schubert Winterreise Dataset helpers for DeepAlign-26."""

from __future__ import annotations

import math
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from functools import cached_property
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

SWD_URL = "https://zenodo.org/records/5139893/files/Schubert_Winterreise_Dataset_v2-0.zip"
EXPECTED_SIZE_MB = 506

_LIED_RE = re.compile(r"(D911-\d{2})", re.IGNORECASE)


@dataclass(frozen=True)
class SWDPiece:
    """One Winterreise recording with optional annotation metadata."""

    piece_id: str
    lied_id: str
    performance_id: str
    audio_path: Path
    annotation_path: Path | None = None
    title: str | None = None


@dataclass(frozen=True)
class SWDPair:
    """A pair of recordings of the same lied."""

    pair_id: str
    lied_id: str
    piece_a: SWDPiece
    piece_b: SWDPiece


class SWDDataset:
    """Access the Schubert Winterreise Dataset with pairing helpers."""

    def __init__(self, root: str | Path):
        self.root = _resolve_swd_root(root)
        self.audio_dir = self.root / "01_RawData" / "audio_wav"
        self.score_musicxml_dir = self.root / "01_RawData" / "score_musicxml"
        self.score_midi_dir = self.root / "01_RawData" / "score_midi"
        self.annotation_dir = self.root / "02_Annotations" / "ann_audio_measure"

        if not self.audio_dir.exists():
            raise FileNotFoundError(
                f"Could not find SWD audio at {self.audio_dir}. "
                "Point the dataset at the extracted SWD root or its parent directory."
            )

    @cached_property
    def pieces(self) -> list[SWDPiece]:
        """All discovered recordings in the dataset."""
        pieces = [
            _build_piece(audio_path, self.annotation_dir)
            for audio_path in sorted(self.audio_dir.glob("*.wav"))
        ]
        if not pieces:
            raise FileNotFoundError(f"No SWD .wav files found under {self.audio_dir}")
        return pieces

    @cached_property
    def available_performances(self) -> list[str]:
        """Sorted list of discovered performance identifiers."""
        return sorted({piece.performance_id for piece in self.pieces})

    @cached_property
    def available_lieder(self) -> list[str]:
        """Sorted list of discovered Winterreise lied identifiers."""
        return sorted({piece.lied_id for piece in self.pieces})

    def iter_pieces(
        self,
        performances: Iterable[str] | None = None,
        lieder: Iterable[str] | None = None,
    ) -> Iterable[SWDPiece]:
        """Yield pieces filtered by performance ids and/or lied ids."""
        perf_filter = set(performances) if performances is not None else None
        lied_filter = {_normalise_lied_id(lied) for lied in lieder} if lieder is not None else None

        for piece in self.pieces:
            if perf_filter is not None and piece.performance_id not in perf_filter:
                continue
            if lied_filter is not None and piece.lied_id not in lied_filter:
                continue
            yield piece

    def iter_pairs(
        self,
        performances: Iterable[str] | None = None,
        lieder: Iterable[str] | None = None,
    ) -> Iterable[SWDPair]:
        """Yield all pairwise combinations of recordings for each lied."""
        by_lied: dict[str, list[SWDPiece]] = {}
        for piece in self.iter_pieces(performances=performances, lieder=lieder):
            by_lied.setdefault(piece.lied_id, []).append(piece)

        for lied_id in sorted(by_lied):
            lied_pieces = sorted(by_lied[lied_id], key=lambda piece: piece.performance_id)
            for piece_a, piece_b in combinations(lied_pieces, 2):
                yield SWDPair(
                    pair_id=f"{lied_id}_{piece_a.performance_id}_{piece_b.performance_id}",
                    lied_id=lied_id,
                    piece_a=piece_a,
                    piece_b=piece_b,
                )

    def get_score_path(self, lied_id: str, *, prefer: str = "musicxml") -> Path | None:
        """Resolve the reference score file for one lied."""
        candidates: list[Path] = []
        search_dirs = []
        if prefer == "musicxml":
            search_dirs.extend([self.score_musicxml_dir, self.score_midi_dir])
        else:
            search_dirs.extend([self.score_midi_dir, self.score_musicxml_dir])

        lied_key = _normalise_lied_id(lied_id)
        for directory in search_dirs:
            if not directory.exists():
                continue
            candidates.extend(
                sorted(path for path in directory.iterdir() if path.is_file() and lied_key in path.stem.upper())
            )
        return candidates[0] if candidates else None


def download_swd_dataset(
    output_dir: str | Path,
    *,
    cleanup_zip: bool = False,
    reporthook: Callable[[int, int, int], None] | None = None,
) -> Path:
    """Download and extract the public SWD archive if needed."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    zip_path = output / "Schubert_Winterreise_Dataset_v2-0.zip"
    extracted_path = output / "Schubert_Winterreise_Dataset_v2-0"

    if (extracted_path / "01_RawData" / "audio_wav").exists():
        return extracted_path

    if not zip_path.exists():
        urllib.request.urlretrieve(SWD_URL, zip_path, reporthook=reporthook)

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(output)

    if cleanup_zip and zip_path.exists():
        zip_path.unlink()

    return extracted_path


def verify_swd_dataset(path: str | Path) -> dict[str, Any]:
    """Return a lightweight integrity summary for an SWD directory."""
    dataset = SWDDataset(path)
    wav_files = sorted(dataset.audio_dir.glob("*.wav"))
    annotation_files = (
        sorted(dataset.annotation_dir.glob("*.csv")) if dataset.annotation_dir.exists() else []
    )

    performance_counts: dict[str, int] = {}
    for piece in dataset.pieces:
        performance_counts[piece.performance_id] = performance_counts.get(piece.performance_id, 0) + 1

    pair_count = sum(1 for _ in dataset.iter_pairs())

    return {
        "root": str(dataset.root),
        "audio_files": len(wav_files),
        "annotations": len(annotation_files),
        "performances": dict(sorted(performance_counts.items())),
        "lieder": len(dataset.available_lieder),
        "pairs": pair_count,
    }


def load_swd_audio(
    piece_or_path: SWDPiece | str | Path,
    *,
    sr: int = 22050,
    mono: bool = True,
) -> tuple[NDArray[np.floating], int]:
    """Load an SWD waveform via librosa."""
    import librosa

    audio_path = piece_or_path.audio_path if isinstance(piece_or_path, SWDPiece) else Path(piece_or_path)
    audio, sample_rate = librosa.load(audio_path, sr=sr, mono=mono)
    return audio, sample_rate


def compute_ground_truth_alignment(
    pair: SWDPair,
) -> tuple[NDArray[np.floating], NDArray[np.floating]] | None:
    """Build measure-level correspondence targets for one SWD pair."""
    annotations_a = load_swd_measure_annotations(pair.piece_a)
    annotations_b = load_swd_measure_annotations(pair.piece_b)
    if annotations_a is None or annotations_b is None:
        return None

    lookup_b = dict(zip(annotations_b["event_id"], annotations_b["time_s"], strict=False))
    aligned_a = annotations_a[annotations_a["event_id"].isin(lookup_b)].copy()
    if aligned_a.empty:
        return None

    times_a = aligned_a["time_s"].to_numpy(dtype=float)
    times_b = np.array([lookup_b[event_id] for event_id in aligned_a["event_id"]], dtype=float)

    filtered_a: list[float] = []
    filtered_b: list[float] = []
    last_a = -math.inf
    last_b = -math.inf
    for time_a, time_b in zip(times_a, times_b, strict=False):
        if time_a >= last_a and time_b >= last_b:
            filtered_a.append(time_a)
            filtered_b.append(time_b)
            last_a = time_a
            last_b = time_b

    if len(filtered_a) < 2:
        return None

    return np.asarray(filtered_a, dtype=float), np.asarray(filtered_b, dtype=float)


def compute_ground_truth_measure_alignment(pair: SWDPair) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Return aligned measure ids with measure times for both pieces."""
    annotations_a = load_swd_measure_annotations(pair.piece_a)
    annotations_b = load_swd_measure_annotations(pair.piece_b)
    if annotations_a is None or annotations_b is None:
        return None

    lookup_b = dict(zip(annotations_b["event_id"], annotations_b["time_s"], strict=False))
    aligned_a = annotations_a[annotations_a["event_id"].isin(lookup_b)].copy()
    if aligned_a.empty:
        return None

    aligned_a["other_time_s"] = [lookup_b[event_id] for event_id in aligned_a["event_id"]]
    aligned_a = aligned_a.rename(columns={"time_s": "time_a_s", "other_time_s": "time_b_s"})

    filtered_records: list[tuple[str, float, float]] = []
    last_a = -math.inf
    last_b = -math.inf
    for event_id, time_a, time_b in aligned_a[["event_id", "time_a_s", "time_b_s"]].itertuples(index=False):
        if time_a >= last_a and time_b >= last_b:
            filtered_records.append((event_id, float(time_a), float(time_b)))
            last_a = float(time_a)
            last_b = float(time_b)

    if len(filtered_records) < 2:
        return None

    pair_frame = pd.DataFrame(filtered_records, columns=["event_id", "time_a_s", "time_b_s"])
    audio_a = pair_frame.rename(columns={"time_a_s": "time_s"})[["event_id", "time_s"]]
    audio_b = pair_frame.rename(columns={"time_b_s": "time_s"})[["event_id", "time_s"]]
    return audio_a, audio_b


def load_swd_measure_annotations(piece: SWDPiece) -> pd.DataFrame | None:
    """Load normalized SWD measure annotations with stable event ids."""
    frame = _load_measure_annotations(piece)
    if frame is None:
        return None
    return frame.rename(columns={"measure_id": "event_id"})[["event_id", "time_s"]]


def _resolve_swd_root(root: str | Path) -> Path:
    candidate = Path(root)
    search_roots = [
        candidate,
        candidate / "Schubert_Winterreise_Dataset_v2-0",
        candidate / "Schubert_Winterreise_Dataset_v1-0",
    ]

    for path in search_roots:
        if (path / "01_RawData" / "audio_wav").exists():
            return path

    raise FileNotFoundError(
        f"Could not resolve SWD root from {candidate}. "
        "Expected an extracted SWD directory containing 01_RawData/audio_wav."
    )


def _build_piece(audio_path: Path, annotation_dir: Path) -> SWDPiece:
    lied_id = _extract_lied_id(audio_path.stem)
    performance_id = audio_path.stem.split("_")[-1]
    title = _extract_title(audio_path.stem, lied_id, performance_id)
    annotation_path = _find_annotation_path(audio_path.stem, lied_id, performance_id, annotation_dir)
    return SWDPiece(
        piece_id=audio_path.stem,
        lied_id=lied_id,
        performance_id=performance_id,
        audio_path=audio_path,
        annotation_path=annotation_path,
        title=title,
    )


def _extract_lied_id(stem: str) -> str:
    match = _LIED_RE.search(stem)
    if not match:
        raise ValueError(f"Could not parse SWD lied id from {stem}")
    return _normalise_lied_id(match.group(1))


def _normalise_lied_id(value: str | int) -> str:
    if isinstance(value, int):
        return f"D911-{value:02d}"

    text = str(value).strip().upper()
    if text.startswith("D911-") and len(text) == 7:
        return text
    if text.isdigit():
        return f"D911-{int(text):02d}"
    return text


def _extract_title(stem: str, lied_id: str, performance_id: str) -> str | None:
    parts = stem.split("_")
    filtered = [part for part in parts if part not in {lied_id, performance_id}]
    return " ".join(filtered) if filtered else None


def _find_annotation_path(
    stem: str,
    lied_id: str,
    performance_id: str,
    annotation_dir: Path,
) -> Path | None:
    if not annotation_dir.exists():
        return None

    exact_candidates = [
        annotation_dir / f"{stem}.csv",
        annotation_dir / f"{stem}_measure.csv",
        annotation_dir / f"{stem}_measures.csv",
    ]
    for candidate in exact_candidates:
        if candidate.exists():
            return candidate

    loose_candidates = sorted(
        path
        for path in annotation_dir.glob("*.csv")
        if lied_id in path.stem.upper() and performance_id.upper() in path.stem.upper()
    )
    if len(loose_candidates) == 1:
        return loose_candidates[0]

    return None


def _load_measure_annotations(piece: SWDPiece) -> pd.DataFrame | None:
    if piece.annotation_path is None or not piece.annotation_path.exists():
        return None

    frame = _read_annotation_csv(piece.annotation_path)
    if frame.empty:
        return None

    normalised = {_normalise_column_name(column): column for column in frame.columns}
    measure_col = _pick_column(
        normalised,
        [
            "measure",
            "measure_number",
            "measure_no",
            "measure_idx",
            "measure_index",
            "bar",
            "bar_number",
            "bar_no",
            "mn",
        ],
    )
    time_col = _pick_column(
        normalised,
        [
            "start_sec",
            "start_seconds",
            "start_time",
            "time_sec",
            "time_seconds",
            "timestamp",
            "time",
            "sec",
            "seconds",
            "start",
            "onset",
        ],
    )

    if time_col is None:
        return None

    if measure_col is None:
        frame = frame.copy()
        frame["_measure_id"] = [str(index + 1) for index in range(len(frame))]
        measure_col = "_measure_id"

    records: list[tuple[str, float]] = []
    seen: set[str] = set()
    for _, row in frame.iterrows():
        measure_id = _normalise_measure_id(row.get(measure_col))
        time_value = _coerce_time_seconds(row.get(time_col))
        if measure_id is None or time_value is None:
            continue
        if measure_id in seen:
            continue
        seen.add(measure_id)
        records.append((measure_id, time_value))

    if not records:
        return None

    annotation_frame = pd.DataFrame(records, columns=["measure_id", "time_s"])
    annotation_frame = annotation_frame.sort_values("time_s", kind="stable").reset_index(drop=True)
    return annotation_frame


def _read_annotation_csv(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
        if frame.shape[1] > 1:
            return frame
    except Exception:
        pass

    return pd.read_csv(path, sep=None, engine="python")


def _pick_column(columns: dict[str, str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]
    return None


def _normalise_column_name(column: Any) -> str:
    text = str(column).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _normalise_measure_id(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None

    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float) and float(value).is_integer():
        return str(int(value))

    text = str(value).strip()
    return text or None


def _coerce_time_seconds(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None

    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(seconds):
        return None
    return seconds
