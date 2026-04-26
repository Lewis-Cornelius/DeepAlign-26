"""MazurkaBL dataset helpers for pairwise robustness evaluation."""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from functools import cached_property
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from numpy.typing import NDArray

_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}
_SCORE_EXTS = {".musicxml", ".xml", ".mxl", ".mid", ".midi"}
_ANNOTATION_EXTS = {".csv", ".tsv", ".txt"}


@dataclass(frozen=True)
class MazurkaPerformance:
    """One MazurkaBL performance with aligned beat annotations."""

    performance_id: str
    work_id: str
    audio_path: Path
    annotation_path: Path | None
    score_path: Path | None = None


@dataclass(frozen=True)
class MazurkaPair:
    """A pair of performances from the same work."""

    pair_id: str
    work_id: str
    piece_a: MazurkaPerformance
    piece_b: MazurkaPerformance


class MazurkaDataset:
    """Flexible loader for MazurkaBL-style pairwise evaluation."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"Could not find Mazurka root at {self.root}")

    @cached_property
    def performances(self) -> list[MazurkaPerformance]:
        metadata_records = _load_metadata_records(self.root)
        if metadata_records:
            performances = [_build_performance_from_metadata(self.root, record) for record in metadata_records]
        else:
            performances = _discover_performances(self.root)

        if not performances:
            raise FileNotFoundError(
                "Could not discover any Mazurka performances. "
                "Provide metadata.csv or a layout containing audio, annotation, and score files."
            )
        return performances

    @cached_property
    def available_works(self) -> list[str]:
        """Sorted list of discovered work ids."""
        return sorted({performance.work_id for performance in self.performances})

    def iter_performances(self, works: Iterable[str] | None = None) -> Iterable[MazurkaPerformance]:
        """Yield performances filtered by work ids."""
        work_filter = {str(work).strip() for work in works} if works is not None else None
        for performance in self.performances:
            if work_filter is not None and performance.work_id not in work_filter:
                continue
            yield performance

    def iter_pairs(self, works: Iterable[str] | None = None) -> Iterable[MazurkaPair]:
        """Yield all pairwise performance combinations for each work."""
        by_work: dict[str, list[MazurkaPerformance]] = {}
        for performance in self.iter_performances(works=works):
            by_work.setdefault(performance.work_id, []).append(performance)

        for work_id in sorted(by_work):
            performances = sorted(by_work[work_id], key=lambda performance: performance.performance_id)
            for piece_a, piece_b in combinations(performances, 2):
                yield MazurkaPair(
                    pair_id=f"{work_id}_{piece_a.performance_id}_{piece_b.performance_id}",
                    work_id=work_id,
                    piece_a=piece_a,
                    piece_b=piece_b,
                )


def load_mazurka_audio(
    performance_or_path: MazurkaPerformance | str | Path,
    *,
    sr: int = 22050,
    mono: bool = True,
) -> tuple[NDArray[Any], int]:
    """Load a Mazurka performance waveform via librosa."""
    import librosa

    audio_path = (
        performance_or_path.audio_path
        if isinstance(performance_or_path, MazurkaPerformance)
        else Path(performance_or_path)
    )
    audio, sample_rate = librosa.load(audio_path, sr=sr, mono=mono)
    return audio, sample_rate


def compute_ground_truth_alignment(
    pair: MazurkaPair,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Build beat-level correspondence targets for one Mazurka pair."""
    annotations_a = load_mazurka_annotations(pair.piece_a)
    annotations_b = load_mazurka_annotations(pair.piece_b)
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


def load_mazurka_annotations(performance: MazurkaPerformance) -> pd.DataFrame | None:
    """Load beat annotations for one performance."""
    path = performance.annotation_path
    if path is None or not path.exists():
        return None

    frame = _read_annotation_table(path)
    if frame.empty:
        return None

    normalised = {_normalise_column_name(column): column for column in frame.columns}
    event_col = _pick_column(
        normalised,
        ["beat", "beat_id", "beat_number", "beat_index", "onset_id", "event", "event_id"],
    )
    time_col = _pick_column(
        normalised,
        ["time", "time_s", "time_sec", "seconds", "sec", "onset", "onset_sec", "timestamp"],
    )

    if time_col is None:
        return None

    if event_col is None:
        frame = frame.copy()
        frame["_event_id"] = [str(index + 1) for index in range(len(frame))]
        event_col = "_event_id"

    records: list[tuple[str, float]] = []
    seen: set[str] = set()
    for _, row in frame.iterrows():
        event_id = _normalise_event_id(row.get(event_col))
        time_value = _coerce_time_seconds(row.get(time_col))
        if event_id is None or time_value is None or event_id in seen:
            continue
        seen.add(event_id)
        records.append((event_id, time_value))

    if not records:
        return None

    annotation_frame = pd.DataFrame(records, columns=["event_id", "time_s"])
    annotation_frame = annotation_frame.sort_values("time_s", kind="stable").reset_index(drop=True)
    return annotation_frame


def verify_mazurka_dataset(path: str | Path) -> dict[str, Any]:
    """Return a lightweight integrity summary for a Mazurka directory."""
    dataset = MazurkaDataset(path)
    pair_count = sum(1 for _ in dataset.iter_pairs())
    with_scores = sum(1 for performance in dataset.performances if performance.score_path is not None)
    with_annotations = sum(
        1 for performance in dataset.performances if performance.annotation_path is not None
    )
    return {
        "root": str(dataset.root),
        "performances": len(dataset.performances),
        "works": len(dataset.available_works),
        "pairs": pair_count,
        "performances_with_scores": with_scores,
        "performances_with_annotations": with_annotations,
    }


def _load_metadata_records(root: Path) -> list[dict[str, str]]:
    for metadata_name in ("metadata.csv", "mazurka_metadata.csv"):
        metadata_path = root / metadata_name
        if not metadata_path.exists():
            continue
        with open(metadata_path, encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    return []


def _build_performance_from_metadata(root: Path, record: dict[str, str]) -> MazurkaPerformance:
    lowered = {key.lower(): value for key, value in record.items()}
    work_id = (
        lowered.get("work_id")
        or lowered.get("piece_id")
        or lowered.get("mazurka_id")
        or lowered.get("score_id")
    )
    performance_id = lowered.get("performance_id") or lowered.get("recording_id") or lowered.get("take_id")
    audio_rel = lowered.get("audio_path") or lowered.get("performance_path")
    annotation_rel = lowered.get("annotation_path") or lowered.get("beat_path")
    score_rel = lowered.get("score_path") or lowered.get("musicxml_path") or lowered.get("midi_path")

    if not work_id or not performance_id or not audio_rel:
        raise ValueError(
            "Mazurka metadata rows must contain work_id/performance_id/audio_path or equivalent aliases."
        )

    audio_path = (root / audio_rel).resolve()
    annotation_path = (root / annotation_rel).resolve() if annotation_rel else None
    score_path = (root / score_rel).resolve() if score_rel else None
    return MazurkaPerformance(
        performance_id=str(performance_id),
        work_id=str(work_id),
        audio_path=audio_path,
        annotation_path=annotation_path,
        score_path=score_path,
    )


def _discover_performances(root: Path) -> list[MazurkaPerformance]:
    audio_files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in _AUDIO_EXTS
    ]
    annotation_files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in _ANNOTATION_EXTS
    ]
    score_files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in _SCORE_EXTS
    ]

    performances: list[MazurkaPerformance] = []
    for audio_path in sorted(audio_files):
        work_id, performance_id = _infer_ids(audio_path)
        annotation_path = _match_annotation(audio_path, annotation_files)
        score_path = _match_score(work_id, score_files)
        performances.append(
            MazurkaPerformance(
                performance_id=performance_id,
                work_id=work_id,
                audio_path=audio_path,
                annotation_path=annotation_path,
                score_path=score_path,
            )
        )
    return performances


def _infer_ids(audio_path: Path) -> tuple[str, str]:
    stem = audio_path.stem
    parent = audio_path.parent.name
    if parent and parent.lower() not in {"audio", "performances", "wav"} and parent != audio_path.anchor:
        work_id = parent
    else:
        tokens = re.split(r"[_\-]", stem)
        work_id = "_".join(tokens[:2]) if len(tokens) >= 2 else stem
    performance_id = stem
    return work_id, performance_id


def _match_annotation(audio_path: Path, annotation_files: list[Path]) -> Path | None:
    candidates = []
    stem = audio_path.stem.lower()
    for path in annotation_files:
        path_stem = path.stem.lower()
        if stem == path_stem or stem in path_stem or path_stem in stem:
            candidates.append(path)
    if len(candidates) == 1:
        return candidates[0]
    exact_suffixes = ["_beats", "_beat", ".beats", ".beat"]
    for suffix in exact_suffixes:
        for path in annotation_files:
            if path.stem.lower() == f"{stem}{suffix}":
                return path
    return sorted(candidates)[0] if candidates else None


def _match_score(work_id: str, score_files: list[Path]) -> Path | None:
    work_key = work_id.lower()
    candidates = [path for path in score_files if work_key in path.stem.lower()]
    return sorted(candidates)[0] if candidates else None


def _read_annotation_table(path: Path) -> pd.DataFrame:
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


def _normalise_event_id(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer():
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
