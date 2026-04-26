"""Matchmaker wrapper for score-following baselines."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class MatchmakerResult:
    """Collected frame-wise score positions from one Matchmaker run."""

    score_positions: NDArray[np.floating]
    times_s: NDArray[np.floating]
    runtime_seconds: float
    method: str
    feature_type: str


def run_matchmaker(
    score_file: str | Path,
    performance_file: str | Path,
    *,
    input_type: str = "audio",
    method: str = "arzt",
    feature_type: str = "chroma",
    sample_rate: int = 22050,
    frame_rate: int = 100,
    **kwargs: Any,
) -> MatchmakerResult:
    """
    Run Matchmaker on one performance file and collect score positions.

    The returned positions follow Matchmaker's score coordinate system
    (beat-like score positions used by partitura).
    """
    try:
        from matchmaker import Matchmaker
    except ImportError as exc:
        raise ImportError(
            "matchmaker is required for the Matchmaker baseline. "
            "Install the official pymatchmaker package from source in a Python 3.12 environment, "
            "for example `pip install git+https://github.com/pymatchmaker/matchmaker.git@v0.2.1`."
        ) from exc

    start = time.perf_counter()
    follower = Matchmaker(
        score_file=str(score_file),
        performance_file=str(performance_file),
        input_type=input_type,
        method=method,
        feature_type=feature_type,
        sample_rate=sample_rate,
        frame_rate=frame_rate,
        **kwargs,
    )
    positions = np.asarray(list(follower.run()), dtype=np.float64)
    runtime = time.perf_counter() - start
    if positions.size == 0:
        raise RuntimeError(f"Matchmaker produced no score positions for {performance_file}")

    times = np.arange(len(positions), dtype=np.float64) / float(frame_rate)
    positions = np.maximum.accumulate(positions)
    return MatchmakerResult(
        score_positions=positions,
        times_s=times,
        runtime_seconds=runtime,
        method=method,
        feature_type=feature_type,
    )
