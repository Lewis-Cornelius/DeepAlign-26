from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

FALSE_DESTINATION_COLUMNS = [
    "pair_id",
    "group_id",
    "piece_b_id",
    "event_id",
    "gt_a_s",
    "gt_b_s",
    "pred_b_s",
    "abs_error_ms",
    "error_sign",
    "gt_b_frame",
    "pred_b_frame",
]


def _parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    label, path_text = value.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Invalid empty source label in {value!r}")
    return label, Path(path_text)


def _read_source(label: str, path: Path, *, min_error_ms: float) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = [column for column in FALSE_DESTINATION_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    frame = frame[FALSE_DESTINATION_COLUMNS].copy()
    frame["abs_error_ms"] = pd.to_numeric(frame["abs_error_ms"], errors="coerce")
    frame = frame[np.isfinite(frame["abs_error_ms"].astype(float))]
    frame = frame[frame["abs_error_ms"].astype(float) >= float(min_error_ms)]
    frame["source"] = label
    frame["_source_order"] = 0
    return frame.sort_values(["abs_error_ms", "pair_id"], ascending=[False, True]).reset_index(drop=True)


def build_replay_buffer(
    sources: Sequence[tuple[str, pd.DataFrame]],
    *,
    nms_window_sec: float = 2.0,
    max_per_pair: int | None = 2,
    max_per_lied: int | None = 6,
) -> pd.DataFrame:
    """Merge mined false destinations with source-balanced replay caps."""
    prepared: list[tuple[str, pd.DataFrame]] = []
    for source_index, (label, frame) in enumerate(sources):
        current = frame.copy()
        current["source"] = label
        current["_source_order"] = source_index
        current["_row_order"] = np.arange(len(current), dtype=int)
        prepared.append((label, current))

    kept_rows: list[pd.Series] = []
    kept_pred_times_by_pair: dict[str, list[float]] = {}
    kept_count_by_pair: dict[str, int] = {}
    kept_count_by_lied: dict[str, int] = {}
    cursors = {label: 0 for label, _ in prepared}
    exhausted: set[str] = set()

    while len(exhausted) < len(prepared):
        progressed = False
        for label, frame in prepared:
            if label in exhausted:
                continue
            cursor = cursors[label]
            accepted = False
            while cursor < len(frame):
                row = frame.iloc[cursor]
                cursor += 1
                pair_id = str(row["pair_id"])
                lied_id = str(row["group_id"])
                pred_b_s = float(row["pred_b_s"])
                if max_per_pair is not None and kept_count_by_pair.get(pair_id, 0) >= max_per_pair:
                    continue
                if max_per_lied is not None and kept_count_by_lied.get(lied_id, 0) >= max_per_lied:
                    continue
                kept_pred_times = kept_pred_times_by_pair.setdefault(pair_id, [])
                if any(abs(pred_b_s - kept_time) < nms_window_sec for kept_time in kept_pred_times):
                    continue
                kept_rows.append(row)
                kept_pred_times.append(pred_b_s)
                kept_count_by_pair[pair_id] = kept_count_by_pair.get(pair_id, 0) + 1
                kept_count_by_lied[lied_id] = kept_count_by_lied.get(lied_id, 0) + 1
                accepted = True
                progressed = True
                break
            cursors[label] = cursor
            if cursor >= len(frame):
                exhausted.add(label)
            if accepted:
                continue
        if not progressed:
            break

    if not kept_rows:
        return pd.DataFrame(columns=FALSE_DESTINATION_COLUMNS + ["source"])

    replay = pd.DataFrame(kept_rows)
    replay = replay[FALSE_DESTINATION_COLUMNS + ["source"]].copy()
    replay["abs_error_ms"] = replay["abs_error_ms"].astype(float)
    return replay.sort_values(["abs_error_ms", "pair_id"], ascending=[False, True]).reset_index(drop=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a source-balanced false-destination replay buffer.")
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Replay source as label=csv_path. Pass multiple times.",
    )
    parser.add_argument("--output", required=True, help="Output replay CSV")
    parser.add_argument("--min-error-ms", type=float, default=500.0)
    parser.add_argument("--nms-window-sec", type=float, default=2.0)
    parser.add_argument("--max-per-pair", type=int, default=2)
    parser.add_argument("--max-per-lied", type=int, default=6)
    args = parser.parse_args(argv)

    sources = []
    for source in args.source:
        label, path = _parse_source(source)
        sources.append((label, _read_source(label, path, min_error_ms=args.min_error_ms)))

    replay = build_replay_buffer(
        sources,
        nms_window_sec=args.nms_window_sec,
        max_per_pair=args.max_per_pair,
        max_per_lied=args.max_per_lied,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    replay.to_csv(output_path, index=False)
    print(f"Saved {len(replay)} replay false destinations to {output_path}")
    if not replay.empty:
        print(replay.groupby("source").size().rename("rows").to_string())
        print(replay.groupby("group_id").size().rename("rows").to_string())
        print(replay.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
