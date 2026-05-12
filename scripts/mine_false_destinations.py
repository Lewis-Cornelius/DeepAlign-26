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


def mine_false_destinations(
    event_errors: pd.DataFrame,
    *,
    min_error_ms: float = 500.0,
    nms_window_sec: float = 2.0,
    max_per_pair: int | None = None,
) -> pd.DataFrame:
    missing = [column for column in FALSE_DESTINATION_COLUMNS if column not in event_errors.columns]
    if missing:
        raise ValueError("Event error CSV is missing required columns: " + ", ".join(missing))

    candidates = event_errors.copy()
    candidates = candidates[np.isfinite(candidates["abs_error_ms"].astype(float))]
    candidates = candidates[candidates["abs_error_ms"].astype(float) >= float(min_error_ms)]
    candidates = candidates.sort_values(["pair_id", "abs_error_ms"], ascending=[True, False])

    kept_rows: list[pd.Series] = []
    for _, pair_rows in candidates.groupby("pair_id", sort=True):
        kept_pred_times: list[float] = []
        kept_for_pair = 0
        for _, row in pair_rows.iterrows():
            pred_b_s = float(row["pred_b_s"])
            if any(abs(pred_b_s - kept_time) < nms_window_sec for kept_time in kept_pred_times):
                continue
            kept_rows.append(row)
            kept_pred_times.append(pred_b_s)
            kept_for_pair += 1
            if max_per_pair is not None and kept_for_pair >= max_per_pair:
                break

    if not kept_rows:
        return pd.DataFrame(columns=FALSE_DESTINATION_COLUMNS)

    mined = pd.DataFrame(kept_rows)
    mined = mined[FALSE_DESTINATION_COLUMNS].copy()
    mined["abs_error_ms"] = mined["abs_error_ms"].astype(float)
    mined = mined.sort_values(["abs_error_ms", "pair_id"], ascending=[False, True]).reset_index(drop=True)
    return mined


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mine model false-destination negatives from event error rows.")
    parser.add_argument("event_errors", help="Event-level failure CSV from report_deepalign_failures.py")
    parser.add_argument("--output", required=True, help="Output false-destination CSV")
    parser.add_argument("--min-error-ms", type=float, default=500.0)
    parser.add_argument("--nms-window-sec", type=float, default=2.0)
    parser.add_argument("--max-per-pair", type=int, default=None)
    args = parser.parse_args(argv)

    event_errors = pd.read_csv(args.event_errors)
    mined = mine_false_destinations(
        event_errors,
        min_error_ms=args.min_error_ms,
        nms_window_sec=args.nms_window_sec,
        max_per_pair=args.max_per_pair,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mined.to_csv(output_path, index=False)
    print(f"Saved {len(mined)} false destinations to {output_path}")
    if not mined.empty:
        print(mined.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
