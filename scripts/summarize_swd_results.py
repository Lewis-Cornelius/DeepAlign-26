"""Summarize SWD evaluation CSVs for the timeboxed model sprint.

The project CSVs store one row per evaluated pair.  This helper keeps the
promotion criterion explicit: mean pair-level AR@50 first, then median AE/MAE.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


def _parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        return label.strip(), Path(path.strip())
    path = Path(value)
    return path.stem, path


def _mean_column(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns or df.empty:
        return float("nan")
    return float(df[column].mean())


def _weighted_ar(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns or "n_gt_points" not in df.columns or df.empty:
        return float("nan")
    weights = pd.to_numeric(df["n_gt_points"], errors="coerce").fillna(0.0)
    values = pd.to_numeric(df[column], errors="coerce")
    total = float(weights.sum())
    if total <= 0:
        return float("nan")
    return float((values * weights).sum() / total)


def summarize_csv(label: str, path: Path) -> dict[str, object]:
    row: dict[str, object] = {
        "label": label,
        "csv": str(path),
    }
    if not path.exists():
        row["status"] = "missing"
        return row

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive CLI reporting
        row["status"] = f"read_failed: {exc}"
        return row

    method_label = ""
    if "method" in df.columns:
        deepalign_rows = df[df["method"].astype(str).eq("deepalign")].copy()
        if not deepalign_rows.empty:
            df = deepalign_rows
        else:
            df = df[~df["method"].astype(str).eq("oracle")].copy()
        method_label = "|".join(sorted(str(v) for v in df["method"].dropna().unique()))
    if df.empty:
        row["status"] = "no_deepalign_rows"
        return row

    decode = ""
    if "deep_decode" in df.columns:
        decode = "|".join(sorted(str(v) for v in df["deep_decode"].dropna().unique()))

    row.update(
        {
            "status": "ok",
            "pairs": int(len(df)),
            "method": method_label,
            "deep_decode": decode,
            "mae_ms": round(_mean_column(df, "mae") * 1000.0, 3),
            "median_ae_ms": round(_mean_column(df, "median_ae") * 1000.0, 3),
            "ar_50_pct": round(_mean_column(df, "ar_50ms") * 100.0, 3),
            "ar_100_pct": round(_mean_column(df, "ar_100ms") * 100.0, 3),
            "ar_200_pct": round(_mean_column(df, "ar_200ms") * 100.0, 3),
            "weighted_ar_50_pct": round(_weighted_ar(df, "ar_50ms") * 100.0, 3),
            "weighted_ar_100_pct": round(_weighted_ar(df, "ar_100ms") * 100.0, 3),
            "weighted_ar_200_pct": round(_weighted_ar(df, "ar_200ms") * 100.0, 3),
            "runtime_s": round(_mean_column(df, "runtime_s"), 3),
            "n_gt_points": int(df["n_gt_points"].sum()) if "n_gt_points" in df.columns else "",
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csvs", nargs="*", help="CSV paths, optionally label=path")
    parser.add_argument("--results-dir", type=Path, help="Add all CSVs in this directory")
    parser.add_argument("--reference", action="append", default=[], help="Reference label=path")
    parser.add_argument("--output", type=Path, required=True, help="Summary CSV path")
    args = parser.parse_args()

    labeled_paths: list[tuple[str, Path]] = []
    labeled_paths.extend(_parse_labeled_path(item) for item in args.reference)
    labeled_paths.extend(_parse_labeled_path(item) for item in args.csvs)
    if args.results_dir and args.results_dir.exists():
        labeled_paths.extend((path.stem, path) for path in sorted(args.results_dir.glob("*.csv")))

    seen: set[Path] = set()
    rows = []
    for label, path in labeled_paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        rows.append(summarize_csv(label, path))

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    other_rows = [row for row in rows if row.get("status") != "ok"]
    ok_rows.sort(
        key=lambda row: (
            -float(row["ar_50_pct"]),
            float(row["median_ae_ms"]),
            float(row["mae_ms"]),
        )
    )
    rows = ok_rows + other_rows

    fieldnames = [
        "label",
        "status",
        "pairs",
        "method",
        "deep_decode",
        "mae_ms",
        "median_ae_ms",
        "ar_50_pct",
        "ar_100_pct",
        "ar_200_pct",
        "weighted_ar_50_pct",
        "weighted_ar_100_pct",
        "weighted_ar_200_pct",
        "runtime_s",
        "n_gt_points",
        "csv",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Summary written: {args.output}")
    for row in ok_rows[:8]:
        print(
            f"{row['label']}: AR@50={row['ar_50_pct']}% "
            f"median={row['median_ae_ms']}ms MAE={row['mae_ms']}ms "
            f"pairs={row['pairs']}"
        )


if __name__ == "__main__":
    main()
