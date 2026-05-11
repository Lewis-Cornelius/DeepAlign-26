"""Validate strict DeepAlign gate CSVs.

This tiny helper exists because passing a multi-line Python snippet through
PowerShell and `python -c` is fragile on Windows.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("max_mae_ms", type=float)
    parser.add_argument("min_ar50_pct", type=float)
    parser.add_argument("expected_pairs", type=int)
    args = parser.parse_args(argv)

    df = pd.read_csv(args.csv_path)
    if "method" not in df:
        raise SystemExit("Missing method column")

    df = df[df["method"].astype(str).eq("deepalign")].copy()
    if df.empty:
        raise SystemExit("No deepalign rows found")

    if "deep_decode" not in df or set(df["deep_decode"].astype(str)) != {"unconstrained"}:
        raise SystemExit("Strict headline gate requires deep_decode=unconstrained only")
    if "pool_size" not in df or set(df["pool_size"].astype(float)) != {1.0}:
        raise SystemExit("Strict headline gate requires pool_size=1")

    pairs = len(df)
    mae_ms = float(df["mae"].mean()) * 1000.0
    ar50_pct = float(df["ar_50ms"].mean()) * 100.0
    print(f"pairs={pairs} MAE={mae_ms:.3f} ms AR@50={ar50_pct:.3f}%")

    if pairs != args.expected_pairs:
        raise SystemExit(f"Expected {args.expected_pairs} pairs, found {pairs}")
    if not mae_ms < args.max_mae_ms:
        raise SystemExit(
            f"MAE gate failed: {mae_ms:.3f} ms is not below {args.max_mae_ms:.3f} ms"
        )
    if not ar50_pct > args.min_ar50_pct:
        raise SystemExit(
            f"AR@50 gate failed: {ar50_pct:.3f}% is not above {args.min_ar50_pct:.3f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
