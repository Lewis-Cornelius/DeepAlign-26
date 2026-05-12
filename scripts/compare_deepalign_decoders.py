from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


IDENTITY_COLUMNS = ("pair_id", "group_id", "piece_a_id", "piece_b_id")
METRIC_SPECS = {
    "mae": ("mae_ms", 1000.0),
    "median_ae": ("median_ae_ms", 1000.0),
    "ar_50ms": ("ar50_pct", 100.0),
    "ar_100ms": ("ar100_pct", 100.0),
    "ar_200ms": ("ar200_pct", 100.0),
}
NON_DIAGNOSTIC_COLUMNS = {
    "dataset",
    "method",
    "deep_decode",
    "duration_s",
    "runtime_s",
    "memory_mb",
    "n_gt_points",
    *IDENTITY_COLUMNS,
    *METRIC_SPECS,
}


def _load_deep_rows(path: str | Path, *, deep_decode: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "method" not in frame:
        raise ValueError(f"{path} has no method column.")
    rows = frame[frame["method"].astype(str).eq("deepalign")].copy()
    if "deep_decode" not in rows:
        raise ValueError(f"{path} has no deep_decode column.")
    rows = rows[rows["deep_decode"].astype(str).eq(deep_decode)].copy()
    if rows.empty:
        raise ValueError(f"No deepalign rows with deep_decode={deep_decode!r} found in {path}.")
    return rows


def _rename_for_label(rows: pd.DataFrame, *, label: str) -> pd.DataFrame:
    missing_identity = [column for column in ("pair_id", "group_id") if column not in rows]
    if missing_identity:
        raise ValueError("Missing required identity columns: " + ", ".join(missing_identity))

    selected = rows[[column for column in IDENTITY_COLUMNS if column in rows]].copy()
    if "deep_decode" in rows:
        selected[f"deep_decode_{label}"] = rows["deep_decode"].astype(str).to_numpy()
    if "runtime_s" in rows:
        selected[f"runtime_s_{label}"] = rows["runtime_s"].to_numpy(dtype=float)

    for source, (target, scale) in METRIC_SPECS.items():
        if source in rows:
            selected[f"{target}_{label}"] = rows[source].to_numpy(dtype=float) * scale

    for column in rows.columns:
        if column in NON_DIAGNOSTIC_COLUMNS:
            continue
        if not pd.api.types.is_numeric_dtype(rows[column]):
            continue
        selected[f"{column}_{label}"] = rows[column].to_numpy(dtype=float)
    return selected


def _add_canonical_selector_columns(
    comparison: pd.DataFrame,
    *,
    unconstrained_label: str,
    candidate_label: str,
) -> pd.DataFrame:
    result = comparison.copy()
    aliases = {
        "unconstrained_cost_per_step": f"deep_path_cost_per_step_{unconstrained_label}",
        "candidate_cost_per_step": f"deep_path_cost_per_step_{candidate_label}",
        "ctf_cost_per_step": f"deep_path_cost_per_step_{candidate_label}",
        "ctf_band_edge_fraction": f"ctf_band_edge_fraction_{candidate_label}",
        "ctf_band_lower_edge_fraction": f"ctf_band_lower_edge_fraction_{candidate_label}",
        "ctf_band_upper_edge_fraction": f"ctf_band_upper_edge_fraction_{candidate_label}",
        "ctf_band_width_mean": f"ctf_band_width_mean_{candidate_label}",
        "ctf_coarse_large_jump_count": f"ctf_coarse_large_jump_count_{candidate_label}",
        "ctf_coarse_slope_std": f"ctf_coarse_slope_std_{candidate_label}",
        "ctf_coarse_endpoint_strain_frames": f"ctf_coarse_endpoint_strain_frames_{candidate_label}",
    }
    for target, source in aliases.items():
        if source in result and target not in result:
            result[target] = result[source]
    return result


def _event_disagreement(
    unconstrained_events: str | Path,
    candidate_events: str | Path,
) -> pd.DataFrame:
    left = pd.read_csv(unconstrained_events)
    right = pd.read_csv(candidate_events)
    required = {"pair_id", "event_id", "pred_b_s"}
    for path, frame in ((unconstrained_events, left), (candidate_events, right)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing required event columns: {', '.join(missing)}")

    merged = left[["pair_id", "event_id", "pred_b_s"]].merge(
        right[["pair_id", "event_id", "pred_b_s"]],
        on=["pair_id", "event_id"],
        suffixes=("_unconstrained", "_candidate"),
    )
    if merged.empty:
        return pd.DataFrame(
            columns=[
                "pair_id",
                "decoder_disagreement_mean_ms",
                "decoder_disagreement_median_ms",
                "decoder_disagreement_max_ms",
            ]
        )
    merged["abs_disagreement_ms"] = (
        merged["pred_b_s_candidate"].to_numpy(dtype=float)
        - merged["pred_b_s_unconstrained"].to_numpy(dtype=float)
    )
    merged["abs_disagreement_ms"] = merged["abs_disagreement_ms"].abs() * 1000.0
    return (
        merged.groupby("pair_id", as_index=False)["abs_disagreement_ms"]
        .agg(
            decoder_disagreement_mean_ms="mean",
            decoder_disagreement_median_ms="median",
            decoder_disagreement_max_ms="max",
        )
        .reset_index(drop=True)
    )


def build_decoder_comparison(
    *,
    unconstrained_csv: str | Path,
    candidate_csv: str | Path,
    unconstrained_decode: str = "unconstrained",
    candidate_decode: str = "deepalign_coarse_to_fine",
    unconstrained_label: str = "unconstrained",
    candidate_label: str = "coarse_to_fine",
    unconstrained_events: str | Path | None = None,
    candidate_events: str | Path | None = None,
) -> pd.DataFrame:
    unconstrained = _rename_for_label(
        _load_deep_rows(unconstrained_csv, deep_decode=unconstrained_decode),
        label=unconstrained_label,
    )
    candidate = _rename_for_label(
        _load_deep_rows(candidate_csv, deep_decode=candidate_decode),
        label=candidate_label,
    )
    comparison = unconstrained.merge(
        candidate,
        on=[column for column in IDENTITY_COLUMNS if column in unconstrained and column in candidate],
        how="inner",
        validate="one_to_one",
    )
    if comparison.empty:
        raise ValueError("No overlapping pair_id rows found between decoder CSVs.")

    mae_u = f"mae_ms_{unconstrained_label}"
    mae_c = f"mae_ms_{candidate_label}"
    ar_u = f"ar50_pct_{unconstrained_label}"
    ar_c = f"ar50_pct_{candidate_label}"
    if mae_u in comparison and mae_c in comparison:
        choose_candidate = comparison[mae_c] < comparison[mae_u]
        comparison["oracle_choice"] = np.where(choose_candidate, candidate_label, unconstrained_label)
        comparison["oracle_mae_ms"] = np.where(choose_candidate, comparison[mae_c], comparison[mae_u])
        comparison["oracle_gain_ms"] = comparison[mae_u] - comparison["oracle_mae_ms"]
    if ar_u in comparison and ar_c in comparison and "oracle_choice" in comparison:
        comparison["oracle_ar50_pct"] = np.where(
            comparison["oracle_choice"].eq(candidate_label),
            comparison[ar_c],
            comparison[ar_u],
        )

    if unconstrained_events is not None and candidate_events is not None:
        disagreement = _event_disagreement(unconstrained_events, candidate_events)
        comparison = comparison.merge(disagreement, on="pair_id", how="left")

    comparison = _add_canonical_selector_columns(
        comparison,
        unconstrained_label=unconstrained_label,
        candidate_label=candidate_label,
    )
    return comparison.sort_values(["group_id", "pair_id"], kind="stable").reset_index(drop=True)


def _print_summary(
    comparison: pd.DataFrame,
    *,
    unconstrained_label: str,
    candidate_label: str,
) -> None:
    mae_u = f"mae_ms_{unconstrained_label}"
    mae_c = f"mae_ms_{candidate_label}"
    ar_u = f"ar50_pct_{unconstrained_label}"
    ar_c = f"ar50_pct_{candidate_label}"
    if mae_u in comparison and mae_c in comparison:
        print(f"pairs={len(comparison)}")
        print(f"{unconstrained_label} mean MAE: {comparison[mae_u].mean():.3f} ms")
        print(f"{candidate_label} mean MAE: {comparison[mae_c].mean():.3f} ms")
    if ar_u in comparison and ar_c in comparison:
        print(f"{unconstrained_label} mean AR@50: {comparison[ar_u].mean():.3f}%")
        print(f"{candidate_label} mean AR@50: {comparison[ar_c].mean():.3f}%")
    if "oracle_mae_ms" in comparison:
        print(f"Oracle selector mean MAE: {comparison['oracle_mae_ms'].mean():.3f} ms")
    if "oracle_ar50_pct" in comparison:
        print(f"Oracle selector mean AR@50: {comparison['oracle_ar50_pct'].mean():.3f}%")
    if "oracle_choice" in comparison:
        print("Oracle choices:")
        print(comparison["oracle_choice"].value_counts().to_string())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare unconstrained and candidate DeepAlign decoder CSVs.")
    parser.add_argument("--unconstrained-csv", required=True)
    parser.add_argument("--candidate-csv", required=True)
    parser.add_argument("--output", default="results/selector/deepalign_decoder_comparison.csv")
    parser.add_argument("--unconstrained-decode", default="unconstrained")
    parser.add_argument("--candidate-decode", default="deepalign_coarse_to_fine")
    parser.add_argument("--unconstrained-label", default="unconstrained")
    parser.add_argument("--candidate-label", default="coarse_to_fine")
    parser.add_argument("--unconstrained-events", default=None)
    parser.add_argument("--candidate-events", default=None)
    args = parser.parse_args(argv)

    comparison = build_decoder_comparison(
        unconstrained_csv=args.unconstrained_csv,
        candidate_csv=args.candidate_csv,
        unconstrained_decode=args.unconstrained_decode,
        candidate_decode=args.candidate_decode,
        unconstrained_label=args.unconstrained_label,
        candidate_label=args.candidate_label,
        unconstrained_events=args.unconstrained_events,
        candidate_events=args.candidate_events,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output, index=False)
    print(f"Saved decoder comparison to {output}")
    _print_summary(
        comparison,
        unconstrained_label=args.unconstrained_label,
        candidate_label=args.candidate_label,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
