from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


def _finite_float(value: Any) -> float | None:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(resolved):
        return None
    return resolved


def choose_decoder(
    row: pd.Series,
    *,
    candidate_label: str = "coarse_to_fine",
    unconstrained_label: str = "unconstrained",
    band_edge_max: float = 0.15,
    coarse_large_jump_max: int = 3,
    cost_ratio_max: float = 1.20,
    disagreement_max_ms: float = 750.0,
    disagreement_band_edge_min: float = 0.05,
    fallback: str = "unconstrained",
) -> str:
    """Select a decoder using only decoder-internal confidence columns."""
    seen_feature = False
    band_edge = _finite_float(row.get("ctf_band_edge_fraction"))
    if band_edge is not None:
        seen_feature = True
        if band_edge > band_edge_max:
            return unconstrained_label

    large_jumps = _finite_float(row.get("ctf_coarse_large_jump_count"))
    if large_jumps is not None:
        seen_feature = True
        if large_jumps > coarse_large_jump_max:
            return unconstrained_label

    ctf_cost = _finite_float(row.get("ctf_cost_per_step"))
    unconstrained_cost = _finite_float(row.get("unconstrained_cost_per_step"))
    if ctf_cost is not None and unconstrained_cost is not None and unconstrained_cost > 0:
        seen_feature = True
        if ctf_cost > unconstrained_cost * cost_ratio_max:
            return unconstrained_label

    disagreement = _finite_float(row.get("decoder_disagreement_median_ms"))
    if disagreement is not None:
        seen_feature = True
        edge_for_disagreement = band_edge if band_edge is not None else 0.0
        if disagreement > disagreement_max_ms and edge_for_disagreement > disagreement_band_edge_min:
            return unconstrained_label

    if not seen_feature:
        if fallback not in {candidate_label, unconstrained_label}:
            raise ValueError("fallback must match either the candidate or unconstrained label.")
        return fallback
    return candidate_label


def apply_selector(
    comparison: pd.DataFrame,
    *,
    candidate_label: str = "coarse_to_fine",
    unconstrained_label: str = "unconstrained",
    band_edge_max: float = 0.15,
    coarse_large_jump_max: int = 3,
    cost_ratio_max: float = 1.20,
    disagreement_max_ms: float = 750.0,
    disagreement_band_edge_min: float = 0.05,
    fallback: str = "unconstrained",
) -> pd.DataFrame:
    result = comparison.copy()
    choices = [
        choose_decoder(
            row,
            candidate_label=candidate_label,
            unconstrained_label=unconstrained_label,
            band_edge_max=band_edge_max,
            coarse_large_jump_max=coarse_large_jump_max,
            cost_ratio_max=cost_ratio_max,
            disagreement_max_ms=disagreement_max_ms,
            disagreement_band_edge_min=disagreement_band_edge_min,
            fallback=fallback,
        )
        for _, row in result.iterrows()
    ]
    result["selector_choice"] = choices

    for metric in ("mae_ms", "median_ae_ms", "ar50_pct", "ar100_pct", "ar200_pct"):
        candidate_col = f"{metric}_{candidate_label}"
        unconstrained_col = f"{metric}_{unconstrained_label}"
        if candidate_col in result and unconstrained_col in result:
            result[f"selector_{metric}"] = np.where(
                result["selector_choice"].eq(candidate_label),
                result[candidate_col],
                result[unconstrained_col],
            )
    return result


def selector_summary(selected: pd.DataFrame) -> dict[str, float]:
    summary: dict[str, float] = {"pairs": float(len(selected))}
    for column, output in (
        ("selector_mae_ms", "mae_ms"),
        ("selector_median_ae_ms", "median_ae_ms"),
        ("selector_ar50_pct", "ar50_pct"),
        ("selector_ar100_pct", "ar100_pct"),
        ("selector_ar200_pct", "ar200_pct"),
    ):
        if column in selected:
            summary[output] = float(selected[column].mean())
    return summary


def _print_summary(
    selected: pd.DataFrame,
    *,
    candidate_label: str,
    unconstrained_label: str,
) -> None:
    summary = selector_summary(selected)
    print(f"pairs={int(summary['pairs'])}")
    if "mae_ms" in summary:
        print(f"Selector mean MAE: {summary['mae_ms']:.3f} ms")
    if "ar50_pct" in summary:
        print(f"Selector mean AR@50: {summary['ar50_pct']:.3f}%")
    print("Selector choices:")
    print(selected["selector_choice"].value_counts().reindex([unconstrained_label, candidate_label]).fillna(0).astype(int).to_string())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a non-leaky confidence gate for DeepAlign decoders.")
    parser.add_argument("comparison_csv")
    parser.add_argument("--output", default="results/selector/deepalign_decoder_selector.csv")
    parser.add_argument("--candidate-label", default="coarse_to_fine")
    parser.add_argument("--unconstrained-label", default="unconstrained")
    parser.add_argument("--band-edge-max", type=float, default=0.15)
    parser.add_argument("--coarse-large-jump-max", type=int, default=3)
    parser.add_argument("--cost-ratio-max", type=float, default=1.20)
    parser.add_argument("--disagreement-max-ms", type=float, default=750.0)
    parser.add_argument("--disagreement-band-edge-min", type=float, default=0.05)
    parser.add_argument("--fallback", default="unconstrained")
    args = parser.parse_args(argv)

    selected = apply_selector(
        pd.read_csv(args.comparison_csv),
        candidate_label=args.candidate_label,
        unconstrained_label=args.unconstrained_label,
        band_edge_max=args.band_edge_max,
        coarse_large_jump_max=args.coarse_large_jump_max,
        cost_ratio_max=args.cost_ratio_max,
        disagreement_max_ms=args.disagreement_max_ms,
        disagreement_band_edge_min=args.disagreement_band_edge_min,
        fallback=args.fallback,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output, index=False)
    print(f"Saved selector evaluation to {output}")
    _print_summary(
        selected,
        candidate_label=args.candidate_label,
        unconstrained_label=args.unconstrained_label,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
