#!/usr/bin/env python
"""Thin wrapper around the package SWD evaluator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DeepAlign-26 on SWD")
    parser.add_argument("--swd-path", type=str, required=True, help="Path to the extracted SWD root")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional DeepAlign checkpoint. Omit for baseline-only chroma DTW evaluation.",
    )
    parser.add_argument("--output", type=str, default="results/swd_evaluation.csv")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--quiet", action="store_true", help="Disable the progress bar")
    args = parser.parse_args()

    from dis_alignment.data import SWDDataset
    from dis_alignment.evaluation import (
        check_success_criteria,
        evaluate_swd_dataset,
        save_evaluation_results,
        summarize_evaluation,
    )

    dataset = SWDDataset(args.swd_path)
    results = evaluate_swd_dataset(
        dataset,
        checkpoint_path=args.checkpoint,
        device=args.device,
        show_progress=not args.quiet,
    )
    output_path = save_evaluation_results(results, args.output)
    print(f"Saved results to {output_path}")

    print("\n=== SWD Summary ===")
    for method, stats in summarize_evaluation(results).items():
        print(f"\n{method}")
        print(f"  pairs:       {int(stats['pairs'])}")
        print(f"  MAE:         {stats['mae_ms']:.1f} ms")
        print(f"  Median AE:   {stats['median_ae_ms']:.1f} ms")
        print(f"  AR @ 50ms:   {stats['ar_50ms_pct']:.1f} %")
        print(f"  Runtime:     {stats['runtime_s']:.2f} s")

    success = check_success_criteria(results)
    if success["available"]:
        print("\n=== Success Criteria ===")
        print(
            f"  Criterion 1 (MAE < 50ms):    "
            f"{'PASS' if success['criterion_1_pass'] else 'FAIL'} "
            f"(MAE = {success['mae_seconds'] * 1000:.1f} ms)"
        )
        print(
            f"  Criterion 2 (MAE < 20ms):    "
            f"{'PASS' if success['criterion_2_mae_pass'] else 'FAIL'}"
        )
        print(
            f"  Criterion 2 (AR@50ms > 98%): "
            f"{'PASS' if success['criterion_2_ar_pass'] else 'FAIL'} "
            f"(AR = {success['ar_50ms'] * 100:.1f} %)"
        )


if __name__ == "__main__":
    main()
