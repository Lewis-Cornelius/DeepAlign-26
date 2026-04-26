#!/usr/bin/env python
"""Thin wrapper around the package Mazurka evaluator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DeepAlign-26 on MazurkaBL-style data")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to the Mazurka root")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional DeepAlign checkpoint")
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated methods: chroma_dtw,mrmsdtw,deepalign,matchmaker",
    )
    parser.add_argument("--output", type=str, default="results/mazurka_evaluation.csv")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--quiet", action="store_true", help="Disable the progress bar")
    parser.add_argument("--matchmaker-method", type=str, default="arzt")
    parser.add_argument("--matchmaker-feature-type", type=str, default="chroma")
    parser.add_argument("--matchmaker-frame-rate", type=int, default=100)
    args = parser.parse_args()

    from dis_alignment.data import MazurkaDataset
    from dis_alignment.evaluation import (
        check_success_criteria,
        evaluate_mazurka_dataset,
        save_evaluation_results,
        summarize_evaluation,
    )

    dataset = MazurkaDataset(args.dataset_path)
    results = evaluate_mazurka_dataset(
        dataset,
        checkpoint_path=args.checkpoint,
        device=args.device,
        methods=args.methods,
        show_progress=not args.quiet,
        matchmaker_method=args.matchmaker_method,
        matchmaker_feature_type=args.matchmaker_feature_type,
        matchmaker_frame_rate=args.matchmaker_frame_rate,
    )
    output_path = save_evaluation_results(results, args.output)
    print(f"Saved results to {output_path}")

    print("\n=== Mazurka Summary ===")
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
