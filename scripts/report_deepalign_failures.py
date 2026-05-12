from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from dis_alignment.data.swd import SWDDataset, SWDPair, compute_ground_truth_measure_alignment, load_swd_audio
from dis_alignment.evaluation.common import _interp_monotonic, fast_dtw_align, parse_deep_decode, temporal_pool
from dis_alignment.model.inference import extract_deep_features, load_trained_encoder


def _alignment_rate_from_errors(abs_errors_s: np.ndarray, threshold_s: float) -> float:
    if abs_errors_s.size == 0:
        return float("nan")
    return float(np.mean(abs_errors_s <= threshold_s))


def _error_sign(error_s: float) -> str:
    if error_s > 0:
        return "late"
    if error_s < 0:
        return "early"
    return "exact"


def summarize_pair_errors(
    *,
    pair: SWDPair,
    event_ids: Sequence[str],
    gt_a_s: np.ndarray,
    gt_b_s: np.ndarray,
    pred_b_s: np.ndarray,
    checkpoint: str | Path,
    sr: int,
    deep_hop: int,
    pool_size: int,
    deep_distance: str,
    deep_decode: str,
    path_cost: float,
    runtime_s: float,
    path_length: int,
) -> dict[str, Any]:
    signed_errors_s = np.asarray(pred_b_s, dtype=np.float64) - np.asarray(gt_b_s, dtype=np.float64)
    abs_errors_s = np.abs(signed_errors_s)
    if abs_errors_s.size == 0:
        raise ValueError(f"No event errors available for {pair.pair_id}.")

    worst_idx = int(np.argmax(abs_errors_s))
    frame_duration_s = (deep_hop * pool_size) / float(sr)
    return {
        "dataset": "swd",
        "pair_id": pair.pair_id,
        "group_id": pair.lied_id,
        "piece_a_id": pair.piece_a.piece_id,
        "piece_b_id": pair.piece_b.piece_id,
        "method": "deepalign",
        "checkpoint": str(checkpoint),
        "deep_decode": deep_decode,
        "deep_hop": int(deep_hop),
        "pool_size": int(pool_size),
        "deep_distance": deep_distance,
        "n_gt_points": int(abs_errors_s.size),
        "mae_ms": float(np.mean(abs_errors_s) * 1000.0),
        "median_ae_ms": float(np.median(abs_errors_s) * 1000.0),
        "p90_error_ms": float(np.percentile(abs_errors_s, 90) * 1000.0),
        "max_error_ms": float(abs_errors_s[worst_idx] * 1000.0),
        "ar50_pct": _alignment_rate_from_errors(abs_errors_s, 0.05) * 100.0,
        "ar100_pct": _alignment_rate_from_errors(abs_errors_s, 0.10) * 100.0,
        "ar200_pct": _alignment_rate_from_errors(abs_errors_s, 0.20) * 100.0,
        "worst_event_id": str(event_ids[worst_idx]) if worst_idx < len(event_ids) else str(worst_idx),
        "worst_gt_a_s": float(gt_a_s[worst_idx]),
        "worst_gt_b_s": float(gt_b_s[worst_idx]),
        "worst_pred_b_s": float(pred_b_s[worst_idx]),
        "worst_signed_error_ms": float(signed_errors_s[worst_idx] * 1000.0),
        "worst_error_sign": _error_sign(float(signed_errors_s[worst_idx])),
        "worst_gt_b_frame": int(round(float(gt_b_s[worst_idx]) / frame_duration_s)),
        "worst_pred_b_frame": int(round(float(pred_b_s[worst_idx]) / frame_duration_s)),
        "path_cost": float(path_cost),
        "path_length": int(path_length),
        "runtime_s": float(runtime_s),
    }


def event_error_rows(
    *,
    pair: SWDPair,
    event_ids: Sequence[str],
    gt_a_s: np.ndarray,
    gt_b_s: np.ndarray,
    pred_b_s: np.ndarray,
    deep_hop: int,
    pool_size: int,
) -> list[dict[str, Any]]:
    signed_errors_s = np.asarray(pred_b_s, dtype=np.float64) - np.asarray(gt_b_s, dtype=np.float64)
    abs_errors_s = np.abs(signed_errors_s)
    frame_duration_s = (deep_hop * pool_size) / 22050.0
    rows: list[dict[str, Any]] = []
    for idx, abs_error_s in enumerate(abs_errors_s):
        signed_error_s = float(signed_errors_s[idx])
        event_id = str(event_ids[idx]) if idx < len(event_ids) else str(idx)
        rows.append(
            {
                "dataset": "swd",
                "pair_id": pair.pair_id,
                "group_id": pair.lied_id,
                "piece_a_id": pair.piece_a.piece_id,
                "piece_b_id": pair.piece_b.piece_id,
                "event_index": int(idx),
                "event_id": event_id,
                "gt_a_s": float(gt_a_s[idx]),
                "gt_b_s": float(gt_b_s[idx]),
                "pred_b_s": float(pred_b_s[idx]),
                "signed_error_ms": signed_error_s * 1000.0,
                "abs_error_ms": float(abs_error_s * 1000.0),
                "error_sign": _error_sign(signed_error_s),
                "gt_b_frame": int(round(float(gt_b_s[idx]) / frame_duration_s)),
                "pred_b_frame": int(round(float(pred_b_s[idx]) / frame_duration_s)),
                "off_50ms": bool(abs_error_s > 0.05),
                "off_100ms": bool(abs_error_s > 0.10),
                "off_200ms": bool(abs_error_s > 0.20),
            }
        )
    return rows


def report_pair_failures(
    pair: SWDPair,
    *,
    encoder: Any,
    checkpoint: str | Path,
    sr: int,
    deep_hop: int,
    pool_size: int,
    deep_distance: str,
    deep_decode: str,
    device: str | None,
    cache_root: str | Path | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ground_truth = compute_ground_truth_measure_alignment(pair)
    if ground_truth is None:
        raise ValueError(f"No SWD ground truth alignment for {pair.pair_id}.")
    annotations_a, annotations_b = ground_truth
    gt_a_s = annotations_a["time_s"].to_numpy(dtype=float)
    gt_b_s = annotations_b["time_s"].to_numpy(dtype=float)
    event_ids = annotations_a["event_id"].astype(str).tolist()

    audio_a, _ = load_swd_audio(pair.piece_a, sr=sr)
    audio_b, _ = load_swd_audio(pair.piece_b, sr=sr)
    feat_a = extract_deep_features(
        audio_a,
        encoder,
        sr=sr,
        hop_length=deep_hop,
        device=device,
        audio_path=pair.piece_a.audio_path,
        cache_root=cache_root,
    )
    feat_b = extract_deep_features(
        audio_b,
        encoder,
        sr=sr,
        hop_length=deep_hop,
        device=device,
        audio_path=pair.piece_b.audio_path,
        cache_root=cache_root,
    )
    pooled_a = temporal_pool(feat_a, pool_size=pool_size)
    pooled_b = temporal_pool(feat_b, pool_size=pool_size)
    path, path_cost, runtime_s = fast_dtw_align(pooled_a, pooled_b, distance=deep_distance)
    frame_duration_s = (deep_hop * pool_size) / sr
    pred_b_s = _interp_monotonic(gt_a_s, path[0] * frame_duration_s, path[1] * frame_duration_s)
    summary = summarize_pair_errors(
        pair=pair,
        event_ids=event_ids,
        gt_a_s=gt_a_s,
        gt_b_s=gt_b_s,
        pred_b_s=pred_b_s,
        checkpoint=checkpoint,
        sr=sr,
        deep_hop=deep_hop,
        pool_size=pool_size,
        deep_distance=deep_distance,
        deep_decode=deep_decode,
        path_cost=path_cost,
        runtime_s=runtime_s,
        path_length=path.shape[1],
    )
    events = event_error_rows(
        pair=pair,
        event_ids=event_ids,
        gt_a_s=gt_a_s,
        gt_b_s=gt_b_s,
        pred_b_s=pred_b_s,
        deep_hop=deep_hop,
        pool_size=pool_size,
    )
    return summary, events


def build_failure_report(
    *,
    swd_path: str | Path,
    checkpoint: str | Path,
    output: str | Path,
    event_output: str | Path | None,
    lieder: Sequence[str] | None,
    sr: int,
    deep_hop: int,
    pool_size: int,
    deep_distance: str,
    deep_decode: str,
    device: str | None,
    cache_root: str | Path | None,
) -> pd.DataFrame:
    resolved_decode = parse_deep_decode(deep_decode)
    if resolved_decode != "unconstrained":
        raise ValueError("Failure reports currently support only unconstrained DeepAlign decoding.")

    dataset = SWDDataset(swd_path)
    encoder, _ = load_trained_encoder(checkpoint, device=device)
    summaries: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for pair in dataset.iter_pairs(lieder=list(lieder) if lieder else None):
        summary, events = report_pair_failures(
            pair,
            encoder=encoder,
            checkpoint=checkpoint,
            sr=sr,
            deep_hop=deep_hop,
            pool_size=pool_size,
            deep_distance=deep_distance,
            deep_decode=resolved_decode,
            device=device,
            cache_root=cache_root,
        )
        summaries.append(summary)
        event_rows.extend(events)

    report = pd.DataFrame(summaries).sort_values(["max_error_ms", "mae_ms"], ascending=False)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    if event_output is not None:
        event_path = Path(event_output)
        event_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(event_rows).sort_values(["abs_error_ms", "pair_id"], ascending=False).to_csv(
            event_path,
            index=False,
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report DeepAlign per-pair and per-event failure excursions.")
    parser.add_argument("swd_path", help="Path to the SWD dataset root")
    parser.add_argument("--checkpoint", required=True, help="DeepAlign checkpoint to inspect")
    parser.add_argument("--output", default="results/failure_reports/deepalign_failure_report.csv")
    parser.add_argument("--event-output", default=None, help="Optional per-event CSV output")
    parser.add_argument("--lied", dest="lieder", action="append", default=None, help="Restrict to one SWD lied")
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--deep-hop", type=int, default=110)
    parser.add_argument("--pool-size", type=int, default=1)
    parser.add_argument("--deep-distance", default="sqeuclidean")
    parser.add_argument("--deep-decode", default="unconstrained")
    parser.add_argument("--device", default=None)
    parser.add_argument("--cache-root", default=".cache/cqt")
    args = parser.parse_args(argv)

    report = build_failure_report(
        swd_path=args.swd_path,
        checkpoint=args.checkpoint,
        output=args.output,
        event_output=args.event_output,
        lieder=args.lieder,
        sr=args.sr,
        deep_hop=args.deep_hop,
        pool_size=args.pool_size,
        deep_distance=args.deep_distance,
        deep_decode=args.deep_decode,
        device=args.device,
        cache_root=args.cache_root,
    )
    print(f"Saved failure report to {args.output}")
    if args.event_output:
        print(f"Saved event failure report to {args.event_output}")
    if not report.empty:
        columns = ["pair_id", "mae_ms", "median_ae_ms", "p90_error_ms", "max_error_ms", "ar50_pct"]
        print(report[columns].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
