#!/usr/bin/env python
"""Evaluate DeepAlign-26 vs classical baselines on SWD.

Compares alignment accuracy (MAE, Alignment Rate) between:
  1. DeepAlign: trained CRNN encoder features + DTW
  2. Chroma baseline: hand-crafted CQT chroma features + DTW

Uses ground truth measure-level annotations from SWD.

Usage:
    python scripts/evaluate_swd.py --swd-path data/swd --checkpoint checkpoints/best_model.pt
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import librosa
import numpy as np
from scipy.spatial.distance import cdist

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dis_alignment.data.swd import (
    SWDDataset,
    SWDPair,
    compute_ground_truth_alignment,
    load_swd_audio,
)
from dis_alignment.evaluation.metrics import (
    alignment_rate,
    mean_absolute_error,
    median_absolute_error,
)
from dis_alignment.features.chroma import extract_chroma_cqt
from dis_alignment.model.inference import (
    extract_deep_features,
    load_trained_encoder,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def temporal_pool(features, pool_size=2):
    """Pool features across time to reduce memory without changing temporal dynamics."""
    if pool_size <= 1:
        return features
    n_features, n_frames = features.shape
    pooled_frames = n_frames // pool_size
    features_crop = features[:, :pooled_frames * pool_size]
    return features_crop.reshape(n_features, pooled_frames, pool_size).mean(axis=2)


def fast_dtw_align(features_a, features_b, distance="cosine"):
    """
    Fast DTW alignment using librosa's dtw (C-accelerated).

    Args:
        features_a: Shape (n_features, n_frames_a).
        features_b: Shape (n_features, n_frames_b).
        distance: Distance metric.

    Returns:
        Tuple of (warping_path, cost). Path is shape (2, path_len).
    """
    start = time.perf_counter()

    # Compute cost matrix
    C = cdist(features_a.T, features_b.T, metric=distance).astype(np.float64)

    # Use librosa's DTW (C-accelerated via numba)
    D, wp = librosa.sequence.dtw(C=C, backtrack=True)

    # wp is (path_len, 2) in reverse order; flip and transpose
    wp = wp[::-1].T  # -> (2, path_len)

    runtime = time.perf_counter() - start
    return wp, D[-1, -1], runtime


def evaluate_pair(
    pair: SWDPair,
    encoder=None,
    sr: int = 22050,
    chroma_hop: int = 512,
    deep_hop: int = 512,  # Matched with chroma to prevent 8GB+ OOM on large audio files
    device=None,
) -> list[dict]:
    """Evaluate a pair with both methods. Returns list of result dicts."""
    gt = compute_ground_truth_alignment(pair)
    if gt is None:
        logger.warning(f"  No GT annotations for {pair.pair_id}, skipping")
        return []

    gt_times_a, gt_times_b = gt
    results = []

    # Load audio
    audio_a, _ = load_swd_audio(pair.piece_a, sr=sr)
    audio_b, _ = load_swd_audio(pair.piece_b, sr=sr)

    # --- Chroma baseline ---
    chroma_hop = 440  # Directly extract at 440 for baseline
    chroma_a = extract_chroma_cqt(audio_a, sr=sr, hop_length=chroma_hop)
    chroma_b = extract_chroma_cqt(audio_b, sr=sr, hop_length=chroma_hop)

    wp, cost, runtime = fast_dtw_align(chroma_a, chroma_b, distance="cosine")

    # Convert path indices to time
    frame_dur = chroma_hop / sr
    path_times_a = wp[0] * frame_dur
    path_times_b = wp[1] * frame_dur

    # Interpolate predicted times at GT positions
    pred_b = np.interp(gt_times_a, path_times_a, path_times_b)

    results.append({
        "pair_id": pair.pair_id,
        "method": "chroma_dtw",
        "mae": mean_absolute_error(pred_b, gt_times_b),
        "median_ae": median_absolute_error(pred_b, gt_times_b),
        "ar_50ms": alignment_rate(pred_b, gt_times_b, 0.05),
        "ar_100ms": alignment_rate(pred_b, gt_times_b, 0.10),
        "ar_200ms": alignment_rate(pred_b, gt_times_b, 0.20),
        "runtime_s": runtime,
        "n_gt_points": len(gt_times_a),
    })

    # --- DeepAlign ---
    if encoder is not None:
        # Extract at native 220 
        feat_a = extract_deep_features(audio_a, encoder, sr=sr, hop_length=220, device=device)
        feat_b = extract_deep_features(audio_b, encoder, sr=sr, hop_length=220, device=device)

        # Pool by 2 -> effective hop 440 (prevents OOM)
        pool_size = 2
        feat_a_pool = temporal_pool(feat_a, pool_size=pool_size)
        feat_b_pool = temporal_pool(feat_b, pool_size=pool_size)

        wp_d, cost_d, runtime_d = fast_dtw_align(feat_a_pool, feat_b_pool, distance="sqeuclidean")

        frame_dur_d = (220 * pool_size) / sr
        path_times_a_d = wp_d[0] * frame_dur_d
        path_times_b_d = wp_d[1] * frame_dur_d

        pred_b_d = np.interp(gt_times_a, path_times_a_d, path_times_b_d)

        results.append({
            "pair_id": pair.pair_id,
            "method": "deepalign",
            "mae": mean_absolute_error(pred_b_d, gt_times_b),
            "median_ae": median_absolute_error(pred_b_d, gt_times_b),
            "ar_50ms": alignment_rate(pred_b_d, gt_times_b, 0.05),
            "ar_100ms": alignment_rate(pred_b_d, gt_times_b, 0.10),
            "ar_200ms": alignment_rate(pred_b_d, gt_times_b, 0.20),
            "runtime_s": runtime_d,
            "n_gt_points": len(gt_times_a),
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate DeepAlign on SWD")
    parser.add_argument("--swd-path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # Load SWD
    swd = SWDDataset(args.swd_path)
    pairs = list(swd.iter_pairs())
    logger.info(f"Found {len(pairs)} pairs across {swd.available_performances}")

    # Load trained encoder
    logger.info(f"Loading encoder from {args.checkpoint}")
    encoder, config = load_trained_encoder(args.checkpoint, device=args.device)

    # Evaluate all pairs
    all_results = []
    for i, pair in enumerate(pairs):
        logger.info(f"[{i+1}/{len(pairs)}] {pair.pair_id}")
        pair_results = evaluate_pair(pair, encoder=encoder, device=args.device)
        for r in pair_results:
            logger.info(f"  {r['method']:12s} MAE={r['mae']*1000:6.1f}ms  AR@50ms={r['ar_50ms']*100:5.1f}%")
        all_results.extend(pair_results)

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS — SWD (Schubert Winterreise Dataset)")
    print("=" * 70)

    for method_name in ["chroma_dtw", "deepalign"]:
        results = [r for r in all_results if r["method"] == method_name]
        if not results:
            continue

        label = "Chroma + DTW (baseline)" if method_name == "chroma_dtw" else "DeepAlign-26"
        maes = [r["mae"] for r in results]
        med_aes = [r["median_ae"] for r in results]
        ar50s = [r["ar_50ms"] for r in results]
        ar100s = [r["ar_100ms"] for r in results]
        ar200s = [r["ar_200ms"] for r in results]

        print(f"\n  {label} ({len(results)} pairs)")
        print(f"  {'─' * 50}")
        print(f"  MAE:         {np.mean(maes)*1000:7.1f} ms  (median: {np.median(maes)*1000:.1f} ms)")
        print(f"  Median AE:   {np.mean(med_aes)*1000:7.1f} ms")
        print(f"  AR @ 50ms:   {np.mean(ar50s)*100:7.1f} %")
        print(f"  AR @ 100ms:  {np.mean(ar100s)*100:7.1f} %")
        print(f"  AR @ 200ms:  {np.mean(ar200s)*100:7.1f} %")

    # Success criteria
    deep_results = [r for r in all_results if r["method"] == "deepalign"]
    if deep_results:
        avg_mae = np.mean([r["mae"] for r in deep_results])
        avg_ar50 = np.mean([r["ar_50ms"] for r in deep_results])
        print(f"\n{'=' * 70}")
        print("SUCCESS CRITERIA CHECK")
        print(f"{'=' * 70}")
        c1 = avg_mae < 0.05
        c2a = avg_mae < 0.02
        c2b = avg_ar50 > 0.98
        print(f"  Criterion 1 (MAE < 50ms):    {'PASS' if c1 else 'FAIL'}  (MAE = {avg_mae*1000:.1f} ms)")
        print(f"  Criterion 2 (MAE < 20ms):    {'PASS' if c2a else 'FAIL'}  (MAE = {avg_mae*1000:.1f} ms)")
        print(f"  Criterion 2 (AR@50ms > 98%): {'PASS' if c2b else 'FAIL'}  (AR  = {avg_ar50*100:.1f} %)")


if __name__ == "__main__":
    main()
