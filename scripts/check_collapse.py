#!/usr/bin/env python
"""Inspect a trained DeepAlign checkpoint for feature collapse."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check a DeepAlign checkpoint for temporal collapse")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt")
    parser.add_argument("--swd-path", type=str, required=True, help="Path to the SWD root")
    parser.add_argument("--pair-index", type=int, default=0, help="Which SWD pair to inspect")
    args = parser.parse_args()

    from dis_alignment.data import SWDDataset, load_swd_audio
    from dis_alignment.model.inference import extract_deep_features, load_trained_encoder

    print("Loading model...")
    encoder, _ = load_trained_encoder(args.checkpoint)

    print("Loading SWD pair...")
    swd = SWDDataset(args.swd_path)
    pairs = list(swd.iter_pairs())
    if not pairs:
        raise RuntimeError("No SWD pairs found")
    pair = pairs[args.pair_index]
    audio_a, _ = load_swd_audio(pair.piece_a)

    print("Extracting DeepAlign features...")
    features = extract_deep_features(audio_a, encoder, sr=22050, hop_length=220)

    std_across_time = np.std(features, axis=1)
    mean_std = float(np.mean(std_across_time))
    distance = 0.0
    if features.shape[1] > 1000:
        distance = float(np.linalg.norm(features[:, 0] - features[:, 1000]))

    print(f"Feature shape: {features.shape}")
    print(f"Mean std deviation across time: {mean_std:.6f}")
    print(f"Distance between frame 0 and 1000: {distance:.6f}")

    if mean_std < 1e-3:
        print("WARNING: embeddings look nearly collapsed across time.")
    else:
        print("Model shows temporal variance; no obvious complete collapse.")


if __name__ == "__main__":
    main()
