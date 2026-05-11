"""Tests for the DeepAlign target-performance recovery workflow."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.io import wavfile

from dis_alignment.data.swd import SWDPair, SWDPiece
from dis_alignment.evaluation import common as eval_common
from dis_alignment.model.anchor_loss import AnchorContrastiveLoss
from dis_alignment.model.cqt_cache import SpectrogramCache, compute_log_cqt
from dis_alignment.model.encoder import CRNNEncoder, DeepAlignModel
from dis_alignment.model.recovery import normalize_debug_lied_ids
from dis_alignment.model.train import (
    _build_checkpoint_payload,
    _build_selection_snapshot,
    _build_selection_key,
    _empty_history,
    _load_resume_state,
)


def _make_swd_pair(tmp_path, lied_id: str = "D911-01") -> SWDPair:
    return SWDPair(
        pair_id=f"{lied_id}_HU33_SC06",
        lied_id=lied_id,
        piece_a=SWDPiece(
            piece_id=f"{lied_id}_HU33",
            lied_id=lied_id,
            performance_id="HU33",
            audio_path=tmp_path / f"{lied_id}_HU33.wav",
        ),
        piece_b=SWDPiece(
            piece_id=f"{lied_id}_SC06",
            lied_id=lied_id,
            performance_id="SC06",
            audio_path=tmp_path / f"{lied_id}_SC06.wav",
        ),
    )


def test_spectrogram_cache_matches_direct_cqt_and_keys_include_frontend(tmp_path):
    sr = 22050
    t = np.linspace(0.0, 1.0, sr, endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 220.0 * t)
    audio_path = tmp_path / "example.wav"
    wavfile.write(audio_path, sr, (audio * 32767).astype(np.int16))

    import librosa

    loaded_audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    direct = compute_log_cqt(
        loaded_audio,
        sr=sr,
        hop_length=220,
        n_bins=24,
        bins_per_octave=12,
    )

    cache = SpectrogramCache(tmp_path / "cache", sr=sr, hop_length=220, n_bins=24, bins_per_octave=12)
    cached = cache.load_or_compute(audio_path)
    np.testing.assert_allclose(cached, direct, atol=1e-5)

    other_cache = SpectrogramCache(tmp_path / "cache", sr=sr, hop_length=110, n_bins=24, bins_per_octave=12)
    other_cache.load_or_compute(audio_path)
    assert len(list((tmp_path / "cache").glob("*.npy"))) == 2


def test_resume_state_restores_optimizer_and_scheduler_state(tmp_path):
    encoder = CRNNEncoder(
        n_freq_bins=24,
        embed_dim=8,
        num_conv_channels=[8, 8],
        gru_hidden_size=16,
        num_gru_layers=1,
        dropout=0.0,
    )
    model = DeepAlignModel(encoder)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    spec_a = torch.randn(1, 1, 24, 10)
    spec_b = torch.randn(1, 1, 24, 10)
    emb_a, emb_b = model(spec_a, spec_b)
    loss = (emb_a - emb_b).pow(2).mean()
    loss.backward()
    optimizer.step()
    scheduler.step()

    payload = _build_checkpoint_payload(
        epoch=0,
        model=model,
        encoder=encoder,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        scaler=scaler,
        val_loss=1.23,
        history=_empty_history(),
        best_selection={"metric": "debug_mae", "selection_key": [0.5, -0.2], "epoch": 0},
        best_selections={"debug_mae": {"metric": "debug_mae", "selection_key": [0.5, -0.2], "epoch": 0}},
        current_selection={"metric": "debug_mae", "selection_key": [0.5, -0.2], "epoch": 0},
        selection_snapshots={"debug_mae": {"metric": "debug_mae", "selection_key": [0.5, -0.2], "epoch": 0}},
        training_state_signature={"epochs": 2, "lr": 1e-3, "start_gamma": 1.0, "end_gamma": 0.01, "selection_metric": "debug_mae", "anchor_loss_weight": 0.0},
        config={"embed_dim": 8, "n_freq_bins": 24, "gru_hidden_size": 16, "num_gru_layers": 1, "dropout": 0.0},
    )
    checkpoint_path = tmp_path / "resume.pt"
    torch.save(payload, checkpoint_path)

    encoder_resumed = CRNNEncoder(
        n_freq_bins=24,
        embed_dim=8,
        num_conv_channels=[8, 8],
        gru_hidden_size=16,
        num_gru_layers=1,
        dropout=0.0,
    )
    model_resumed = DeepAlignModel(encoder_resumed)
    optimizer_resumed = torch.optim.AdamW(model_resumed.parameters(), lr=1e-3)
    scheduler_resumed = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_resumed, T_max=2)
    scaler_resumed = torch.amp.GradScaler("cuda", enabled=False)

    start_epoch, _, best_selection, best_selections = _load_resume_state(
        checkpoint_path=checkpoint_path,
        model=model_resumed,
        encoder=encoder_resumed,
        optimizer=optimizer_resumed,
        lr_scheduler=scheduler_resumed,
        scaler=scaler_resumed,
        current_signature={"epochs": 2, "lr": 1e-3, "start_gamma": 1.0, "end_gamma": 0.01, "selection_metric": "debug_mae", "anchor_loss_weight": 0.0},
    )

    assert start_epoch == 1
    assert optimizer_resumed.state_dict()["param_groups"][0]["lr"] == optimizer.state_dict()["param_groups"][0]["lr"]
    assert scheduler_resumed.state_dict() == scheduler.state_dict()
    assert best_selection == {"metric": "debug_mae", "selection_key": [0.5, -0.2], "epoch": 0}
    assert best_selections == {"debug_mae": {"metric": "debug_mae", "selection_key": [0.5, -0.2], "epoch": 0}}


def test_selection_metric_prefers_debug_mae_over_lower_val_loss():
    better_debug = _build_selection_key(
        selection_metric="debug_mae",
        val_loss=10.0,
        debug_metrics={"mae": 0.5, "ar_50ms": 0.2, "ar_100ms": 0.3, "ar_200ms": 0.4},
        val_metrics={"mae": 1.0, "ar_50ms": 0.1, "ar_100ms": 0.2, "ar_200ms": 0.3},
    )
    worse_debug = _build_selection_key(
        selection_metric="debug_mae",
        val_loss=1.0,
        debug_metrics={"mae": 0.8, "ar_50ms": 0.2, "ar_100ms": 0.3, "ar_200ms": 0.4},
        val_metrics={"mae": 0.9, "ar_50ms": 0.1, "ar_100ms": 0.2, "ar_200ms": 0.3},
    )
    assert better_debug < worse_debug


def test_selection_metric_prefers_debug_ar50_then_mae_then_ar100():
    better_ar100_tiebreak = _build_selection_key(
        selection_metric="debug_ar50",
        val_loss=1.0,
        debug_metrics={"mae": 0.6, "ar_50ms": 0.3, "ar_100ms": 0.5, "ar_200ms": 0.7},
        val_metrics={"mae": 0.9, "ar_50ms": 0.2, "ar_100ms": 0.4, "ar_200ms": 0.6},
    )
    worse_ar100_tiebreak = _build_selection_key(
        selection_metric="debug_ar50",
        val_loss=1.0,
        debug_metrics={"mae": 0.6, "ar_50ms": 0.3, "ar_100ms": 0.4, "ar_200ms": 0.7},
        val_metrics={"mae": 0.9, "ar_50ms": 0.2, "ar_100ms": 0.4, "ar_200ms": 0.6},
    )
    assert better_ar100_tiebreak < worse_ar100_tiebreak


def test_balanced_selection_uses_accuracy_bands_before_mae():
    more_balanced = _build_selection_key(
        selection_metric="debug_balanced",
        val_loss=1.0,
        debug_metrics={"mae": 0.7, "ar_50ms": 0.35, "ar_100ms": 0.55, "ar_200ms": 0.75},
        val_metrics={"mae": 0.9, "ar_50ms": 0.2, "ar_100ms": 0.4, "ar_200ms": 0.6},
    )
    less_balanced = _build_selection_key(
        selection_metric="debug_balanced",
        val_loss=1.0,
        debug_metrics={"mae": 0.5, "ar_50ms": 0.30, "ar_100ms": 0.50, "ar_200ms": 0.70},
        val_metrics={"mae": 0.9, "ar_50ms": 0.2, "ar_100ms": 0.4, "ar_200ms": 0.6},
    )
    assert more_balanced < less_balanced


def test_selection_snapshot_includes_all_debug_metrics():
    snapshot = _build_selection_snapshot(
        selection_metric="debug_ar50",
        val_loss=1.0,
        debug_metrics={"mae": 0.7, "ar_50ms": 0.3, "ar_100ms": 0.5, "ar_200ms": 0.7},
        val_metrics={"mae": 0.9, "ar_50ms": 0.2, "ar_100ms": 0.4, "ar_200ms": 0.6},
        epoch=3,
    )
    assert snapshot["metric"] == "debug_ar50"
    assert snapshot["epoch"] == 3
    assert snapshot["debug_metrics"]["ar_100ms"] == 0.5


def test_banded_dtw_path_is_monotonic():
    features = np.eye(6, dtype=np.float32)
    lower, upper = eval_common._diagonal_band_bounds(6, 6, 1)
    path, _, _ = eval_common.banded_dtw_align(
        features,
        features,
        lower_bounds=lower,
        upper_bounds=upper,
        distance="sqeuclidean",
    )
    assert np.all(np.diff(path[0]) >= 0)
    assert np.all(np.diff(path[1]) >= 0)


def test_pairwise_dtw_evaluation_collapses_duplicate_query_frames(monkeypatch):
    """Direct DTW paths can repeat query frames; evaluation should average them safely."""

    monkeypatch.setattr(
        "dis_alignment.features.chroma.extract_chroma_cqt",
        lambda audio, sr=1, hop_length=1: np.eye(3, dtype=np.float32),
    )
    monkeypatch.setattr(
        eval_common,
        "fast_dtw_align",
        lambda features_a, features_b, distance="cosine": (
            np.array([[0, 1, 1, 2], [0, 0, 2, 2]], dtype=np.intp),
            0.0,
            0.0,
        ),
    )

    rows = eval_common.evaluate_pairwise_methods(
        methods=["chroma_dtw"],
        pair_id="pair",
        group_id="group",
        dataset_name="test",
        piece_a_id="a",
        piece_b_id="b",
        audio_a=np.zeros(3, dtype=np.float32),
        audio_b=np.zeros(3, dtype=np.float32),
        gt_a=np.array([1.0], dtype=np.float64),
        gt_b=np.array([1.0], dtype=np.float64),
        chroma_hop=1,
        sr=1,
    )

    assert rows[0]["mae"] == 0.0
    assert rows[0]["ar_50ms"] == 1.0


def test_chroma_guided_band_decode_produces_valid_alignment(monkeypatch, tmp_path):
    pair = _make_swd_pair(tmp_path)

    monkeypatch.setattr(
        "dis_alignment.evaluation.swd.compute_ground_truth_measure_alignment",
        lambda _pair: (
            pd.DataFrame({"event_id": ["1", "2", "3"], "time_s": [0.0, 1.0, 2.0]}),
            pd.DataFrame({"event_id": ["1", "2", "3"], "time_s": [0.0, 1.0, 2.0]}),
        ),
    )
    monkeypatch.setattr(
        "dis_alignment.evaluation.swd.load_swd_audio",
        lambda piece, sr=10: (np.linspace(0.0, 1.0, 30, dtype=np.float32), sr),
    )
    monkeypatch.setattr(
        "dis_alignment.features.chroma.extract_chroma_cqt",
        lambda audio, sr=10, hop_length=1: np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], dtype=np.float32),
    )
    monkeypatch.setattr(
        "dis_alignment.model.inference.extract_deep_features",
        lambda audio, encoder, sr=10, hop_length=1, device=None, audio_path=None, cache_root=None: np.array(
            [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
            dtype=np.float32,
        ),
    )

    from dis_alignment.evaluation import swd as swd_eval

    unconstrained = swd_eval.evaluate_pair(
        pair,
        methods=["deepalign"],
        encoder=object(),
        sr=10,
        chroma_hop=1,
        deep_hop=1,
        pool_size=1,
        deep_decode="unconstrained",
    )
    guided = swd_eval.evaluate_pair(
        pair,
        methods=["deepalign"],
        encoder=object(),
        sr=10,
        chroma_hop=1,
        deep_hop=1,
        pool_size=1,
        deep_decode="chroma_guided_band",
        band_radius_frames=1,
    )

    assert len(unconstrained) == 1
    assert len(guided) == 1
    assert guided[0]["deep_decode"] == "chroma_guided_band"


def test_anchor_contrastive_loss_handles_sparse_and_dense_anchors():
    loss_fn = AnchorContrastiveLoss(temperature=0.1, min_anchor_gap=1)
    emb_a = torch.randn(1, 5, 4)
    emb_b = torch.randn(1, 5, 4)

    dense_loss = loss_fn(emb_a, emb_b, [[0, 2, 4]], [[0, 2, 4]])
    sparse_loss = loss_fn(emb_a, emb_b, [[1]], [[1]])

    assert torch.isfinite(dense_loss)
    assert sparse_loss.item() == 0.0


def test_normalize_debug_lied_ids_defaults_to_representative_gate_set():
    assert normalize_debug_lied_ids(None) == (
        "D911-07",
        "D911-18",
        "D911-20",
        "D911-22",
        "D911-24",
        "D911-17",
        "D911-06",
        "D911-02",
    )
