"""Tests for audio-only teacher path artifacts."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _make_pair(tmp_path, lied_id: str = "D911-01"):
    from dis_alignment.data.swd import SWDPair, SWDPiece

    piece_a = SWDPiece(
        piece_id=f"{lied_id}_HU33",
        lied_id=lied_id,
        performance_id="HU33",
        audio_path=tmp_path / f"{lied_id}_HU33.wav",
    )
    piece_b = SWDPiece(
        piece_id=f"{lied_id}_SC06",
        lied_id=lied_id,
        performance_id="SC06",
        audio_path=tmp_path / f"{lied_id}_SC06.wav",
    )
    return SWDPair(
        pair_id=f"{lied_id}_HU33_SC06",
        lied_id=lied_id,
        piece_a=piece_a,
        piece_b=piece_b,
    )


def test_teacher_path_round_trip(tmp_path):
    from dis_alignment.alignment.teacher import TeacherPath, load_teacher_path, save_teacher_path

    path = np.array([[0, 1, 2], [0, 2, 4]], dtype=np.intp)
    teacher = TeacherPath(
        pair_id="D911-01_HU33_SC06",
        lied_id="D911-01",
        piece_a_id="D911-01_HU33",
        piece_b_id="D911-01_SC06",
        frame_hop=110,
        sr=22050,
        path=path,
        time_a_s=np.array([0.0, 0.1, 0.2]),
        time_b_s=np.array([0.0, 0.2, 0.4]),
        chroma_shift=10,
        anchor_calibrated=True,
        confidence=np.array([0.1, 0.7, 0.9], dtype=np.float32),
    )

    saved = save_teacher_path(teacher, tmp_path)
    loaded = load_teacher_path(saved)

    assert loaded.pair_id == teacher.pair_id
    assert loaded.frame_hop == 110
    assert loaded.chroma_shift == 10
    assert loaded.anchor_calibrated is True
    np.testing.assert_array_equal(loaded.path, path)
    np.testing.assert_allclose(loaded.time_b_s, [0.0, 0.2, 0.4])
    np.testing.assert_allclose(loaded.confidence, [0.1, 0.7, 0.9])


def test_oracle_teacher_metric_is_perfect(monkeypatch, tmp_path):
    from dis_alignment.alignment import teacher as teacher_module

    pair = _make_pair(tmp_path)
    monkeypatch.setattr(
        teacher_module,
        "compute_ground_truth_measure_alignment",
        lambda pair_arg: (
            pd.DataFrame({"event_id": ["1", "2", "3"], "time_s": [0.0, 1.0, 2.0]}),
            pd.DataFrame({"event_id": ["1", "2", "3"], "time_s": [0.1, 1.2, 2.3]}),
        ),
    )

    row = teacher_module.evaluate_oracle_alignment(pair)

    assert row["method"] == "oracle"
    assert row["mae"] == 0.0
    assert row["ar_50ms"] == 1.0


def test_trim_active_audio_preserves_absolute_offset():
    from dis_alignment.alignment.teacher import trim_active_audio

    audio = np.concatenate(
        [
            np.zeros(200, dtype=np.float32),
            np.ones(600, dtype=np.float32),
            np.zeros(200, dtype=np.float32),
        ]
    )

    trimmed, start_s, end_s = trim_active_audio(
        audio,
        sr=100,
        trim_top_db=40,
        frame_length=64,
        hop_length=16,
    )

    assert trimmed.size < audio.size
    assert start_s > 0
    assert end_s < len(audio) / 100


def test_chroma_shift_estimation_returns_shift_for_first_sequence():
    from dis_alignment.alignment.teacher import estimate_chroma_shift_to_first

    rng = np.random.default_rng(7)
    chroma_b = rng.random((12, 24), dtype=np.float64)
    chroma_a = np.roll(chroma_b, 5, axis=0)

    shift = estimate_chroma_shift_to_first(chroma_a, chroma_b, max_frames=0)

    assert shift == 7


def test_chroma_shift_rolls_only_chroma_sensitive_blocks():
    from dis_alignment.alignment.teacher import apply_chroma_shift_to_blocks

    chroma = np.arange(24, dtype=np.float64).reshape(12, 2)
    dlnco = np.arange(24, 48, dtype=np.float64).reshape(12, 2)
    flux = np.asarray([[99.0, 100.0]])
    onset = np.vstack([dlnco, flux])

    shifted_chroma, shifted_onset = apply_chroma_shift_to_blocks(chroma, onset, 2)

    np.testing.assert_array_equal(shifted_chroma, np.roll(chroma, 2, axis=0))
    np.testing.assert_array_equal(shifted_onset[:12], np.roll(dlnco, 2, axis=0))
    np.testing.assert_array_equal(shifted_onset[12:], flux)


def test_teacher_feature_blocks_allow_chroma_only_onsets():
    from dis_alignment.alignment.teacher import extract_audio_teacher_feature_blocks

    audio = np.sin(np.linspace(0.0, 8.0 * np.pi, 4096, dtype=np.float32))

    chroma, onset = extract_audio_teacher_feature_blocks(
        audio,
        sr=22050,
        hop_length=512,
        chroma_weight=1.0,
        dlnco_weight=0.0,
        spectral_flux_weight=0.0,
    )

    assert chroma.shape[0] == 12
    assert onset.shape == (1, chroma.shape[1])
    assert np.all(onset == 0.0)


def test_teacher_path_confidence_prefers_local_positive_match():
    from dis_alignment.alignment.teacher import compute_teacher_path_confidence

    features_a = np.eye(5, dtype=np.float64)
    features_b = np.eye(5, dtype=np.float64)
    path = np.array([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], dtype=np.intp)

    confidence = compute_teacher_path_confidence(
        features_a,
        features_b,
        path,
        local_radius_frames=2,
        exclusion_radius_frames=0,
    )

    assert confidence.shape == (5,)
    assert float(confidence[2]) > 0.95


def test_teacher_path_confidence_drops_ambiguous_region():
    from dis_alignment.alignment.teacher import compute_teacher_path_confidence

    features_a = np.ones((3, 4), dtype=np.float64)
    features_b = np.ones((3, 4), dtype=np.float64)
    path = np.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.intp)

    confidence = compute_teacher_path_confidence(
        features_a,
        features_b,
        path,
        local_radius_frames=2,
        exclusion_radius_frames=0,
    )

    assert np.all(confidence < 0.6)


def test_reciprocal_pseudo_path_keeps_monotonic_matches():
    from dis_alignment.alignment.teacher import mine_reciprocal_pseudo_path

    features_a = np.eye(8, dtype=np.float64)
    features_b = np.eye(8, dtype=np.float64)

    path, confidence = mine_reciprocal_pseudo_path(
        features_a,
        features_b,
        min_confidence=0.5,
        min_gap_frames=2,
        max_anchors=8,
        chunk_size=3,
    )

    np.testing.assert_array_equal(path, np.array([[0, 2, 4, 6], [0, 2, 4, 6]], dtype=np.intp))
    assert confidence.shape == (4,)
    assert np.all(confidence > 0.5)


def test_reciprocal_pseudo_path_returns_empty_for_ambiguous_features():
    from dis_alignment.alignment.teacher import mine_reciprocal_pseudo_path

    features_a = np.ones((3, 6), dtype=np.float64)
    features_b = np.ones((3, 6), dtype=np.float64)

    path, confidence = mine_reciprocal_pseudo_path(
        features_a,
        features_b,
        min_confidence=0.8,
        min_gap_frames=1,
        max_anchors=8,
        chunk_size=2,
    )

    assert path.shape == (2, 0)
    assert confidence.size == 0


def test_guided_pseudo_path_uses_coarse_corridor():
    from dis_alignment.alignment.teacher import mine_reciprocal_pseudo_path

    features_a = np.eye(6, dtype=np.float64)
    features_b = np.roll(np.eye(6, dtype=np.float64), 1, axis=1)
    guide = np.array([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 5]], dtype=np.intp)

    path, confidence = mine_reciprocal_pseudo_path(
        features_a,
        features_b,
        min_confidence=0.5,
        min_gap_frames=1,
        max_anchors=8,
        guide_path=guide,
        guide_radius_frames=0,
        chunk_size=3,
    )

    assert path.shape[1] >= 4
    np.testing.assert_array_less(np.abs(path[1] - np.minimum(path[0] + 1, 5)), 1)
    assert np.all(confidence > 0.5)


def test_sample_coarse_teacher_path_respects_spacing_and_limit(tmp_path):
    from dis_alignment.alignment.teacher import _sample_coarse_teacher_path

    pair = _make_pair(tmp_path)
    guide_path = np.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        ],
        dtype=np.intp,
    )

    teacher = _sample_coarse_teacher_path(
        pair,
        guide_path=guide_path,
        guide_time_a_s=np.arange(10, dtype=np.float64) * 0.1 + 1.5,
        guide_time_b_s=np.arange(10, dtype=np.float64) * 0.1 + 2.0,
        guide_confidence=np.linspace(0.1, 1.0, 10, dtype=np.float32),
        sr=100,
        hop_length=10,
        min_gap_frames=3,
        max_anchors=3,
    )

    np.testing.assert_array_equal(teacher.path, np.array([[0, 3, 9], [0, 3, 9]], dtype=np.intp))
    np.testing.assert_allclose(teacher.time_a_s, [1.5, 1.8, 2.4])
    np.testing.assert_allclose(teacher.time_b_s, [2.0, 2.3, 2.9])
    np.testing.assert_allclose(teacher.confidence, [0.1, 0.4, 1.0])


def test_anchor_calibrated_teacher_hits_measure_anchors(monkeypatch, tmp_path):
    from dis_alignment.alignment import teacher as teacher_module

    pair = _make_pair(tmp_path)
    monkeypatch.setattr(
        teacher_module,
        "compute_ground_truth_measure_alignment",
        lambda pair_arg: (
            pd.DataFrame({"event_id": ["1", "2", "3"], "time_s": [0.0, 1.0, 2.0]}),
            pd.DataFrame({"event_id": ["1", "2", "3"], "time_s": [0.0, 1.5, 3.0]}),
        ),
    )
    raw_teacher = teacher_module.TeacherPath(
        pair_id=pair.pair_id,
        lied_id=pair.lied_id,
        piece_a_id=pair.piece_a.piece_id,
        piece_b_id=pair.piece_b.piece_id,
        frame_hop=10,
        sr=100,
        path=np.array([[0, 10, 20], [0, 10, 20]], dtype=np.intp),
        time_a_s=np.array([0.0, 1.0, 2.0]),
        time_b_s=np.array([0.0, 1.0, 2.0]),
    )

    calibrated = teacher_module.calibrate_teacher_path_to_measure_anchors(pair, raw_teacher)
    row = teacher_module.evaluate_teacher_path(pair, calibrated, method="calibrated")

    assert calibrated.anchor_calibrated is True
    np.testing.assert_allclose(calibrated.time_b_s, [0.0, 1.5, 3.0])
    assert row["mae"] == 0.0
    assert row["ar_50ms"] == 1.0
