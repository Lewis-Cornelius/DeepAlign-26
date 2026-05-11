"""Tests for the expanded results-first evaluation workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dis_alignment.analysis.statistics import friedman_nemenyi_analysis
from dis_alignment.data.mazurka import MazurkaDataset, MazurkaPair, MazurkaPerformance
from dis_alignment.data.swd import SWDPair, SWDPiece
from dis_alignment.evaluation import common as eval_common
from dis_alignment.evaluation import mazurka as mazurka_eval
from dis_alignment.evaluation import swd as swd_eval


def test_parse_methods_defaults_and_validation():
    assert eval_common.parse_methods(None, checkpoint_path=None) == ["chroma_dtw"]
    assert eval_common.parse_methods(None, checkpoint_path="checkpoint.pt") == [
        "chroma_dtw",
        "deepalign",
    ]
    assert eval_common.parse_methods("chroma_dtw,mrmsdtw") == ["chroma_dtw", "mrmsdtw"]

    try:
        eval_common.parse_methods("deepalign")
    except ValueError as exc:
        assert "--checkpoint" in str(exc)
    else:
        raise AssertionError("Expected deepalign without checkpoint to fail")

    assert eval_common.parse_deep_decode("deepalign_transcription_fused") == "deepalign_transcription_fused"
    assert eval_common.parse_deep_decode("deepalign_transcription_fused_refined") == "deepalign_transcription_fused_refined"
    assert eval_common.parse_deep_decode("deepalign_transcription_guided") == "deepalign_transcription_guided"
    assert eval_common.parse_deep_decode("deepalign_score_guided_refined") == "deepalign_score_guided_refined"


def test_transcription_fusion_weights_affect_feature_matrix(monkeypatch, tmp_path):
    from dis_alignment.features import chroma as chroma_module
    from dis_alignment.features import dlnco as dlnco_module
    from dis_alignment.features import transcription as transcription_module

    audio_path = tmp_path / "a.wav"
    audio_path.write_bytes(b"placeholder")
    deep = np.ones((2, 4), dtype=np.float64)

    monkeypatch.setattr(
        transcription_module,
        "extract_basic_pitch_features",
        lambda *args, **kwargs: {
            "note": np.ones((3, 4), dtype=np.float64),
            "onset": np.ones((3, 4), dtype=np.float64) * 2.0,
            "contour": np.ones((3, 4), dtype=np.float64) * 3.0,
            "pitch_class_note": np.ones((12, 4), dtype=np.float64),
            "pitch_class_onset": np.ones((12, 4), dtype=np.float64),
        },
    )
    monkeypatch.setattr(chroma_module, "extract_chroma_cqt", lambda *args, **kwargs: np.ones((12, 4)))
    monkeypatch.setattr(dlnco_module, "extract_dlnco", lambda *args, **kwargs: np.ones((12, 4)))

    base = eval_common._build_transcription_fused_features(
        audio=np.ones(100, dtype=np.float32),
        audio_path=audio_path,
        deep_features=deep,
        sr=10,
        frame_hop=1,
        transcription_cache_root=tmp_path,
        fusion_deep_weight=1.0,
        fusion_onset_weight=1.0,
        fusion_note_weight=0.75,
        fusion_dlnco_weight=0.5,
        fusion_chroma_weight=0.25,
    )
    changed = eval_common._build_transcription_fused_features(
        audio=np.ones(100, dtype=np.float32),
        audio_path=audio_path,
        deep_features=deep,
        sr=10,
        frame_hop=1,
        transcription_cache_root=tmp_path,
        fusion_deep_weight=1.0,
        fusion_onset_weight=0.5,
        fusion_note_weight=0.75,
        fusion_dlnco_weight=0.5,
        fusion_chroma_weight=0.25,
    )

    assert base.shape[1] == 4
    assert np.isfinite(base).all()
    assert base.shape == changed.shape
    assert not np.allclose(base, changed)


def test_transcription_guided_decode_returns_deepalign_row(monkeypatch, tmp_path):
    from dis_alignment.features import chroma as chroma_module
    from dis_alignment.features import dlnco as dlnco_module
    from dis_alignment.features import transcription as transcription_module
    from dis_alignment.model import inference as inference_module

    path_a = tmp_path / "a.wav"
    path_b = tmp_path / "b.wav"
    path_a.write_bytes(b"a")
    path_b.write_bytes(b"b")

    monkeypatch.setattr(
        inference_module,
        "extract_deep_features",
        lambda *args, **kwargs: np.array(
            [[1.0, 0.8, 0.2, 0.0], [0.0, 0.2, 0.8, 1.0]],
            dtype=np.float64,
        ),
    )
    monkeypatch.setattr(chroma_module, "extract_chroma_cqt", lambda *args, **kwargs: np.eye(4, dtype=np.float64))
    monkeypatch.setattr(dlnco_module, "extract_dlnco", lambda *args, **kwargs: np.eye(4, dtype=np.float64))
    monkeypatch.setattr(
        transcription_module,
        "extract_basic_pitch_features",
        lambda *args, **kwargs: {
            "note": np.eye(4, dtype=np.float64),
            "onset": np.eye(4, dtype=np.float64),
            "contour": np.eye(4, dtype=np.float64),
            "pitch_class_note": np.eye(12, 4, dtype=np.float64),
            "pitch_class_onset": np.eye(12, 4, dtype=np.float64),
        },
    )

    rows = eval_common.evaluate_pairwise_methods(
        methods=["deepalign"],
        pair_id="p",
        group_id="g",
        dataset_name="swd",
        piece_a_id="a",
        piece_b_id="b",
        audio_a=np.ones(100, dtype=np.float32),
        audio_b=np.ones(100, dtype=np.float32),
        gt_a=np.array([0.0, 0.1, 0.2]),
        gt_b=np.array([0.0, 0.1, 0.2]),
        chroma_hop=1,
        sr=10,
        encoder=object(),
        deep_hop=1,
        pool_size=1,
        audio_a_path=path_a,
        audio_b_path=path_b,
        deep_decode="deepalign_transcription_guided",
        refine_window_sec=0.2,
    )

    assert len(rows) == 1
    assert rows[0]["method"] == "deepalign"
    assert rows[0]["deep_decode"] == "deepalign_transcription_guided"
    assert rows[0]["ar_50ms"] == 1.0


def test_score_guided_decode_returns_deepalign_row(monkeypatch, tmp_path):
    from dis_alignment.features import chroma as chroma_module
    from dis_alignment.features import dlnco as dlnco_module
    from dis_alignment.features import transcription as transcription_module
    from dis_alignment.model import inference as inference_module

    path_a = tmp_path / "a.wav"
    path_b = tmp_path / "b.wav"
    score_path = tmp_path / "score.xml"
    path_a.write_bytes(b"a")
    path_b.write_bytes(b"b")
    score_path.write_text("<score-partwise/>", encoding="utf-8")

    monkeypatch.setattr(
        inference_module,
        "extract_deep_features",
        lambda *args, **kwargs: np.eye(4, dtype=np.float64),
    )
    monkeypatch.setattr(chroma_module, "extract_chroma_cqt", lambda *args, **kwargs: np.eye(12, 4, dtype=np.float64))
    monkeypatch.setattr(dlnco_module, "extract_dlnco", lambda *args, **kwargs: np.eye(12, 4, dtype=np.float64))
    monkeypatch.setattr(
        transcription_module,
        "extract_basic_pitch_features",
        lambda *args, **kwargs: {
            "note": np.eye(4, dtype=np.float64),
            "onset": np.eye(4, dtype=np.float64),
            "contour": np.eye(4, dtype=np.float64),
            "pitch_class_note": np.eye(12, 4, dtype=np.float64),
            "pitch_class_onset": np.eye(12, 4, dtype=np.float64),
        },
    )
    monkeypatch.setattr(
        eval_common,
        "_build_score_reference_features",
        lambda **kwargs: (np.eye(48, 4, dtype=np.float64), np.array([0.0, 0.1, 0.2]), 0.1),
    )
    monkeypatch.setattr(
        eval_common,
        "fast_dtw_align",
        lambda a, b, distance="sqeuclidean": (
            np.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.intp),
            0.0,
            0.01,
        ),
    )

    rows = eval_common.evaluate_pairwise_methods(
        methods=["deepalign"],
        pair_id="p",
        group_id="g",
        dataset_name="swd",
        piece_a_id="a",
        piece_b_id="b",
        audio_a=np.ones(100, dtype=np.float32),
        audio_b=np.ones(100, dtype=np.float32),
        gt_a=np.array([0.0, 0.1, 0.2]),
        gt_b=np.array([0.0, 0.1, 0.2]),
        chroma_hop=1,
        sr=10,
        encoder=object(),
        deep_hop=1,
        pool_size=1,
        audio_a_path=path_a,
        audio_b_path=path_b,
        score_path=score_path,
        event_ids=["1", "2", "3"],
        deep_decode="deepalign_score_guided_refined",
        score_refine_radius_sec=0.0,
    )

    assert len(rows) == 1
    assert rows[0]["method"] == "deepalign"
    assert rows[0]["deep_decode"] == "deepalign_score_guided_refined"
    assert rows[0]["ar_50ms"] == 1.0


def test_extract_musicxml_measure_positions(tmp_path):
    score_path = tmp_path / "score.musicxml"
    score_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
  <part-list>
    <score-part id="P1"><part-name>Music</part-name></score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes><divisions>2</divisions></attributes>
      <note><rest/><duration>8</duration></note>
    </measure>
    <measure number="2">
      <note><rest/><duration>4</duration></note>
    </measure>
  </part>
</score-partwise>
""",
        encoding="utf-8",
    )

    positions = eval_common.extract_musicxml_measure_positions(score_path)
    assert positions["event_id"].tolist() == ["1", "2"]
    np.testing.assert_allclose(positions["score_position"].to_numpy(), np.array([0.0, 4.0]))


def test_swd_evaluate_pair_supports_all_methods(monkeypatch, tmp_path):
    pair = SWDPair(
        pair_id="D911-01_AL98_SC06",
        lied_id="D911-01",
        piece_a=SWDPiece(
            piece_id="D911-01_AL98",
            lied_id="D911-01",
            performance_id="AL98",
            audio_path=tmp_path / "a.wav",
        ),
        piece_b=SWDPiece(
            piece_id="D911-01_SC06",
            lied_id="D911-01",
            performance_id="SC06",
            audio_path=tmp_path / "b.wav",
        ),
    )

    dataset = type(
        "Dataset",
        (),
        {"get_score_path": lambda self, lied_id: tmp_path / "score.musicxml"},
    )()

    monkeypatch.setattr(
        swd_eval,
        "compute_ground_truth_measure_alignment",
        lambda _pair: (
            pd.DataFrame({"event_id": ["1", "2", "3"], "time_s": [0.0, 1.0, 2.0]}),
            pd.DataFrame({"event_id": ["1", "2", "3"], "time_s": [0.0, 1.0, 2.0]}),
        ),
    )
    monkeypatch.setattr(
        swd_eval,
        "load_swd_audio",
        lambda piece, sr=10: (np.linspace(0.0, 1.0, 30, dtype=np.float32), sr),
    )
    monkeypatch.setattr(
        eval_common,
        "fast_dtw_align",
        lambda a, b, distance="cosine": (
            np.array([[0, 1, 2], [0, 1, 2]], dtype=np.intp),
            0.0,
            0.01,
        ),
    )
    monkeypatch.setattr(
        eval_common,
        "align_mrmsdtw",
        lambda a, b, memory_limit_mb=500, **kwargs: type(
            "Result",
            (),
            {
                "path": np.array([[0, 1, 2], [0, 1, 2]], dtype=np.intp),
                "runtime_seconds": 0.02,
                "memory_bytes": 1024,
            },
        )(),
    )
    monkeypatch.setattr(
        swd_eval,
        "_get_matchmaker_predictions",
        lambda **kwargs: (
            pd.DataFrame({"event_id": ["1", "2", "3"], "time_s": [0.0, 1.0, 2.0]}),
            type("MatchmakerResult", (), {"times_s": np.array([0.0, 2.0]), "runtime_seconds": 0.03})(),
        ),
    )

    from dis_alignment.features import chroma as chroma_module
    from dis_alignment.model import inference as inference_module

    monkeypatch.setattr(
        chroma_module,
        "extract_chroma_cqt",
        lambda audio, sr=10, hop_length=1: np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], dtype=np.float32),
    )
    monkeypatch.setattr(
        inference_module,
        "extract_deep_features",
        lambda audio, encoder, sr=10, hop_length=1, device=None: np.array(
            [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
            dtype=np.float32,
        ),
    )

    results = swd_eval.evaluate_pair(
        pair,
        dataset=dataset,
        methods=["chroma_dtw", "mrmsdtw", "deepalign", "matchmaker"],
        encoder=object(),
        sr=10,
        chroma_hop=1,
        deep_hop=1,
        pool_size=1,
    )

    assert {row["method"] for row in results} == {"chroma_dtw", "mrmsdtw", "deepalign", "matchmaker"}
    assert all(row["dataset"] == "swd" for row in results)


def test_swd_evaluate_pair_skips_matchmaker_failure(monkeypatch, tmp_path):
    pair = SWDPair(
        pair_id="D911-01_AL98_SC06",
        lied_id="D911-01",
        piece_a=SWDPiece(
            piece_id="D911-01_AL98",
            lied_id="D911-01",
            performance_id="AL98",
            audio_path=tmp_path / "a.wav",
        ),
        piece_b=SWDPiece(
            piece_id="D911-01_SC06",
            lied_id="D911-01",
            performance_id="SC06",
            audio_path=tmp_path / "b.wav",
        ),
    )

    dataset = type(
        "Dataset",
        (),
        {"get_score_path": lambda self, lied_id: tmp_path / "score.musicxml"},
    )()

    monkeypatch.setattr(
        swd_eval,
        "compute_ground_truth_measure_alignment",
        lambda _pair: (
            pd.DataFrame({"event_id": ["1", "2", "3"], "time_s": [0.0, 1.0, 2.0]}),
            pd.DataFrame({"event_id": ["1", "2", "3"], "time_s": [0.0, 1.0, 2.0]}),
        ),
    )
    monkeypatch.setattr(
        swd_eval,
        "load_swd_audio",
        lambda piece, sr=10: (np.linspace(0.0, 1.0, 30, dtype=np.float32), sr),
    )

    from dis_alignment.features import chroma as chroma_module

    monkeypatch.setattr(
        chroma_module,
        "extract_chroma_cqt",
        lambda audio, sr=10, hop_length=1: np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], dtype=np.float32),
    )
    monkeypatch.setattr(
        eval_common,
        "fast_dtw_align",
        lambda a, b, distance="cosine": (
            np.array([[0, 1, 2], [0, 1, 2]], dtype=np.intp),
            0.0,
            0.01,
        ),
    )
    monkeypatch.setattr(
        swd_eval,
        "_get_matchmaker_predictions",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("queue timeout")),
    )

    results = swd_eval.evaluate_pair(
        pair,
        dataset=dataset,
        methods=["chroma_dtw", "matchmaker"],
        sr=10,
        chroma_hop=1,
        deep_hop=1,
        pool_size=1,
    )

    assert [row["method"] for row in results] == ["chroma_dtw"]


def test_swd_dataset_evaluation_fails_on_pair_error(monkeypatch, tmp_path):
    pair = SWDPair(
        pair_id="D911-01_AL98_SC06",
        lied_id="D911-01",
        piece_a=SWDPiece(
            piece_id="D911-01_AL98",
            lied_id="D911-01",
            performance_id="AL98",
            audio_path=tmp_path / "a.wav",
        ),
        piece_b=SWDPiece(
            piece_id="D911-01_SC06",
            lied_id="D911-01",
            performance_id="SC06",
            audio_path=tmp_path / "b.wav",
        ),
    )
    dataset = type("Dataset", (), {"iter_pairs": lambda self, **kwargs: [pair]})()

    monkeypatch.setattr(
        swd_eval,
        "evaluate_pair",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("decoder failed")),
    )

    with pytest.raises(RuntimeError, match="D911-01_AL98_SC06"):
        swd_eval.evaluate_swd_dataset(
            dataset,
            methods="chroma_dtw",
            show_progress=False,
        )


def test_swd_dataset_evaluation_records_allowed_skips(monkeypatch, tmp_path):
    pair = SWDPair(
        pair_id="D911-01_AL98_SC06",
        lied_id="D911-01",
        piece_a=SWDPiece(
            piece_id="D911-01_AL98",
            lied_id="D911-01",
            performance_id="AL98",
            audio_path=tmp_path / "a.wav",
        ),
        piece_b=SWDPiece(
            piece_id="D911-01_SC06",
            lied_id="D911-01",
            performance_id="SC06",
            audio_path=tmp_path / "b.wav",
        ),
    )
    dataset = type("Dataset", (), {"iter_pairs": lambda self, **kwargs: [pair]})()

    monkeypatch.setattr(
        swd_eval,
        "evaluate_pair",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("decoder failed")),
    )

    results = swd_eval.evaluate_swd_dataset(
        dataset,
        methods="chroma_dtw",
        show_progress=False,
        allow_skips=True,
    )

    failures = results.attrs["failures"]
    assert results.empty
    assert failures.loc[0, "pair_id"] == pair.pair_id
    assert failures.loc[0, "error_type"] == "RuntimeError"


def test_swd_dataset_evaluation_fails_when_requested_method_is_missing(monkeypatch, tmp_path):
    pair = SWDPair(
        pair_id="D911-01_AL98_SC06",
        lied_id="D911-01",
        piece_a=SWDPiece(
            piece_id="D911-01_AL98",
            lied_id="D911-01",
            performance_id="AL98",
            audio_path=tmp_path / "a.wav",
        ),
        piece_b=SWDPiece(
            piece_id="D911-01_SC06",
            lied_id="D911-01",
            performance_id="SC06",
            audio_path=tmp_path / "b.wav",
        ),
    )
    dataset = type("Dataset", (), {"iter_pairs": lambda self, **kwargs: [pair]})()

    monkeypatch.setattr(
        swd_eval,
        "evaluate_pair",
        lambda *args, **kwargs: [
            {
                "method": "chroma_dtw",
                "mae": 0.0,
                "median_ae": 0.0,
                "ar_50ms": 1.0,
                "ar_100ms": 1.0,
                "ar_200ms": 1.0,
                "runtime_s": 0.0,
            }
        ],
    )

    with pytest.raises(RuntimeError, match="matchmaker"):
        swd_eval.evaluate_swd_dataset(
            dataset,
            methods="chroma_dtw,matchmaker",
            show_progress=False,
        )


def test_mazurka_dataset_discovers_pairs_from_metadata(tmp_path):
    audio_dir = tmp_path / "audio"
    ann_dir = tmp_path / "annotations"
    score_dir = tmp_path / "scores"
    audio_dir.mkdir()
    ann_dir.mkdir()
    score_dir.mkdir()

    for name in ("perf_a.wav", "perf_b.wav"):
        (audio_dir / name).write_bytes(b"")
        (ann_dir / f"{Path(name).stem}.csv").write_text("beat,time\n1,0.0\n2,1.0\n", encoding="utf-8")
    (score_dir / "mazurka-01.musicxml").write_text("<score-partwise/>", encoding="utf-8")
    (tmp_path / "metadata.csv").write_text(
        "work_id,performance_id,audio_path,annotation_path,score_path\n"
        "mazurka-01,perf_a,audio/perf_a.wav,annotations/perf_a.csv,scores/mazurka-01.musicxml\n"
        "mazurka-01,perf_b,audio/perf_b.wav,annotations/perf_b.csv,scores/mazurka-01.musicxml\n",
        encoding="utf-8",
    )

    dataset = MazurkaDataset(tmp_path)
    pairs = list(dataset.iter_pairs())
    assert len(dataset.performances) == 2
    assert len(pairs) == 1
    assert pairs[0].work_id == "mazurka-01"


def test_mazurka_evaluate_pair_supports_matchmaker(monkeypatch, tmp_path):
    pair = MazurkaPair(
        pair_id="mazurka-01_a_b",
        work_id="mazurka-01",
        piece_a=MazurkaPerformance(
            performance_id="a",
            work_id="mazurka-01",
            audio_path=tmp_path / "a.wav",
            annotation_path=tmp_path / "a.csv",
            score_path=tmp_path / "score.musicxml",
        ),
        piece_b=MazurkaPerformance(
            performance_id="b",
            work_id="mazurka-01",
            audio_path=tmp_path / "b.wav",
            annotation_path=tmp_path / "b.csv",
            score_path=tmp_path / "score.musicxml",
        ),
    )

    monkeypatch.setattr(
        mazurka_eval,
        "compute_ground_truth_alignment",
        lambda _pair: (
            pd.DataFrame({"event_id": ["1", "2", "3"], "time_s": [0.0, 1.0, 2.0]}),
            pd.DataFrame({"event_id": ["1", "2", "3"], "time_s": [0.0, 1.0, 2.0]}),
        ),
    )
    monkeypatch.setattr(
        mazurka_eval,
        "load_mazurka_audio",
        lambda piece, sr=10: (np.linspace(0.0, 1.0, 30, dtype=np.float32), sr),
    )
    monkeypatch.setattr(
        eval_common,
        "fast_dtw_align",
        lambda a, b, distance="cosine": (
            np.array([[0, 1, 2], [0, 1, 2]], dtype=np.intp),
            0.0,
            0.01,
        ),
    )
    monkeypatch.setattr(
        eval_common,
        "align_mrmsdtw",
        lambda a, b, memory_limit_mb=500, **kwargs: type(
            "Result",
            (),
            {
                "path": np.array([[0, 1, 2], [0, 1, 2]], dtype=np.intp),
                "runtime_seconds": 0.02,
                "memory_bytes": 1024,
            },
        )(),
    )
    monkeypatch.setattr(
        mazurka_eval,
        "_get_matchmaker_predictions",
        lambda **kwargs: (
            pd.DataFrame({"event_id": ["1", "2", "3"], "time_s": [0.0, 1.0, 2.0]}),
            type("MatchmakerResult", (), {"times_s": np.array([0.0, 2.0]), "runtime_seconds": 0.03})(),
        ),
    )

    from dis_alignment.features import chroma as chroma_module

    monkeypatch.setattr(
        chroma_module,
        "extract_chroma_cqt",
        lambda audio, sr=10, hop_length=1: np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], dtype=np.float32),
    )

    results = mazurka_eval.evaluate_pair(
        pair,
        methods=["chroma_dtw", "mrmsdtw", "matchmaker"],
        sr=10,
        chroma_hop=1,
        deep_hop=1,
        pool_size=1,
    )
    assert {row["method"] for row in results} == {"chroma_dtw", "mrmsdtw", "matchmaker"}
    assert all(row["dataset"] == "mazurka" for row in results)


def test_mazurka_evaluate_pair_skips_matchmaker_failure(monkeypatch, tmp_path):
    pair = MazurkaPair(
        pair_id="mazurka-01_a_b",
        work_id="mazurka-01",
        piece_a=MazurkaPerformance(
            performance_id="a",
            work_id="mazurka-01",
            audio_path=tmp_path / "a.wav",
            annotation_path=tmp_path / "a.csv",
            score_path=tmp_path / "score.musicxml",
        ),
        piece_b=MazurkaPerformance(
            performance_id="b",
            work_id="mazurka-01",
            audio_path=tmp_path / "b.wav",
            annotation_path=tmp_path / "b.csv",
            score_path=tmp_path / "score.musicxml",
        ),
    )

    monkeypatch.setattr(
        mazurka_eval,
        "compute_ground_truth_alignment",
        lambda _pair: (
            pd.DataFrame({"event_id": ["1", "2", "3"], "time_s": [0.0, 1.0, 2.0]}),
            pd.DataFrame({"event_id": ["1", "2", "3"], "time_s": [0.0, 1.0, 2.0]}),
        ),
    )
    monkeypatch.setattr(
        mazurka_eval,
        "load_mazurka_audio",
        lambda piece, sr=10: (np.linspace(0.0, 1.0, 30, dtype=np.float32), sr),
    )

    from dis_alignment.features import chroma as chroma_module

    monkeypatch.setattr(
        chroma_module,
        "extract_chroma_cqt",
        lambda audio, sr=10, hop_length=1: np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], dtype=np.float32),
    )
    monkeypatch.setattr(
        eval_common,
        "fast_dtw_align",
        lambda a, b, distance="cosine": (
            np.array([[0, 1, 2], [0, 1, 2]], dtype=np.intp),
            0.0,
            0.01,
        ),
    )
    monkeypatch.setattr(
        mazurka_eval,
        "_get_matchmaker_predictions",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("queue timeout")),
    )

    results = mazurka_eval.evaluate_pair(
        pair,
        methods=["chroma_dtw", "matchmaker"],
        sr=10,
        chroma_hop=1,
        deep_hop=1,
        pool_size=1,
    )

    assert [row["method"] for row in results] == ["chroma_dtw"]


def test_friedman_nemenyi_analysis_runs():
    results = pd.DataFrame(
        {
            "pair_id": ["p1", "p1", "p1", "p2", "p2", "p2", "p3", "p3", "p3", "p4", "p4", "p4", "p5", "p5", "p5"],
            "method": [
                "chroma_dtw",
                "mrmsdtw",
                "deepalign",
                "chroma_dtw",
                "mrmsdtw",
                "deepalign",
                "chroma_dtw",
                "mrmsdtw",
                "deepalign",
                "chroma_dtw",
                "mrmsdtw",
                "deepalign",
                "chroma_dtw",
                "mrmsdtw",
                "deepalign",
            ],
            "mae": [0.10, 0.08, 0.05, 0.12, 0.09, 0.04, 0.11, 0.07, 0.03, 0.10, 0.08, 0.02, 0.09, 0.07, 0.03],
            "ar_50ms": [0.8] * 15,
            "duration_s": [10.0] * 15,
            "runtime_s": [1.0] * 15,
            "memory_mb": [100.0] * 15,
        }
    )

    analysis = friedman_nemenyi_analysis(results, metric="mae")
    assert analysis["available"] is True
    assert analysis["friedman_significant"] in {True, False}
    assert set(analysis["average_ranks"]) == {"chroma_dtw", "mrmsdtw", "deepalign"}
    assert len(analysis["nemenyi"]) == 3


def test_merge_evaluation_results_deduplicates_by_pair_and_method(tmp_path):
    first = pd.DataFrame(
        [
            {
                "dataset": "swd",
                "pair_id": "pair-1",
                "group_id": "lied-1",
                "piece_a_id": "a",
                "piece_b_id": "b",
                "method": "chroma_dtw",
                "mae": 0.30,
                "median_ae": 0.20,
                "ar_50ms": 0.50,
                "ar_100ms": 0.60,
                "ar_200ms": 0.70,
                "runtime_s": 1.0,
                "duration_s": 10.0,
                "memory_mb": 100.0,
                "n_gt_points": 32,
            },
            {
                "dataset": "swd",
                "pair_id": "pair-1",
                "group_id": "lied-1",
                "piece_a_id": "a",
                "piece_b_id": "b",
                "method": "deepalign",
                "mae": 0.10,
                "median_ae": 0.08,
                "ar_50ms": 0.90,
                "ar_100ms": 0.95,
                "ar_200ms": 0.98,
                "runtime_s": 2.0,
                "duration_s": 10.0,
                "memory_mb": np.nan,
                "n_gt_points": 32,
            },
        ]
    )
    second = pd.DataFrame(
        [
            {
                "dataset": "swd",
                "pair_id": "pair-1",
                "group_id": "lied-1",
                "piece_a_id": "a",
                "piece_b_id": "b",
                "method": "mrmsdtw",
                "mae": 0.12,
                "median_ae": 0.11,
                "ar_50ms": 0.85,
                "ar_100ms": 0.90,
                "ar_200ms": 0.96,
                "runtime_s": 0.8,
                "duration_s": 10.0,
                "memory_mb": 64.0,
                "n_gt_points": 32,
            },
            {
                "dataset": "swd",
                "pair_id": "pair-1",
                "group_id": "lied-1",
                "piece_a_id": "a",
                "piece_b_id": "b",
                "method": "deepalign",
                "mae": 0.09,
                "median_ae": 0.07,
                "ar_50ms": 0.92,
                "ar_100ms": 0.96,
                "ar_200ms": 0.99,
                "runtime_s": 1.9,
                "duration_s": 10.0,
                "memory_mb": np.nan,
                "n_gt_points": 32,
            },
        ]
    )

    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    first.to_csv(first_path, index=False)
    second.to_csv(second_path, index=False)

    merged = eval_common.merge_evaluation_results([first_path, second_path])

    assert len(merged) == 3
    assert set(merged["method"]) == {"chroma_dtw", "mrmsdtw", "deepalign"}
    deepalign_row = merged[merged["method"] == "deepalign"].iloc[0]
    assert deepalign_row["mae"] == 0.09


def test_deepalign_decode_variants_are_not_collapsed(tmp_path):
    base_row = {
        "dataset": "swd",
        "pair_id": "pair-1",
        "group_id": "lied-1",
        "piece_a_id": "a",
        "piece_b_id": "b",
        "method": "deepalign",
        "median_ae": 0.08,
        "ar_50ms": 0.90,
        "ar_100ms": 0.95,
        "ar_200ms": 0.98,
        "runtime_s": 2.0,
        "duration_s": 10.0,
        "memory_mb": np.nan,
        "n_gt_points": 32,
    }
    unconstrained = pd.DataFrame([{**base_row, "mae": 0.10, "deep_decode": "unconstrained"}])
    fused = pd.DataFrame(
        [{**base_row, "mae": 0.05, "deep_decode": "deepalign_transcription_fused"}]
    )

    unconstrained_path = tmp_path / "unconstrained.csv"
    fused_path = tmp_path / "fused.csv"
    unconstrained.to_csv(unconstrained_path, index=False)
    fused.to_csv(fused_path, index=False)

    merged = eval_common.merge_evaluation_results([unconstrained_path, fused_path])
    assert len(merged) == 2
    assert set(merged["deep_decode"]) == {"unconstrained", "deepalign_transcription_fused"}

    summary = eval_common.summarize_evaluation(merged)
    assert set(summary) == {
        "deepalign:unconstrained",
        "deepalign:deepalign_transcription_fused",
    }

    ambiguous = eval_common.check_success_criteria(merged)
    assert not ambiguous["available"]
    assert "Multiple DeepAlign variants" in ambiguous["error"]

    fused_success = eval_common.check_success_criteria(
        merged,
        candidate_deep_decode="deepalign_transcription_fused",
    )
    assert fused_success["available"]
    assert fused_success["mae_seconds"] == 0.05
