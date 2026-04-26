"""Smoke tests for the DeepAlign-first public project surface."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from click.testing import CliRunner


def test_package_import_smoke():
    package = importlib.import_module("dis_alignment")
    assert hasattr(package, "SWDDataset")
    assert hasattr(package, "align_with_deep_features")
    assert hasattr(package, "align_global_dtw")


def test_cli_help_mentions_deepalign_workflow():
    from dis_alignment.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "prepare-swd" in result.output
    assert "evaluate-swd" in result.output
    assert "evaluate-mazurka" in result.output
    assert "legacy-maestro-benchmark" in result.output
    assert "DeepAlign-26 dissertation workflow" in result.output


def test_train_dry_run_creates_artifacts(tmp_path):
    from dis_alignment.model.train import train

    history = train(
        swd_path=None,
        output_dir=str(tmp_path),
        epochs=1,
        batch_size=2,
        embed_dim=8,
        n_freq_bins=24,
        num_conv_channels=[8, 8],
        gru_hidden_size=16,
        num_gru_layers=1,
        dropout=0.0,
        augment=False,
        device="cpu",
        save_every_n_epochs=0,
        dry_run=True,
    )

    assert len(history["train_loss"]) == 1
    assert len(history["val_loss"]) == 1
    assert (tmp_path / "best_model.pt").exists()
    assert (tmp_path / "best_model_debug_ar50.pt").exists()
    assert (tmp_path / "best_model_debug_mae.pt").exists()
    assert (tmp_path / "best_model_balanced.pt").exists()
    assert (tmp_path / "final_model.pt").exists()
    assert (tmp_path / "training_history.json").exists()

    with open(tmp_path / "training_history.json", encoding="utf-8") as handle:
        saved_history = json.load(handle)
    assert saved_history["gamma"]


def test_evaluate_pair_smoke(monkeypatch, tmp_path):
    from dis_alignment.data.swd import SWDPair, SWDPiece
    from dis_alignment.evaluation import common as eval_common
    from dis_alignment.evaluation import swd as swd_eval
    from dis_alignment.features import chroma as chroma_module
    from dis_alignment.model import inference as inference_module

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
        chroma_module,
        "extract_chroma_cqt",
        lambda audio, sr=10, hop_length=1: np.array(
            [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
            dtype=np.float32,
        ),
    )
    monkeypatch.setattr(
        inference_module,
        "extract_deep_features",
        lambda audio, encoder, sr=10, hop_length=1, device=None: np.array(
            [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
            dtype=np.float32,
        ),
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

    results = swd_eval.evaluate_pair(
        pair,
        methods=["chroma_dtw", "deepalign"],
        encoder=object(),
        sr=10,
        chroma_hop=1,
        deep_hop=1,
        pool_size=1,
    )

    assert {row["method"] for row in results} == {"chroma_dtw", "deepalign"}
    assert all(row["duration_s"] == 3.0 for row in results)
    assert all("mae" in row for row in results)


def test_scripts_do_not_use_hardcoded_personal_paths():
    repo_root = Path(__file__).resolve().parents[1]
    script_paths = [
        repo_root / "scripts" / "check_collapse.py",
        repo_root / "scripts" / "download_swd.py",
        repo_root / "scripts" / "evaluate_swd.py",
        repo_root / "scripts" / "evaluate_mazurka.py",
    ]

    for script_path in script_paths:
        contents = script_path.read_text(encoding="utf-8")
        assert "c:/code/Dis" not in contents
        assert "C:\\code\\Dis" not in contents
