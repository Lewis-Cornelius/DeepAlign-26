"""Tests for deep learning components (encoder, loss, dataset, augmentation).

All tests use synthetic data — no SWD download required.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from dis_alignment.model.encoder import ConvBlock, CRNNEncoder, DeepAlignModel
from dis_alignment.model.soft_dtw_loss import GammaScheduler, SoftDTWLoss


def _make_pair(tmp_path, lied_id: str, performance_a: str = "HU33", performance_b: str = "SC06"):
    from dis_alignment.data.swd import SWDPair, SWDPiece

    piece_a = SWDPiece(
        piece_id=f"{lied_id}_{performance_a}",
        lied_id=lied_id,
        performance_id=performance_a,
        audio_path=tmp_path / f"{lied_id}_{performance_a}.wav",
    )
    piece_b = SWDPiece(
        piece_id=f"{lied_id}_{performance_b}",
        lied_id=lied_id,
        performance_id=performance_b,
        audio_path=tmp_path / f"{lied_id}_{performance_b}.wav",
    )
    return SWDPair(
        pair_id=f"{lied_id}_{performance_a}_{performance_b}",
        lied_id=lied_id,
        piece_a=piece_a,
        piece_b=piece_b,
    )


def _patch_aligned_sampling(monkeypatch):
    from dis_alignment.model import dataset as dataset_module

    monkeypatch.setattr(
        dataset_module,
        "compute_ground_truth_measure_alignment",
        lambda pair: (
            pd.DataFrame(
                {"event_id": ["1", "2", "3", "4"], "time_s": [0.0, 8.0, 18.0, 28.0]}
            ),
            pd.DataFrame(
                {"event_id": ["1", "2", "3", "4"], "time_s": [0.0, 9.0, 19.0, 29.0]}
            ),
        ),
    )
    monkeypatch.setattr(
        dataset_module,
        "load_swd_audio",
        lambda piece, sr=10: (np.zeros(400, dtype=np.float32), sr),
    )
    monkeypatch.setattr(
        dataset_module.SWDPairDataset,
        "_compute_cqt",
        lambda self, audio: np.ones((2, max(1, len(audio) // 10)), dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Encoder tests
# ---------------------------------------------------------------------------

class TestConvBlock:
    """Tests for the ConvBlock building block."""

    def test_output_shape(self):
        block = ConvBlock(1, 32, kernel_size=(3, 3), pool_size=(2, 1))
        x = torch.randn(2, 1, 84, 100)
        out = block(x)
        assert out.shape == (2, 32, 42, 100)  # freq halved, time preserved

    def test_no_pool(self):
        block = ConvBlock(1, 16, pool_size=None)
        x = torch.randn(2, 1, 84, 50)
        out = block(x)
        assert out.shape == (2, 16, 84, 50)


class TestCRNNEncoder:
    """Tests for the CRNN encoder architecture."""

    @pytest.fixture
    def encoder(self):
        return CRNNEncoder(
            n_freq_bins=84,
            embed_dim=64,
            num_conv_channels=[32, 64],
            gru_hidden_size=64,
            num_gru_layers=1,
            dropout=0.0,
        )

    def test_forward_shape(self, encoder):
        """Output should be (batch, time, embed_dim)."""
        x = torch.randn(2, 1, 84, 100)
        out = encoder(x)
        assert out.shape == (2, 100, 64)

    def test_output_is_l2_normalised(self, encoder):
        """Embeddings should be L2-normalised along the feature dim."""
        x = torch.randn(2, 1, 84, 50)
        out = encoder(x)
        norms = torch.norm(out, p=2, dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)

    def test_different_time_lengths(self, encoder):
        """Encoder should handle varying time dimensions."""
        x1 = torch.randn(1, 1, 84, 30)
        x2 = torch.randn(1, 1, 84, 200)
        out1 = encoder(x1)
        out2 = encoder(x2)
        assert out1.shape == (1, 30, 64)
        assert out2.shape == (1, 200, 64)

    def test_extract_features_no_grad(self, encoder):
        """extract_features should not track gradients."""
        x = torch.randn(1, 1, 84, 20)
        out = encoder.extract_features(x)
        assert not out.requires_grad

    def test_parameter_count_positive(self, encoder):
        n_params = sum(p.numel() for p in encoder.parameters())
        assert n_params > 0


class TestDeepAlignModel:
    """Tests for the full Siamese DeepAlign model."""

    @pytest.fixture
    def model(self):
        enc = CRNNEncoder(
            n_freq_bins=84, embed_dim=32,
            num_conv_channels=[16, 32], gru_hidden_size=32,
            num_gru_layers=1, dropout=0.0,
        )
        return DeepAlignModel(enc)

    def test_forward_returns_pair(self, model):
        sa = torch.randn(2, 1, 84, 40)
        sb = torch.randn(2, 1, 84, 60)
        emb_a, emb_b = model(sa, sb)
        assert emb_a.shape == (2, 40, 32)
        assert emb_b.shape == (2, 60, 32)

    def test_shared_weights(self, model):
        """Both streams should share the same encoder parameters."""
        sa = torch.randn(1, 1, 84, 30)
        # Running same input through both streams should give identical output
        emb_a, emb_b = model(sa, sa)
        torch.testing.assert_close(emb_a, emb_b)


# ---------------------------------------------------------------------------
# Soft-DTW loss tests
# ---------------------------------------------------------------------------

class TestSoftDTWLoss:
    """Tests for the differentiable Soft-DTW loss."""

    def test_identical_sequences_low_loss(self):
        """Identical sequences should produce near-zero normalised loss."""
        loss_fn = SoftDTWLoss(gamma=1.0, normalize=True)
        seq = torch.randn(1, 10, 8)
        loss = loss_fn(seq, seq.clone())
        assert loss.item() < 1e-3

    def test_different_sequences_positive_loss(self):
        """Different sequences should have higher loss than identical."""
        loss_fn = SoftDTWLoss(gamma=1.0, normalize=True)
        a = torch.randn(1, 10, 8)
        b = torch.randn(1, 10, 8)
        loss = loss_fn(a, b)
        assert loss.item() > 0

    def test_gradient_flows(self):
        """Gradients should propagate through Soft-DTW."""
        loss_fn = SoftDTWLoss(gamma=1.0, normalize=False)
        a = torch.randn(1, 5, 4, requires_grad=True)
        b = torch.randn(1, 5, 4)
        loss = loss_fn(a, b)
        loss.backward()
        assert a.grad is not None
        assert not torch.all(a.grad == 0)

    def test_unnormalized_non_negative(self):
        """Unnormalized Soft-DTW should be non-negative for sqeuclidean."""
        loss_fn = SoftDTWLoss(gamma=0.5, normalize=False, dist_func="sqeuclidean")
        a = torch.randn(2, 8, 4)
        b = torch.randn(2, 8, 4)
        loss = loss_fn(a, b)
        assert loss.item() >= 0

    def test_cosine_distance(self):
        """Cosine distance mode should work without errors."""
        loss_fn = SoftDTWLoss(gamma=1.0, normalize=False, dist_func="cosine")
        a = torch.randn(1, 6, 4)
        b = torch.randn(1, 6, 4)
        loss = loss_fn(a, b)
        assert torch.isfinite(loss)

    def test_batch_dimension(self):
        """Loss should handle batched inputs."""
        loss_fn = SoftDTWLoss(gamma=1.0, normalize=True)
        a = torch.randn(4, 8, 6)
        b = torch.randn(4, 8, 6)
        loss = loss_fn(a, b)
        assert loss.shape == ()  # scalar

    def test_asymmetric_lengths(self):
        """Loss should handle different sequence lengths."""
        loss_fn = SoftDTWLoss(gamma=1.0, normalize=True)
        a = torch.randn(1, 10, 4)
        b = torch.randn(1, 15, 4)
        loss = loss_fn(a, b)
        assert torch.isfinite(loss)

    def test_lengths_ignore_padded_values(self):
        """Length-aware loss must not depend on padded frames."""
        loss_fn = SoftDTWLoss(gamma=0.5, normalize=True)
        a = torch.randn(2, 6, 3)
        b = torch.randn(2, 7, 3)
        lengths_a = torch.tensor([4, 5])
        lengths_b = torch.tensor([5, 4])

        base_loss = loss_fn(a, b, lengths_a=lengths_a, lengths_b=lengths_b)

        a_with_bad_padding = a.clone()
        b_with_bad_padding = b.clone()
        a_with_bad_padding[0, 4:] = 1000.0
        a_with_bad_padding[1, 5:] = -1000.0
        b_with_bad_padding[0, 5:] = -500.0
        b_with_bad_padding[1, 4:] = 500.0

        padded_loss = loss_fn(
            a_with_bad_padding,
            b_with_bad_padding,
            lengths_a=lengths_a,
            lengths_b=lengths_b,
        )
        torch.testing.assert_close(base_loss, padded_loss)

    def test_batched_length_aware_loss_matches_manual_slices(self):
        """Batched variable-length loss should equal manual per-item slicing."""
        loss_fn = SoftDTWLoss(gamma=0.5, normalize=True)
        a = torch.randn(2, 6, 3)
        b = torch.randn(2, 7, 3)
        lengths_a = torch.tensor([4, 6])
        lengths_b = torch.tensor([7, 5])

        batched = loss_fn(a, b, lengths_a=lengths_a, lengths_b=lengths_b)
        manual = torch.stack(
            [
                loss_fn(a[0:1, :4], b[0:1, :7]),
                loss_fn(a[1:2, :6], b[1:2, :5]),
            ]
        ).mean()

        torch.testing.assert_close(batched, manual)

    def test_full_lengths_match_existing_batch_behavior(self):
        """Supplying full lengths should preserve the old unpadded behavior."""
        loss_fn = SoftDTWLoss(gamma=0.5, normalize=True)
        a = torch.randn(2, 5, 3)
        b = torch.randn(2, 6, 3)

        without_lengths = loss_fn(a, b)
        with_lengths = loss_fn(
            a,
            b,
            lengths_a=torch.tensor([5, 5]),
            lengths_b=torch.tensor([6, 6]),
        )

        torch.testing.assert_close(without_lengths, with_lengths)


class TestGammaScheduler:
    """Tests for the gamma annealing scheduler."""

    def test_initial_gamma(self):
        loss_fn = SoftDTWLoss(gamma=1.0)
        sched = GammaScheduler(loss_fn, start_gamma=1.0, end_gamma=0.01, num_epochs=50)
        gamma = sched.step(0)
        assert gamma == pytest.approx(1.0, abs=1e-6)

    def test_gamma_decreases(self):
        loss_fn = SoftDTWLoss(gamma=1.0)
        sched = GammaScheduler(loss_fn, start_gamma=1.0, end_gamma=0.01, num_epochs=50)
        gammas = [sched.step(e) for e in range(50)]
        # Should be strictly decreasing
        for i in range(1, len(gammas)):
            assert gammas[i] <= gammas[i - 1]

    def test_final_gamma_near_target(self):
        loss_fn = SoftDTWLoss(gamma=1.0)
        sched = GammaScheduler(loss_fn, start_gamma=1.0, end_gamma=0.01, num_epochs=50)
        gamma = sched.step(49)
        assert gamma == pytest.approx(0.01, rel=0.05)

    def test_updates_loss_fn(self):
        loss_fn = SoftDTWLoss(gamma=1.0)
        sched = GammaScheduler(loss_fn, start_gamma=1.0, end_gamma=0.01, num_epochs=10)
        sched.step(5)
        assert loss_fn.gamma < 1.0


class TestPathDistillationLoss:
    """Tests for dense teacher-path contrastive supervision."""

    def test_identical_teacher_frames_have_low_loss(self):
        from dis_alignment.model.anchor_loss import PathDistillationLoss

        loss_fn = PathDistillationLoss(
            temperature=0.05,
            min_anchor_gap=1,
            local_radius=6,
            local_step=3,
        )
        emb = torch.eye(40).unsqueeze(0)

        loss = loss_fn(emb, emb.clone(), [[12, 18, 24]], [[12, 18, 24]])

        assert loss.item() < 0.01

    def test_distillation_loss_masks_when_no_valid_local_negatives(self):
        from dis_alignment.model.anchor_loss import PathDistillationLoss

        loss_fn = PathDistillationLoss(temperature=0.1, min_anchor_gap=1)
        emb = torch.ones(1, 1, 2)

        loss = loss_fn(emb, emb.clone(), [[0]], [[0]])

        assert loss.item() == pytest.approx(0.0)

    def test_soft_path_distillation_prefers_teacher_route(self):
        from dis_alignment.model.anchor_loss import SoftPathDistillationLoss

        loss_fn = SoftPathDistillationLoss(temperature=0.05, target_sigma_frames=0.0)
        emb = torch.eye(8).unsqueeze(0)

        good = loss_fn(emb, emb.clone(), [[1, 3, 5]], [[1, 3, 5]])
        bad = loss_fn(emb, emb.flip(dims=[1]), [[1, 3, 5]], [[1, 3, 5]])

        assert good.item() < 0.01
        assert bad.item() > good.item()

    def test_soft_path_distillation_uses_lengths(self):
        from dis_alignment.model.anchor_loss import SoftPathDistillationLoss

        loss_fn = SoftPathDistillationLoss(temperature=0.1, target_sigma_frames=1.0)
        emb = torch.eye(6).unsqueeze(0)

        loss = loss_fn(
            emb,
            emb.clone(),
            [[1, 2, 5]],
            [[1, 2, 5]],
            lengths_a=torch.tensor([3]),
            lengths_b=torch.tensor([3]),
        )

        assert torch.isfinite(loss)


class TestStrictAudioOnlyLosses:
    """Tests for strict self-supervised headline losses."""

    def test_sequence_contrastive_loss_prefers_rowwise_pairs(self):
        from dis_alignment.model.anchor_loss import SequenceContrastiveLoss

        loss_fn = SequenceContrastiveLoss(temperature=0.05)
        emb_a = torch.eye(4).view(4, 1, 4).repeat(1, 3, 1)
        emb_b = emb_a.clone()

        loss = loss_fn(
            emb_a,
            emb_b,
            lengths_a=torch.tensor([3, 3, 3, 3]),
            lengths_b=torch.tensor([3, 3, 3, 3]),
            group_ids=["D911-01", "D911-02", "D911-03", "D911-04"],
        )

        assert loss.item() < 0.01

    def test_sequence_contrastive_masks_same_lied_false_negatives(self):
        from dis_alignment.model.anchor_loss import SequenceContrastiveLoss

        loss_fn = SequenceContrastiveLoss(temperature=0.1)
        emb_a = torch.eye(2).view(2, 1, 2)
        emb_b = emb_a.clone()

        loss = loss_fn(emb_a, emb_b, group_ids=["D911-01", "D911-01"])

        assert loss.item() == pytest.approx(0.0)

    def test_anti_collapse_loss_penalizes_constant_embeddings_more(self):
        from dis_alignment.model.anchor_loss import EmbeddingAntiCollapseLoss

        loss_fn = EmbeddingAntiCollapseLoss(covariance_weight=0.0)
        collapsed = torch.ones(2, 8, 4)
        varied = torch.randn(2, 8, 4) * 2.0

        collapsed_loss = loss_fn(collapsed, collapsed)
        varied_loss = loss_fn(varied, varied)

        assert collapsed_loss.item() > varied_loss.item()

    def test_hard_negative_loss_is_low_when_positive_is_clear(self):
        from dis_alignment.model.anchor_loss import HardNegativeContrastiveLoss

        loss_fn = HardNegativeContrastiveLoss(margin=0.2)
        emb_a = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
        emb_b = emb_a.clone()
        emb_neg = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])

        loss = loss_fn(emb_a, emb_b, emb_neg)

        assert loss.item() == pytest.approx(0.0)

    def test_hard_negative_loss_masks_invalid_rows(self):
        from dis_alignment.model.anchor_loss import HardNegativeContrastiveLoss

        loss_fn = HardNegativeContrastiveLoss(margin=0.2)
        emb_a = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]])
        emb_b = emb_a.clone()
        emb_neg = torch.tensor([[[0.0, 1.0]], [[1.0, 0.0]]])

        masked = loss_fn(emb_a, emb_b, emb_neg, valid_mask=torch.tensor([True, False]))
        unmasked = loss_fn(emb_a, emb_b, emb_neg)

        assert masked.item() == pytest.approx(0.0)
        assert unmasked.item() > masked.item()

    def test_masked_reconstruction_ignores_unmasked_frames(self):
        from dis_alignment.model.anchor_loss import MaskedReconstructionLoss

        loss_fn = MaskedReconstructionLoss()
        prediction = torch.zeros(1, 3, 2)
        target = torch.zeros(1, 1, 2, 3)
        target[:, :, :, 0] = 10.0
        mask = torch.zeros_like(target)
        mask[:, :, :, 1] = 1.0

        loss = loss_fn(prediction, target, mask)

        assert loss.item() == pytest.approx(0.0)

    def test_cycle_loss_is_low_for_identity_features(self):
        from dis_alignment.model.anchor_loss import CycleAlignmentLoss

        loss_fn = CycleAlignmentLoss(temperature=0.01)
        emb = torch.eye(5).unsqueeze(0)

        components = loss_fn(emb, emb.clone())

        assert components["cycle"].item() < 1e-3
        assert components["monotonicity"].item() < 1e-3
        assert components["smoothness"].item() < 1e-3


class TestMemoryEfficientDtw:
    """Tests for strict full-DTW decoding without dense float64 matrices."""

    def test_memory_efficient_dtw_matches_dense_cost(self):
        from dis_alignment.evaluation.common import _memory_efficient_dtw_align, fast_dtw_align

        rng = np.random.default_rng(123)
        features_a = rng.normal(size=(5, 11)).astype(np.float32)
        features_b = rng.normal(size=(5, 13)).astype(np.float32)

        dense_path, dense_cost, _ = fast_dtw_align(features_a, features_b, distance="sqeuclidean")
        mem_path, mem_cost, _ = _memory_efficient_dtw_align(
            features_a,
            features_b,
            distance="sqeuclidean",
        )

        assert mem_cost == pytest.approx(dense_cost, rel=1e-5, abs=1e-5)
        assert tuple(mem_path[:, 0]) == (0, 0)
        assert tuple(mem_path[:, -1]) == (features_a.shape[1] - 1, features_b.shape[1] - 1)
        assert np.all(np.diff(mem_path[0]) >= 0)
        assert np.all(np.diff(mem_path[1]) >= 0)
        assert dense_path.shape[0] == mem_path.shape[0] == 2

    def test_large_pool_one_pairs_use_memory_efficient_dtw(self):
        from dis_alignment.evaluation.common import _should_use_memory_efficient_dtw

        assert _should_use_memory_efficient_dtw(50_000, 45_000, "sqeuclidean")
        assert _should_use_memory_efficient_dtw(50_000, 45_000, "cosine")
        assert not _should_use_memory_efficient_dtw(2_000, 2_000, "sqeuclidean")
        assert not _should_use_memory_efficient_dtw(50_000, 45_000, "cityblock")


# ---------------------------------------------------------------------------
# Augmentation tests
# ---------------------------------------------------------------------------

class TestAudioAugmentor:
    """Tests for audio augmentation."""

    def test_additive_noise(self):
        from dis_alignment.model.augmentation import AudioAugmentor

        aug = AudioAugmentor(
            prob=1.0,
            time_stretch_range=(1.0, 1.0),   # disable stretch
            pitch_shift_range=(0, 0),          # disable pitch shift
            noise_snr_range=(10.0, 10.0),      # force noise
        )
        audio = torch.randn(22050)
        augmented = aug(audio)
        # Audio should be different (noise was added)
        assert not torch.allclose(audio, augmented, atol=1e-6)
        # But should be roughly the same length
        assert abs(len(augmented) - len(audio)) < 100

    def test_no_augmentation_when_disabled(self):
        from dis_alignment.model.augmentation import AudioAugmentor

        aug = AudioAugmentor(prob=0.0)
        audio = torch.randn(22050)
        augmented = aug(audio)
        torch.testing.assert_close(audio, augmented)

    def test_output_shape_1d(self):
        from dis_alignment.model.augmentation import AudioAugmentor

        aug = AudioAugmentor(
            prob=1.0,
            time_stretch_range=(1.0, 1.0),
            pitch_shift_range=(0, 0),
            noise_snr_range=(20.0, 20.0),
        )
        audio = torch.randn(11025)
        out = aug(audio)
        assert out.dim() == 1


class TestSpecAugment:
    """Tests for SpecAugment masking."""

    def test_introduces_zeros(self):
        from dis_alignment.model.augmentation import SpecAugment

        spec_aug = SpecAugment(
            freq_mask_param=10,
            time_mask_param=20,
            num_freq_masks=2,
            num_time_masks=2,
        )
        spec = torch.ones(1, 84, 100)  # all ones
        masked = spec_aug(spec)
        # At least some values should be zeroed
        assert (masked == 0).any()

    def test_output_shape_unchanged(self):
        from dis_alignment.model.augmentation import SpecAugment

        spec_aug = SpecAugment()
        spec = torch.randn(1, 84, 100)
        out = spec_aug(spec)
        assert out.shape == spec.shape


# ---------------------------------------------------------------------------
# Dataset / collate tests
# ---------------------------------------------------------------------------

class TestCollateVariableLength:
    """Tests for the variable-length collate function."""

    def test_pads_to_max_length(self):
        from dis_alignment.model.dataset import collate_variable_length

        batch = [
            {
                "spec_a": torch.randn(1, 84, 50),
                "spec_b": torch.randn(1, 84, 30),
                "pair_id": "test_pair_0",
            },
            {
                "spec_a": torch.randn(1, 84, 80),
                "spec_b": torch.randn(1, 84, 60),
                "pair_id": "test_pair_1",
            },
        ]

        collated = collate_variable_length(batch)

        assert collated["spec_a"].shape == (2, 1, 84, 80)   # padded to max
        assert collated["spec_b"].shape == (2, 1, 84, 60)
        assert collated["lengths_a"].tolist() == [50, 80]
        assert collated["lengths_b"].tolist() == [30, 60]
        assert len(collated["pair_ids"]) == 2

    def test_pads_optional_hard_negative_and_reconstruction_tensors(self):
        from dis_alignment.model.dataset import collate_variable_length

        batch = [
            {
                "spec_a": torch.randn(1, 2, 5),
                "spec_b": torch.randn(1, 2, 5),
                "spec_neg": torch.randn(1, 2, 3),
                "reconstruction_target_a": torch.randn(1, 2, 5),
                "reconstruction_mask_a": torch.ones(1, 2, 5),
                "pair_id": "a",
            },
            {
                "spec_a": torch.randn(1, 2, 7),
                "spec_b": torch.randn(1, 2, 4),
                "spec_neg": torch.randn(1, 2, 6),
                "reconstruction_target_a": torch.randn(1, 2, 7),
                "reconstruction_mask_a": torch.ones(1, 2, 7),
                "pair_id": "b",
            },
            {
                "spec_a": torch.randn(1, 2, 9),
                "spec_b": torch.randn(1, 2, 4),
                "reconstruction_target_a": torch.randn(1, 2, 9),
                "reconstruction_mask_a": torch.ones(1, 2, 9),
                "pair_id": "c",
            },
        ]

        collated = collate_variable_length(batch)

        assert collated["spec_neg"].shape == (3, 1, 2, 9)
        assert collated["lengths_neg"].tolist() == [3, 6, 9]
        assert collated["reconstruction_target_a"].shape == (3, 1, 2, 9)
        assert collated["reconstruction_mask_a"].shape == (3, 1, 2, 9)


class TestSWDPairDatasetSampling:
    """Tests for aligned-window SWD segment sampling."""

    def test_aligned_sampling_returns_shared_measure_window(self, monkeypatch, tmp_path):
        from dis_alignment.model.dataset import SWDPairDataset

        _patch_aligned_sampling(monkeypatch)
        pair = _make_pair(tmp_path, "D911-01")

        dataset = SWDPairDataset(
            pairs=[pair],
            sr=10,
            max_length_sec=12.0,
            segment_sampling="aligned_measures",
            deterministic=True,
        )

        item = dataset[0]

        assert item["pair_id"] == pair.pair_id
        assert item["segment_sampling"] == "aligned_measures"
        assert item["window_start_event_id"] == "2"
        assert item["window_end_boundary_event_id"] == "3"
        assert item["window_duration_a_s"] <= 12.0
        assert item["window_duration_b_s"] <= 12.0
        assert item["anchor_event_ids"] == ["2", "3"]

    def test_aligned_sampling_can_emit_distant_hard_negative(self, monkeypatch, tmp_path):
        from dis_alignment.model import dataset as dataset_module
        from dis_alignment.model.dataset import SWDPairDataset

        _patch_aligned_sampling(monkeypatch)
        pair = _make_pair(tmp_path, "D911-02")
        monkeypatch.setattr(
            dataset_module,
            "load_swd_audio",
            lambda piece, sr=10: (np.arange(400, dtype=np.float32), sr),
        )
        monkeypatch.setattr(
            dataset_module.SWDPairDataset,
            "_compute_cqt",
            lambda self, audio: np.tile(audio[::10][: max(1, len(audio) // 10)], (2, 1)).astype(np.float32),
        )

        dataset = SWDPairDataset(
            pairs=[pair],
            sr=10,
            hop_length=1,
            max_length_sec=12.0,
            segment_sampling="aligned_measures",
            deterministic=True,
            hard_negative_radius_frames=80,
            emit_repeated_hard_negatives=True,
        )

        item = dataset[0]

        assert "spec_neg" in item
        assert item["spec_neg"].shape[-1] > 0
        assert item["repeated_negative_start_s"] >= 27.0

    def test_aligned_sampling_can_emit_false_destination_negative(self, monkeypatch, tmp_path):
        from dis_alignment.model import dataset as dataset_module
        from dis_alignment.model.dataset import SWDPairDataset

        _patch_aligned_sampling(monkeypatch)
        pair = _make_pair(tmp_path, "D911-03")
        false_destinations = tmp_path / "false_destinations.csv"
        false_destinations.write_text(
            "pair_id,event_id,gt_a_s,gt_b_s,pred_b_s,abs_error_ms,error_sign\n"
            f"{pair.pair_id},2,9.0,10.0,25.0,1500.0,late\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            dataset_module,
            "load_swd_audio",
            lambda piece, sr=10: (np.arange(400, dtype=np.float32), sr),
        )
        monkeypatch.setattr(
            dataset_module.SWDPairDataset,
            "_compute_cqt",
            lambda self, audio: np.tile(audio[::10][: max(1, len(audio) // 10)], (2, 1)).astype(np.float32),
        )

        dataset = SWDPairDataset(
            pairs=[pair],
            sr=10,
            hop_length=1,
            max_length_sec=12.0,
            segment_sampling="aligned_measures",
            deterministic=True,
            false_destination_negative_root=false_destinations,
            false_destination_min_error_ms=500.0,
        )

        item = dataset[0]

        assert item["negative_source"] == "false_destination"
        assert item["negative_valid"] is True
        assert item["false_destination_event_id"] == "2"
        assert item["repeated_negative_start_s"] == pytest.approx(24.0)

    def test_fallback_window_includes_end_boundary_anchor(self, monkeypatch, tmp_path):
        from dis_alignment.model.dataset import SWDPairDataset

        _patch_aligned_sampling(monkeypatch)
        pair = _make_pair(tmp_path, "D911-06")

        dataset = SWDPairDataset(
            pairs=[pair],
            sr=10,
            max_length_sec=0.5,
            segment_sampling="aligned_measures",
            deterministic=True,
        )

        windows = dataset._windows_by_pair_id[pair.pair_id]
        assert windows[0].anchor_event_ids == ("1", "2")

    def test_teacher_path_samples_are_local_to_window(self, monkeypatch, tmp_path):
        from dis_alignment.alignment.teacher import TeacherPath, save_teacher_path
        from dis_alignment.model.dataset import SWDPairDataset

        _patch_aligned_sampling(monkeypatch)
        pair = _make_pair(tmp_path, "D911-08")
        teacher_root = tmp_path / "teacher_paths"
        times = np.arange(0, 40, dtype=np.float64)
        path = np.vstack([np.arange(times.size), np.arange(times.size)])
        save_teacher_path(
            TeacherPath(
                pair_id=pair.pair_id,
                lied_id=pair.lied_id,
                piece_a_id=pair.piece_a.piece_id,
                piece_b_id=pair.piece_b.piece_id,
                frame_hop=10,
                sr=10,
                path=path,
                time_a_s=times,
                time_b_s=times,
            ),
            teacher_root,
        )

        dataset = SWDPairDataset(
            pairs=[pair],
            sr=10,
            hop_length=10,
            max_length_sec=20.0,
            segment_sampling="aligned_measures",
            deterministic=True,
            teacher_path_root=teacher_root,
            num_teacher_samples=5,
        )

        item = dataset[0]

        assert len(item["teacher_frame_indices_a"]) == 5
        assert len(item["teacher_frame_indices_b"]) == 5
        assert min(item["teacher_frame_indices_a"]) >= 0
        assert max(item["teacher_frame_indices_a"]) < item["spec_a"].shape[-1]
        assert min(item["teacher_frame_indices_b"]) >= 0
        assert max(item["teacher_frame_indices_b"]) < item["spec_b"].shape[-1]

    def test_teacher_path_sampling_uses_no_measure_anchors(self, monkeypatch, tmp_path):
        from dis_alignment.alignment.teacher import TeacherPath, save_teacher_path
        from dis_alignment.model import dataset as dataset_module
        from dis_alignment.model.dataset import SWDPairDataset

        pair = _make_pair(tmp_path, "D911-09")
        teacher_root = tmp_path / "teacher_paths"
        times = np.arange(0, 40, dtype=np.float64)
        confidence = np.full(times.shape, 0.2, dtype=np.float32)
        confidence[10:31] = 0.95
        save_teacher_path(
            TeacherPath(
                pair_id=pair.pair_id,
                lied_id=pair.lied_id,
                piece_a_id=pair.piece_a.piece_id,
                piece_b_id=pair.piece_b.piece_id,
                frame_hop=10,
                sr=10,
                path=np.vstack([np.arange(times.size), np.arange(times.size)]),
                time_a_s=times,
                time_b_s=times,
                confidence=confidence,
            ),
            teacher_root,
        )
        monkeypatch.setattr(
            dataset_module,
            "compute_ground_truth_measure_alignment",
            lambda pair_arg: (_ for _ in ()).throw(AssertionError("measure anchors used")),
        )
        monkeypatch.setattr(
            dataset_module,
            "load_swd_audio",
            lambda piece, sr=10: (np.zeros(500, dtype=np.float32), sr),
        )
        monkeypatch.setattr(
            dataset_module.SWDPairDataset,
            "_compute_cqt",
            lambda self, audio: np.ones((2, max(1, len(audio) // 10)), dtype=np.float32),
        )

        dataset = SWDPairDataset(
            pairs=[pair],
            sr=10,
            hop_length=10,
            max_length_sec=12.0,
            segment_sampling="teacher_path",
            deterministic=True,
            teacher_path_root=teacher_root,
            teacher_min_confidence=0.9,
            num_teacher_samples=6,
        )

        item = dataset[0]

        assert item["segment_sampling"] == "teacher_path"
        assert item["anchor_frame_indices_a"] == []
        assert item["anchor_frame_indices_b"] == []
        assert len(item["teacher_frame_indices_a"]) > 0
        assert min(item["teacher_frame_indices_a"]) >= 0
        assert max(item["teacher_frame_indices_a"]) < item["spec_a"].shape[-1]

    def test_self_audio_sampling_builds_identity_positives_without_anchors(self, monkeypatch, tmp_path):
        from dis_alignment.model import dataset as dataset_module
        from dis_alignment.model.dataset import SWDPairDataset

        pair = _make_pair(tmp_path, "D911-10")
        monkeypatch.setattr(
            dataset_module,
            "compute_ground_truth_measure_alignment",
            lambda pair_arg: (_ for _ in ()).throw(AssertionError("measure anchors used")),
        )
        monkeypatch.setattr(
            dataset_module,
            "load_swd_audio",
            lambda piece, sr=10: (np.arange(120, dtype=np.float32), sr),
        )
        monkeypatch.setattr(
            dataset_module.SWDPairDataset,
            "_compute_cqt",
            lambda self, audio: np.ones((2, max(1, len(audio) // 10)), dtype=np.float32),
        )

        dataset = SWDPairDataset(
            pairs=[pair],
            sr=10,
            hop_length=10,
            max_length_sec=6.0,
            segment_sampling="self_audio",
            deterministic=True,
            num_teacher_samples=5,
        )

        item = dataset[0]

        assert item["segment_sampling"] == "self_audio"
        assert item["anchor_frame_indices_a"] == []
        assert item["anchor_frame_indices_b"] == []
        assert item["teacher_frame_indices_a"] == item["teacher_frame_indices_b"]
        assert len(item["teacher_frame_indices_a"]) == 5

    def test_self_audio_ordered_sampling_uses_raw_time_indices_only(self, monkeypatch, tmp_path):
        from dis_alignment.model import dataset as dataset_module
        from dis_alignment.model.dataset import SWDPairDataset

        pair = _make_pair(tmp_path, "D911-13")
        monkeypatch.setattr(
            dataset_module,
            "compute_ground_truth_measure_alignment",
            lambda pair_arg: (_ for _ in ()).throw(AssertionError("measure anchors used")),
        )
        monkeypatch.setattr(
            dataset_module,
            "load_swd_audio",
            lambda piece, sr=10: (np.arange(240, dtype=np.float32), sr),
        )
        monkeypatch.setattr(
            dataset_module.SWDPairDataset,
            "_compute_cqt",
            lambda self, audio: np.tile(audio[: max(1, len(audio) // 10)], (2, 1)).astype(np.float32),
        )

        dataset = SWDPairDataset(
            pairs=[pair],
            sr=10,
            hop_length=10,
            max_length_sec=6.0,
            segment_sampling="self_audio_ordered",
            deterministic=True,
            num_teacher_samples=5,
            hard_negative_radius_frames=4,
            relative_offset_bins=[1, 2, 4],
        )

        item = dataset[0]

        assert item["segment_sampling"] == "self_audio_ordered"
        assert item["anchor_frame_indices_a"] == []
        assert item["anchor_frame_indices_b"] == []
        assert item["positive_start_sample"] != item["anchor_start_sample"]
        frame_offset = round(
            (item["anchor_start_sample"] - item["positive_start_sample"]) / dataset.hop_length
        )
        assert all(
            (frame_b - frame_a) == frame_offset
            for frame_a, frame_b in zip(
                item["teacher_frame_indices_a"],
                item["teacher_frame_indices_b"],
                strict=False,
            )
        )
        assert item["spec_neg"].shape[-1] > 0
        assert item["temporal_order_label"] in {0, 1, 2}
        assert 0 <= item["relative_offset_label"] <= 3
        assert item["reconstruction_target_a"].shape == item["spec_a"].shape
        assert item["reconstruction_mask_a"].sum() > 0

    def test_self_audio_ordered_raw_shift_frames_follow_overlap(self, tmp_path):
        from dis_alignment.model.dataset import SWDPairDataset

        dataset = SWDPairDataset(
            pairs=[_make_pair(tmp_path, "D911-15")],
            sr=10,
            hop_length=10,
            max_length_sec=6.0,
            segment_sampling="self_audio_ordered",
            deterministic=True,
            num_teacher_samples=4,
        )

        frames_a, frames_b = dataset._raw_offset_frames_for_specs(
            8,
            8,
            start_a_sample=50,
            start_b_sample=70,
        )

        assert frames_a == [2, 3, 5, 7]
        assert frames_b == [0, 1, 3, 5]

    def test_relative_offset_bins_are_coarse_raw_frame_distances(self, tmp_path):
        from dis_alignment.model.dataset import SWDPairDataset

        dataset = SWDPairDataset(
            pairs=[_make_pair(tmp_path, "D911-14")],
            segment_sampling="self_audio_ordered",
            relative_offset_bins=[2, 5, 9],
        )

        assert dataset._relative_offset_bin(0) == 0
        assert dataset._relative_offset_bin(2) == 1
        assert dataset._relative_offset_bin(6) == 2
        assert dataset._relative_offset_bin(12) == 3

    def test_same_lied_pair_sampling_uses_no_measure_anchors(self, monkeypatch, tmp_path):
        from dis_alignment.model import dataset as dataset_module
        from dis_alignment.model.dataset import SWDPairDataset

        pair = _make_pair(tmp_path, "D911-11")
        monkeypatch.setattr(
            dataset_module,
            "compute_ground_truth_measure_alignment",
            lambda pair_arg: (_ for _ in ()).throw(AssertionError("measure anchors used")),
        )
        monkeypatch.setattr(
            dataset_module,
            "load_swd_audio",
            lambda piece, sr=10: (np.arange(200, dtype=np.float32), sr),
        )
        monkeypatch.setattr(
            dataset_module.SWDPairDataset,
            "_compute_cqt",
            lambda self, audio: np.ones((2, max(1, len(audio) // 10)), dtype=np.float32),
        )

        dataset = SWDPairDataset(
            pairs=[pair],
            sr=10,
            hop_length=10,
            max_length_sec=6.0,
            segment_sampling="same_lied_pair",
            deterministic=True,
        )

        item = dataset[0]

        assert item["segment_sampling"] == "same_lied_pair"
        assert item["lied_id"] == "D911-11"
        assert item["positive_pair"] is True
        assert item["anchor_frame_indices_a"] == []
        assert item["teacher_frame_indices_a"] == []

    def test_self_mined_path_sampling_uses_separate_path_root(self, monkeypatch, tmp_path):
        from dis_alignment.alignment.teacher import TeacherPath, save_teacher_path
        from dis_alignment.model import dataset as dataset_module
        from dis_alignment.model.dataset import SWDPairDataset

        pair = _make_pair(tmp_path, "D911-12")
        mined_root = tmp_path / "self_mined"
        times = np.arange(0, 20, dtype=np.float64)
        save_teacher_path(
            TeacherPath(
                pair_id=pair.pair_id,
                lied_id=pair.lied_id,
                piece_a_id=pair.piece_a.piece_id,
                piece_b_id=pair.piece_b.piece_id,
                frame_hop=10,
                sr=10,
                path=np.vstack([np.arange(times.size), np.arange(times.size)]),
                time_a_s=times,
                time_b_s=times,
            ),
            mined_root,
        )
        monkeypatch.setattr(
            dataset_module,
            "compute_ground_truth_measure_alignment",
            lambda pair_arg: (_ for _ in ()).throw(AssertionError("measure anchors used")),
        )
        monkeypatch.setattr(
            dataset_module,
            "load_swd_audio",
            lambda piece, sr=10: (np.zeros(300, dtype=np.float32), sr),
        )
        monkeypatch.setattr(
            dataset_module.SWDPairDataset,
            "_compute_cqt",
            lambda self, audio: np.ones((2, max(1, len(audio) // 10)), dtype=np.float32),
        )

        dataset = SWDPairDataset(
            pairs=[pair],
            sr=10,
            hop_length=10,
            max_length_sec=10.0,
            segment_sampling="self_mined_path",
            deterministic=True,
            self_mined_path_root=mined_root,
            num_teacher_samples=4,
        )

        item = dataset[0]

        assert item["segment_sampling"] == "self_mined_path"
        assert item["anchor_frame_indices_a"] == []
        assert len(item["teacher_frame_indices_a"]) == 4

    def test_self_mined_path_sampling_filters_low_confidence_frames(self, monkeypatch, tmp_path):
        from dis_alignment.alignment.teacher import TeacherPath, save_teacher_path
        from dis_alignment.model import dataset as dataset_module
        from dis_alignment.model.dataset import SWDPairDataset

        pair = _make_pair(tmp_path, "D911-16")
        mined_root = tmp_path / "self_mined_confidence"
        times = np.arange(0, 20, dtype=np.float64)
        confidence = np.zeros(times.shape, dtype=np.float32)
        confidence[8:13] = 0.9
        save_teacher_path(
            TeacherPath(
                pair_id=pair.pair_id,
                lied_id=pair.lied_id,
                piece_a_id=pair.piece_a.piece_id,
                piece_b_id=pair.piece_b.piece_id,
                frame_hop=10,
                sr=10,
                path=np.vstack([np.arange(times.size), np.arange(times.size)]),
                time_a_s=times,
                time_b_s=times,
                confidence=confidence,
            ),
            mined_root,
        )
        monkeypatch.setattr(
            dataset_module,
            "load_swd_audio",
            lambda piece, sr=10: (np.zeros(300, dtype=np.float32), sr),
        )
        monkeypatch.setattr(
            dataset_module.SWDPairDataset,
            "_compute_cqt",
            lambda self, audio: np.ones((2, max(1, len(audio) // 10)), dtype=np.float32),
        )

        dataset = SWDPairDataset(
            pairs=[pair],
            sr=10,
            hop_length=10,
            max_length_sec=10.0,
            segment_sampling="self_mined_path",
            deterministic=True,
            self_mined_path_root=mined_root,
            teacher_min_confidence=0.8,
            num_teacher_samples=8,
        )

        item = dataset[0]

        assert item["teacher_frame_indices_a"]
        assert len(item["teacher_frame_indices_a"]) <= 5

    def test_self_mined_confidence_prefers_local_margin(self):
        from dis_alignment.cli import _self_mined_path_confidence

        features = np.eye(5, dtype=np.float32)
        path = np.vstack([np.arange(5), np.arange(5)])

        confidence = _self_mined_path_confidence(
            features,
            features,
            path,
            distance="sqeuclidean",
            local_radius=1,
            margin_scale=0.25,
        )

        assert confidence[2] > 0.95
        assert confidence.shape == (5,)

    def test_waveform_augmentation_is_applied_to_both_sides(self, monkeypatch, tmp_path):
        from dis_alignment.model import dataset as dataset_module
        from dis_alignment.model.dataset import SWDPairDataset

        class AddOneAugmentor:
            def __init__(self):
                self.calls = 0

            def __call__(self, audio):
                self.calls += 1
                return audio + 1.0

        monkeypatch.setattr(
            dataset_module,
            "compute_ground_truth_measure_alignment",
            lambda pair: (
                pd.DataFrame({"event_id": ["1", "2"], "time_s": [0.0, 1.0]}),
                pd.DataFrame({"event_id": ["1", "2"], "time_s": [0.0, 1.0]}),
            ),
        )
        monkeypatch.setattr(
            dataset_module,
            "load_swd_audio",
            lambda piece, sr=10: (np.zeros(20, dtype=np.float32), sr),
        )
        monkeypatch.setattr(
            dataset_module.SWDPairDataset,
            "_compute_cqt",
            lambda self, audio: np.full((2, 2), float(np.mean(audio)), dtype=np.float32),
        )

        augmentor = AddOneAugmentor()
        pair = _make_pair(tmp_path, "D911-07")
        dataset = SWDPairDataset(
            pairs=[pair],
            sr=10,
            max_length_sec=1.0,
            augmentor=augmentor,
            segment_sampling="aligned_measures",
            deterministic=True,
        )

        item = dataset[0]

        assert augmentor.calls == 2
        assert torch.all(item["spec_a"] == 1.0)
        assert torch.all(item["spec_b"] == 1.0)

    def test_validation_sampling_is_deterministic(self, monkeypatch, tmp_path):
        from dis_alignment.model.dataset import SWDPairDataset

        _patch_aligned_sampling(monkeypatch)
        pair = _make_pair(tmp_path, "D911-02")

        dataset = SWDPairDataset(
            pairs=[pair],
            sr=10,
            max_length_sec=12.0,
            segment_sampling="aligned_measures",
            deterministic=True,
        )

        first = dataset[0]
        second = dataset[0]

        assert first["window_start_event_id"] == second["window_start_event_id"]
        assert first["window_end_boundary_event_id"] == second["window_end_boundary_event_id"]
        assert first["window_duration_a_s"] == second["window_duration_a_s"]
        assert first["window_duration_b_s"] == second["window_duration_b_s"]

    def test_training_sampling_randomizes_aligned_windows(self, monkeypatch, tmp_path):
        from dis_alignment.model.dataset import SWDPairDataset

        _patch_aligned_sampling(monkeypatch)
        pair = _make_pair(tmp_path, "D911-03")

        dataset = SWDPairDataset(
            pairs=[pair],
            sr=10,
            max_length_sec=12.0,
            segment_sampling="aligned_measures",
            deterministic=False,
            random_seed=7,
        )

        seen = {
            (dataset[0]["window_start_event_id"], dataset[0]["window_end_boundary_event_id"])
            for _ in range(12)
        }

        assert len(seen) > 1

    def test_samples_per_epoch_repeats_pair_ownership_without_leakage(self, monkeypatch, tmp_path):
        from dis_alignment.model.dataset import SWDPairDataset

        _patch_aligned_sampling(monkeypatch)
        pair_a = _make_pair(tmp_path, "D911-04")
        pair_b = _make_pair(tmp_path, "D911-05")

        dataset = SWDPairDataset(
            pairs=[pair_a, pair_b],
            sr=10,
            max_length_sec=12.0,
            segment_sampling="aligned_measures",
            deterministic=True,
            samples_per_epoch=6,
        )

        observed_pair_ids = [dataset[idx]["pair_id"] for idx in range(len(dataset))]

        assert len(dataset) == 6
        assert set(observed_pair_ids) == {pair_a.pair_id, pair_b.pair_id}


class TestTrainingPairSplit:
    """Tests for leakage-safe train/validation splitting."""

    def test_split_swd_pairs_has_no_overlap(self, tmp_path):
        from dis_alignment.model.train import _split_swd_pairs

        pairs = [_make_pair(tmp_path, f"D911-{idx:02d}") for idx in range(1, 6)]

        train_pairs, val_pairs = _split_swd_pairs(pairs, val_split=0.4, random_seed=3)

        train_ids = {pair.pair_id for pair in train_pairs}
        val_ids = {pair.pair_id for pair in val_pairs}

        assert train_ids
        assert val_ids
        assert train_ids.isdisjoint(val_ids)
        assert train_ids | val_ids == {pair.pair_id for pair in pairs}

    def test_split_swd_pairs_keeps_lieder_disjoint(self, tmp_path):
        from dis_alignment.model.train import _split_swd_pairs

        pairs = [
            _make_pair(tmp_path, "D911-01", "A", "B"),
            _make_pair(tmp_path, "D911-01", "A", "C"),
            _make_pair(tmp_path, "D911-01", "B", "C"),
            _make_pair(tmp_path, "D911-02", "A", "B"),
            _make_pair(tmp_path, "D911-03", "A", "B"),
            _make_pair(tmp_path, "D911-04", "A", "B"),
        ]

        train_pairs, val_pairs = _split_swd_pairs(pairs, val_split=0.25, random_seed=1)

        train_lieder = {pair.lied_id for pair in train_pairs}
        val_lieder = {pair.lied_id for pair in val_pairs}
        train_piece_ids = {
            piece_id
            for pair in train_pairs
            for piece_id in (pair.piece_a.piece_id, pair.piece_b.piece_id)
        }
        val_piece_ids = {
            piece_id
            for pair in val_pairs
            for piece_id in (pair.piece_a.piece_id, pair.piece_b.piece_id)
        }

        assert train_lieder.isdisjoint(val_lieder)
        assert train_piece_ids.isdisjoint(val_piece_ids)

    def test_split_swd_pairs_forces_debug_lieder_into_validation(self, tmp_path):
        from dis_alignment.model.train import _split_swd_pairs

        pairs = [_make_pair(tmp_path, f"D911-{idx:02d}") for idx in range(1, 6)]

        train_pairs, val_pairs = _split_swd_pairs(
            pairs,
            val_split=0.2,
            random_seed=3,
            heldout_lieder=["D911-03"],
        )

        assert all(pair.lied_id != "D911-03" for pair in train_pairs)
        assert any(pair.lied_id == "D911-03" for pair in val_pairs)

    def test_training_kwargs_reads_audio_frontend_from_config(self, tmp_path):
        from dis_alignment.model.train import _build_training_kwargs

        config_path = tmp_path / "train.yaml"
        config_path.write_text(
            "audio:\n"
            "  sr: 16000\n"
            "  hop_length: 160\n"
            "  n_bins: 72\n",
            encoding="utf-8",
        )
        args = SimpleNamespace(
            config=str(config_path),
            swd_path=None,
            output_dir=None,
            epochs=None,
            batch_size=None,
            lr=None,
            embed_dim=None,
            sr=None,
            hop_length=None,
            n_freq_bins=None,
            conv_channels=None,
            gru_hidden_size=None,
            num_gru_layers=None,
            dropout=None,
            temporal_attention_heads=None,
            max_length_sec=None,
            segment_sampling=None,
            samples_per_epoch=None,
            device=None,
            start_gamma=None,
            end_gamma=None,
            soft_dtw_loss_weight=None,
            gradient_clip=None,
            weight_decay=None,
            val_split=None,
            save_every_n_epochs=None,
            dist_func=None,
            resume_from=None,
            selection_metric=None,
            cache_root=None,
            deep_decode=None,
            band_radius_frames=None,
            anchor_loss_weight=None,
            dense_anchor_loss_weight=None,
            path_distill_loss_weight=None,
            teacher_path_root=None,
            num_anchor_samples=None,
            teacher_min_confidence=None,
            disable_time_stretch_for_anchors=None,
            eval_pool_size=None,
            alignment_eval_every_n_epochs=None,
            anchor_temperature=None,
            anchor_min_anchor_gap=None,
            no_augment=False,
            dry_run=False,
            cache_spectrograms=None,
            normalize_loss=None,
        )

        kwargs = _build_training_kwargs(args)

        assert kwargs["sr"] == 16000
        assert kwargs["hop_length"] == 160
        assert kwargs["n_freq_bins"] == 72

    def test_training_gate_passes_dataset_and_hop_length(self, monkeypatch, tmp_path):
        from dis_alignment.model import train as train_module

        pair = _make_pair(tmp_path, "D911-01")
        dataset = object()
        seen = {}

        def fake_evaluate_pair(pair_arg, **kwargs):
            seen.update(kwargs)
            assert pair_arg is pair
            return [
                {
                    "mae": 0.1,
                    "ar_50ms": 0.2,
                    "ar_100ms": 0.3,
                    "ar_200ms": 0.4,
                }
            ]

        monkeypatch.setattr("dis_alignment.evaluation.swd.evaluate_pair", fake_evaluate_pair)

        metrics = train_module._evaluate_swd_pairs(
            [pair],
            dataset=dataset,
            sr=16000,
            hop_length=160,
            encoder=object(),
            device="cpu",
            cache_root=None,
            deep_decode="deepalign_score_guided_refined",
            band_radius_frames=None,
            pool_size=1,
        )

        assert seen["dataset"] is dataset
        assert seen["sr"] == 16000
        assert seen["deep_hop"] == 160
        assert seen["pool_size"] == 1
        assert metrics["pairs"] == 1.0

    def test_training_kwargs_reads_sota_teacher_config(self, tmp_path):
        from dis_alignment.model.train import _build_training_kwargs

        config_path = tmp_path / "train.yaml"
        config_path.write_text(
            "training:\n"
            "  dense_anchor_loss_weight: 0.15\n"
            "  path_distill_loss_weight: 0.75\n"
            "  soft_path_distill_loss_weight: 0.25\n"
            "  soft_path_temperature: 0.07\n"
            "  soft_path_target_sigma_frames: 3.0\n"
            "  teacher_path_root: results/teacher/paths\n"
            "  num_anchor_samples: 96\n"
            "  teacher_min_confidence: 0.6\n"
            "  path_distill_local_radius_frames: 10\n"
            "  path_distill_local_step_frames: 2\n"
            "  disable_time_stretch_for_anchors: true\n"
            "soft_dtw:\n"
            "  loss_weight: 0.0\n"
            "evaluation:\n"
            "  pool_size: 1\n"
            "  alignment_eval_every_n_epochs: 0\n",
            encoding="utf-8",
        )
        args = SimpleNamespace(
            config=str(config_path),
            swd_path=None,
            output_dir=None,
            epochs=None,
            batch_size=None,
            lr=None,
            embed_dim=None,
            sr=None,
            hop_length=None,
            n_freq_bins=None,
            conv_channels=None,
            gru_hidden_size=None,
            num_gru_layers=None,
            dropout=None,
            temporal_attention_heads=None,
            max_length_sec=None,
            segment_sampling=None,
            samples_per_epoch=None,
            device=None,
            start_gamma=None,
            end_gamma=None,
            soft_dtw_loss_weight=None,
            gradient_clip=None,
            weight_decay=None,
            val_split=None,
            save_every_n_epochs=None,
            dist_func=None,
            resume_from=None,
            selection_metric=None,
            cache_root=None,
            deep_decode=None,
            band_radius_frames=None,
            anchor_loss_weight=None,
            dense_anchor_loss_weight=None,
            path_distill_loss_weight=None,
            teacher_path_root=None,
            num_anchor_samples=None,
            teacher_min_confidence=None,
            path_distill_local_radius_frames=None,
            path_distill_local_step_frames=None,
            disable_time_stretch_for_anchors=None,
            eval_pool_size=None,
            alignment_eval_every_n_epochs=None,
            anchor_temperature=None,
            anchor_min_anchor_gap=None,
            no_augment=False,
            dry_run=False,
            cache_spectrograms=None,
            normalize_loss=None,
        )

        kwargs = _build_training_kwargs(args)

        assert kwargs["dense_anchor_loss_weight"] == pytest.approx(0.15)
        assert kwargs["path_distill_loss_weight"] == pytest.approx(0.75)
        assert kwargs["soft_path_distill_loss_weight"] == pytest.approx(0.25)
        assert kwargs["soft_path_temperature"] == pytest.approx(0.07)
        assert kwargs["soft_path_target_sigma_frames"] == pytest.approx(3.0)
        assert kwargs["soft_dtw_loss_weight"] == 0.0
        assert kwargs["teacher_path_root"] == "results/teacher/paths"
        assert kwargs["num_anchor_samples"] == 96
        assert kwargs["teacher_min_confidence"] == pytest.approx(0.6)
        assert kwargs["path_distill_local_radius_frames"] == 10
        assert kwargs["path_distill_local_step_frames"] == 2
        assert kwargs["disable_time_stretch_for_anchors"] is True
        assert kwargs["eval_pool_size"] == 1
        assert kwargs["alignment_eval_every_n_epochs"] == 0

    def test_training_kwargs_reads_strict_stage1_v2_keys(self, tmp_path):
        from dis_alignment.model.train import _build_training_kwargs

        config_path = tmp_path / "train.yaml"
        config_path.write_text(
            "dataset:\n"
            "  segment_sampling: self_audio_ordered\n"
            "training:\n"
            "  temporal_order_loss_weight: 0.4\n"
            "  relative_offset_loss_weight: 0.5\n"
            "  relative_offset_bins: [4, 8, 16]\n"
            "  masked_reconstruction_loss_weight: 0.1\n"
            "  hard_negative_loss_weight: 0.3\n"
            "  hard_negative_radius_frames: 32\n"
            "  false_destination_negative_loss_weight: 0.07\n"
            "  false_destination_negative_root: results/failure_reports/mined.csv\n"
            "  false_destination_min_error_ms: 750\n"
            "  memory_bank_size: 128\n"
            "soft_dtw:\n"
            "  loss_weight: 0.0\n",
            encoding="utf-8",
        )
        args = SimpleNamespace(
            config=str(config_path),
            swd_path=None,
            output_dir=None,
            epochs=None,
            batch_size=None,
            lr=None,
            embed_dim=None,
            sr=None,
            hop_length=None,
            n_freq_bins=None,
            conv_channels=None,
            gru_hidden_size=None,
            num_gru_layers=None,
            dropout=None,
            temporal_attention_heads=None,
            max_length_sec=None,
            segment_sampling=None,
            samples_per_epoch=None,
            device=None,
            start_gamma=None,
            end_gamma=None,
            soft_dtw_loss_weight=None,
            gradient_clip=None,
            weight_decay=None,
            val_split=None,
            save_every_n_epochs=None,
            dist_func=None,
            resume_from=None,
            selection_metric=None,
            cache_root=None,
            deep_decode=None,
            band_radius_frames=None,
            anchor_loss_weight=None,
            dense_anchor_loss_weight=None,
            path_distill_loss_weight=None,
            temporal_order_loss_weight=None,
            relative_offset_loss_weight=None,
            relative_offset_bins=None,
            masked_reconstruction_loss_weight=None,
            cycle_consistency_loss_weight=None,
            cycle_entropy_loss_weight=None,
            cycle_monotonicity_loss_weight=None,
            cycle_smoothness_loss_weight=None,
            hard_negative_loss_weight=None,
            hard_negative_radius_frames=None,
            false_destination_negative_loss_weight=None,
            false_destination_negative_root=None,
            false_destination_min_error_ms=None,
            memory_bank_size=None,
            teacher_path_root=None,
            self_mined_path_root=None,
            num_anchor_samples=None,
            teacher_min_confidence=None,
            disable_time_stretch_for_anchors=None,
            eval_pool_size=None,
            alignment_eval_every_n_epochs=None,
            sequence_contrastive_loss_weight=None,
            anti_collapse_loss_weight=None,
            anti_collapse_covariance_weight=None,
            anchor_temperature=None,
            anchor_min_anchor_gap=None,
            track_debug_checkpoints=None,
            no_augment=False,
            dry_run=False,
            cache_spectrograms=None,
            normalize_loss=None,
        )

        kwargs = _build_training_kwargs(args)

        assert kwargs["segment_sampling"] == "self_audio_ordered"
        assert kwargs["temporal_order_loss_weight"] == pytest.approx(0.4)
        assert kwargs["relative_offset_loss_weight"] == pytest.approx(0.5)
        assert kwargs["relative_offset_bins"] == [4, 8, 16]
        assert kwargs["masked_reconstruction_loss_weight"] == pytest.approx(0.1)
        assert kwargs["hard_negative_loss_weight"] == pytest.approx(0.3)
        assert kwargs["hard_negative_radius_frames"] == 32
        assert kwargs["false_destination_negative_loss_weight"] == pytest.approx(0.07)
        assert kwargs["false_destination_negative_root"] == "results/failure_reports/mined.csv"
        assert kwargs["false_destination_min_error_ms"] == pytest.approx(750.0)
        assert kwargs["memory_bank_size"] == 128

    def test_headline_claim_config_accepts_strict_audio_only_route(self):
        from dis_alignment.model.train import _validate_headline_claim_config

        config = {
            "claim": {"name": "initial_audio_only_unconstrained", "headline": True},
            "dataset": {"segment_sampling": "same_lied_pair"},
            "training": {
                "selection_metric": "val_loss",
                "anchor_loss_weight": 0.0,
                "dense_anchor_loss_weight": 0.0,
                "path_distill_loss_weight": 0.0,
                "sequence_contrastive_loss_weight": 1.0,
                "teacher_path_root": None,
            },
            "evaluation": {
                "deep_decode": "unconstrained",
                "pool_size": 1,
                "band_radius_frames": None,
            },
        }

        _validate_headline_claim_config(config)

    def test_headline_claim_config_accepts_stage1_v2_sampling(self):
        from dis_alignment.model.train import _validate_headline_claim_config

        config = {
            "claim": {"name": "initial_audio_only_unconstrained", "headline": True},
            "dataset": {"segment_sampling": "self_audio_ordered"},
            "training": {
                "selection_metric": "val_loss",
                "track_debug_checkpoints": False,
                "anchor_loss_weight": 0.0,
                "dense_anchor_loss_weight": 0.0,
                "teacher_path_root": None,
                "temporal_order_loss_weight": 0.4,
                "relative_offset_loss_weight": 0.4,
                "hard_negative_loss_weight": 0.3,
            },
            "evaluation": {
                "deep_decode": "unconstrained",
                "pool_size": 1,
                "band_radius_frames": None,
            },
        }

        _validate_headline_claim_config(config)

    @pytest.mark.parametrize(
        "bad_decode",
        [
            "diagonal_band",
            "chroma_guided_band",
            "deepalign_transcription_fused",
            "deepalign_score_guided_refined",
        ],
    )
    def test_headline_claim_config_rejects_guided_decoders(self, bad_decode):
        from dis_alignment.model.train import _validate_headline_claim_config

        config = {
            "claim": {"name": "initial_audio_only_unconstrained", "headline": True},
            "dataset": {"segment_sampling": "self_audio"},
            "training": {
                "selection_metric": "val_loss",
                "anchor_loss_weight": 0.0,
                "dense_anchor_loss_weight": 0.0,
                "teacher_path_root": None,
            },
            "evaluation": {
                "deep_decode": bad_decode,
                "pool_size": 1,
                "band_radius_frames": None,
            },
        }

        with pytest.raises(ValueError, match="deep_decode"):
            _validate_headline_claim_config(config)

    def test_headline_claim_config_rejects_teacher_path_training(self):
        from dis_alignment.model.train import _validate_headline_claim_config

        config = {
            "claim": {"name": "initial_audio_only_unconstrained", "headline": True},
            "dataset": {"segment_sampling": "teacher_path"},
            "training": {
                "selection_metric": "val_loss",
                "anchor_loss_weight": 0.0,
                "dense_anchor_loss_weight": 0.0,
                "teacher_path_root": "results/teacher_initial_claim_audio_only/paths",
            },
            "evaluation": {
                "deep_decode": "unconstrained",
                "pool_size": 1,
                "band_radius_frames": None,
            },
        }

        with pytest.raises(ValueError, match="segment_sampling"):
            _validate_headline_claim_config(config)

    def test_headline_claim_config_rejects_debug_checkpoint_selection(self):
        from dis_alignment.model.train import _validate_headline_claim_config

        config = {
            "claim": {"name": "initial_audio_only_unconstrained", "headline": True},
            "dataset": {"segment_sampling": "self_audio"},
            "training": {
                "selection_metric": "debug_ar50",
                "anchor_loss_weight": 0.0,
                "dense_anchor_loss_weight": 0.0,
                "teacher_path_root": None,
            },
            "evaluation": {
                "deep_decode": "unconstrained",
                "pool_size": 1,
                "band_radius_frames": None,
            },
        }

        with pytest.raises(ValueError, match="selection_metric"):
            _validate_headline_claim_config(config)

    def test_tracked_selection_metrics_can_disable_debug_best_checkpoints(self):
        from dis_alignment.model.train import _tracked_selection_metrics

        assert _tracked_selection_metrics("val_loss", track_debug_checkpoints=False) == ("val_loss",)

    def test_headline_claim_config_rejects_teacher_root_even_with_strict_sampling(self):
        from dis_alignment.model.train import _validate_headline_claim_config

        config = {
            "claim": {"name": "initial_audio_only_unconstrained", "headline": True},
            "dataset": {"segment_sampling": "self_audio"},
            "training": {
                "selection_metric": "val_loss",
                "anchor_loss_weight": 0.0,
                "dense_anchor_loss_weight": 0.0,
                "teacher_path_root": "results/pseudo_initial_claim_audio_only/paths",
            },
            "evaluation": {
                "deep_decode": "unconstrained",
                "pool_size": 1,
                "band_radius_frames": None,
            },
        }

        with pytest.raises(ValueError, match="teacher_path_root"):
            _validate_headline_claim_config(config)
