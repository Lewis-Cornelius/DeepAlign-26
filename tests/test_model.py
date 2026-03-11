"""Tests for deep learning components (encoder, loss, dataset, augmentation).

All tests use synthetic data — no SWD download required.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from dis_alignment.model.encoder import CRNNEncoder, ConvBlock, DeepAlignModel
from dis_alignment.model.soft_dtw_loss import GammaScheduler, SoftDTWLoss


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
