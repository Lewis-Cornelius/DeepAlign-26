"""Tests for feature extraction modules."""

import numpy as np
import pytest

from dis_alignment.features.chroma import extract_chroma_cqt
from dis_alignment.features.dlnco import extract_dlnco, _apply_decay_envelope
from dis_alignment.features.transcription import (
    build_transcription_feature_blocks,
    extract_basic_pitch_features,
    fold_pitch_to_pitch_class,
    resample_feature_frames,
)


class TestChromaExtraction:
    """Tests for CQT chroma feature extraction."""
    
    def test_output_shape(self):
        """Test that chroma has expected shape."""
        # Create synthetic audio (1 second of noise)
        sr = 22050
        audio = np.random.randn(sr).astype(np.float32)
        
        chroma = extract_chroma_cqt(audio, sr=sr, hop_length=512)
        
        # Should have 12 chroma bins
        assert chroma.shape[0] == 12
        # Should have expected number of frames
        expected_frames = 1 + len(audio) // 512
        assert abs(chroma.shape[1] - expected_frames) <= 1
    
    def test_normalization(self):
        """Test that L2 normalization is applied."""
        sr = 22050
        audio = np.random.randn(sr).astype(np.float32)
        
        chroma = extract_chroma_cqt(audio, sr=sr, norm="l2")
        
        # Each frame should have unit L2 norm (or zero)
        norms = np.linalg.norm(chroma, axis=0)
        non_zero = norms > 1e-6
        np.testing.assert_allclose(norms[non_zero], 1.0, rtol=1e-5)
    
    def test_no_normalization(self):
        """Test that normalization can be disabled."""
        sr = 22050
        audio = np.random.randn(sr).astype(np.float32)
        
        chroma = extract_chroma_cqt(audio, sr=sr, norm=None)
        
        # Should not have unit norm
        norms = np.linalg.norm(chroma, axis=0)
        assert not np.allclose(norms[norms > 0], 1.0)


class TestDLNCO:
    """Tests for DLNCO feature extraction."""
    
    def test_output_shape(self):
        """Test that DLNCO has expected shape."""
        sr = 22050
        audio = np.random.randn(sr).astype(np.float32)
        
        dlnco = extract_dlnco(audio, sr=sr, hop_length=512)
        
        assert dlnco.shape[0] == 12
        expected_frames = 1 + len(audio) // 512
        assert abs(dlnco.shape[1] - expected_frames) <= 1
    
    def test_decay_envelope(self):
        """Test exponential decay envelope application."""
        # Create simple test input with one onset per feature
        n_features, n_frames = 3, 10
        features = np.zeros((n_features, n_frames))
        features[0, 2] = 1.0  # Onset at frame 2
        features[1, 5] = 1.0  # Onset at frame 5
        
        decay_rate = 0.5
        output = _apply_decay_envelope(features, decay_rate)
        
        # Check decay behavior
        # Feature 0: onset at frame 2, should decay exponentially after
        assert output[0, 2] == 1.0
        assert output[0, 3] == pytest.approx(0.5, rel=1e-5)
        assert output[0, 4] == pytest.approx(0.25, rel=1e-5)
        
        # Feature 1: should be zero before onset
        assert output[1, 4] == 0.0
        assert output[1, 5] == 1.0
    
    def test_non_negative(self):
        """Test that DLNCO features are non-negative."""
        sr = 22050
        audio = np.random.randn(sr).astype(np.float32)
        
        dlnco = extract_dlnco(audio, sr=sr)
        
        assert np.all(dlnco >= 0)


class TestFeatureConsistency:
    """Tests for consistent behavior between feature types."""
    
    def test_same_frame_count(self):
        """Test that chroma and DLNCO produce same number of frames."""
        sr = 22050
        audio = np.random.randn(sr * 2).astype(np.float32)
        hop_length = 512
        
        chroma = extract_chroma_cqt(audio, sr=sr, hop_length=hop_length)
        dlnco = extract_dlnco(audio, sr=sr, hop_length=hop_length)
        
        assert chroma.shape[1] == dlnco.shape[1]


class TestTranscriptionFeatures:
    """Tests for optional Basic Pitch feature adaptation."""

    def test_build_transcription_blocks_resamples_and_folds_pitch_classes(self):
        frames = 5
        pitches = 88
        output = {
            "note": np.ones((frames, pitches), dtype=np.float32),
            "onset": np.eye(frames, pitches, dtype=np.float32),
            "contour": np.full((frames, pitches), 0.5, dtype=np.float32),
        }

        blocks = build_transcription_feature_blocks(output, target_frames=7)

        assert blocks["note"].shape == (pitches, 7)
        assert blocks["onset"].shape == (pitches, 7)
        assert blocks["contour"].shape == (pitches, 7)
        assert blocks["pitch_class_note"].shape == (12, 7)
        assert blocks["pitch_class_onset"].shape == (12, 7)
        assert np.isfinite(blocks["note"]).all()

    def test_fold_pitch_to_pitch_class_preserves_time_axis(self):
        features = np.zeros((24, 3), dtype=np.float32)
        features[0] = 1.0
        features[12] = 2.0

        folded = fold_pitch_to_pitch_class(features)

        assert folded.shape == (12, 3)
        np.testing.assert_allclose(folded[0], np.full(3, 3.0))

    def test_resample_feature_frames_handles_single_frame(self):
        features = np.array([[1.0], [2.0]], dtype=np.float32)

        resampled = resample_feature_frames(features, target_frames=4)

        np.testing.assert_allclose(resampled, np.array([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]]))

    def test_extract_basic_pitch_features_uses_cached_outputs(self, tmp_path):
        audio_path = tmp_path / "audio.wav"
        audio_path.write_bytes(b"not real audio; cache avoids decoding")
        cache_root = tmp_path / "cache"
        cached = cache_root / f"{audio_path.stem}_dummy.npz"
        cached.parent.mkdir(parents=True)

        from dis_alignment.features import transcription as transcription_module

        cache_path = transcription_module._cache_path(audio_path, cache_root)
        np.savez_compressed(
            cache_path,
            note=np.ones((4, 88), dtype=np.float32),
            onset=np.ones((4, 88), dtype=np.float32),
            contour=np.ones((4, 88), dtype=np.float32),
        )

        blocks = extract_basic_pitch_features(audio_path, cache_root=cache_root, target_frames=6)

        assert blocks["note"].shape == (88, 6)
        assert cached.name != cache_path.name

    def test_missing_basic_pitch_dependency_has_actionable_error(self, monkeypatch, tmp_path):
        from dis_alignment.features import transcription as transcription_module

        original_import = __import__

        def blocked_import(name, *args, **kwargs):
            if name.startswith("basic_pitch"):
                raise ImportError("blocked for test")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", blocked_import)

        with pytest.raises(ImportError, match="Basic Pitch"):
            transcription_module._predict_basic_pitch(tmp_path / "missing.wav")
