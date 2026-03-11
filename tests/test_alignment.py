"""Tests for alignment algorithms."""

import numpy as np
import pytest

from dis_alignment.alignment.baseline_dtw import (
    AlignmentResult,
    align_global_dtw,
    path_to_alignment,
)


class TestGlobalDTW:
    """Tests for standard Global DTW implementation."""
    
    def test_identical_sequences(self):
        """Test alignment of identical sequences produces diagonal path."""
        n_features, n_frames = 12, 50
        features = np.random.randn(n_features, n_frames).astype(np.float32)
        
        result = align_global_dtw(features, features.copy())
        
        # Path should be approximately diagonal
        assert result.path.shape[0] == 2
        # Start at (0, 0)
        assert result.path[0, 0] == 0
        assert result.path[1, 0] == 0
        # End at (n-1, n-1)
        assert result.path[0, -1] == n_frames - 1
        assert result.path[1, -1] == n_frames - 1
    
    def test_cost_is_zero_for_identical(self):
        """Test that identical sequences have near-zero cost."""
        n_features, n_frames = 12, 30
        features = np.random.randn(n_features, n_frames).astype(np.float32)
        # Normalize for cosine distance
        features = features / np.linalg.norm(features, axis=0, keepdims=True)
        
        result = align_global_dtw(features, features.copy(), distance="cosine")
        
        # Cosine distance of identical normalized vectors is 0
        assert result.cost < 0.01 * n_frames
    
    def test_path_is_monotonic(self):
        """Test that DTW path is monotonically increasing."""
        n_features, n_frames = 12, 40
        features_q = np.random.randn(n_features, n_frames).astype(np.float32)
        features_r = np.random.randn(n_features, n_frames + 10).astype(np.float32)
        
        result = align_global_dtw(features_q, features_r)
        
        # Both dimensions should be monotonically non-decreasing
        assert np.all(np.diff(result.path[0, :]) >= 0)
        assert np.all(np.diff(result.path[1, :]) >= 0)
    
    def test_sakoe_chiba_constraint(self):
        """Test Sakoe-Chiba band constraint is respected."""
        n_features, n_frames = 12, 50
        features = np.random.randn(n_features, n_frames).astype(np.float32)
        radius = 5
        
        result = align_global_dtw(features, features.copy(), sakoe_chiba_radius=radius)
        
        # Path should stay within band
        for i in range(result.path.shape[1]):
            q_idx, r_idx = result.path[0, i], result.path[1, i]
            assert abs(q_idx - r_idx) <= radius + 1  # +1 for boundary effects
    
    def test_result_metadata(self):
        """Test that result contains expected metadata."""
        features = np.random.randn(12, 20).astype(np.float32)
        
        result = align_global_dtw(features, features)
        
        assert isinstance(result, AlignmentResult)
        assert result.runtime_seconds >= 0
        assert result.memory_bytes > 0
        assert result.algorithm == "global_dtw"
    
    def test_cost_matrix_retention(self):
        """Test optional cost matrix retention."""
        n_features, n_frames = 12, 15
        features = np.random.randn(n_features, n_frames).astype(np.float32)
        
        result_with = align_global_dtw(features, features, retain_cost_matrix=True)
        result_without = align_global_dtw(features, features, retain_cost_matrix=False)
        
        assert result_with.cost_matrix is not None
        assert result_with.cost_matrix.shape == (n_frames, n_frames)
        assert result_without.cost_matrix is None


class TestPathConversion:
    """Tests for path conversion utilities."""
    
    def test_path_to_time(self):
        """Test conversion of frame path to time alignment."""
        # Simple path: frames 0, 1, 2 -> 0, 1, 2
        path = np.array([[0, 1, 2], [0, 1, 2]], dtype=np.intp)
        sr = 22050
        hop_length = 512
        
        time_path = path_to_alignment(path, hop_length, sr)
        
        expected_frame_duration = hop_length / sr
        np.testing.assert_allclose(
            time_path[0, :],
            np.array([0, 1, 2]) * expected_frame_duration,
        )


class TestSyntheticWarp:
    """Tests using synthetic warped sequences."""
    
    def test_warp_recovery(self):
        """Test that DTW can recover a known time warp."""
        np.random.seed(42)
        n_features, n_frames = 12, 100
        
        # Create reference features
        reference = np.random.randn(n_features, n_frames).astype(np.float32)
        
        # Create warped query by duplicating some frames (tempo variation)
        warp_indices = []
        for i in range(n_frames):
            # Randomly repeat some frames to simulate rubato
            repeats = 1 if np.random.rand() > 0.3 else 2
            warp_indices.extend([i] * repeats)
        
        warp_indices = np.array(warp_indices)
        query = reference[:, warp_indices]
        
        # Align
        result = align_global_dtw(query, reference)
        
        # Verify path approximately recovers the warp
        # For each query frame, the aligned reference frame should match
        for i in range(result.path.shape[1]):
            q_idx = result.path[0, i]
            r_idx = result.path[1, i]
            expected_r = warp_indices[min(q_idx, len(warp_indices) - 1)]
            # Allow some tolerance due to DTW flexibility
            assert abs(r_idx - expected_r) <= 2, f"Frame {q_idx}: expected ~{expected_r}, got {r_idx}"
