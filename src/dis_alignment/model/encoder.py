"""Dual-stream CRNN encoder for learning alignment features.

The encoder maps audio spectrograms to time-indexed embedding sequences
that are optimised for alignment via Soft-DTW loss.

Architecture:
    Input: CQT/Mel spectrogram (batch, 1, freq_bins, time_frames)
    → Conv blocks: feature extraction from spectral patterns
    → BiGRU layers: temporal modelling across frames
    → Output: embedding sequence (batch, time_frames, embed_dim)
"""

import torch
import torch.nn as nn
from torch import Tensor


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm, ReLU, and optional pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] = (3, 3),
        pool_size: tuple[int, int] | None = (2, 1),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout)
        self.pool = nn.MaxPool2d(pool_size) if pool_size else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool(x)
        return x


class CRNNEncoder(nn.Module):
    """
    Convolutional Recurrent Neural Network encoder for audio alignment.

    Maps spectrograms to time-indexed embedding sequences. Uses a Siamese
    architecture where both audio streams share the same encoder weights.

    Args:
        n_freq_bins: Number of frequency bins in input spectrogram.
            Default 84 for CQT with 7 octaves × 12 bins.
        embed_dim: Dimensionality of output embeddings.
        num_conv_channels: List of channel sizes for conv blocks.
        gru_hidden_size: Hidden size for BiGRU layers.
        num_gru_layers: Number of stacked GRU layers.
        dropout: Dropout rate.

    Input:
        Spectrogram tensor of shape (batch, 1, n_freq_bins, n_time_frames).

    Output:
        Embedding tensor of shape (batch, n_time_frames, embed_dim).
    """

    def __init__(
        self,
        n_freq_bins: int = 84,
        embed_dim: int = 64,
        num_conv_channels: list[int] | None = None,
        gru_hidden_size: int = 128,
        num_gru_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        if num_conv_channels is None:
            num_conv_channels = [32, 64, 128, 128]

        self.n_freq_bins = n_freq_bins
        self.embed_dim = embed_dim

        # Build convolutional feature extractor
        # Pool only along frequency axis to preserve time resolution
        conv_layers = []
        in_ch = 1
        for out_ch in num_conv_channels:
            conv_layers.append(ConvBlock(
                in_ch, out_ch,
                kernel_size=(3, 3),
                pool_size=(2, 1),  # Downsample frequency, keep time
                dropout=dropout,
            ))
            in_ch = out_ch

        self.conv = nn.Sequential(*conv_layers)

        # Calculate frequency dimension after pooling
        freq_after_conv = n_freq_bins
        for _ in num_conv_channels:
            freq_after_conv = freq_after_conv // 2

        # Input size to GRU = last conv channels × remaining freq bins
        gru_input_size = num_conv_channels[-1] * freq_after_conv

        # Recurrent layers for temporal modelling
        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=gru_hidden_size,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_gru_layers > 1 else 0,
        )

        # Project to embedding dimension
        # BiGRU output is 2 * hidden_size
        self.projection = nn.Sequential(
            nn.Linear(gru_hidden_size * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the encoder.

        Args:
            x: Input spectrogram of shape (batch, 1, freq, time).

        Returns:
            Embeddings of shape (batch, time, embed_dim).
        """
        batch_size = x.shape[0]

        # Conv feature extraction: (B, 1, F, T) → (B, C, F', T)
        features = self.conv(x)

        # Reshape for GRU: (B, C, F', T) → (B, T, C*F')
        features = features.permute(0, 3, 1, 2)  # (B, T, C, F')
        features = features.reshape(batch_size, features.shape[1], -1)  # (B, T, C*F')

        # Temporal modelling: (B, T, C*F') → (B, T, 2*H)
        features, _ = self.gru(features)

        # Project to embedding space: (B, T, 2*H) → (B, T, E)
        embeddings = self.projection(features)

        # Prevent temporal collapse by normalizing variance across time
        # InstanceNorm expects (B, C, T)
        embeddings = embeddings.transpose(1, 2)
        embeddings = nn.functional.instance_norm(embeddings)
        embeddings = embeddings.transpose(1, 2)

        # L2 normalize embeddings for cosine-based DTW
        embeddings = nn.functional.normalize(embeddings, p=2, dim=-1)

        return embeddings

    @torch.no_grad()
    def extract_features(self, x: Tensor) -> Tensor:
        """Extract features without gradient computation (for inference)."""
        self.eval()
        return self.forward(x)


class DeepAlignModel(nn.Module):
    """
    Full DeepAlign-26 model with shared Siamese encoder.

    Takes two spectrograms and produces two embedding sequences
    for alignment via Soft-DTW.

    Args:
        encoder: CRNNEncoder instance (weights shared between streams).
    """

    def __init__(self, encoder: CRNNEncoder):
        super().__init__()
        self.encoder = encoder

    def forward(
        self,
        spec_a: Tensor,
        spec_b: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Encode both audio streams through the shared encoder.

        Args:
            spec_a: Spectrogram A of shape (batch, 1, freq, time_a).
            spec_b: Spectrogram B of shape (batch, 1, freq, time_b).

        Returns:
            Tuple of (embeddings_a, embeddings_b), each of shape
            (batch, time, embed_dim).
        """
        emb_a = self.encoder(spec_a)
        emb_b = self.encoder(spec_b)
        return emb_a, emb_b
