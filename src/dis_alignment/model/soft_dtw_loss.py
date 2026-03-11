"""Differentiable Soft-DTW loss function for end-to-end alignment training.

Soft-DTW replaces the hard minimum in classical DTW with a smoothed
(soft) minimum operator, enabling gradient backpropagation through
the alignment computation.

Reference:
    M. Cuturi and M. Blondel, "Soft-DTW: a Differentiable Loss Function
    for Time-Series," ICML, 2017.
"""

import torch
import torch.nn as nn
from torch import Tensor


class SoftDTWLoss(nn.Module):
    """
    Soft-DTW loss for differentiable sequence alignment.

    Computes the soft minimum cost of aligning two embedding sequences,
    with a smoothing parameter γ that controls the softness.

    As γ → 0, Soft-DTW approaches standard DTW.
    As γ → ∞, Soft-DTW approaches a mean-pooled distance.

    Args:
        gamma: Smoothing parameter. Start high (1.0) for stable gradients,
            anneal to low (0.01) for sharper alignment.
        normalize: If True, compute normalized Soft-DTW divergence
            (subtracts self-alignment to ensure non-negative loss).
        dist_func: Distance function for comparing embeddings.
            Default is squared Euclidean distance.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        normalize: bool = True,
        dist_func: str = "sqeuclidean",
    ):
        super().__init__()
        self.gamma = gamma
        self.normalize = normalize
        self.dist_func = dist_func

    def forward(self, seq_a: Tensor, seq_b: Tensor) -> Tensor:
        """
        Compute Soft-DTW loss between two embedding sequences.

        Args:
            seq_a: Embeddings of shape (batch, time_a, embed_dim).
            seq_b: Embeddings of shape (batch, time_b, embed_dim).

        Returns:
            Scalar loss value (mean over batch).
        """
        # Compute pairwise distance matrix
        D = self._pairwise_distances(seq_a, seq_b)

        # Compute Soft-DTW
        sdtw = self._compute_soft_dtw(D)

        if self.normalize:
            # Normalized divergence: sdtw(a,b) - 0.5*(sdtw(a,a) + sdtw(b,b))
            D_aa = self._pairwise_distances(seq_a, seq_a)
            D_bb = self._pairwise_distances(seq_b, seq_b)
            sdtw_aa = self._compute_soft_dtw(D_aa)
            sdtw_bb = self._compute_soft_dtw(D_bb)
            sdtw = sdtw - 0.5 * (sdtw_aa + sdtw_bb)

        return sdtw.mean()

    def _pairwise_distances(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Compute pairwise distance matrix between sequences.

        Args:
            x: Shape (batch, n, d).
            y: Shape (batch, m, d).

        Returns:
            Distance matrix of shape (batch, n, m).
        """
        if self.dist_func == "sqeuclidean":
            # ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x·y
            x_sq = (x ** 2).sum(dim=-1, keepdim=True)  # (B, N, 1)
            y_sq = (y ** 2).sum(dim=-1, keepdim=True)  # (B, M, 1)
            xy = torch.bmm(x, y.transpose(1, 2))  # (B, N, M)
            D = x_sq + y_sq.transpose(1, 2) - 2 * xy
            return D.clamp(min=0)  # Numerical stability
        elif self.dist_func == "cosine":
            # Cosine distance = 1 - cosine_similarity
            x_norm = nn.functional.normalize(x, p=2, dim=-1)
            y_norm = nn.functional.normalize(y, p=2, dim=-1)
            sim = torch.bmm(x_norm, y_norm.transpose(1, 2))
            return 1 - sim
        else:
            raise ValueError(f"Unknown distance function: {self.dist_func}")

    def _compute_soft_dtw(self, D: Tensor) -> Tensor:
        """
        Compute Soft-DTW alignment cost using anti-diagonal wavefront.

        Instead of iterating cell-by-cell (N*M Python iterations), processes
        entire anti-diagonals at once. Cells on the same anti-diagonal are
        independent, enabling full vectorization across batch and diagonal.

        This reduces Python loop iterations from N*M to N+M (e.g., from
        9M to 6K for 3000×3000 sequences — a ~1500x speedup).

        Args:
            D: Distance matrix of shape (batch, n, m).

        Returns:
            Soft-DTW cost for each batch element, shape (batch,).
        """
        batch_size, n, m = D.shape
        device = D.device
        dtype = D.dtype

        # Initialize accumulated cost matrix with infinity
        # Use 1-indexed R with border of inf for boundary conditions
        R = torch.full((batch_size, n + 1, m + 1), float("inf"),
                        device=device, dtype=dtype)
        R[:, 0, 0] = 0

        # Process anti-diagonals: d = i + j, ranging from 2 to n+m
        # On anti-diagonal d, valid cells satisfy:
        #   1 <= i <= n, 1 <= j <= m, i + j = d
        for d in range(2, n + m + 1):
            # Range of i values on this anti-diagonal
            i_min = max(1, d - m)
            i_max = min(n, d - 1)
            i_vals = torch.arange(i_min, i_max + 1, device=device)
            j_vals = d - i_vals  # j = d - i

            # Gather distance costs: D is 0-indexed
            cost = D[:, i_vals - 1, j_vals - 1]  # (batch, diag_len)

            # Gather the three predecessors for each cell
            r_diag = R[:, i_vals - 1, j_vals - 1]  # diagonal
            r_left = R[:, i_vals, j_vals - 1]       # left
            r_up = R[:, i_vals - 1, j_vals]         # up

            # Stack and compute soft-min: (batch, diag_len, 3)
            r_stack = torch.stack([r_up, r_left, r_diag], dim=-1)
            smin = -self.gamma * torch.logsumexp(-r_stack / self.gamma, dim=-1)

            # Scatter results back into R
            R[:, i_vals, j_vals] = cost + smin

        return R[:, n, m]

    def _soft_min(self, x: Tensor) -> Tensor:
        """
        Compute soft minimum: -γ * log(Σ exp(-x/γ)).

        Args:
            x: Values of shape (..., k).

        Returns:
            Soft minimum, shape (...).
        """
        return -self.gamma * torch.logsumexp(-x / self.gamma, dim=-1)

    def set_gamma(self, gamma: float) -> None:
        """Update the smoothing parameter (for annealing)."""
        self.gamma = gamma


class GammaScheduler:
    """
    Scheduler for annealing the Soft-DTW γ parameter.

    Exponentially decays γ from start_gamma to end_gamma over
    the specified number of epochs.

    Args:
        loss_fn: SoftDTWLoss instance to update.
        start_gamma: Initial γ value.
        end_gamma: Final γ value.
        num_epochs: Total training epochs.
    """

    def __init__(
        self,
        loss_fn: SoftDTWLoss,
        start_gamma: float = 1.0,
        end_gamma: float = 0.01,
        num_epochs: int = 50,
    ):
        self.loss_fn = loss_fn
        self.start_gamma = start_gamma
        self.end_gamma = end_gamma
        self.num_epochs = num_epochs
        self._decay_rate = (end_gamma / start_gamma) ** (1.0 / max(num_epochs - 1, 1))

    def step(self, epoch: int) -> float:
        """
        Update γ for the given epoch.

        Args:
            epoch: Current epoch (0-indexed).

        Returns:
            New γ value.
        """
        gamma = self.start_gamma * (self._decay_rate ** epoch)
        gamma = max(gamma, self.end_gamma)
        self.loss_fn.set_gamma(gamma)
        return gamma
