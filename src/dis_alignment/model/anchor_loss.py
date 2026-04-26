"""Auxiliary anchor-based contrastive loss for aligned SWD windows."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor


class AnchorContrastiveLoss(nn.Module):
    """
    Contrastive alignment loss over shared measure anchors.

    Positive pairs are matched measure anchors across the two performances.
    Negatives are other anchors that are sufficiently separated in event order.
    """

    def __init__(self, temperature: float = 0.1, min_anchor_gap: int = 1):
        super().__init__()
        self.temperature = temperature
        self.min_anchor_gap = min_anchor_gap

    def forward(
        self,
        emb_a: Tensor,
        emb_b: Tensor,
        anchor_frames_a: Sequence[Sequence[int]] | None,
        anchor_frames_b: Sequence[Sequence[int]] | None,
    ) -> Tensor:
        if not anchor_frames_a or not anchor_frames_b:
            return emb_a.new_zeros(())

        losses: list[Tensor] = []
        batch_size = emb_a.shape[0]
        for batch_idx in range(batch_size):
            if batch_idx >= len(anchor_frames_a) or batch_idx >= len(anchor_frames_b):
                break

            idx_a = torch.as_tensor(anchor_frames_a[batch_idx], device=emb_a.device, dtype=torch.long)
            idx_b = torch.as_tensor(anchor_frames_b[batch_idx], device=emb_b.device, dtype=torch.long)
            n_anchors = min(idx_a.numel(), idx_b.numel())
            if n_anchors < 2:
                continue

            idx_a = idx_a[:n_anchors].clamp_(0, emb_a.shape[1] - 1)
            idx_b = idx_b[:n_anchors].clamp_(0, emb_b.shape[1] - 1)

            anchors_a = emb_a[batch_idx, idx_a]
            anchors_b = emb_b[batch_idx, idx_b]
            logits_ab = anchors_a @ anchors_b.T / self.temperature
            logits_ba = anchors_b @ anchors_a.T / self.temperature

            valid_mask = self._build_negative_mask(n_anchors, device=emb_a.device)
            if valid_mask is None:
                continue

            large_neg = torch.finfo(logits_ab.dtype).min
            logits_ab = logits_ab.masked_fill(~valid_mask, large_neg)
            logits_ba = logits_ba.masked_fill(~valid_mask, large_neg)

            targets = torch.arange(n_anchors, device=emb_a.device)
            losses.append(nn.functional.cross_entropy(logits_ab, targets))
            losses.append(nn.functional.cross_entropy(logits_ba, targets))

        if not losses:
            return emb_a.new_zeros(())
        return torch.stack(losses).mean()

    def _build_negative_mask(self, n_anchors: int, *, device: torch.device) -> Tensor | None:
        indices = torch.arange(n_anchors, device=device)
        distance = (indices[:, None] - indices[None, :]).abs()
        mask = distance >= self.min_anchor_gap
        mask.fill_diagonal_(True)

        off_diagonal = mask.clone()
        off_diagonal.fill_diagonal_(False)
        if not off_diagonal.any():
            return None
        return mask
