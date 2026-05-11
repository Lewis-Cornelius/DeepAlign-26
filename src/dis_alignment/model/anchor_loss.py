"""Auxiliary anchor-based contrastive loss for aligned SWD windows."""

from __future__ import annotations

from collections.abc import Sequence

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

            frames_a = anchor_frames_a[batch_idx]
            frames_b = anchor_frames_b[batch_idx]
            if frames_a is None or frames_b is None or len(frames_a) == 0 or len(frames_b) == 0:
                continue

            idx_a = torch.as_tensor(frames_a, device=emb_a.device, dtype=torch.long)
            idx_b = torch.as_tensor(frames_b, device=emb_b.device, dtype=torch.long)
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


class PathDistillationLoss(AnchorContrastiveLoss):
    """
    Contrastive loss over dense teacher-path frame correspondences.

    Sampled teacher frames are positives, and nearby temporal offsets become
    local negatives. This teaches sub-second precision instead of only asking
    the model to separate distant points on the same path.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        min_anchor_gap: int = 1,
        local_radius: int = 12,
        local_step: int = 3,
    ):
        super().__init__(temperature=temperature, min_anchor_gap=min_anchor_gap)
        self.local_radius = max(1, int(local_radius))
        self.local_step = max(1, int(local_step))

    def forward(
        self,
        emb_a: Tensor,
        emb_b: Tensor,
        anchor_frames_a: Sequence[Sequence[int]] | None,
        anchor_frames_b: Sequence[Sequence[int]] | None,
    ) -> Tensor:
        if not anchor_frames_a or not anchor_frames_b:
            return emb_a.new_zeros(())

        offsets = self._candidate_offsets(device=emb_a.device)
        target_index = int((offsets == 0).nonzero(as_tuple=False)[0].item())
        losses: list[Tensor] = []
        for batch_idx in range(emb_a.shape[0]):
            if batch_idx >= len(anchor_frames_a) or batch_idx >= len(anchor_frames_b):
                break

            frames_a = anchor_frames_a[batch_idx]
            frames_b = anchor_frames_b[batch_idx]
            if frames_a is None or frames_b is None or len(frames_a) == 0 or len(frames_b) == 0:
                continue

            idx_a = torch.as_tensor(frames_a, device=emb_a.device, dtype=torch.long)
            idx_b = torch.as_tensor(frames_b, device=emb_b.device, dtype=torch.long)
            n_points = min(idx_a.numel(), idx_b.numel())
            if n_points == 0:
                continue

            idx_a = idx_a[:n_points].clamp_(0, emb_a.shape[1] - 1)
            idx_b = idx_b[:n_points].clamp_(0, emb_b.shape[1] - 1)
            losses.extend(
                self._local_losses_one_direction(
                    queries=emb_a[batch_idx],
                    references=emb_b[batch_idx],
                    query_indices=idx_a,
                    reference_indices=idx_b,
                    offsets=offsets,
                    target_index=target_index,
                )
            )
            losses.extend(
                self._local_losses_one_direction(
                    queries=emb_b[batch_idx],
                    references=emb_a[batch_idx],
                    query_indices=idx_b,
                    reference_indices=idx_a,
                    offsets=offsets,
                    target_index=target_index,
                )
            )

        if not losses:
            return emb_a.new_zeros(())
        return torch.stack(losses).mean()

    def _candidate_offsets(self, *, device: torch.device) -> Tensor:
        offsets = torch.arange(
            -self.local_radius,
            self.local_radius + 1,
            self.local_step,
            device=device,
            dtype=torch.long,
        )
        if not (offsets == 0).any():
            offsets = torch.cat([offsets, torch.zeros(1, device=device, dtype=torch.long)])
            offsets = torch.sort(offsets).values
        return offsets

    def _local_losses_one_direction(
        self,
        *,
        queries: Tensor,
        references: Tensor,
        query_indices: Tensor,
        reference_indices: Tensor,
        offsets: Tensor,
        target_index: int,
    ) -> list[Tensor]:
        query_vecs = nn.functional.normalize(queries[query_indices], dim=-1)
        raw_candidate_indices = reference_indices[:, None] + offsets[None, :]
        valid_candidates = (
            (raw_candidate_indices >= 0)
            & (raw_candidate_indices < references.shape[0])
            & (offsets.abs()[None, :] >= self.min_anchor_gap)
        )
        valid_candidates[:, target_index] = True
        rows_with_negatives = valid_candidates.clone()
        rows_with_negatives[:, target_index] = False
        row_mask = rows_with_negatives.any(dim=1)
        if not row_mask.any():
            return []

        query_vecs = query_vecs[row_mask]
        candidate_indices = raw_candidate_indices[row_mask].clamp_(0, references.shape[0] - 1)
        valid_candidates = valid_candidates[row_mask]
        candidate_vecs = nn.functional.normalize(references[candidate_indices], dim=-1)
        logits = torch.einsum("nd,nkd->nk", query_vecs, candidate_vecs) / self.temperature
        logits = logits.masked_fill(~valid_candidates, torch.finfo(logits.dtype).min)
        targets = torch.full(
            (query_vecs.shape[0],),
            target_index,
            device=queries.device,
            dtype=torch.long,
        )
        return [nn.functional.cross_entropy(logits, targets)]
