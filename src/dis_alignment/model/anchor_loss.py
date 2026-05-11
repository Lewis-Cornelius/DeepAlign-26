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


class SequenceContrastiveLoss(nn.Module):
    """
    Contrast whole same-lied performance pairs against other lieder in the batch.

    The positive is the row-wise pair (A_i, B_i). Other batch items are negatives,
    except items with the same lied id, which are masked to avoid false negatives.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = float(temperature)

    def forward(
        self,
        emb_a: Tensor,
        emb_b: Tensor,
        lengths_a: Tensor | Sequence[int] | None = None,
        lengths_b: Tensor | Sequence[int] | None = None,
        group_ids: Sequence[str] | None = None,
    ) -> Tensor:
        if emb_a.shape[0] < 2 or emb_b.shape[0] < 2:
            return emb_a.new_zeros(())

        pooled_a = self._masked_mean_pool(emb_a, lengths_a)
        pooled_b = self._masked_mean_pool(emb_b, lengths_b)
        pooled_a = nn.functional.normalize(pooled_a, dim=-1)
        pooled_b = nn.functional.normalize(pooled_b, dim=-1)
        logits_ab = pooled_a @ pooled_b.T / self.temperature
        logits_ba = pooled_b @ pooled_a.T / self.temperature
        valid_mask = self._negative_mask(emb_a.shape[0], group_ids=group_ids, device=emb_a.device)
        targets = torch.arange(emb_a.shape[0], device=emb_a.device)
        losses: list[Tensor] = []
        for logits in (logits_ab, logits_ba):
            row_has_negative = valid_mask.clone()
            row_has_negative.fill_diagonal_(False)
            row_mask = row_has_negative.any(dim=1)
            if not row_mask.any():
                continue
            masked_logits = logits.masked_fill(~valid_mask, torch.finfo(logits.dtype).min)
            losses.append(nn.functional.cross_entropy(masked_logits[row_mask], targets[row_mask]))
        if not losses:
            return emb_a.new_zeros(())
        return torch.stack(losses).mean()

    @staticmethod
    def _masked_mean_pool(emb: Tensor, lengths: Tensor | Sequence[int] | None) -> Tensor:
        if lengths is None:
            return emb.mean(dim=1)
        length_tensor = torch.as_tensor(lengths, device=emb.device, dtype=torch.long)
        length_tensor = length_tensor.clamp(min=1, max=emb.shape[1])
        frames = torch.arange(emb.shape[1], device=emb.device)[None, :]
        mask = frames < length_tensor[:, None]
        masked = emb * mask.unsqueeze(-1).to(dtype=emb.dtype)
        return masked.sum(dim=1) / length_tensor.to(dtype=emb.dtype).unsqueeze(-1)

    @staticmethod
    def _negative_mask(
        batch_size: int,
        *,
        group_ids: Sequence[str] | None,
        device: torch.device,
    ) -> Tensor:
        mask = torch.ones(batch_size, batch_size, dtype=torch.bool, device=device)
        if group_ids is not None and len(group_ids) >= batch_size:
            labels = [str(value) for value in group_ids[:batch_size]]
            for row in range(batch_size):
                for col in range(batch_size):
                    if row != col and labels[row] == labels[col]:
                        mask[row, col] = False
        mask.fill_diagonal_(True)
        return mask


class HardNegativeContrastiveLoss(nn.Module):
    """
    Contrast same-audio positives against nearby confusable windows.

    The positive is the row-wise pair (A_i, B_i); ``emb_neg`` contains a third
    crop from the same recording, sampled close enough in time to be musically
    plausible but not identical. The loss asks the positive similarity to beat
    the hard negative by a margin.
    """

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = float(margin)

    def forward(
        self,
        emb_a: Tensor,
        emb_b: Tensor,
        emb_neg: Tensor,
        lengths_a: Tensor | Sequence[int] | None = None,
        lengths_b: Tensor | Sequence[int] | None = None,
        lengths_neg: Tensor | Sequence[int] | None = None,
    ) -> Tensor:
        pooled_a = SequenceContrastiveLoss._masked_mean_pool(emb_a, lengths_a)
        pooled_b = SequenceContrastiveLoss._masked_mean_pool(emb_b, lengths_b)
        pooled_neg = SequenceContrastiveLoss._masked_mean_pool(emb_neg, lengths_neg)
        pooled_a = nn.functional.normalize(pooled_a, dim=-1)
        pooled_b = nn.functional.normalize(pooled_b, dim=-1)
        pooled_neg = nn.functional.normalize(pooled_neg, dim=-1)

        positive = (pooled_a * pooled_b).sum(dim=-1)
        negative_a = (pooled_a * pooled_neg).sum(dim=-1)
        negative_b = (pooled_b * pooled_neg).sum(dim=-1)
        loss_a = torch.relu(self.margin + negative_a - positive)
        loss_b = torch.relu(self.margin + negative_b - positive)
        return torch.stack([loss_a, loss_b], dim=0).mean()


class MaskedReconstructionLoss(nn.Module):
    """Reconstruct only the masked CQT frames used by Stage 1 v2."""

    def forward(self, prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        if target.ndim == 4:
            target = target.squeeze(1).transpose(1, 2)
        if mask.ndim == 4:
            mask = mask.squeeze(1).transpose(1, 2)
        mask = mask.to(device=prediction.device, dtype=prediction.dtype)
        target = target.to(device=prediction.device, dtype=prediction.dtype)
        if mask.shape != prediction.shape:
            mask = mask.expand_as(prediction)
        denom = mask.sum().clamp_min(1.0)
        return ((prediction - target).pow(2) * mask).sum() / denom


class CycleAlignmentLoss(nn.Module):
    """
    Strict same-lied soft alignment loss without timestamps or decode bands.

    It builds unconstrained soft A->B and B->A alignments from learned feature
    similarities, then penalizes cycles that fail to return to the source frame.
    Optional entropy and monotonicity components are exposed separately so the
    trainer can weight them independently.
    """

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = float(temperature)

    def forward(
        self,
        emb_a: Tensor,
        emb_b: Tensor,
        lengths_a: Tensor | Sequence[int] | None = None,
        lengths_b: Tensor | Sequence[int] | None = None,
    ) -> dict[str, Tensor]:
        losses: list[Tensor] = []
        entropies: list[Tensor] = []
        monotonicities: list[Tensor] = []
        smoothnesses: list[Tensor] = []
        device = emb_a.device

        len_a = self._length_tensor(lengths_a, batch_size=emb_a.shape[0], max_len=emb_a.shape[1], device=device)
        len_b = self._length_tensor(lengths_b, batch_size=emb_b.shape[0], max_len=emb_b.shape[1], device=device)

        for batch_idx in range(emb_a.shape[0]):
            n_a = int(len_a[batch_idx].item())
            n_b = int(len_b[batch_idx].item())
            if n_a < 2 or n_b < 2:
                continue

            a = nn.functional.normalize(emb_a[batch_idx, :n_a], dim=-1)
            b = nn.functional.normalize(emb_b[batch_idx, :n_b], dim=-1)
            logits_ab = (a @ b.T) / self.temperature
            logits_ba = logits_ab.T
            prob_ab = nn.functional.softmax(logits_ab, dim=-1)
            prob_ba = nn.functional.softmax(logits_ba, dim=-1)

            pos_a = torch.linspace(0.0, 1.0, n_a, device=device, dtype=emb_a.dtype)
            pos_b = torch.linspace(0.0, 1.0, n_b, device=device, dtype=emb_a.dtype)
            cycle_a = prob_ab @ prob_ba
            cycle_b = prob_ba @ prob_ab
            expected_back_a = cycle_a @ pos_a
            expected_back_b = cycle_b @ pos_b
            cycle_loss = nn.functional.mse_loss(expected_back_a, pos_a) + nn.functional.mse_loss(
                expected_back_b,
                pos_b,
            )
            losses.append(cycle_loss)

            ent_ab = self._normalised_entropy(prob_ab)
            ent_ba = self._normalised_entropy(prob_ba)
            entropies.append((ent_ab + ent_ba) * 0.5)

            expected_b = prob_ab @ pos_b
            expected_a = prob_ba @ pos_a
            mono_ab = torch.relu(-(expected_b[1:] - expected_b[:-1])).mean()
            mono_ba = torch.relu(-(expected_a[1:] - expected_a[:-1])).mean()
            monotonicities.append((mono_ab + mono_ba) * 0.5)

            smooth_ab = self._slope_smoothness(expected_b)
            smooth_ba = self._slope_smoothness(expected_a)
            smoothnesses.append((smooth_ab + smooth_ba) * 0.5)

        zero = emb_a.new_zeros(())
        return {
            "cycle": torch.stack(losses).mean() if losses else zero,
            "entropy": torch.stack(entropies).mean() if entropies else zero,
            "monotonicity": torch.stack(monotonicities).mean() if monotonicities else zero,
            "smoothness": torch.stack(smoothnesses).mean() if smoothnesses else zero,
        }

    @staticmethod
    def _normalised_entropy(prob: Tensor) -> Tensor:
        entropy = -(prob * prob.clamp_min(1e-8).log()).sum(dim=-1)
        normaliser = torch.log(torch.as_tensor(prob.shape[-1], device=prob.device, dtype=prob.dtype)).clamp_min(1e-8)
        return (entropy / normaliser).mean()

    @staticmethod
    def _slope_smoothness(expected_positions: Tensor) -> Tensor:
        if expected_positions.numel() < 3:
            return expected_positions.new_zeros(())
        slopes = (expected_positions[1:] - expected_positions[:-1]) * max(expected_positions.numel() - 1, 1)
        return (slopes[1:] - slopes[:-1]).pow(2).mean()

    @staticmethod
    def _length_tensor(
        lengths: Tensor | Sequence[int] | None,
        *,
        batch_size: int,
        max_len: int,
        device: torch.device,
    ) -> Tensor:
        if lengths is None:
            return torch.full((batch_size,), max_len, device=device, dtype=torch.long)
        return torch.as_tensor(lengths, device=device, dtype=torch.long).clamp(min=0, max=max_len)


class EmbeddingAntiCollapseLoss(nn.Module):
    """
    Penalize collapsed embeddings with a VICReg-style variance/covariance term.
    """

    def __init__(self, target_std: float = 1.0, covariance_weight: float = 0.01, eps: float = 1e-4):
        super().__init__()
        self.target_std = float(target_std)
        self.covariance_weight = float(covariance_weight)
        self.eps = float(eps)

    def forward(
        self,
        emb_a: Tensor,
        emb_b: Tensor,
        lengths_a: Tensor | Sequence[int] | None = None,
        lengths_b: Tensor | Sequence[int] | None = None,
    ) -> Tensor:
        frames = [
            self._valid_frames(emb_a, lengths_a),
            self._valid_frames(emb_b, lengths_b),
        ]
        values = torch.cat([frame for frame in frames if frame.numel() > 0], dim=0)
        if values.shape[0] < 2:
            return emb_a.new_zeros(())

        centered = values - values.mean(dim=0, keepdim=True)
        std = torch.sqrt(centered.var(dim=0, unbiased=False) + self.eps)
        variance_loss = torch.relu(self.target_std - std).mean()

        covariance_loss = emb_a.new_zeros(())
        if values.shape[0] > 2 and values.shape[1] > 1 and self.covariance_weight > 0:
            cov = centered.T @ centered / (values.shape[0] - 1)
            cov = cov - torch.diag(torch.diag(cov))
            covariance_loss = cov.pow(2).sum() / values.shape[1]
        return variance_loss + (self.covariance_weight * covariance_loss)

    @staticmethod
    def _valid_frames(emb: Tensor, lengths: Tensor | Sequence[int] | None) -> Tensor:
        if lengths is None:
            return emb.reshape(-1, emb.shape[-1])
        length_tensor = torch.as_tensor(lengths, device=emb.device, dtype=torch.long)
        length_tensor = length_tensor.clamp(min=0, max=emb.shape[1])
        frames = torch.arange(emb.shape[1], device=emb.device)[None, :]
        mask = frames < length_tensor[:, None]
        return emb[mask]
