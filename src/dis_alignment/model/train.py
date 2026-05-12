"""Training entrypoint for the DeepAlign-26 SWD workflow."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from dis_alignment.model.anchor_loss import (
    AnchorContrastiveLoss,
    CycleAlignmentLoss,
    EmbeddingAntiCollapseLoss,
    HardNegativeContrastiveLoss,
    MaskedReconstructionLoss,
    PathDistillationLoss,
    SequenceContrastiveLoss,
    SoftPathDistillationLoss,
)
from dis_alignment.model.encoder import CRNNEncoder, DeepAlignModel
from dis_alignment.model.recovery import normalize_debug_lied_ids
from dis_alignment.model.soft_dtw_loss import GammaScheduler, SoftDTWLoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SELECTION_METRICS = ("debug_mae", "debug_ar50", "debug_balanced", "val_loss", "val_mae")
TRACKED_SELECTION_METRICS = ("debug_ar50", "debug_mae", "debug_balanced")
TRAIN_DEEP_DECODE_CHOICES = (
    "unconstrained",
    "diagonal_band",
    "chroma_guided_band",
    "deepalign_transcription_fused",
    "deepalign_transcription_fused_refined",
    "deepalign_transcription_guided",
    "deepalign_score_guided_refined",
)
BEST_CHECKPOINT_FILENAMES = {
    "debug_ar50": "best_model_debug_ar50.pt",
    "debug_mae": "best_model_debug_mae.pt",
    "debug_balanced": "best_model_balanced.pt",
}
HEADLINE_CLAIM_NAMES = {"initial_audio_only_unconstrained", "audio_only_unconstrained"}


class _SyntheticPairDataset(Dataset):
    """Tiny synthetic dataset for dry-run validation."""

    def __init__(self, n_pairs: int = 4, n_freq: int = 84, n_time: int = 50):
        self.n_pairs = n_pairs
        self.n_freq = n_freq
        self.n_time = n_time

    def __len__(self) -> int:
        return self.n_pairs

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        spec_a = torch.randn(1, self.n_freq, self.n_time)
        spec_b = spec_a.clone()
        mask = torch.zeros_like(spec_a)
        mask[:, :, self.n_time // 2 :: max(1, self.n_time // 8)] = 1.0
        spec_masked = spec_a.masked_fill(mask.bool(), 0.0)
        return {
            "spec_a": spec_masked,
            "spec_b": spec_b,
            "spec_neg": torch.randn(1, self.n_freq, self.n_time),
            "pair_id": f"synthetic_{idx}",
            "lied_id": f"synthetic_lied_{idx % 2}",
            "anchor_frame_indices_a": [],
            "anchor_frame_indices_b": [],
            "teacher_frame_indices_a": list(range(0, self.n_time, max(1, self.n_time // 8))),
            "teacher_frame_indices_b": list(range(0, self.n_time, max(1, self.n_time // 8))),
            "temporal_order_label": idx % 3,
            "relative_offset_label": idx % 5,
            "reconstruction_target_a": spec_a,
            "reconstruction_mask_a": mask,
        }


class _StrictAuxiliaryHeads(nn.Module):
    """Training-only heads for strict Stage 1 v2 objectives."""

    def __init__(self, *, embed_dim: int, n_freq_bins: int, relative_offset_classes: int):
        super().__init__()
        pair_dim = embed_dim * 4
        hidden = max(embed_dim, 32)
        self.temporal_order = nn.Sequential(
            nn.Linear(pair_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )
        self.relative_offset = nn.Sequential(
            nn.Linear(pair_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, max(1, int(relative_offset_classes))),
        )
        self.reconstruction = nn.Linear(embed_dim, n_freq_bins)

    def pair_features(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        return torch.cat([emb_a, emb_b, (emb_a - emb_b).abs(), emb_a * emb_b], dim=-1)


def train(
    swd_path: str | None,
    output_dir: str = "checkpoints",
    epochs: int = 50,
    batch_size: int = 4,
    lr: float = 1e-3,
    embed_dim: int = 64,
    n_freq_bins: int = 84,
    num_conv_channels: list[int] | None = None,
    gru_hidden_size: int = 128,
    num_gru_layers: int = 2,
    dropout: float = 0.1,
    temporal_attention_heads: int = 0,
    sr: int = 22050,
    hop_length: int = 220,
    max_length_sec: float = 30.0,
    segment_sampling: str = "aligned_measures",
    samples_per_epoch: int | None = None,
    start_gamma: float = 1.0,
    end_gamma: float = 0.01,
    soft_dtw_loss_weight: float = 1.0,
    normalize_loss: bool = True,
    dist_func: str = "sqeuclidean",
    gradient_clip: float = 1.0,
    weight_decay: float = 1e-4,
    augment: bool = True,
    augmentor_kwargs: dict[str, Any] | None = None,
    device: str | None = None,
    val_split: float = 0.2,
    save_every_n_epochs: int = 10,
    cache_spectrograms: bool = False,
    cache_root: str | None = None,
    resume_from: str | None = None,
    selection_metric: str = "debug_ar50",
    debug_subset_lieder: Sequence[str] | None = None,
    deep_decode: str = "unconstrained",
    band_radius_frames: int | None = None,
    anchor_loss_weight: float = 0.0,
    dense_anchor_loss_weight: float = 0.0,
    path_distill_loss_weight: float = 0.0,
    soft_path_distill_loss_weight: float = 0.0,
    soft_path_temperature: float = 0.05,
    soft_path_target_sigma_frames: float = 2.0,
    sequence_contrastive_loss_weight: float = 0.0,
    anti_collapse_loss_weight: float = 0.0,
    anti_collapse_covariance_weight: float = 0.01,
    temporal_order_loss_weight: float = 0.0,
    relative_offset_loss_weight: float = 0.0,
    relative_offset_bins: Sequence[int] | None = None,
    masked_reconstruction_loss_weight: float = 0.0,
    cycle_consistency_loss_weight: float = 0.0,
    cycle_entropy_loss_weight: float = 0.0,
    cycle_monotonicity_loss_weight: float = 0.0,
    cycle_smoothness_loss_weight: float = 0.0,
    hard_negative_loss_weight: float = 0.0,
    hard_negative_radius_frames: int = 96,
    false_destination_negative_loss_weight: float = 0.0,
    false_destination_negative_root: str | None = None,
    false_destination_min_error_ms: float = 500.0,
    memory_bank_size: int = 0,
    teacher_path_root: str | None = None,
    self_mined_path_root: str | None = None,
    num_anchor_samples: int = 64,
    teacher_min_confidence: float = 0.0,
    path_distill_local_radius_frames: int = 12,
    path_distill_local_step_frames: int = 3,
    disable_time_stretch_for_anchors: bool = True,
    eval_pool_size: int = 2,
    alignment_eval_every_n_epochs: int = 1,
    track_debug_checkpoints: bool = True,
    anchor_temperature: float = 0.1,
    anchor_min_anchor_gap: int = 1,
    dry_run: bool = False,
) -> dict[str, list[float]]:
    """
    Train DeepAlign-26 on SWD or synthetic dry-run data.

    Returns:
        Training history with loss, selection, and alignment-gate metrics.
    """
    if selection_metric not in SELECTION_METRICS:
        raise ValueError(
            f"Unsupported selection metric: {selection_metric}. "
            f"Supported metrics: {', '.join(SELECTION_METRICS)}."
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    logger.info("Using device: %s", dev)

    if dev.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        logger.info("VRAM: %.1f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    resolved_cache_root = str(Path(cache_root or ".cache/cqt").resolve()) if cache_spectrograms else None
    debug_lied_ids = normalize_debug_lied_ids(debug_subset_lieder)
    resolved_relative_offset_bins = tuple(sorted(int(value) for value in (relative_offset_bins or (8, 24, 64, 128))))
    augmentor_kwargs = augmentor_kwargs or {}
    supervised_anchor_training = (
        anchor_loss_weight > 0
        or dense_anchor_loss_weight > 0
        or path_distill_loss_weight > 0
        or soft_path_distill_loss_weight > 0
    )
    if false_destination_negative_loss_weight > 0 and false_destination_negative_root is None:
        raise ValueError("false_destination_negative_loss_weight requires false_destination_negative_root.")
    if augment and disable_time_stretch_for_anchors and supervised_anchor_training:
        augmentor_kwargs = dict(augmentor_kwargs)
        augmentor_kwargs["time_stretch_range"] = (1.0, 1.0)
    collate_fn = None

    history = _empty_history()
    debug_pairs: list[Any] = []
    val_pairs: list[Any] = []
    n_train = 0
    n_val = 0
    resolved_samples_per_epoch = samples_per_epoch

    if dry_run:
        logger.info("DRY RUN: using synthetic data (no SWD needed)")
        epochs = min(epochs, 2)
        dataset = _SyntheticPairDataset(n_pairs=4, n_freq=n_freq_bins, n_time=50)
        n_val = 1
        n_train = len(dataset) - n_val
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])
    else:
        from dis_alignment.data.swd import SWDDataset
        from dis_alignment.model.augmentation import AudioAugmentor
        from dis_alignment.model.dataset import SWDPairDataset, collate_variable_length

        logger.info("Loading SWD from %s", swd_path)
        swd = SWDDataset(swd_path)
        logger.info("Available performances: %s", swd.available_performances)
        logger.info("Available lieder: %s", len(swd.available_lieder))

        all_pairs = list(swd.iter_pairs())
        train_pairs, val_pairs = _split_swd_pairs(
            all_pairs,
            val_split=val_split,
            heldout_lieder=debug_lied_ids,
        )
        debug_lied_set = {str(lied_id).strip().upper() for lied_id in debug_lied_ids}
        debug_pairs = [
            pair for pair in val_pairs if str(pair.lied_id).strip().upper() in debug_lied_set
        ]
        if selection_metric.startswith("debug_") and not debug_pairs:
            raise ValueError(
                "The configured debug subset did not resolve to any held-out SWD pairs, "
                "so debug-based checkpoint selection cannot run without leakage."
            )

        resolved_samples_per_epoch = (
            samples_per_epoch if samples_per_epoch is not None else max(256, len(train_pairs) * 16)
        )
        val_samples_per_epoch = len(val_pairs)

        train_augmentor = AudioAugmentor(**augmentor_kwargs) if augment else None
        emit_repeated_hard_negatives = (
            hard_negative_loss_weight > 0
            and false_destination_negative_loss_weight <= 0
            and segment_sampling in {"aligned_measures", "teacher_path", "self_mined_path"}
        )
        emit_false_destination_negatives = (
            false_destination_negative_loss_weight > 0
            and false_destination_negative_root is not None
        )
        train_dataset = SWDPairDataset(
            swd,
            sr=sr,
            hop_length=hop_length,
            max_length_sec=max_length_sec,
            augmentor=train_augmentor,
            n_bins=n_freq_bins,
            pairs=train_pairs,
            segment_sampling=segment_sampling,
            samples_per_epoch=resolved_samples_per_epoch,
            deterministic=False,
            cache_spectrograms=cache_spectrograms,
            cache_root=resolved_cache_root,
            teacher_path_root=teacher_path_root
            if (path_distill_loss_weight > 0 or soft_path_distill_loss_weight > 0)
            else None,
            self_mined_path_root=self_mined_path_root if segment_sampling == "self_mined_path" else None,
            false_destination_negative_root=false_destination_negative_root if emit_false_destination_negatives else None,
            false_destination_min_error_ms=false_destination_min_error_ms,
            num_teacher_samples=num_anchor_samples,
            teacher_min_confidence=teacher_min_confidence,
            relative_offset_bins=list(resolved_relative_offset_bins),
            hard_negative_radius_frames=hard_negative_radius_frames,
            emit_repeated_hard_negatives=emit_repeated_hard_negatives,
        )
        val_dataset = SWDPairDataset(
            swd,
            sr=sr,
            hop_length=hop_length,
            max_length_sec=max_length_sec,
            augmentor=None,
            n_bins=n_freq_bins,
            pairs=val_pairs,
            segment_sampling=segment_sampling,
            samples_per_epoch=val_samples_per_epoch,
            deterministic=True,
            cache_spectrograms=cache_spectrograms,
            cache_root=resolved_cache_root,
            teacher_path_root=teacher_path_root
            if (
                segment_sampling == "teacher_path"
                or path_distill_loss_weight > 0
                or soft_path_distill_loss_weight > 0
            )
            else None,
            self_mined_path_root=self_mined_path_root if segment_sampling == "self_mined_path" else None,
            false_destination_negative_root=false_destination_negative_root if emit_false_destination_negatives else None,
            false_destination_min_error_ms=false_destination_min_error_ms,
            num_teacher_samples=num_anchor_samples,
            teacher_min_confidence=teacher_min_confidence,
            relative_offset_bins=list(resolved_relative_offset_bins),
            hard_negative_radius_frames=hard_negative_radius_frames,
            emit_repeated_hard_negatives=emit_repeated_hard_negatives,
        )
        n_train = len(train_pairs)
        n_val = len(val_pairs)
        collate_fn = collate_variable_length

    logger.info(
        "Training pairs: %s, Validation pairs: %s, Train samples/epoch: %s, Sampling: %s, Cache: %s",
        n_train,
        n_val,
        len(train_dataset),
        segment_sampling,
        cache_spectrograms,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    encoder = CRNNEncoder(
        n_freq_bins=n_freq_bins,
        embed_dim=embed_dim,
        num_conv_channels=num_conv_channels,
        gru_hidden_size=gru_hidden_size,
        num_gru_layers=num_gru_layers,
        dropout=dropout,
        temporal_attention_heads=temporal_attention_heads,
    )
    model = DeepAlignModel(encoder).to(dev)
    logger.info("Model parameters: %s", f"{sum(p.numel() for p in model.parameters()):,}")

    criterion = (
        SoftDTWLoss(
            gamma=start_gamma,
            normalize=normalize_loss,
            dist_func=dist_func,
        ).to(dev)
        if soft_dtw_loss_weight > 0
        else None
    )
    anchor_loss_fn = (
        AnchorContrastiveLoss(temperature=anchor_temperature, min_anchor_gap=anchor_min_anchor_gap).to(dev)
        if anchor_loss_weight > 0
        else None
    )
    dense_anchor_loss_fn = (
        AnchorContrastiveLoss(temperature=anchor_temperature, min_anchor_gap=anchor_min_anchor_gap).to(dev)
        if dense_anchor_loss_weight > 0
        else None
    )
    path_distill_loss_fn = (
        PathDistillationLoss(
            temperature=anchor_temperature,
            min_anchor_gap=anchor_min_anchor_gap,
            local_radius=path_distill_local_radius_frames,
            local_step=path_distill_local_step_frames,
        ).to(dev)
        if path_distill_loss_weight > 0
        else None
    )
    soft_path_distill_loss_fn = (
        SoftPathDistillationLoss(
            temperature=soft_path_temperature,
            target_sigma_frames=soft_path_target_sigma_frames,
        ).to(dev)
        if soft_path_distill_loss_weight > 0
        else None
    )
    sequence_contrastive_loss_fn = (
        SequenceContrastiveLoss(temperature=anchor_temperature).to(dev)
        if sequence_contrastive_loss_weight > 0
        else None
    )
    anti_collapse_loss_fn = (
        EmbeddingAntiCollapseLoss(covariance_weight=anti_collapse_covariance_weight).to(dev)
        if anti_collapse_loss_weight > 0
        else None
    )
    hard_negative_loss_fn = (
        HardNegativeContrastiveLoss().to(dev)
        if hard_negative_loss_weight > 0 or false_destination_negative_loss_weight > 0
        else None
    )
    reconstruction_loss_fn = (
        MaskedReconstructionLoss().to(dev)
        if masked_reconstruction_loss_weight > 0
        else None
    )
    cycle_loss_fn = (
        CycleAlignmentLoss(temperature=anchor_temperature).to(dev)
        if (
            cycle_consistency_loss_weight > 0
            or cycle_entropy_loss_weight > 0
            or cycle_monotonicity_loss_weight > 0
            or cycle_smoothness_loss_weight > 0
        )
        else None
    )
    use_auxiliary_heads = (
        temporal_order_loss_weight > 0
        or relative_offset_loss_weight > 0
        or masked_reconstruction_loss_weight > 0
    )
    auxiliary_heads = (
        _StrictAuxiliaryHeads(
            embed_dim=embed_dim,
            n_freq_bins=n_freq_bins,
            relative_offset_classes=len(resolved_relative_offset_bins) + 1,
        ).to(dev)
        if use_auxiliary_heads
        else None
    )

    gamma_scheduler = (
        GammaScheduler(
            criterion,
            start_gamma=start_gamma,
            end_gamma=end_gamma,
            num_epochs=epochs,
        )
        if criterion is not None
        else None
    )
    optim_parameters = list(model.parameters())
    if auxiliary_heads is not None:
        optim_parameters.extend(auxiliary_heads.parameters())
    optimizer = AdamW(optim_parameters, lr=lr, weight_decay=weight_decay)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=(dev.type == "cuda"))

    training_state_signature = {
        "epochs": epochs,
        "lr": lr,
        "sr": sr,
        "hop_length": hop_length,
        "start_gamma": start_gamma,
        "end_gamma": end_gamma,
        "soft_dtw_loss_weight": soft_dtw_loss_weight,
        "selection_metric": selection_metric,
        "anchor_loss_weight": anchor_loss_weight,
        "dense_anchor_loss_weight": dense_anchor_loss_weight,
        "path_distill_loss_weight": path_distill_loss_weight,
        "soft_path_distill_loss_weight": soft_path_distill_loss_weight,
        "soft_path_temperature": soft_path_temperature,
        "soft_path_target_sigma_frames": soft_path_target_sigma_frames,
        "sequence_contrastive_loss_weight": sequence_contrastive_loss_weight,
        "anti_collapse_loss_weight": anti_collapse_loss_weight,
        "anti_collapse_covariance_weight": anti_collapse_covariance_weight,
        "temporal_order_loss_weight": temporal_order_loss_weight,
        "relative_offset_loss_weight": relative_offset_loss_weight,
        "relative_offset_bins": list(resolved_relative_offset_bins),
        "masked_reconstruction_loss_weight": masked_reconstruction_loss_weight,
        "cycle_consistency_loss_weight": cycle_consistency_loss_weight,
        "cycle_entropy_loss_weight": cycle_entropy_loss_weight,
        "cycle_monotonicity_loss_weight": cycle_monotonicity_loss_weight,
        "cycle_smoothness_loss_weight": cycle_smoothness_loss_weight,
        "hard_negative_loss_weight": hard_negative_loss_weight,
        "hard_negative_radius_frames": hard_negative_radius_frames,
        "false_destination_negative_loss_weight": false_destination_negative_loss_weight,
        "false_destination_negative_root": (
            str(false_destination_negative_root) if false_destination_negative_root is not None else None
        ),
        "false_destination_min_error_ms": false_destination_min_error_ms,
        "memory_bank_size": memory_bank_size,
        "teacher_path_root": str(teacher_path_root) if teacher_path_root is not None else None,
        "self_mined_path_root": str(self_mined_path_root) if self_mined_path_root is not None else None,
        "num_anchor_samples": num_anchor_samples,
        "teacher_min_confidence": teacher_min_confidence,
        "path_distill_local_radius_frames": path_distill_local_radius_frames,
        "path_distill_local_step_frames": path_distill_local_step_frames,
        "disable_time_stretch_for_anchors": disable_time_stretch_for_anchors,
        "eval_pool_size": eval_pool_size,
        "alignment_eval_every_n_epochs": alignment_eval_every_n_epochs,
        "track_debug_checkpoints": track_debug_checkpoints,
    }
    best_selections: dict[str, dict[str, Any]] = {}
    start_epoch = 0

    if resume_from is not None:
        start_epoch, history, best_selection, best_selections = _load_resume_state(
            checkpoint_path=resume_from,
            model=model,
            encoder=encoder,
            auxiliary_heads=auxiliary_heads,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
            current_signature=training_state_signature,
        )
    else:
        best_selection = None
    history = _ensure_history_keys(history)

    for epoch in range(start_epoch, epochs):
        epoch_start = time.perf_counter()
        gamma = gamma_scheduler.step(epoch) if gamma_scheduler is not None else start_gamma
        history["gamma"].append(gamma)

        model.train()
        train_loss = 0.0
        train_soft_dtw_loss = 0.0
        train_anchor_loss = 0.0
        train_dense_anchor_loss = 0.0
        train_path_distill_loss = 0.0
        train_soft_path_distill_loss = 0.0
        train_sequence_contrastive_loss = 0.0
        train_anti_collapse_loss = 0.0
        train_temporal_order_loss = 0.0
        train_relative_offset_loss = 0.0
        train_masked_reconstruction_loss = 0.0
        train_cycle_consistency_loss = 0.0
        train_cycle_entropy_loss = 0.0
        train_cycle_monotonicity_loss = 0.0
        train_cycle_smoothness_loss = 0.0
        train_hard_negative_loss = 0.0
        train_false_destination_negative_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            spec_a = batch["spec_a"].to(dev)
            spec_b = batch["spec_b"].to(dev)
            spec_neg = batch.get("spec_neg")
            if spec_neg is not None:
                spec_neg = spec_neg.to(dev)
            lengths_a = batch.get("lengths_a")
            lengths_b = batch.get("lengths_b")
            lengths_neg = batch.get("lengths_neg")
            negative_valid = batch.get("negative_valid")
            anchor_frames_a = batch.get("anchor_frame_indices_a")
            anchor_frames_b = batch.get("anchor_frame_indices_b")
            teacher_frames_a = batch.get("teacher_frame_indices_a")
            teacher_frames_b = batch.get("teacher_frame_indices_b")
            group_ids = batch.get("lied_id")

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(dev.type == "cuda")):
                emb_a, emb_b = model(spec_a, spec_b)
                soft_dtw_component = emb_a.new_zeros(())
                loss = emb_a.new_zeros(())
                if criterion is not None:
                    soft_dtw_component = criterion(
                        emb_a,
                        emb_b,
                        lengths_a=lengths_a,
                        lengths_b=lengths_b,
                    )
                    loss = loss + (soft_dtw_loss_weight * soft_dtw_component)
                anchor_component = emb_a.new_zeros(())
                if anchor_loss_fn is not None:
                    anchor_component = anchor_loss_fn(emb_a, emb_b, anchor_frames_a, anchor_frames_b)
                    loss = loss + (anchor_loss_weight * anchor_component)
                dense_anchor_component = emb_a.new_zeros(())
                if dense_anchor_loss_fn is not None:
                    dense_anchor_component = dense_anchor_loss_fn(emb_a, emb_b, anchor_frames_a, anchor_frames_b)
                    loss = loss + (dense_anchor_loss_weight * dense_anchor_component)
                path_distill_component = emb_a.new_zeros(())
                if path_distill_loss_fn is not None:
                    path_distill_component = path_distill_loss_fn(emb_a, emb_b, teacher_frames_a, teacher_frames_b)
                    loss = loss + (path_distill_loss_weight * path_distill_component)
                soft_path_distill_component = emb_a.new_zeros(())
                if soft_path_distill_loss_fn is not None:
                    soft_path_distill_component = soft_path_distill_loss_fn(
                        emb_a,
                        emb_b,
                        teacher_frames_a,
                        teacher_frames_b,
                        lengths_a=lengths_a,
                        lengths_b=lengths_b,
                    )
                    loss = loss + (soft_path_distill_loss_weight * soft_path_distill_component)
                sequence_component = emb_a.new_zeros(())
                if sequence_contrastive_loss_fn is not None:
                    sequence_component = sequence_contrastive_loss_fn(
                        emb_a,
                        emb_b,
                        lengths_a=lengths_a,
                        lengths_b=lengths_b,
                        group_ids=group_ids,
                    )
                    loss = loss + (sequence_contrastive_loss_weight * sequence_component)
                anti_collapse_component = emb_a.new_zeros(())
                if anti_collapse_loss_fn is not None:
                    anti_collapse_component = anti_collapse_loss_fn(
                        emb_a,
                        emb_b,
                        lengths_a=lengths_a,
                        lengths_b=lengths_b,
                    )
                    loss = loss + (anti_collapse_loss_weight * anti_collapse_component)
                temporal_order_component = emb_a.new_zeros(())
                relative_offset_component = emb_a.new_zeros(())
                masked_reconstruction_component = emb_a.new_zeros(())
                hard_negative_component = emb_a.new_zeros(())
                false_destination_negative_component = emb_a.new_zeros(())
                emb_neg = None
                if spec_neg is not None and (
                    hard_negative_loss_fn is not None
                    or temporal_order_loss_weight > 0
                    or relative_offset_loss_weight > 0
                ):
                    emb_neg = model.encoder(spec_neg)
                if auxiliary_heads is not None and emb_neg is not None:
                    pooled_a = SequenceContrastiveLoss._masked_mean_pool(emb_a, lengths_a)
                    pooled_neg = SequenceContrastiveLoss._masked_mean_pool(emb_neg, lengths_neg)
                    pair_features = auxiliary_heads.pair_features(pooled_a, pooled_neg)
                    if temporal_order_loss_weight > 0 and batch.get("temporal_order_label") is not None:
                        temporal_labels = torch.as_tensor(
                            batch["temporal_order_label"],
                            device=dev,
                            dtype=torch.long,
                        )
                        temporal_order_component = nn.functional.cross_entropy(
                            auxiliary_heads.temporal_order(pair_features),
                            temporal_labels,
                        )
                        loss = loss + (temporal_order_loss_weight * temporal_order_component)
                    if relative_offset_loss_weight > 0 and batch.get("relative_offset_label") is not None:
                        offset_labels = torch.as_tensor(
                            batch["relative_offset_label"],
                            device=dev,
                            dtype=torch.long,
                        )
                        relative_offset_component = nn.functional.cross_entropy(
                            auxiliary_heads.relative_offset(pair_features),
                            offset_labels,
                        )
                        loss = loss + (relative_offset_loss_weight * relative_offset_component)
                if auxiliary_heads is not None and reconstruction_loss_fn is not None:
                    reconstruction_target = batch.get("reconstruction_target_a")
                    reconstruction_mask = batch.get("reconstruction_mask_a")
                    if reconstruction_target is not None and reconstruction_mask is not None:
                        reconstructed = auxiliary_heads.reconstruction(emb_a)
                        masked_reconstruction_component = reconstruction_loss_fn(
                            reconstructed,
                            reconstruction_target.to(dev),
                            reconstruction_mask.to(dev),
                        )
                        loss = loss + (masked_reconstruction_loss_weight * masked_reconstruction_component)
                if hard_negative_loss_fn is not None and emb_neg is not None:
                    negative_component = hard_negative_loss_fn(
                        emb_a,
                        emb_b,
                        emb_neg,
                        lengths_a=lengths_a,
                        lengths_b=lengths_b,
                        lengths_neg=lengths_neg,
                        valid_mask=negative_valid,
                    )
                    if hard_negative_loss_weight > 0:
                        hard_negative_component = negative_component
                        loss = loss + (hard_negative_loss_weight * hard_negative_component)
                    if false_destination_negative_loss_weight > 0:
                        false_destination_negative_component = negative_component
                        loss = loss + (
                            false_destination_negative_loss_weight
                            * false_destination_negative_component
                        )
                cycle_consistency_component = emb_a.new_zeros(())
                cycle_entropy_component = emb_a.new_zeros(())
                cycle_monotonicity_component = emb_a.new_zeros(())
                cycle_smoothness_component = emb_a.new_zeros(())
                if cycle_loss_fn is not None:
                    cycle_components = cycle_loss_fn(
                        emb_a,
                        emb_b,
                        lengths_a=lengths_a,
                        lengths_b=lengths_b,
                    )
                    cycle_consistency_component = cycle_components["cycle"]
                    cycle_entropy_component = cycle_components["entropy"]
                    cycle_monotonicity_component = cycle_components["monotonicity"]
                    cycle_smoothness_component = cycle_components["smoothness"]
                    loss = loss + (cycle_consistency_loss_weight * cycle_consistency_component)
                    loss = loss + (cycle_entropy_loss_weight * cycle_entropy_component)
                    loss = loss + (cycle_monotonicity_loss_weight * cycle_monotonicity_component)
                    loss = loss + (cycle_smoothness_loss_weight * cycle_smoothness_component)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(optim_parameters, gradient_clip)
            scaler.step(optimizer)
            scaler.update()

            train_loss += float(loss.item())
            train_soft_dtw_loss += float(soft_dtw_component.item())
            train_anchor_loss += float(anchor_component.item())
            train_dense_anchor_loss += float(dense_anchor_component.item())
            train_path_distill_loss += float(path_distill_component.item())
            train_soft_path_distill_loss += float(soft_path_distill_component.item())
            train_sequence_contrastive_loss += float(sequence_component.item())
            train_anti_collapse_loss += float(anti_collapse_component.item())
            train_temporal_order_loss += float(temporal_order_component.item())
            train_relative_offset_loss += float(relative_offset_component.item())
            train_masked_reconstruction_loss += float(masked_reconstruction_component.item())
            train_cycle_consistency_loss += float(cycle_consistency_component.item())
            train_cycle_entropy_loss += float(cycle_entropy_component.item())
            train_cycle_monotonicity_loss += float(cycle_monotonicity_component.item())
            train_cycle_smoothness_loss += float(cycle_smoothness_component.item())
            train_hard_negative_loss += float(hard_negative_component.item())
            train_false_destination_negative_loss += float(false_destination_negative_component.item())
            n_batches += 1

        train_loss /= max(n_batches, 1)
        train_soft_dtw_loss /= max(n_batches, 1)
        train_anchor_loss /= max(n_batches, 1)
        train_dense_anchor_loss /= max(n_batches, 1)
        train_path_distill_loss /= max(n_batches, 1)
        train_soft_path_distill_loss /= max(n_batches, 1)
        train_sequence_contrastive_loss /= max(n_batches, 1)
        train_anti_collapse_loss /= max(n_batches, 1)
        train_temporal_order_loss /= max(n_batches, 1)
        train_relative_offset_loss /= max(n_batches, 1)
        train_masked_reconstruction_loss /= max(n_batches, 1)
        train_cycle_consistency_loss /= max(n_batches, 1)
        train_cycle_entropy_loss /= max(n_batches, 1)
        train_cycle_monotonicity_loss /= max(n_batches, 1)
        train_cycle_smoothness_loss /= max(n_batches, 1)
        train_hard_negative_loss /= max(n_batches, 1)
        train_false_destination_negative_loss /= max(n_batches, 1)
        history["train_loss"].append(train_loss)
        history["train_soft_dtw_loss"].append(train_soft_dtw_loss)
        history["train_anchor_loss"].append(train_anchor_loss)
        history["train_dense_anchor_loss"].append(train_dense_anchor_loss)
        history["train_path_distill_loss"].append(train_path_distill_loss)
        history["train_soft_path_distill_loss"].append(train_soft_path_distill_loss)
        history["train_sequence_contrastive_loss"].append(train_sequence_contrastive_loss)
        history["train_anti_collapse_loss"].append(train_anti_collapse_loss)
        history["train_temporal_order_loss"].append(train_temporal_order_loss)
        history["train_relative_offset_loss"].append(train_relative_offset_loss)
        history["train_masked_reconstruction_loss"].append(train_masked_reconstruction_loss)
        history["train_cycle_consistency_loss"].append(train_cycle_consistency_loss)
        history["train_cycle_entropy_loss"].append(train_cycle_entropy_loss)
        history["train_cycle_monotonicity_loss"].append(train_cycle_monotonicity_loss)
        history["train_cycle_smoothness_loss"].append(train_cycle_smoothness_loss)
        history["train_hard_negative_loss"].append(train_hard_negative_loss)
        history["train_false_destination_negative_loss"].append(train_false_destination_negative_loss)

        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                spec_a = batch["spec_a"].to(dev)
                spec_b = batch["spec_b"].to(dev)
                spec_neg = batch.get("spec_neg")
                if spec_neg is not None:
                    spec_neg = spec_neg.to(dev)
                lengths_a = batch.get("lengths_a")
                lengths_b = batch.get("lengths_b")
                lengths_neg = batch.get("lengths_neg")
                negative_valid = batch.get("negative_valid")
                emb_a, emb_b = model(spec_a, spec_b)
                if criterion is not None:
                    loss = criterion(emb_a, emb_b, lengths_a=lengths_a, lengths_b=lengths_b)
                else:
                    loss = emb_a.new_zeros(())
                anchor_frames_a = batch.get("anchor_frame_indices_a")
                anchor_frames_b = batch.get("anchor_frame_indices_b")
                teacher_frames_a = batch.get("teacher_frame_indices_a")
                teacher_frames_b = batch.get("teacher_frame_indices_b")
                group_ids = batch.get("lied_id")
                if anchor_loss_fn is not None:
                    loss = loss + (
                        anchor_loss_weight
                        * anchor_loss_fn(emb_a, emb_b, anchor_frames_a, anchor_frames_b)
                    )
                if dense_anchor_loss_fn is not None:
                    loss = loss + (
                        dense_anchor_loss_weight
                        * dense_anchor_loss_fn(emb_a, emb_b, anchor_frames_a, anchor_frames_b)
                    )
                if path_distill_loss_fn is not None:
                    loss = loss + (
                        path_distill_loss_weight
                        * path_distill_loss_fn(emb_a, emb_b, teacher_frames_a, teacher_frames_b)
                    )
                if soft_path_distill_loss_fn is not None:
                    loss = loss + (
                        soft_path_distill_loss_weight
                        * soft_path_distill_loss_fn(
                            emb_a,
                            emb_b,
                            teacher_frames_a,
                            teacher_frames_b,
                            lengths_a=lengths_a,
                            lengths_b=lengths_b,
                        )
                    )
                if sequence_contrastive_loss_fn is not None:
                    loss = loss + (
                        sequence_contrastive_loss_weight
                        * sequence_contrastive_loss_fn(
                            emb_a,
                            emb_b,
                            lengths_a=lengths_a,
                            lengths_b=lengths_b,
                            group_ids=group_ids,
                        )
                    )
                if anti_collapse_loss_fn is not None:
                    loss = loss + (
                        anti_collapse_loss_weight
                        * anti_collapse_loss_fn(
                            emb_a,
                            emb_b,
                            lengths_a=lengths_a,
                            lengths_b=lengths_b,
                        )
                    )
                emb_neg = None
                if spec_neg is not None and (
                    hard_negative_loss_fn is not None
                    or temporal_order_loss_weight > 0
                    or relative_offset_loss_weight > 0
                ):
                    emb_neg = model.encoder(spec_neg)
                if auxiliary_heads is not None and emb_neg is not None:
                    pooled_a = SequenceContrastiveLoss._masked_mean_pool(emb_a, lengths_a)
                    pooled_neg = SequenceContrastiveLoss._masked_mean_pool(emb_neg, lengths_neg)
                    pair_features = auxiliary_heads.pair_features(pooled_a, pooled_neg)
                    if temporal_order_loss_weight > 0 and batch.get("temporal_order_label") is not None:
                        temporal_labels = torch.as_tensor(
                            batch["temporal_order_label"],
                            device=dev,
                            dtype=torch.long,
                        )
                        loss = loss + (
                            temporal_order_loss_weight
                            * nn.functional.cross_entropy(
                                auxiliary_heads.temporal_order(pair_features),
                                temporal_labels,
                            )
                        )
                    if relative_offset_loss_weight > 0 and batch.get("relative_offset_label") is not None:
                        offset_labels = torch.as_tensor(
                            batch["relative_offset_label"],
                            device=dev,
                            dtype=torch.long,
                        )
                        loss = loss + (
                            relative_offset_loss_weight
                            * nn.functional.cross_entropy(
                                auxiliary_heads.relative_offset(pair_features),
                                offset_labels,
                            )
                        )
                if auxiliary_heads is not None and reconstruction_loss_fn is not None:
                    reconstruction_target = batch.get("reconstruction_target_a")
                    reconstruction_mask = batch.get("reconstruction_mask_a")
                    if reconstruction_target is not None and reconstruction_mask is not None:
                        loss = loss + (
                            masked_reconstruction_loss_weight
                            * reconstruction_loss_fn(
                                auxiliary_heads.reconstruction(emb_a),
                                reconstruction_target.to(dev),
                                reconstruction_mask.to(dev),
                            )
                        )
                if hard_negative_loss_fn is not None and emb_neg is not None:
                    negative_loss = hard_negative_loss_fn(
                        emb_a,
                        emb_b,
                        emb_neg,
                        lengths_a=lengths_a,
                        lengths_b=lengths_b,
                        lengths_neg=lengths_neg,
                        valid_mask=negative_valid,
                    )
                    loss = loss + (hard_negative_loss_weight * negative_loss)
                    loss = loss + (false_destination_negative_loss_weight * negative_loss)
                if cycle_loss_fn is not None:
                    cycle_components = cycle_loss_fn(
                        emb_a,
                        emb_b,
                        lengths_a=lengths_a,
                        lengths_b=lengths_b,
                    )
                    loss = loss + (cycle_consistency_loss_weight * cycle_components["cycle"])
                    loss = loss + (cycle_entropy_loss_weight * cycle_components["entropy"])
                    loss = loss + (cycle_monotonicity_loss_weight * cycle_components["monotonicity"])
                    loss = loss + (cycle_smoothness_loss_weight * cycle_components["smoothness"])
                val_loss += float(loss.item())
                n_val_batches += 1

        val_loss /= max(n_val_batches, 1)
        history["val_loss"].append(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(current_lr)
        lr_scheduler.step()

        run_alignment_eval = (
            alignment_eval_every_n_epochs > 0
            and ((epoch + 1) % alignment_eval_every_n_epochs == 0 or epoch + 1 == epochs)
        )
        if dry_run or not run_alignment_eval:
            debug_metrics = {"mae": np.nan, "ar_50ms": np.nan, "ar_100ms": np.nan, "ar_200ms": np.nan, "pairs": 0.0}
            val_metrics = {"mae": np.nan, "ar_50ms": np.nan, "ar_100ms": np.nan, "ar_200ms": np.nan, "pairs": 0.0}
        else:
            debug_metrics = _evaluate_swd_pairs(
                debug_pairs,
                dataset=swd,
                sr=sr,
                hop_length=hop_length,
                encoder=model.encoder,
                device=device,
                cache_root=resolved_cache_root,
                deep_decode=deep_decode,
                band_radius_frames=band_radius_frames,
                pool_size=eval_pool_size,
            )
            val_metrics = _evaluate_swd_pairs(
                val_pairs,
                dataset=swd,
                sr=sr,
                hop_length=hop_length,
                encoder=model.encoder,
                device=device,
                cache_root=resolved_cache_root,
                deep_decode=deep_decode,
                band_radius_frames=band_radius_frames,
                pool_size=eval_pool_size,
            )

        history["debug_mae"].append(float(debug_metrics["mae"]))
        history["debug_ar50"].append(float(debug_metrics["ar_50ms"]))
        history["debug_ar100"].append(float(debug_metrics["ar_100ms"]))
        history["debug_ar200"].append(float(debug_metrics["ar_200ms"]))
        history["val_mae"].append(float(val_metrics["mae"]))
        history["val_ar50"].append(float(val_metrics["ar_50ms"]))
        history["val_ar100"].append(float(val_metrics["ar_100ms"]))
        history["val_ar200"].append(float(val_metrics["ar_200ms"]))

        selection_snapshots = {
            metric: _build_selection_snapshot(
                selection_metric=metric,
                val_loss=val_loss,
                debug_metrics=debug_metrics,
                val_metrics=val_metrics,
                epoch=epoch,
            )
            for metric in _tracked_selection_metrics(
                selection_metric,
                track_debug_checkpoints=track_debug_checkpoints,
            )
        }
        current_selection = selection_snapshots[selection_metric]
        selection_key = tuple(float(value) for value in current_selection["selection_key"])
        history["selection_primary"].append(float(selection_key[0]))
        history["selection_secondary"].append(float(selection_key[1]) if len(selection_key) > 1 else np.nan)

        epoch_time = time.perf_counter() - epoch_start
        history["epoch_time"].append(epoch_time)

        logger.info(
            "Epoch %s/%s | Train: %.4f | SoftDTW: %.4f | Anchor: %.4f | "
            "Dense: %.4f | Distill: %.4f | SoftPath: %.4f | Seq: %.4f | AntiCollapse: %.4f | "
            "Order: %.4f | Offset: %.4f | Recon: %.4f | Cycle: %.4f/%.4f/%.4f/%.4f | HardNeg: %.4f | "
            "FalseDest: %.4f | Val: %.4f | Debug MAE: %.4f | Debug AR@50: %.4f | "
            "Debug AR@100: %.4f | Val MAE: %.4f | Val AR@50: %.4f | Val AR@100: %.4f | "
            "gamma: %.4f | LR: %.2e | Time: %.1fs",
            epoch + 1,
            epochs,
            train_loss,
            train_soft_dtw_loss,
            train_anchor_loss,
            train_dense_anchor_loss,
            train_path_distill_loss,
            train_soft_path_distill_loss,
            train_sequence_contrastive_loss,
            train_anti_collapse_loss,
            train_temporal_order_loss,
            train_relative_offset_loss,
            train_masked_reconstruction_loss,
            train_cycle_consistency_loss,
            train_cycle_entropy_loss,
            train_cycle_monotonicity_loss,
            train_cycle_smoothness_loss,
            train_hard_negative_loss,
            train_false_destination_negative_loss,
            val_loss,
            debug_metrics["mae"],
            debug_metrics["ar_50ms"],
            debug_metrics["ar_100ms"],
            val_metrics["mae"],
            val_metrics["ar_50ms"],
            val_metrics["ar_100ms"],
            gamma,
            current_lr,
            epoch_time,
        )

        checkpoint_payload = _build_checkpoint_payload(
            epoch=epoch,
            model=model,
            encoder=encoder,
            auxiliary_heads=auxiliary_heads,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
            val_loss=val_loss,
            history=history,
            best_selection=best_selections.get(selection_metric),
            best_selections=best_selections,
            current_selection=current_selection,
            selection_snapshots=selection_snapshots,
            training_state_signature=training_state_signature,
            config={
                "swd_path": swd_path,
                "embed_dim": embed_dim,
                "n_freq_bins": n_freq_bins,
                "sr": sr,
                "hop_length": hop_length,
                "num_conv_channels": num_conv_channels,
                "gru_hidden_size": gru_hidden_size,
                "num_gru_layers": num_gru_layers,
                "dropout": dropout,
                "temporal_attention_heads": temporal_attention_heads,
                "max_length_sec": max_length_sec,
                "segment_sampling": segment_sampling,
                "samples_per_epoch": resolved_samples_per_epoch,
                "start_gamma": start_gamma,
                "end_gamma": end_gamma,
                "soft_dtw_loss_weight": soft_dtw_loss_weight,
                "normalize_loss": normalize_loss,
                "dist_func": dist_func,
                "gradient_clip": gradient_clip,
                "weight_decay": weight_decay,
                "augment": augment,
                "augmentor_kwargs": augmentor_kwargs,
                "val_split": val_split,
                "cache_spectrograms": cache_spectrograms,
                "cache_root": resolved_cache_root,
                "selection_metric": selection_metric,
                "debug_subset_lieder": list(debug_lied_ids),
                "deep_decode": deep_decode,
                "band_radius_frames": band_radius_frames,
                "anchor_loss_weight": anchor_loss_weight,
                "dense_anchor_loss_weight": dense_anchor_loss_weight,
                "path_distill_loss_weight": path_distill_loss_weight,
                "soft_path_distill_loss_weight": soft_path_distill_loss_weight,
                "soft_path_temperature": soft_path_temperature,
                "soft_path_target_sigma_frames": soft_path_target_sigma_frames,
                "sequence_contrastive_loss_weight": sequence_contrastive_loss_weight,
                "anti_collapse_loss_weight": anti_collapse_loss_weight,
                "anti_collapse_covariance_weight": anti_collapse_covariance_weight,
                "temporal_order_loss_weight": temporal_order_loss_weight,
                "relative_offset_loss_weight": relative_offset_loss_weight,
                "relative_offset_bins": list(resolved_relative_offset_bins),
                "masked_reconstruction_loss_weight": masked_reconstruction_loss_weight,
                "cycle_consistency_loss_weight": cycle_consistency_loss_weight,
                "cycle_entropy_loss_weight": cycle_entropy_loss_weight,
                "cycle_monotonicity_loss_weight": cycle_monotonicity_loss_weight,
                "cycle_smoothness_loss_weight": cycle_smoothness_loss_weight,
                "hard_negative_loss_weight": hard_negative_loss_weight,
                "hard_negative_radius_frames": hard_negative_radius_frames,
                "false_destination_negative_loss_weight": false_destination_negative_loss_weight,
                "false_destination_negative_root": false_destination_negative_root,
                "false_destination_min_error_ms": false_destination_min_error_ms,
                "memory_bank_size": memory_bank_size,
                "teacher_path_root": teacher_path_root,
                "self_mined_path_root": self_mined_path_root,
                "num_anchor_samples": num_anchor_samples,
                "teacher_min_confidence": teacher_min_confidence,
                "path_distill_local_radius_frames": path_distill_local_radius_frames,
                "path_distill_local_step_frames": path_distill_local_step_frames,
                "disable_time_stretch_for_anchors": disable_time_stretch_for_anchors,
                "eval_pool_size": eval_pool_size,
                "alignment_eval_every_n_epochs": alignment_eval_every_n_epochs,
                "track_debug_checkpoints": track_debug_checkpoints,
                "anchor_temperature": anchor_temperature,
                "anchor_min_anchor_gap": anchor_min_anchor_gap,
            },
        )

        for tracked_metric, snapshot in selection_snapshots.items():
            best_snapshot = best_selections.get(tracked_metric)
            if best_snapshot is None or _is_better_selection(snapshot["selection_key"], best_snapshot["selection_key"]):
                best_selections[tracked_metric] = snapshot
                metric_payload = _build_checkpoint_payload(
                    epoch=epoch,
                    model=model,
                    encoder=encoder,
                    auxiliary_heads=auxiliary_heads,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    scaler=scaler,
                    val_loss=val_loss,
                    history=history,
                    best_selection=snapshot,
                    best_selections=best_selections,
                    current_selection=current_selection,
                    selection_snapshots=selection_snapshots,
                    training_state_signature=training_state_signature,
                    config={
                        "swd_path": swd_path,
                        "embed_dim": embed_dim,
                        "n_freq_bins": n_freq_bins,
                        "sr": sr,
                        "hop_length": hop_length,
                        "num_conv_channels": num_conv_channels,
                        "gru_hidden_size": gru_hidden_size,
                        "num_gru_layers": num_gru_layers,
                        "dropout": dropout,
                        "temporal_attention_heads": temporal_attention_heads,
                        "max_length_sec": max_length_sec,
                        "segment_sampling": segment_sampling,
                        "samples_per_epoch": resolved_samples_per_epoch,
                        "start_gamma": start_gamma,
                        "end_gamma": end_gamma,
                        "soft_dtw_loss_weight": soft_dtw_loss_weight,
                        "normalize_loss": normalize_loss,
                        "dist_func": dist_func,
                        "gradient_clip": gradient_clip,
                        "weight_decay": weight_decay,
                        "augment": augment,
                        "augmentor_kwargs": augmentor_kwargs,
                        "val_split": val_split,
                        "cache_spectrograms": cache_spectrograms,
                        "cache_root": resolved_cache_root,
                        "selection_metric": selection_metric,
                        "debug_subset_lieder": list(debug_lied_ids),
                        "deep_decode": deep_decode,
                        "band_radius_frames": band_radius_frames,
                        "anchor_loss_weight": anchor_loss_weight,
                        "dense_anchor_loss_weight": dense_anchor_loss_weight,
                        "path_distill_loss_weight": path_distill_loss_weight,
                        "soft_path_distill_loss_weight": soft_path_distill_loss_weight,
                        "soft_path_temperature": soft_path_temperature,
                        "soft_path_target_sigma_frames": soft_path_target_sigma_frames,
                        "sequence_contrastive_loss_weight": sequence_contrastive_loss_weight,
                        "anti_collapse_loss_weight": anti_collapse_loss_weight,
                        "anti_collapse_covariance_weight": anti_collapse_covariance_weight,
                        "temporal_order_loss_weight": temporal_order_loss_weight,
                        "relative_offset_loss_weight": relative_offset_loss_weight,
                        "relative_offset_bins": list(resolved_relative_offset_bins),
                        "masked_reconstruction_loss_weight": masked_reconstruction_loss_weight,
                        "cycle_consistency_loss_weight": cycle_consistency_loss_weight,
                        "cycle_entropy_loss_weight": cycle_entropy_loss_weight,
                        "cycle_monotonicity_loss_weight": cycle_monotonicity_loss_weight,
                        "cycle_smoothness_loss_weight": cycle_smoothness_loss_weight,
                        "hard_negative_loss_weight": hard_negative_loss_weight,
                        "hard_negative_radius_frames": hard_negative_radius_frames,
                        "false_destination_negative_loss_weight": false_destination_negative_loss_weight,
                        "false_destination_negative_root": false_destination_negative_root,
                        "false_destination_min_error_ms": false_destination_min_error_ms,
                        "memory_bank_size": memory_bank_size,
                        "teacher_path_root": teacher_path_root,
                        "self_mined_path_root": self_mined_path_root,
                        "num_anchor_samples": num_anchor_samples,
                        "teacher_min_confidence": teacher_min_confidence,
                        "path_distill_local_radius_frames": path_distill_local_radius_frames,
                        "path_distill_local_step_frames": path_distill_local_step_frames,
                        "disable_time_stretch_for_anchors": disable_time_stretch_for_anchors,
                        "eval_pool_size": eval_pool_size,
                        "alignment_eval_every_n_epochs": alignment_eval_every_n_epochs,
                        "track_debug_checkpoints": track_debug_checkpoints,
                        "anchor_temperature": anchor_temperature,
                        "anchor_min_anchor_gap": anchor_min_anchor_gap,
                    },
                )
                if tracked_metric == selection_metric:
                    torch.save(metric_payload, out / "best_model.pt")
                metric_filename = BEST_CHECKPOINT_FILENAMES.get(tracked_metric)
                if metric_filename is not None:
                    torch.save(metric_payload, out / metric_filename)
                logger.info(
                    "  -> Saved best %s model (%s)",
                    tracked_metric,
                    ", ".join(f"{value:.4f}" for value in snapshot["selection_key"]),
                )

        if save_every_n_epochs > 0 and (epoch + 1) % save_every_n_epochs == 0:
            torch.save(checkpoint_payload, out / f"checkpoint_epoch{epoch + 1}.pt")

    torch.save(model.state_dict(), out / "final_model.pt")
    with open(out / "training_history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    logger.info("Training complete. Best selection: %s", best_selections.get(selection_metric))
    logger.info("Checkpoints saved to %s", out)
    return history


def _empty_history() -> dict[str, list[float]]:
    return {
        "train_loss": [],
        "train_soft_dtw_loss": [],
        "train_anchor_loss": [],
        "train_dense_anchor_loss": [],
        "train_path_distill_loss": [],
        "train_soft_path_distill_loss": [],
        "train_sequence_contrastive_loss": [],
        "train_anti_collapse_loss": [],
        "train_temporal_order_loss": [],
        "train_relative_offset_loss": [],
        "train_masked_reconstruction_loss": [],
        "train_cycle_consistency_loss": [],
        "train_cycle_entropy_loss": [],
        "train_cycle_monotonicity_loss": [],
        "train_cycle_smoothness_loss": [],
        "train_hard_negative_loss": [],
        "train_false_destination_negative_loss": [],
        "val_loss": [],
        "debug_mae": [],
        "debug_ar50": [],
        "debug_ar100": [],
        "debug_ar200": [],
        "val_mae": [],
        "val_ar50": [],
        "val_ar100": [],
        "val_ar200": [],
        "gamma": [],
        "lr": [],
        "selection_primary": [],
        "selection_secondary": [],
        "epoch_time": [],
    }


def _ensure_history_keys(history: dict[str, list[float]]) -> dict[str, list[float]]:
    """Backfill new history keys when warm-starting older checkpoints."""
    defaults = _empty_history()
    for key, value in defaults.items():
        history.setdefault(key, list(value))
    return history


def _build_checkpoint_payload(
    *,
    epoch: int,
    model: DeepAlignModel,
    encoder: CRNNEncoder,
    optimizer: AdamW,
    lr_scheduler: CosineAnnealingLR,
    scaler: torch.amp.GradScaler,
    val_loss: float,
    history: dict[str, list[float]],
    best_selection: dict[str, Any] | None,
    best_selections: dict[str, dict[str, Any]],
    current_selection: dict[str, Any],
    selection_snapshots: dict[str, dict[str, Any]],
    training_state_signature: dict[str, Any],
    config: dict[str, Any],
    auxiliary_heads: nn.Module | None = None,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "encoder_state_dict": encoder.state_dict(),
        "auxiliary_state_dict": auxiliary_heads.state_dict() if auxiliary_heads is not None else None,
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "val_loss": float(val_loss),
        "history": history,
        "best_selection": best_selection,
        "best_selections": best_selections,
        "current_selection": current_selection,
        "selection_snapshots": selection_snapshots,
        "training_state_signature": training_state_signature,
        "config": config,
    }


def _load_resume_state(
    *,
    checkpoint_path: str | Path,
    model: DeepAlignModel,
    encoder: CRNNEncoder,
    optimizer: AdamW,
    lr_scheduler: CosineAnnealingLR,
    scaler: torch.amp.GradScaler,
    current_signature: dict[str, Any],
    auxiliary_heads: nn.Module | None = None,
) -> tuple[int, dict[str, list[float]], dict[str, Any] | None, dict[str, dict[str, Any]]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "encoder_state_dict" in checkpoint:
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise ValueError(f"Checkpoint at {checkpoint_path} does not contain model weights.")

    if auxiliary_heads is not None and checkpoint.get("auxiliary_state_dict") is not None:
        auxiliary_heads.load_state_dict(checkpoint["auxiliary_state_dict"])

    saved_signature = checkpoint.get("training_state_signature")
    if _normalise_training_state_signature(saved_signature) == _normalise_training_state_signature(
        current_signature
    ):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "lr_scheduler_state_dict" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        history = checkpoint.get("history", _empty_history())
        best_selection = checkpoint.get("best_selection")
        best_selections = checkpoint.get("best_selections") or {}
        if not best_selections and best_selection is not None:
            metric = str(best_selection.get("metric", current_signature.get("selection_metric", "debug_mae")))
            best_selections = {metric: best_selection}
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        logger.info("Resumed full training state from %s at epoch %s", checkpoint_path, start_epoch)
        return start_epoch, history, best_selection, best_selections

    logger.info(
        "Warm-starting weights from %s because the training signature changed.",
        checkpoint_path,
    )
    return 0, _empty_history(), None, {}


def _normalise_training_state_signature(signature: Any) -> dict[str, Any] | None:
    """Preserve resume compatibility with checkpoints saved before frontend keys existed."""
    if not isinstance(signature, dict):
        return None
    normalised = dict(signature)
    normalised.setdefault("sr", 22050)
    normalised.setdefault("hop_length", 220)
    normalised.setdefault("dense_anchor_loss_weight", 0.0)
    normalised.setdefault("soft_dtw_loss_weight", 1.0)
    normalised.setdefault("path_distill_loss_weight", 0.0)
    normalised.setdefault("soft_path_distill_loss_weight", 0.0)
    normalised.setdefault("soft_path_temperature", 0.05)
    normalised.setdefault("soft_path_target_sigma_frames", 2.0)
    normalised.setdefault("sequence_contrastive_loss_weight", 0.0)
    normalised.setdefault("anti_collapse_loss_weight", 0.0)
    normalised.setdefault("anti_collapse_covariance_weight", 0.01)
    normalised.setdefault("temporal_order_loss_weight", 0.0)
    normalised.setdefault("relative_offset_loss_weight", 0.0)
    normalised.setdefault("relative_offset_bins", [8, 24, 64, 128])
    normalised.setdefault("masked_reconstruction_loss_weight", 0.0)
    normalised.setdefault("cycle_consistency_loss_weight", 0.0)
    normalised.setdefault("cycle_entropy_loss_weight", 0.0)
    normalised.setdefault("cycle_monotonicity_loss_weight", 0.0)
    normalised.setdefault("cycle_smoothness_loss_weight", 0.0)
    normalised.setdefault("hard_negative_loss_weight", 0.0)
    normalised.setdefault("hard_negative_radius_frames", 96)
    normalised.setdefault("false_destination_negative_loss_weight", 0.0)
    normalised.setdefault("false_destination_negative_root", None)
    normalised.setdefault("false_destination_min_error_ms", 500.0)
    normalised.setdefault("memory_bank_size", 0)
    normalised.setdefault("teacher_path_root", None)
    normalised.setdefault("self_mined_path_root", None)
    normalised.setdefault("num_anchor_samples", 64)
    normalised.setdefault("teacher_min_confidence", 0.0)
    normalised.setdefault("path_distill_local_radius_frames", 12)
    normalised.setdefault("path_distill_local_step_frames", 3)
    normalised.setdefault("disable_time_stretch_for_anchors", True)
    normalised.setdefault("eval_pool_size", 2)
    normalised.setdefault("alignment_eval_every_n_epochs", 1)
    normalised.setdefault("track_debug_checkpoints", True)
    return normalised


def _evaluate_swd_pairs(
    pairs: Sequence[Any],
    *,
    dataset: Any | None,
    sr: int,
    hop_length: int,
    encoder: CRNNEncoder,
    device: str | None,
    cache_root: str | None,
    deep_decode: str,
    band_radius_frames: int | None,
    pool_size: int = 2,
) -> dict[str, float]:
    if not pairs:
        return {"mae": np.nan, "ar_50ms": np.nan, "ar_100ms": np.nan, "ar_200ms": np.nan, "pairs": 0.0}

    from dis_alignment.evaluation.swd import evaluate_pair

    rows: list[dict[str, Any]] = []
    for pair in pairs:
        pair_rows = evaluate_pair(
            pair,
            dataset=dataset,
            methods=["deepalign"],
            encoder=encoder,
            sr=sr,
            deep_hop=hop_length,
            device=device,
            cache_root=cache_root,
            deep_decode=deep_decode,
            band_radius_frames=band_radius_frames,
            pool_size=pool_size,
        )
        rows.extend(pair_rows)

    if not rows:
        return {"mae": np.nan, "ar_50ms": np.nan, "ar_100ms": np.nan, "ar_200ms": np.nan, "pairs": 0.0}

    mae_values = [float(row["mae"]) for row in rows]
    ar_values = [float(row["ar_50ms"]) for row in rows]
    ar100_values = [float(row["ar_100ms"]) for row in rows]
    ar200_values = [float(row["ar_200ms"]) for row in rows]
    return {
        "mae": float(np.mean(mae_values)),
        "ar_50ms": float(np.mean(ar_values)),
        "ar_100ms": float(np.mean(ar100_values)),
        "ar_200ms": float(np.mean(ar200_values)),
        "pairs": float(len(rows)),
    }


def _tracked_selection_metrics(
    selection_metric: str,
    *,
    track_debug_checkpoints: bool = True,
) -> tuple[str, ...]:
    ordered = list(TRACKED_SELECTION_METRICS) if track_debug_checkpoints else []
    if selection_metric not in ordered:
        ordered.append(selection_metric)
    return tuple(ordered)


def _is_better_selection(candidate: Sequence[float], incumbent: Sequence[float]) -> bool:
    return tuple(float(value) for value in candidate) < tuple(float(value) for value in incumbent)


def _build_selection_snapshot(
    *,
    selection_metric: str,
    val_loss: float,
    debug_metrics: dict[str, float],
    val_metrics: dict[str, float],
    epoch: int,
) -> dict[str, Any]:
    selection_key = _build_selection_key(
        selection_metric=selection_metric,
        val_loss=val_loss,
        debug_metrics=debug_metrics,
        val_metrics=val_metrics,
    )
    return {
        "metric": selection_metric,
        "selection_key": list(selection_key),
        "debug_metrics": debug_metrics,
        "val_metrics": val_metrics,
        "val_loss": float(val_loss),
        "epoch": int(epoch),
    }


def _build_selection_key(
    *,
    selection_metric: str,
    val_loss: float,
    debug_metrics: dict[str, float],
    val_metrics: dict[str, float],
) -> tuple[float, ...]:
    if selection_metric == "debug_mae":
        return (
            _safe_metric(debug_metrics["mae"]),
            _safe_desc_metric(debug_metrics["ar_50ms"]),
            _safe_desc_metric(debug_metrics["ar_100ms"]),
        )
    if selection_metric == "debug_ar50":
        return (
            _safe_desc_metric(debug_metrics["ar_50ms"]),
            _safe_metric(debug_metrics["mae"]),
            _safe_desc_metric(debug_metrics["ar_100ms"]),
        )
    if selection_metric == "debug_balanced":
        balanced_components = np.array(
            [
                float(debug_metrics["ar_50ms"]),
                float(debug_metrics["ar_100ms"]),
                float(debug_metrics["ar_200ms"]),
            ],
            dtype=float,
        )
        finite_components = balanced_components[np.isfinite(balanced_components)]
        balanced_score = float(np.mean(finite_components)) if finite_components.size else float("nan")
        return (
            _safe_desc_metric(float(balanced_score)),
            _safe_metric(debug_metrics["mae"]),
            _safe_desc_metric(debug_metrics["ar_50ms"]),
        )
    if selection_metric == "val_mae":
        return (
            _safe_metric(val_metrics["mae"]),
            _safe_desc_metric(val_metrics["ar_50ms"]),
            _safe_desc_metric(val_metrics["ar_100ms"]),
        )
    return (_safe_metric(val_loss),)


def _safe_metric(value: float) -> float:
    if value is None or not np.isfinite(value):
        return float("inf")
    return float(value)


def _safe_desc_metric(value: float) -> float:
    if value is None or not np.isfinite(value):
        return float("inf")
    return -float(value)


def _load_training_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}

    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise TypeError(f"Training config at {path} must be a mapping")
    return config


def _is_headline_claim_config(config: dict[str, Any]) -> bool:
    claim_cfg = config.get("claim", {})
    if not isinstance(claim_cfg, dict):
        return False
    claim_name = str(claim_cfg.get("name", "")).strip().lower()
    return bool(claim_cfg.get("headline", False)) or claim_name in HEADLINE_CLAIM_NAMES


def _validate_headline_claim_config(config: dict[str, Any], *, path: str | None = None) -> None:
    """
    Ensure the headline initial-claim config cannot silently use off-claim help.

    The headline route is audio-only learned DeepAlign with unconstrained DTW.
    Measure-anchor sampling/losses and guided decoders stay available for
    diagnostics, but they must not be mixed into this config.
    """
    dataset_cfg = config.get("dataset", {})
    training_cfg = config.get("training", {})
    evaluation_cfg = config.get("evaluation", {})
    errors: list[str] = []

    strict_sampling_modes = {"self_audio", "self_audio_ordered", "same_lied_pair", "self_mined_path"}
    if dataset_cfg.get("segment_sampling") not in strict_sampling_modes:
        errors.append(
            "dataset.segment_sampling must be one of "
            "{'self_audio', 'self_audio_ordered', 'same_lied_pair', 'self_mined_path'}"
        )

    for key in ("anchor_loss_weight", "dense_anchor_loss_weight"):
        if float(training_cfg.get(key, 0.0) or 0.0) > 0.0:
            errors.append(f"training.{key} must be 0 for the headline claim")

    teacher_root = str(training_cfg.get("teacher_path_root", "") or "").lower()
    if teacher_root:
        errors.append("training.teacher_path_root must be empty for the strict headline claim")
    if dataset_cfg.get("segment_sampling") == "self_mined_path" and not training_cfg.get("self_mined_path_root"):
        errors.append("training.self_mined_path_root is required for self_mined_path sampling")
    selection_metric = str(training_cfg.get("selection_metric", "val_loss"))
    if selection_metric != "val_loss":
        errors.append("training.selection_metric must be 'val_loss' for strict headline checkpoint selection")
    if bool(training_cfg.get("track_debug_checkpoints", False)):
        errors.append("training.track_debug_checkpoints must be false for strict headline checkpoint selection")

    if evaluation_cfg.get("deep_decode", "unconstrained") != "unconstrained":
        errors.append("evaluation.deep_decode must be 'unconstrained'")
    if int(evaluation_cfg.get("pool_size", 1) or 1) != 1:
        errors.append("evaluation.pool_size must be 1")
    if evaluation_cfg.get("band_radius_frames") is not None:
        errors.append("evaluation.band_radius_frames must be null")

    if errors:
        location = f" at {path}" if path else ""
        raise ValueError(
            "Headline claim config is not initial-claim eligible"
            f"{location}: "
            + "; ".join(errors)
        )


def _resolve_option(cli_value: Any, config_value: Any, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default


def _split_swd_pairs(
    pairs: list[Any],
    *,
    val_split: float,
    random_seed: int = 42,
    heldout_lieder: Sequence[str] | None = None,
) -> tuple[list[Any], list[Any]]:
    """Split SWD pairs at the lied level to avoid shared-recording leakage."""
    if not 0.0 < val_split < 1.0:
        raise ValueError("val_split must be between 0 and 1.")
    if len(pairs) < 2:
        raise ValueError("Need at least two SWD pairs to build train/validation splits.")

    ordered_pairs = sorted(pairs, key=lambda pair: getattr(pair, "pair_id", str(pair)))
    by_lied: dict[str, list[Any]] = {}
    for pair in ordered_pairs:
        lied_id = str(getattr(pair, "lied_id", "")).strip().upper()
        if not lied_id:
            raise ValueError("Every SWD pair must expose a lied_id for leakage-safe splitting.")
        by_lied.setdefault(lied_id, []).append(pair)

    if len(by_lied) < 2:
        raise ValueError("Need at least two lieder to build leakage-safe train/validation splits.")

    ordered_lieder = sorted(by_lied)
    requested_heldout = {
        str(lied_id).strip().upper()
        for lied_id in heldout_lieder or ()
        if str(lied_id).strip()
    }
    val_lied_ids = {lied_id for lied_id in ordered_lieder if lied_id in requested_heldout}

    candidate_lieder = [lied_id for lied_id in ordered_lieder if lied_id not in val_lied_ids]
    rng = np.random.default_rng(random_seed)
    shuffled_candidates = list(candidate_lieder)
    rng.shuffle(shuffled_candidates)

    target_val_lieder = max(1, int(len(ordered_lieder) * val_split))
    extra_needed = max(0, target_val_lieder - len(val_lied_ids))
    max_extra = max(0, len(candidate_lieder) - 1)
    val_lied_ids.update(shuffled_candidates[: min(extra_needed, max_extra)])

    if not val_lied_ids:
        val_lied_ids.add(shuffled_candidates[0])
    if len(val_lied_ids) == len(ordered_lieder):
        raise ValueError(
            "The held-out debug/validation lieder cover the whole dataset; "
            "at least one lied must remain for training."
        )

    train_pairs = [
        pair for pair in ordered_pairs if str(pair.lied_id).strip().upper() not in val_lied_ids
    ]
    val_pairs = [
        pair for pair in ordered_pairs if str(pair.lied_id).strip().upper() in val_lied_ids
    ]
    if not train_pairs or not val_pairs:
        raise ValueError("Leakage-safe split produced an empty train or validation partition.")
    return train_pairs, val_pairs


def _build_training_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_training_config(args.config)
    if _is_headline_claim_config(config):
        _validate_headline_claim_config(config, path=args.config)
    dataset_cfg = config.get("dataset", {})
    audio_cfg = config.get("audio", {})
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    soft_dtw_cfg = config.get("soft_dtw", {})
    aug_cfg = config.get("augmentation", {})
    output_cfg = config.get("output", {})
    evaluation_cfg = config.get("evaluation", {})

    augment_enabled = False if args.no_augment else _resolve_option(None, aug_cfg.get("enabled"), True)
    augmentor_kwargs = (
        {
            "time_stretch_range": tuple(aug_cfg["time_stretch_range"]),
            "pitch_shift_range": tuple(aug_cfg["pitch_shift_range"]),
            "noise_snr_range": tuple(aug_cfg["noise_snr_range"]),
            "prob": aug_cfg["augment_prob"],
            "sr": audio_cfg.get("sr", 22050),
        }
        if augment_enabled and aug_cfg
        else {}
    )

    return {
        "swd_path": _resolve_option(args.swd_path, dataset_cfg.get("swd_path"), None),
        "output_dir": _resolve_option(args.output_dir, output_cfg.get("checkpoint_dir"), "checkpoints"),
        "epochs": _resolve_option(args.epochs, training_cfg.get("epochs"), 50),
        "batch_size": _resolve_option(args.batch_size, training_cfg.get("batch_size"), 4),
        "lr": _resolve_option(args.lr, training_cfg.get("lr"), 1e-3),
        "embed_dim": _resolve_option(args.embed_dim, model_cfg.get("embed_dim"), 64),
        "sr": _resolve_option(args.sr, audio_cfg.get("sr"), 22050),
        "hop_length": _resolve_option(args.hop_length, audio_cfg.get("hop_length"), 220),
        "n_freq_bins": _resolve_option(args.n_freq_bins, audio_cfg.get("n_bins"), 84),
        "num_conv_channels": _resolve_option(args.conv_channels, model_cfg.get("conv_channels"), None),
        "gru_hidden_size": _resolve_option(args.gru_hidden_size, model_cfg.get("gru_hidden_size"), 128),
        "num_gru_layers": _resolve_option(args.num_gru_layers, model_cfg.get("num_gru_layers"), 2),
        "dropout": _resolve_option(args.dropout, model_cfg.get("dropout"), 0.1),
        "temporal_attention_heads": _resolve_option(
            args.temporal_attention_heads,
            model_cfg.get("temporal_attention_heads"),
            0,
        ),
        "max_length_sec": _resolve_option(args.max_length_sec, dataset_cfg.get("max_length_sec"), 30.0),
        "segment_sampling": _resolve_option(
            args.segment_sampling,
            dataset_cfg.get("segment_sampling"),
            "aligned_measures",
        ),
        "samples_per_epoch": _resolve_option(
            args.samples_per_epoch,
            dataset_cfg.get("samples_per_epoch"),
            None,
        ),
        "start_gamma": _resolve_option(args.start_gamma, soft_dtw_cfg.get("start_gamma"), 1.0),
        "end_gamma": _resolve_option(args.end_gamma, soft_dtw_cfg.get("end_gamma"), 0.01),
        "soft_dtw_loss_weight": _resolve_option(
            args.soft_dtw_loss_weight,
            soft_dtw_cfg.get("loss_weight"),
            1.0,
        ),
        "normalize_loss": _resolve_option(args.normalize_loss, soft_dtw_cfg.get("normalize"), True),
        "dist_func": _resolve_option(args.dist_func, soft_dtw_cfg.get("dist_func"), "sqeuclidean"),
        "gradient_clip": _resolve_option(args.gradient_clip, training_cfg.get("gradient_clip"), 1.0),
        "weight_decay": _resolve_option(args.weight_decay, training_cfg.get("weight_decay"), 1e-4),
        "augment": augment_enabled,
        "augmentor_kwargs": augmentor_kwargs,
        "device": args.device,
        "val_split": _resolve_option(args.val_split, dataset_cfg.get("val_split"), 0.2),
        "save_every_n_epochs": _resolve_option(
            args.save_every_n_epochs,
            output_cfg.get("save_every_n_epochs"),
            10,
        ),
        "cache_spectrograms": _resolve_option(
            args.cache_spectrograms,
            dataset_cfg.get("cache_spectrograms"),
            False,
        ),
        "cache_root": _resolve_option(args.cache_root, dataset_cfg.get("cache_root"), None),
        "resume_from": _resolve_option(args.resume_from, training_cfg.get("resume_from"), None),
        "selection_metric": _resolve_option(
            args.selection_metric,
            training_cfg.get("selection_metric"),
            "debug_ar50",
        ),
        "debug_subset_lieder": dataset_cfg.get("debug_subset_lieder"),
        "deep_decode": _resolve_option(args.deep_decode, evaluation_cfg.get("deep_decode"), "unconstrained"),
        "band_radius_frames": _resolve_option(
            args.band_radius_frames,
            evaluation_cfg.get("band_radius_frames"),
            None,
        ),
        "anchor_loss_weight": _resolve_option(
            args.anchor_loss_weight,
            training_cfg.get("anchor_loss_weight"),
            0.0,
        ),
        "dense_anchor_loss_weight": _resolve_option(
            args.dense_anchor_loss_weight,
            training_cfg.get("dense_anchor_loss_weight"),
            0.0,
        ),
        "path_distill_loss_weight": _resolve_option(
            args.path_distill_loss_weight,
            training_cfg.get("path_distill_loss_weight"),
            0.0,
        ),
        "soft_path_distill_loss_weight": _resolve_option(
            getattr(args, "soft_path_distill_loss_weight", None),
            training_cfg.get("soft_path_distill_loss_weight"),
            0.0,
        ),
        "soft_path_temperature": _resolve_option(
            getattr(args, "soft_path_temperature", None),
            training_cfg.get("soft_path_temperature"),
            0.05,
        ),
        "soft_path_target_sigma_frames": _resolve_option(
            getattr(args, "soft_path_target_sigma_frames", None),
            training_cfg.get("soft_path_target_sigma_frames"),
            2.0,
        ),
        "sequence_contrastive_loss_weight": _resolve_option(
            getattr(args, "sequence_contrastive_loss_weight", None),
            training_cfg.get("sequence_contrastive_loss_weight"),
            0.0,
        ),
        "anti_collapse_loss_weight": _resolve_option(
            getattr(args, "anti_collapse_loss_weight", None),
            training_cfg.get("anti_collapse_loss_weight"),
            0.0,
        ),
        "anti_collapse_covariance_weight": _resolve_option(
            getattr(args, "anti_collapse_covariance_weight", None),
            training_cfg.get("anti_collapse_covariance_weight"),
            0.01,
        ),
        "temporal_order_loss_weight": _resolve_option(
            getattr(args, "temporal_order_loss_weight", None),
            training_cfg.get("temporal_order_loss_weight"),
            0.0,
        ),
        "relative_offset_loss_weight": _resolve_option(
            getattr(args, "relative_offset_loss_weight", None),
            training_cfg.get("relative_offset_loss_weight"),
            0.0,
        ),
        "relative_offset_bins": _resolve_option(
            getattr(args, "relative_offset_bins", None),
            training_cfg.get("relative_offset_bins"),
            None,
        ),
        "masked_reconstruction_loss_weight": _resolve_option(
            getattr(args, "masked_reconstruction_loss_weight", None),
            training_cfg.get("masked_reconstruction_loss_weight"),
            0.0,
        ),
        "cycle_consistency_loss_weight": _resolve_option(
            getattr(args, "cycle_consistency_loss_weight", None),
            training_cfg.get("cycle_consistency_loss_weight"),
            0.0,
        ),
        "cycle_entropy_loss_weight": _resolve_option(
            getattr(args, "cycle_entropy_loss_weight", None),
            training_cfg.get("cycle_entropy_loss_weight"),
            0.0,
        ),
        "cycle_monotonicity_loss_weight": _resolve_option(
            getattr(args, "cycle_monotonicity_loss_weight", None),
            training_cfg.get("cycle_monotonicity_loss_weight"),
            0.0,
        ),
        "cycle_smoothness_loss_weight": _resolve_option(
            getattr(args, "cycle_smoothness_loss_weight", None),
            training_cfg.get("cycle_smoothness_loss_weight"),
            0.0,
        ),
        "hard_negative_loss_weight": _resolve_option(
            getattr(args, "hard_negative_loss_weight", None),
            training_cfg.get("hard_negative_loss_weight"),
            0.0,
        ),
        "hard_negative_radius_frames": _resolve_option(
            getattr(args, "hard_negative_radius_frames", None),
            training_cfg.get("hard_negative_radius_frames"),
            96,
        ),
        "false_destination_negative_loss_weight": _resolve_option(
            getattr(args, "false_destination_negative_loss_weight", None),
            training_cfg.get("false_destination_negative_loss_weight"),
            0.0,
        ),
        "false_destination_negative_root": _resolve_option(
            getattr(args, "false_destination_negative_root", None),
            training_cfg.get("false_destination_negative_root"),
            None,
        ),
        "false_destination_min_error_ms": _resolve_option(
            getattr(args, "false_destination_min_error_ms", None),
            training_cfg.get("false_destination_min_error_ms"),
            500.0,
        ),
        "memory_bank_size": _resolve_option(
            getattr(args, "memory_bank_size", None),
            training_cfg.get("memory_bank_size"),
            0,
        ),
        "teacher_path_root": _resolve_option(
            args.teacher_path_root,
            training_cfg.get("teacher_path_root"),
            None,
        ),
        "self_mined_path_root": _resolve_option(
            getattr(args, "self_mined_path_root", None),
            training_cfg.get("self_mined_path_root"),
            None,
        ),
        "num_anchor_samples": _resolve_option(
            args.num_anchor_samples,
            training_cfg.get("num_anchor_samples"),
            64,
        ),
        "teacher_min_confidence": _resolve_option(
            args.teacher_min_confidence,
            training_cfg.get("teacher_min_confidence"),
            0.0,
        ),
        "path_distill_local_radius_frames": _resolve_option(
            getattr(args, "path_distill_local_radius_frames", None),
            training_cfg.get("path_distill_local_radius_frames"),
            12,
        ),
        "path_distill_local_step_frames": _resolve_option(
            getattr(args, "path_distill_local_step_frames", None),
            training_cfg.get("path_distill_local_step_frames"),
            3,
        ),
        "disable_time_stretch_for_anchors": _resolve_option(
            args.disable_time_stretch_for_anchors,
            training_cfg.get("disable_time_stretch_for_anchors"),
            True,
        ),
        "eval_pool_size": _resolve_option(
            args.eval_pool_size,
            evaluation_cfg.get("pool_size"),
            2,
        ),
        "alignment_eval_every_n_epochs": _resolve_option(
            args.alignment_eval_every_n_epochs,
            evaluation_cfg.get("alignment_eval_every_n_epochs"),
            1,
        ),
        "track_debug_checkpoints": _resolve_option(
            getattr(args, "track_debug_checkpoints", None),
            training_cfg.get("track_debug_checkpoints"),
            True,
        ),
        "anchor_temperature": _resolve_option(
            args.anchor_temperature,
            training_cfg.get("anchor_temperature"),
            0.1,
        ),
        "anchor_min_anchor_gap": _resolve_option(
            args.anchor_min_anchor_gap,
            training_cfg.get("anchor_min_anchor_gap"),
            1,
        ),
        "dry_run": args.dry_run,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DeepAlign-26 on SWD")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path")
    parser.add_argument("--swd-path", type=str, default=None, help="Path to the SWD dataset root")
    parser.add_argument("--output-dir", type=str, default=None, help="Checkpoint output directory")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--sr", type=int, default=None)
    parser.add_argument("--hop-length", type=int, default=None)
    parser.add_argument("--n-freq-bins", type=int, default=None)
    parser.add_argument("--conv-channels", type=int, nargs="+", default=None)
    parser.add_argument("--gru-hidden-size", type=int, default=None)
    parser.add_argument("--num-gru-layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--temporal-attention-heads", type=int, default=None)
    parser.add_argument("--max-length-sec", type=float, default=None)
    parser.add_argument(
        "--segment-sampling",
        type=str,
        choices=[
            "aligned_measures",
            "independent_random",
            "teacher_path",
            "self_audio",
            "self_audio_ordered",
            "same_lied_pair",
            "self_mined_path",
        ],
        default=None,
    )
    parser.add_argument("--samples-per-epoch", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--start-gamma", type=float, default=None)
    parser.add_argument("--end-gamma", type=float, default=None)
    parser.add_argument("--soft-dtw-loss-weight", type=float, default=None)
    parser.add_argument("--gradient-clip", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--val-split", type=float, default=None)
    parser.add_argument("--save-every-n-epochs", type=int, default=None)
    parser.add_argument("--dist-func", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument(
        "--selection-metric",
        type=str,
        choices=list(SELECTION_METRICS),
        default=None,
    )
    parser.add_argument("--cache-root", type=str, default=None)
    parser.add_argument("--deep-decode", type=str, choices=list(TRAIN_DEEP_DECODE_CHOICES), default=None)
    parser.add_argument("--band-radius-frames", type=int, default=None)
    parser.add_argument("--anchor-loss-weight", type=float, default=None)
    parser.add_argument("--dense-anchor-loss-weight", type=float, default=None)
    parser.add_argument("--path-distill-loss-weight", type=float, default=None)
    parser.add_argument("--soft-path-distill-loss-weight", type=float, default=None)
    parser.add_argument("--soft-path-temperature", type=float, default=None)
    parser.add_argument("--soft-path-target-sigma-frames", type=float, default=None)
    parser.add_argument("--sequence-contrastive-loss-weight", type=float, default=None)
    parser.add_argument("--anti-collapse-loss-weight", type=float, default=None)
    parser.add_argument("--anti-collapse-covariance-weight", type=float, default=None)
    parser.add_argument("--temporal-order-loss-weight", type=float, default=None)
    parser.add_argument("--relative-offset-loss-weight", type=float, default=None)
    parser.add_argument("--relative-offset-bins", type=int, nargs="+", default=None)
    parser.add_argument("--masked-reconstruction-loss-weight", type=float, default=None)
    parser.add_argument("--cycle-consistency-loss-weight", type=float, default=None)
    parser.add_argument("--cycle-entropy-loss-weight", type=float, default=None)
    parser.add_argument("--cycle-monotonicity-loss-weight", type=float, default=None)
    parser.add_argument("--cycle-smoothness-loss-weight", type=float, default=None)
    parser.add_argument("--hard-negative-loss-weight", type=float, default=None)
    parser.add_argument("--hard-negative-radius-frames", type=int, default=None)
    parser.add_argument("--false-destination-negative-loss-weight", type=float, default=None)
    parser.add_argument("--false-destination-negative-root", type=str, default=None)
    parser.add_argument("--false-destination-min-error-ms", type=float, default=None)
    parser.add_argument("--memory-bank-size", type=int, default=None)
    parser.add_argument("--teacher-path-root", type=str, default=None)
    parser.add_argument("--self-mined-path-root", type=str, default=None)
    parser.add_argument("--num-anchor-samples", type=int, default=None)
    parser.add_argument("--teacher-min-confidence", type=float, default=None)
    parser.add_argument("--path-distill-local-radius-frames", type=int, default=None)
    parser.add_argument("--path-distill-local-step-frames", type=int, default=None)
    parser.add_argument(
        "--disable-time-stretch-for-anchors",
        dest="disable_time_stretch_for_anchors",
        action="store_true",
    )
    parser.add_argument(
        "--allow-time-stretch-for-anchors",
        dest="disable_time_stretch_for_anchors",
        action="store_false",
    )
    parser.add_argument("--eval-pool-size", type=int, default=None)
    parser.add_argument("--alignment-eval-every-n-epochs", type=int, default=None)
    parser.add_argument("--track-debug-checkpoints", dest="track_debug_checkpoints", action="store_true")
    parser.add_argument("--no-track-debug-checkpoints", dest="track_debug_checkpoints", action="store_false")
    parser.set_defaults(track_debug_checkpoints=None)
    parser.add_argument("--anchor-temperature", type=float, default=None)
    parser.add_argument("--anchor-min-anchor-gap", type=int, default=None)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate the training stack on synthetic data")
    parser.add_argument("--cache-spectrograms", dest="cache_spectrograms", action="store_true")
    parser.add_argument("--no-cache-spectrograms", dest="cache_spectrograms", action="store_false")
    parser.add_argument(
        "--normalize-loss",
        dest="normalize_loss",
        action="store_true",
        help="Force normalized Soft-DTW divergence",
    )
    parser.add_argument(
        "--no-normalize-loss",
        dest="normalize_loss",
        action="store_false",
        help="Disable normalized Soft-DTW divergence",
    )
    parser.set_defaults(
        normalize_loss=None,
        cache_spectrograms=None,
        disable_time_stretch_for_anchors=None,
    )

    args = parser.parse_args()
    training_kwargs = _build_training_kwargs(args)
    if not training_kwargs["dry_run"] and not training_kwargs["swd_path"]:
        parser.error("--swd-path is required unless using --dry-run")

    train(**training_kwargs)


if __name__ == "__main__":
    main()
