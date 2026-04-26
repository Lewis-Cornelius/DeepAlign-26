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

from dis_alignment.model.anchor_loss import AnchorContrastiveLoss
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
BEST_CHECKPOINT_FILENAMES = {
    "debug_ar50": "best_model_debug_ar50.pt",
    "debug_mae": "best_model_debug_mae.pt",
    "debug_balanced": "best_model_balanced.pt",
}


class _SyntheticPairDataset(Dataset):
    """Tiny synthetic dataset for dry-run validation."""

    def __init__(self, n_pairs: int = 4, n_freq: int = 84, n_time: int = 50):
        self.n_pairs = n_pairs
        self.n_freq = n_freq
        self.n_time = n_time

    def __len__(self) -> int:
        return self.n_pairs

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        return {
            "spec_a": torch.randn(1, self.n_freq, self.n_time),
            "spec_b": torch.randn(1, self.n_freq, self.n_time),
            "pair_id": f"synthetic_{idx}",
            "anchor_frame_indices_a": [],
            "anchor_frame_indices_b": [],
        }


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
    max_length_sec: float = 30.0,
    segment_sampling: str = "aligned_measures",
    samples_per_epoch: int | None = None,
    start_gamma: float = 1.0,
    end_gamma: float = 0.01,
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
    augmentor_kwargs = augmentor_kwargs or {}
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
        train_pairs, val_pairs = _split_swd_pairs(all_pairs, val_split=val_split)
        debug_pairs = [pair for pair in all_pairs if pair.lied_id in set(debug_lied_ids)]
        if selection_metric.startswith("debug_") and not debug_pairs:
            raise ValueError(
                "The configured debug subset did not resolve to any SWD pairs, "
                "so debug-based checkpoint selection cannot run."
            )

        resolved_samples_per_epoch = (
            samples_per_epoch if samples_per_epoch is not None else max(256, len(train_pairs) * 16)
        )
        val_samples_per_epoch = len(val_pairs)

        train_augmentor = AudioAugmentor(**augmentor_kwargs) if augment else None
        train_dataset = SWDPairDataset(
            swd,
            max_length_sec=max_length_sec,
            augmentor=train_augmentor,
            hop_length=220,
            n_bins=n_freq_bins,
            pairs=train_pairs,
            segment_sampling=segment_sampling,
            samples_per_epoch=resolved_samples_per_epoch,
            deterministic=False,
            cache_spectrograms=cache_spectrograms,
            cache_root=resolved_cache_root,
        )
        val_dataset = SWDPairDataset(
            swd,
            max_length_sec=max_length_sec,
            augmentor=None,
            hop_length=220,
            n_bins=n_freq_bins,
            pairs=val_pairs,
            segment_sampling=segment_sampling,
            samples_per_epoch=val_samples_per_epoch,
            deterministic=True,
            cache_spectrograms=cache_spectrograms,
            cache_root=resolved_cache_root,
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

    criterion = SoftDTWLoss(
        gamma=start_gamma,
        normalize=normalize_loss,
        dist_func=dist_func,
    ).to(dev)
    anchor_loss_fn = (
        AnchorContrastiveLoss(temperature=anchor_temperature, min_anchor_gap=anchor_min_anchor_gap).to(dev)
        if anchor_loss_weight > 0
        else None
    )

    gamma_scheduler = GammaScheduler(
        criterion,
        start_gamma=start_gamma,
        end_gamma=end_gamma,
        num_epochs=epochs,
    )
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=(dev.type == "cuda"))

    training_state_signature = {
        "epochs": epochs,
        "lr": lr,
        "start_gamma": start_gamma,
        "end_gamma": end_gamma,
        "selection_metric": selection_metric,
        "anchor_loss_weight": anchor_loss_weight,
    }
    best_selections: dict[str, dict[str, Any]] = {}
    start_epoch = 0

    if resume_from is not None:
        start_epoch, history, best_selection, best_selections = _load_resume_state(
            checkpoint_path=resume_from,
            model=model,
            encoder=encoder,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
            current_signature=training_state_signature,
        )
    else:
        best_selection = None

    for epoch in range(start_epoch, epochs):
        epoch_start = time.perf_counter()
        gamma = gamma_scheduler.step(epoch)
        history["gamma"].append(gamma)

        model.train()
        train_loss = 0.0
        train_anchor_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            spec_a = batch["spec_a"].to(dev)
            spec_b = batch["spec_b"].to(dev)
            anchor_frames_a = batch.get("anchor_frame_indices_a")
            anchor_frames_b = batch.get("anchor_frame_indices_b")

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(dev.type == "cuda")):
                emb_a, emb_b = model(spec_a, spec_b)
                loss = criterion(emb_a, emb_b)
                anchor_component = emb_a.new_zeros(())
                if anchor_loss_fn is not None:
                    anchor_component = anchor_loss_fn(emb_a, emb_b, anchor_frames_a, anchor_frames_b)
                    loss = loss + (anchor_loss_weight * anchor_component)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()

            train_loss += float(loss.item())
            train_anchor_loss += float(anchor_component.item())
            n_batches += 1

        train_loss /= max(n_batches, 1)
        train_anchor_loss /= max(n_batches, 1)
        history["train_loss"].append(train_loss)
        history["train_anchor_loss"].append(train_anchor_loss)

        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                spec_a = batch["spec_a"].to(dev)
                spec_b = batch["spec_b"].to(dev)
                emb_a, emb_b = model(spec_a, spec_b)
                loss = criterion(emb_a, emb_b)
                val_loss += float(loss.item())
                n_val_batches += 1

        val_loss /= max(n_val_batches, 1)
        history["val_loss"].append(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(current_lr)
        lr_scheduler.step()

        if dry_run:
            debug_metrics = {"mae": np.nan, "ar_50ms": np.nan, "ar_100ms": np.nan, "ar_200ms": np.nan, "pairs": 0.0}
            val_metrics = {"mae": np.nan, "ar_50ms": np.nan, "ar_100ms": np.nan, "ar_200ms": np.nan, "pairs": 0.0}
        else:
            debug_metrics = _evaluate_swd_pairs(
                debug_pairs,
                encoder=model.encoder,
                device=device,
                cache_root=resolved_cache_root,
                deep_decode=deep_decode,
                band_radius_frames=band_radius_frames,
            )
            val_metrics = _evaluate_swd_pairs(
                val_pairs,
                encoder=model.encoder,
                device=device,
                cache_root=resolved_cache_root,
                deep_decode=deep_decode,
                band_radius_frames=band_radius_frames,
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
            for metric in _tracked_selection_metrics(selection_metric)
        }
        current_selection = selection_snapshots[selection_metric]
        selection_key = tuple(float(value) for value in current_selection["selection_key"])
        history["selection_primary"].append(float(selection_key[0]))
        history["selection_secondary"].append(float(selection_key[1]) if len(selection_key) > 1 else np.nan)

        epoch_time = time.perf_counter() - epoch_start
        history["epoch_time"].append(epoch_time)

        logger.info(
            "Epoch %s/%s | Train: %.4f | Val: %.4f | Debug MAE: %.4f | Debug AR@50: %.4f | "
            "Debug AR@100: %.4f | Val MAE: %.4f | Val AR@50: %.4f | Val AR@100: %.4f | "
            "gamma: %.4f | LR: %.2e | Time: %.1fs",
            epoch + 1,
            epochs,
            train_loss,
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
        "train_anchor_loss": [],
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
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "encoder_state_dict": encoder.state_dict(),
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
) -> tuple[int, dict[str, list[float]], dict[str, Any] | None, dict[str, dict[str, Any]]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "encoder_state_dict" in checkpoint:
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise ValueError(f"Checkpoint at {checkpoint_path} does not contain model weights.")

    saved_signature = checkpoint.get("training_state_signature")
    if saved_signature == current_signature:
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


def _evaluate_swd_pairs(
    pairs: Sequence[Any],
    *,
    encoder: CRNNEncoder,
    device: str | None,
    cache_root: str | None,
    deep_decode: str,
    band_radius_frames: int | None,
) -> dict[str, float]:
    if not pairs:
        return {"mae": np.nan, "ar_50ms": np.nan, "ar_100ms": np.nan, "ar_200ms": np.nan, "pairs": 0.0}

    from dis_alignment.evaluation.swd import evaluate_pair

    rows: list[dict[str, Any]] = []
    for pair in pairs:
        pair_rows = evaluate_pair(
            pair,
            methods=["deepalign"],
            encoder=encoder,
            device=device,
            cache_root=cache_root,
            deep_decode=deep_decode,
            band_radius_frames=band_radius_frames,
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


def _tracked_selection_metrics(selection_metric: str) -> tuple[str, ...]:
    ordered = list(TRACKED_SELECTION_METRICS)
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
) -> tuple[list[Any], list[Any]]:
    """Split SWD pairs once at the pair level to avoid train/val leakage."""
    if not 0.0 < val_split < 1.0:
        raise ValueError("val_split must be between 0 and 1.")
    if len(pairs) < 2:
        raise ValueError("Need at least two SWD pairs to build train/validation splits.")

    ordered_pairs = sorted(pairs, key=lambda pair: getattr(pair, "pair_id", str(pair)))
    rng = np.random.default_rng(random_seed)
    indices = np.arange(len(ordered_pairs))
    rng.shuffle(indices)

    n_val = max(1, int(len(ordered_pairs) * val_split))
    n_val = min(n_val, len(ordered_pairs) - 1)
    val_indices = set(indices[:n_val].tolist())

    train_pairs = [pair for idx, pair in enumerate(ordered_pairs) if idx not in val_indices]
    val_pairs = [pair for idx, pair in enumerate(ordered_pairs) if idx in val_indices]
    return train_pairs, val_pairs


def _build_training_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_training_config(args.config)
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
        choices=["aligned_measures", "independent_random"],
        default=None,
    )
    parser.add_argument("--samples-per-epoch", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--start-gamma", type=float, default=None)
    parser.add_argument("--end-gamma", type=float, default=None)
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
    parser.add_argument("--deep-decode", type=str, choices=["unconstrained", "diagonal_band", "chroma_guided_band"], default=None)
    parser.add_argument("--band-radius-frames", type=int, default=None)
    parser.add_argument("--anchor-loss-weight", type=float, default=None)
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
    parser.set_defaults(normalize_loss=None, cache_spectrograms=None)

    args = parser.parse_args()
    training_kwargs = _build_training_kwargs(args)
    if not training_kwargs["dry_run"] and not training_kwargs["swd_path"]:
        parser.error("--swd-path is required unless using --dry-run")

    train(**training_kwargs)


if __name__ == "__main__":
    main()
