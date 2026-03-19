"""Training entrypoint for the DeepAlign-26 SWD workflow."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from dis_alignment.model.encoder import CRNNEncoder, DeepAlignModel
from dis_alignment.model.soft_dtw_loss import GammaScheduler, SoftDTWLoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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
    max_length_sec: float = 30.0,
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
    dry_run: bool = False,
) -> dict[str, list[float]]:
    """
    Train DeepAlign-26 on SWD or synthetic dry-run data.

    Args:
        swd_path: Path to the SWD root directory. Optional for dry runs.
        output_dir: Directory for checkpoints and training history.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        lr: Initial AdamW learning rate.
        embed_dim: Output embedding dimensionality.
        n_freq_bins: Number of CQT bins in the input representation.
        num_conv_channels: Channel sizes for each convolutional block.
        gru_hidden_size: Hidden size for the BiGRU layers.
        num_gru_layers: Number of stacked BiGRU layers.
        dropout: Encoder dropout rate.
        max_length_sec: Maximum sampled segment length.
        start_gamma: Initial Soft-DTW gamma value.
        end_gamma: Final Soft-DTW gamma value.
        normalize_loss: Whether to use normalized Soft-DTW divergence.
        dist_func: Pairwise distance function inside Soft-DTW.
        gradient_clip: Maximum gradient norm.
        weight_decay: AdamW weight decay.
        augment: Whether to apply waveform augmentation.
        augmentor_kwargs: Optional kwargs for AudioAugmentor.
        device: Explicit device string or None for auto-detect.
        val_split: Fraction of dataset reserved for validation.
        save_every_n_epochs: Periodic checkpoint interval.
        dry_run: Use synthetic data and skip SWD access.

    Returns:
        Training history with loss, gamma, learning-rate, and timing curves.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    logger.info(f"Using device: {dev}")

    if dev.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    collate_fn = None
    augmentor_kwargs = augmentor_kwargs or {}

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

        logger.info(f"Loading SWD from {swd_path}")
        swd = SWDDataset(swd_path)
        logger.info(f"Available performances: {swd.available_performances}")
        logger.info(f"Available lieder: {len(swd.available_lieder)}")

        augmentor = AudioAugmentor(**augmentor_kwargs) if augment else None
        dataset = SWDPairDataset(
            swd,
            max_length_sec=max_length_sec,
            augmentor=augmentor,
            hop_length=220,
            n_bins=n_freq_bins,
        )

        n_val = max(1, int(len(dataset) * val_split))
        n_train = len(dataset) - n_val
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])
        collate_fn = collate_variable_length

    logger.info(f"Training pairs: {n_train}, Validation pairs: {n_val}")

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
    )
    model = DeepAlignModel(encoder).to(dev)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = SoftDTWLoss(
        gamma=start_gamma,
        normalize=normalize_loss,
        dist_func=dist_func,
    ).to(dev)
    gamma_scheduler = GammaScheduler(
        criterion,
        start_gamma=start_gamma,
        end_gamma=end_gamma,
        num_epochs=epochs,
    )
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev.type == "cuda"))

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "gamma": [],
        "lr": [],
        "epoch_time": [],
    }
    best_val_loss = float("inf")

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        gamma = gamma_scheduler.step(epoch)
        history["gamma"].append(gamma)

        model.train()
        train_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            spec_a = batch["spec_a"].to(dev)
            spec_b = batch["spec_b"].to(dev)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(dev.type == "cuda")):
                emb_a, emb_b = model(spec_a, spec_b)
                loss = criterion(emb_a, emb_b)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(n_batches, 1)
        history["train_loss"].append(train_loss)

        model.eval()
        val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                spec_a = batch["spec_a"].to(dev)
                spec_b = batch["spec_b"].to(dev)
                emb_a, emb_b = model(spec_a, spec_b)
                loss = criterion(emb_a, emb_b)
                val_loss += loss.item()
                n_val_batches += 1

        val_loss /= max(n_val_batches, 1)
        history["val_loss"].append(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(current_lr)
        lr_scheduler.step()

        epoch_time = time.perf_counter() - epoch_start
        history["epoch_time"].append(epoch_time)

        logger.info(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"gamma: {gamma:.4f} | LR: {current_lr:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "encoder_state_dict": encoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": {
                    "swd_path": swd_path,
                    "embed_dim": embed_dim,
                    "n_freq_bins": n_freq_bins,
                    "num_conv_channels": num_conv_channels,
                    "gru_hidden_size": gru_hidden_size,
                    "num_gru_layers": num_gru_layers,
                    "dropout": dropout,
                    "max_length_sec": max_length_sec,
                    "start_gamma": start_gamma,
                    "end_gamma": end_gamma,
                    "normalize_loss": normalize_loss,
                    "dist_func": dist_func,
                    "gradient_clip": gradient_clip,
                    "weight_decay": weight_decay,
                    "augment": augment,
                    "augmentor_kwargs": augmentor_kwargs,
                    "val_split": val_split,
                },
            }, out / "best_model.pt")
            logger.info(f"  -> Saved best model (val_loss={val_loss:.4f})")

        if save_every_n_epochs > 0 and (epoch + 1) % save_every_n_epochs == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, out / f"checkpoint_epoch{epoch + 1}.pt")

    torch.save(model.state_dict(), out / "final_model.pt")
    with open(out / "training_history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    logger.info(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Checkpoints saved to {out}")
    return history


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


def _build_training_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_training_config(args.config)
    dataset_cfg = config.get("dataset", {})
    audio_cfg = config.get("audio", {})
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    soft_dtw_cfg = config.get("soft_dtw", {})
    aug_cfg = config.get("augmentation", {})
    output_cfg = config.get("output", {})

    augment_enabled = False if args.no_augment else _resolve_option(
        None,
        aug_cfg.get("enabled"),
        True,
    )
    augmentor_kwargs = {
        "time_stretch_range": tuple(aug_cfg["time_stretch_range"]),
        "pitch_shift_range": tuple(aug_cfg["pitch_shift_range"]),
        "noise_snr_range": tuple(aug_cfg["noise_snr_range"]),
        "prob": aug_cfg["augment_prob"],
        "sr": audio_cfg.get("sr", 22050),
    } if augment_enabled and aug_cfg else {}

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
        "max_length_sec": _resolve_option(args.max_length_sec, dataset_cfg.get("max_length_sec"), 30.0),
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
    parser.add_argument("--max-length-sec", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--start-gamma", type=float, default=None)
    parser.add_argument("--end-gamma", type=float, default=None)
    parser.add_argument("--gradient-clip", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--val-split", type=float, default=None)
    parser.add_argument("--save-every-n-epochs", type=int, default=None)
    parser.add_argument("--dist-func", type=str, default=None)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate the training stack on synthetic data")
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
    parser.set_defaults(normalize_loss=None)

    args = parser.parse_args()
    training_kwargs = _build_training_kwargs(args)
    if not training_kwargs["dry_run"] and not training_kwargs["swd_path"]:
        parser.error("--swd-path is required unless using --dry-run")

    train(**training_kwargs)


if __name__ == "__main__":
    main()
