"""Training script for DeepAlign-26 model.

Trains the CRNN encoder using Soft-DTW loss on SWD audio pairs.

Usage:
    python -m dis_alignment.model.train --swd-path /path/to/SWD --epochs 50

    Or with config:
    python -m dis_alignment.model.train --config config/deepalign.yaml

    Dry-run (validates pipeline without SWD):
    python -m dis_alignment.model.train --dry-run
"""

import argparse
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from dis_alignment.data.swd import SWDDataset
from dis_alignment.model.augmentation import AudioAugmentor
from dis_alignment.model.dataset import SWDPairDataset, collate_variable_length
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

    def __getitem__(self, idx: int) -> dict:
        return {
            "spec_a": torch.randn(1, self.n_freq, self.n_time),
            "spec_b": torch.randn(1, self.n_freq, self.n_time),
            "pair_id": f"synthetic_{idx}",
        }


def train(
    swd_path: str,
    output_dir: str = "checkpoints",
    epochs: int = 50,
    batch_size: int = 4,
    lr: float = 1e-3,
    embed_dim: int = 64,
    n_freq_bins: int = 84,
    gru_hidden_size: int = 128,
    max_length_sec: float = 30.0,
    start_gamma: float = 1.0,
    end_gamma: float = 0.01,
    gradient_clip: float = 1.0,
    augment: bool = True,
    device: str | None = None,
    val_split: float = 0.2,
    dry_run: bool = False,
) -> dict:
    """
    Train the DeepAlign-26 model.

    Args:
        swd_path: Path to SWD dataset directory.
        output_dir: Directory for saving checkpoints.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        lr: Initial learning rate.
        embed_dim: Embedding dimensionality.
        n_freq_bins: Number of CQT frequency bins.
        gru_hidden_size: GRU hidden size.
        max_length_sec: Max audio segment length for training.
        start_gamma: Initial Soft-DTW γ value.
        end_gamma: Final Soft-DTW γ value.
        gradient_clip: Max gradient norm.
        augment: Whether to apply data augmentation.
        device: Device string ('cuda', 'cpu', or None for auto).
        val_split: Fraction of pairs for validation.

    Returns:
        Dict with training history.
    """
    # Setup device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    logger.info(f"Using device: {dev}")

    if dev.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create output directory
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if dry_run:
        logger.info("DRY RUN: using synthetic data (no SWD needed)")
        epochs = min(epochs, 2)
        dataset = _SyntheticPairDataset(n_pairs=4, n_freq=n_freq_bins, n_time=50)
        n_val = 1
        n_train = len(dataset) - n_val
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [n_train, n_val]
        )
    else:
        # Load dataset
        logger.info(f"Loading SWD from {swd_path}")
        swd = SWDDataset(swd_path)
        logger.info(f"Available performances: {swd.available_performances}")
        logger.info(f"Available lieder: {len(swd.available_lieder)}")

        # Create augmentor
        augmentor = AudioAugmentor() if augment else None

        # Create training dataset
        dataset = SWDPairDataset(
            swd,
            max_length_sec=max_length_sec,
            augmentor=augmentor,
            hop_length=220,  # ~10ms at 22050 Hz
            n_bins=n_freq_bins,
        )

        # Train/val split
        n_val = max(1, int(len(dataset) * val_split))
        n_train = len(dataset) - n_val
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [n_train, n_val]
        )

    logger.info(f"Training pairs: {n_train}, Validation pairs: {n_val}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_variable_length,
        num_workers=0,  # Windows compatibility
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_variable_length,
        num_workers=0,
        pin_memory=True,
    )

    # Create model
    encoder = CRNNEncoder(
        n_freq_bins=n_freq_bins,
        embed_dim=embed_dim,
        gru_hidden_size=gru_hidden_size,
    )
    model = DeepAlignModel(encoder).to(dev)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = SoftDTWLoss(gamma=start_gamma, normalize=True).to(dev)
    gamma_scheduler = GammaScheduler(
        criterion,
        start_gamma=start_gamma,
        end_gamma=end_gamma,
        num_epochs=epochs,
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Mixed precision for faster training
    scaler = torch.amp.GradScaler("cuda", enabled=(dev.type == "cuda"))

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "gamma": [],
        "lr": [],
        "epoch_time": [],
    }

    best_val_loss = float("inf")

    # Training loop
    for epoch in range(epochs):
        epoch_start = time.perf_counter()

        # Update gamma
        gamma = gamma_scheduler.step(epoch)
        history["gamma"].append(gamma)

        # Train
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

            # Gradient clipping
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(n_batches, 1)
        history["train_loss"].append(train_loss)

        # Validate
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

        # Update LR (after optimizer.step() has been called in training loop)
        current_lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(current_lr)
        lr_scheduler.step()

        epoch_time = time.perf_counter() - epoch_start
        history["epoch_time"].append(epoch_time)

        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"γ: {gamma:.4f} | LR: {current_lr:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "encoder_state_dict": encoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": {
                    "embed_dim": embed_dim,
                    "n_freq_bins": n_freq_bins,
                    "gru_hidden_size": gru_hidden_size,
                },
            }, out / "best_model.pt")
            logger.info(f"  → Saved best model (val_loss={val_loss:.4f})")

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, out / f"checkpoint_epoch{epoch+1}.pt")

    # Save final model and history
    torch.save(model.state_dict(), out / "final_model.pt")

    with open(out / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Checkpoints saved to {out}")

    return history


def main():
    parser = argparse.ArgumentParser(description="Train DeepAlign-26 model")
    parser.add_argument("--swd-path", type=str, default=None,
                        help="Path to SWD dataset directory")
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                        help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--max-length-sec", type=float, default=30.0)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--start-gamma", type=float, default=1.0)
    parser.add_argument("--end-gamma", type=float, default=0.01)
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate pipeline with synthetic data (no SWD needed)")

    args = parser.parse_args()

    if not args.dry_run and not args.swd_path:
        parser.error("--swd-path is required unless using --dry-run")

    train(
        swd_path=args.swd_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        embed_dim=args.embed_dim,
        max_length_sec=args.max_length_sec,
        start_gamma=args.start_gamma,
        end_gamma=args.end_gamma,
        augment=not args.no_augment,
        device=args.device,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
