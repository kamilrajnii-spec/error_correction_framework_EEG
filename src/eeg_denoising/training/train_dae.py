"""Training loop for the Phase 2 DAE."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from eeg_denoising.training.losses import combined_dae_loss


@dataclass(frozen=True)
class TrainingConfig:
    """Small set of training settings kept explicit for reproducibility."""

    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 1e-3
    patience: int = 3
    spectral_weight: float = 0.10
    device: str = "cpu"
    checkpoint_path: Path = Path("results/phase2/dae_best_model.pt")


def train_dae_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: TrainingConfig,
) -> list[dict[str, float]]:
    """Train a DAE and save the best checkpoint by validation loss."""
    device = torch.device(config.device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.epochs + 1):
        train_loss = _run_epoch(
            model=model,
            loader=train_loader,
            config=config,
            optimizer=optimizer,
            train=True,
        )
        validation_loss = _run_epoch(
            model=model,
            loader=validation_loader,
            config=config,
            optimizer=None,
            train=False,
        )

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "validation_loss": validation_loss,
                    "epoch": epoch,
                },
                config.checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            break

    return history


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    config: TrainingConfig,
    optimizer: torch.optim.Optimizer | None,
    train: bool,
) -> float:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_items = 0
    device = torch.device(config.device)

    for noisy, clean in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        if train and optimizer is not None:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            prediction = model(noisy)
            loss = combined_dae_loss(
                prediction,
                clean,
                spectral_weight=config.spectral_weight,
            )

        if train and optimizer is not None:
            loss.backward()
            optimizer.step()

        batch_size = noisy.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

    return total_loss / max(total_items, 1)

