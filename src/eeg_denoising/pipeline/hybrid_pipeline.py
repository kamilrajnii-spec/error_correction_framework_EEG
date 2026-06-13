"""Wavelet to DAE hybrid denoising pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from eeg_denoising.models.dae import ConvDAE, load_dae_checkpoint
from eeg_denoising.training.eeg_dataset import (
    denormalize_epochs,
    normalize_epochs,
)
from eeg_denoising.wavelet.dwt_denoising import denoise_epochs_dwt


def apply_dae_to_epochs(
    epochs: np.ndarray,
    model: ConvDAE,
    device: str = "cpu",
) -> np.ndarray:
    """Run a trained DAE on epochs with per-epoch normalization."""
    normalized, means, stds = normalize_epochs(epochs)
    model_input = torch.tensor(normalized[:, np.newaxis, :], dtype=torch.float32)
    model_input = model_input.to(device)

    model.to(device)
    model.eval()
    with torch.no_grad():
        output = model(model_input).cpu().numpy()[:, 0, :]

    return denormalize_epochs(output, means, stds)


def run_hybrid_pipeline(
    noisy_epochs: np.ndarray,
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Return wavelet-only and hybrid outputs for noisy EEG epochs."""
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(
            "DAE checkpoint not found. Run scripts/train_phase2_dae.py first."
        )

    wavelet_output = denoise_epochs_dwt(noisy_epochs)
    model = load_dae_checkpoint(checkpoint, device=device)
    hybrid_output = apply_dae_to_epochs(wavelet_output, model, device=device)

    return {
        "wavelet": wavelet_output,
        "hybrid": hybrid_output,
    }

