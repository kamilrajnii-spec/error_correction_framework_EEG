"""Loss functions for DAE training."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def time_domain_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error in the time domain."""
    return F.mse_loss(prediction, target)


def spectral_consistency_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """MSE between normalized frequency magnitudes."""
    prediction_spectrum = torch.abs(torch.fft.rfft(prediction, dim=-1))
    target_spectrum = torch.abs(torch.fft.rfft(target, dim=-1))

    prediction_spectrum = prediction_spectrum / (
        prediction_spectrum.mean(dim=-1, keepdim=True) + 1e-8
    )
    target_spectrum = target_spectrum / (
        target_spectrum.mean(dim=-1, keepdim=True) + 1e-8
    )

    return F.mse_loss(prediction_spectrum, target_spectrum)


def combined_dae_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    spectral_weight: float = 0.10,
) -> torch.Tensor:
    """Time-domain MSE plus a small spectral consistency term."""
    return time_domain_mse(prediction, target) + spectral_weight * spectral_consistency_loss(
        prediction,
        target,
    )

