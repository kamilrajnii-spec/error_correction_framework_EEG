"""Simple 1-D convolutional denoising autoencoder."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class ConvDAE(nn.Module):
    """Small encoder-bottleneck-decoder model for 512-sample EEG epochs."""

    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=9, padding=4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        bottleneck = self.bottleneck(encoded)
        return self.decoder(bottleneck)


def count_parameters(model: nn.Module) -> int:
    """Count trainable and non-trainable parameters from the model object."""
    return sum(parameter.numel() for parameter in model.parameters())


def load_dae_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> ConvDAE:
    """Load a ConvDAE checkpoint saved by the Phase 2 training script."""
    model = ConvDAE().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def write_model_summary(
    model: nn.Module,
    output_path: str | Path,
    input_length: int = 512,
    extra_lines: list[str] | None = None,
) -> None:
    """Write a plain text model summary for the thesis folder."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.no_grad():
        dummy_input = torch.zeros(1, 1, input_length)
        dummy_output = model(dummy_input)

    lines = [
        "Phase 2 DAE model summary",
        "Model: 1-D convolutional denoising autoencoder",
        f"Input shape: {tuple(dummy_input.shape)}",
        f"Output shape: {tuple(dummy_output.shape)}",
        f"Parameter count: {count_parameters(model)}",
    ]

    if extra_lines:
        lines.extend(extra_lines)

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")

