"""
dae_model.py — Phase 2 Model Architecture
==========================================
Deep Autoencoder (DAE) for EEG artifact removal.
Implements a 1-D convolutional encoder–decoder with skip connections
and Layer Normalization, designed for 256-sample (1 s @ 256 Hz) windows.

Architecture summary:
    Encoder:  Conv1d(1→32→64→128)  with stride-2 downsampling
    Bottleneck: Linear projection  (128 → 64 → 128)
    Decoder:  ConvTranspose1d(128→64→32→1) + skip adds
    Output:   tanh-scaled residual correction

Usage (standalone):
    python dae_model.py          # prints architecture summary and param count
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1d(nn.Module):
    """1-D residual block: Conv → LayerNorm → GELU → Conv → LayerNorm + skip."""
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.norm1 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.norm2 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.gelu(x + residual)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(8, 32), nn.GELU(),
            ResBlock1d(32),
        )
        self.stage2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 64), nn.GELU(),
            ResBlock1d(64),
        )
        self.stage3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128), nn.GELU(),
            ResBlock1d(128),
        )

    def forward(self, x: torch.Tensor):
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        return s3, s2, s1           # bottleneck + skip features


class Bottleneck(nn.Module):
    def __init__(self, seq_len: int = 32):
        super().__init__()
        flat = 128 * seq_len
        self.fc1 = nn.Linear(flat, 512)
        self.fc2 = nn.Linear(512, flat)
        self.norm = nn.LayerNorm(512)
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        h = x.view(B, -1)
        h = F.gelu(self.norm(self.fc1(h)))
        h = self.fc2(h)
        return h.view(B, 128, self.seq_len)


class Decoder(nn.Module):
    def __init__(self, target_len: int = 256):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64), nn.GELU(),
            ResBlock1d(64),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose1d(64 + 64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 32), nn.GELU(),
            ResBlock1d(32),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose1d(32 + 32, 16, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(4, 16), nn.GELU(),
        )
        self.final = nn.Conv1d(16, 1, kernel_size=7, padding=3)
        self.target_len = target_len

    def forward(self, x: torch.Tensor,
                skip2: torch.Tensor, skip1: torch.Tensor) -> torch.Tensor:
        x = self.up1(x)
        # Align skip connections by cropping/padding to match spatial dim
        x = torch.cat([x[:, :, : skip2.size(2)], skip2], dim=1)
        x = self.up2(x)
        x = torch.cat([x[:, :, : skip1.size(2)], skip1], dim=1)
        x = self.up3(x)
        x = self.final(x)
        # Resize to exact target length
        x = F.interpolate(x, size=self.target_len, mode="linear", align_corners=False)
        return torch.tanh(x)


class HybridDAE(nn.Module):
    """
    Hybrid DAE operating on DWT-pre-processed EEG windows.

    Input:  (batch, 1, window_len)  — single-channel window
    Output: (batch, 1, window_len)  — residual correction in [-1, 1]

    The output is interpreted as a scaled residual:
        cleaned = dwt_output + scale * dae_correction
    """
    def __init__(self, window_len: int = 256, scale: float = 50.0):
        super().__init__()
        self.encoder   = Encoder()
        self.bottleneck = Bottleneck(seq_len=window_len // 8)
        self.decoder   = Decoder(target_len=window_len)
        self.scale     = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc, s2, s1 = self.encoder(x)
        bottleneck  = self.bottleneck(enc)
        correction  = self.decoder(bottleneck, s2, s1)
        return x + self.scale * correction


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = HybridDAE(window_len=256)
    model.eval()
    x_dummy = torch.randn(4, 1, 256)
    with torch.no_grad():
        y = model(x_dummy)
    total = count_parameters(model)
    print("HybridDAE Architecture Summary")
    print("=" * 40)
    print(model)
    print("=" * 40)
    print(f"Input  shape : {tuple(x_dummy.shape)}")
    print(f"Output shape : {tuple(y.shape)}")
    print(f"Total params : {total:,}  ({total/1e6:.3f} M)")
    print("\nAll components verified ✓")
