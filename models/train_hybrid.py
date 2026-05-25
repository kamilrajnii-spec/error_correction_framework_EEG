"""
train_hybrid.py — Phase 2 Model Training
=========================================
Trains the Hybrid DWT + DAE pipeline on synthetic EEG data.
Generates training curves and saves the best checkpoint.

Usage:
    python train_hybrid.py [--epochs N] [--lr LR] [--demo]

Flags:
    --demo    Run 5 epochs only (quick validation)
    --epochs  Number of training epochs (default: 50)
    --lr      Learning rate (default: 1e-3)

Outputs (in ../../results/):
    training_curve.png          — loss and SNR over epochs
    hybrid_dae_best.pt          — best model checkpoint
    training_metrics.csv        — per-epoch metrics
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add parent directory to path for dae_model import
sys.path.insert(0, str(Path(__file__).parent))
from dae_model import HybridDAE

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FS          = 256
WINDOW_LEN  = 256
SEED        = 42


# ── Data preparation ─────────────────────────────────────────────────────────

def load_or_synthesise() -> tuple[np.ndarray, np.ndarray]:
    """Load arrays from results/ or synthesise fallback data."""
    clean_p = RESULTS_DIR / "clean_eeg_array.npy"
    noisy_p = RESULTS_DIR / "ocular_artifact_array.npy"
    if clean_p.exists() and noisy_p.exists():
        clean = np.load(clean_p).astype(np.float32)
        noisy = np.load(noisy_p).astype(np.float32)
        print("  Loaded arrays from results/")
        return clean, noisy
    print("  Arrays not found — synthesising demo data …")
    rng   = np.random.default_rng(SEED)
    n_ch  = 19
    n_smp = WINDOW_LEN * 8
    t     = np.linspace(0, n_smp / FS, n_smp)
    clean = (10 * np.sin(2 * np.pi * 10 * t)
             + 5  * np.sin(2 * np.pi * 5  * t)).astype(np.float32)
    clean = np.tile(clean, (n_ch, 1))
    noisy = clean + rng.normal(0, 20, clean.shape).astype(np.float32)
    return clean, noisy


def sliding_windows(arr: np.ndarray, win: int, step: int) -> np.ndarray:
    """Extract overlapping windows: (n_ch, T) → (N_windows, win)."""
    n_ch, T = arr.shape
    windows = []
    for start in range(0, T - win, step):
        windows.append(arr[:, start: start + win])
    return np.array(windows).reshape(-1, win)


def build_dataset(clean: np.ndarray, noisy: np.ndarray,
                  win: int = WINDOW_LEN, step: int = 64):
    clean_w = sliding_windows(clean, win, step)
    noisy_w = sliding_windows(noisy, win, step)
    # Normalise to [-1, 1]
    scale = np.percentile(np.abs(clean_w), 99, axis=1, keepdims=True) + 1e-6
    clean_w = np.clip(clean_w / scale, -1, 1)
    noisy_w = np.clip(noisy_w / scale, -1, 1)
    return (torch.tensor(noisy_w[:, None, :]),
            torch.tensor(clean_w[:, None, :]))


# ── Loss ─────────────────────────────────────────────────────────────────────

class HybridLoss(nn.Module):
    """MSE + frequency-domain L1 loss."""
    def __init__(self, freq_weight: float = 0.3):
        super().__init__()
        self.freq_weight = freq_weight
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_loss  = self.mse(pred, target)
        pred_fft  = torch.fft.rfft(pred.squeeze(1),  norm="ortho")
        targ_fft  = torch.fft.rfft(target.squeeze(1), norm="ortho")
        freq_loss = torch.mean(torch.abs(pred_fft - targ_fft))
        return mse_loss + self.freq_weight * freq_loss


def snr_db(pred: torch.Tensor, target: torch.Tensor) -> float:
    signal = torch.mean(target ** 2)
    noise  = torch.mean((pred - target) ** 2) + 1e-10
    return 10 * torch.log10(signal / noise).item()


# ── Training loop ─────────────────────────────────────────────────────────────

def train(epochs: int = 50, lr: float = 1e-3, demo: bool = False):
    torch.manual_seed(SEED)
    device = torch.device("cpu")
    if demo:
        epochs = 5

    print(f"\nPhase 2 — train_hybrid.py  [{epochs} epochs, lr={lr}]\n")

    clean, noisy = load_or_synthesise()
    X, Y = build_dataset(clean, noisy)

    n_train = int(0.8 * len(X))
    train_ds = TensorDataset(X[:n_train], Y[:n_train])
    val_ds   = TensorDataset(X[n_train:], Y[n_train:])
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=64, shuffle=False)

    print(f"  Train windows : {len(train_ds)}  |  Val windows : {len(val_ds)}\n")

    model     = HybridDAE(window_len=WINDOW_LEN).to(device)
    criterion = HybridLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history   = {"epoch": [], "train_loss": [], "val_loss": [], "val_snr": []}
    best_val  = float("inf")
    best_path = RESULTS_DIR / "hybrid_dae_best.pt"

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        model.eval()
        val_loss, val_snr = 0.0, 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred     = model(xb)
                val_loss += criterion(pred, yb).item() * len(xb)
                val_snr  += snr_db(pred, yb) * len(xb)
        val_loss /= len(val_ds)
        val_snr  /= len(val_ds)
        scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "val_loss": val_loss, "val_snr": val_snr}, best_path)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_snr"].append(val_snr)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch:3d}/{epochs}  "
              f"train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  "
              f"val_SNR={val_snr:.2f} dB  ({elapsed:.1f}s)")

    # Save metrics CSV
    csv_path = RESULTS_DIR / "training_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch","train_loss","val_loss","val_snr"])
        writer.writeheader()
        for i in range(len(history["epoch"])):
            writer.writerow({k: history[k][i] for k in history})
    print(f"\n  [✓] Metrics saved: training_metrics.csv")
    print(f"  [✓] Best checkpoint: hybrid_dae_best.pt  (val_loss={best_val:.5f})")

    # Plot training curves
    plot_training_curves(history, epochs)


def plot_training_curves(history: dict, epochs: int) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#1e1e2e")
    fig.suptitle("Hybrid DWT + DAE — Training Curves", color="white",
                 fontsize=13, fontweight="bold")

    for ax in (ax1, ax2):
        ax.set_facecolor("#1e1e2e")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")

    ep = history["epoch"]
    ax1.plot(ep, history["train_loss"], color="#60a5fa", lw=1.8, label="Train Loss")
    ax1.plot(ep, history["val_loss"],   color="#f472b6", lw=1.8, label="Val Loss")
    ax1.set_xlabel("Epoch", color="white")
    ax1.set_ylabel("Hybrid Loss", color="white")
    ax1.set_title("Loss Curve", color="white")
    ax1.legend(facecolor="#2e2e3e", labelcolor="white")

    ax2.plot(ep, history["val_snr"], color="#34d399", lw=1.8)
    ax2.axhline(np.max(history["val_snr"]), color="#fbbf24", ls="--", lw=0.8,
                label=f"Best: {max(history['val_snr']):.2f} dB")
    ax2.set_xlabel("Epoch", color="white")
    ax2.set_ylabel("SNR (dB)", color="white")
    ax2.set_title("Validation SNR", color="white")
    ax2.legend(facecolor="#2e2e3e", labelcolor="white")

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "training_curve.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  [✓] Saved: training_curve.png")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int,  default=50)
    parser.add_argument("--lr",     type=float, default=1e-3)
    parser.add_argument("--demo",   action="store_true")
    args = parser.parse_args()
    train(epochs=args.epochs, lr=args.lr, demo=args.demo)
