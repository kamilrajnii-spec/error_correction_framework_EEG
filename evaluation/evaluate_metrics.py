"""
evaluate_metrics.py — Phase 2 Evaluation
=========================================
Computes SNR improvement and RMSE reduction for three pipelines:
    1. Unprocessed (raw noisy EEG)
    2. DWT-only denoising
    3. Hybrid DWT + DAE (trained model)

Evaluates three artifact types: Ocular, EMG, Mixed.
Produces publication-quality comparison bar charts and a CSV table.

Usage:
    python evaluate_metrics.py

Outputs (in ../../results/):
    snr_comparison.png
    rmse_comparison.png
    evaluation_metrics.csv
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from dae_model import HybridDAE
from pathlib import Path as _P

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))
from dwt_preprocessing import dwt_denoise

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
FS          = 256
WINDOW_LEN  = 256
SEED        = 42


# ── Metrics ───────────────────────────────────────────────────────────────────

def snr_db(signal: np.ndarray, noisy: np.ndarray) -> float:
    signal_pwr  = np.mean(signal ** 2)
    artifact_pw = np.mean((noisy - signal) ** 2) + 1e-12
    return 10 * np.log10(signal_pwr / artifact_pw)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


# ── Pipeline wrappers ─────────────────────────────────────────────────────────

def apply_dwt(noisy: np.ndarray) -> np.ndarray:
    return np.stack([dwt_denoise(noisy[ch]) for ch in range(noisy.shape[0])]).astype(np.float32)


def apply_dae(noisy: np.ndarray, model: HybridDAE) -> np.ndarray:
    model.eval()
    out = np.zeros_like(noisy)
    scale = max(np.percentile(np.abs(noisy), 99), 1e-6)
    with torch.no_grad():
        for ch in range(noisy.shape[0]):
            # Sliding window inference
            sig   = noisy[ch].astype(np.float32)
            T     = len(sig)
            win   = WINDOW_LEN
            pred  = np.zeros(T, dtype=np.float32)
            count = np.zeros(T, dtype=np.float32)
            for start in range(0, T - win, win // 2):
                end  = start + win
                win_data = sig[start:end] / scale
                x   = torch.tensor(win_data[None, None, :])
                y   = model(x).squeeze().numpy() * scale
                pred[start:end] += y
                count[start:end] += 1
            count = np.where(count == 0, 1, count)
            out[ch] = pred / count
    return out


def apply_hybrid(noisy: np.ndarray, model: HybridDAE) -> np.ndarray:
    dwt_out = apply_dwt(noisy)
    dae_out = apply_dae(dwt_out, model)
    return dae_out


# ── Load data ─────────────────────────────────────────────────────────────────

def load_data() -> dict[str, tuple]:
    artifact_types = ["ocular", "emg", "mixed"]
    data = {}
    clean = np.load(RESULTS_DIR / "clean_eeg_array.npy").astype(np.float32)
    for art in artifact_types:
        key   = f"{art}_artifact"
        fpath = RESULTS_DIR / f"{key}_array.npy"
        if fpath.exists():
            data[art] = (clean, np.load(fpath).astype(np.float32))
        else:
            print(f"  Warning: {fpath.name} not found, skipping {art}")
    return data


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Phase 2 — evaluate_metrics.py\n")
    torch.manual_seed(SEED)

    if not (RESULTS_DIR / "clean_eeg_array.npy").exists():
        print("  Run inject_artifacts.py first.")
        return

    data = load_data()

    # Load or initialise DAE model
    ckpt_path = RESULTS_DIR / "hybrid_dae_best.pt"
    model     = HybridDAE(window_len=WINDOW_LEN)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        print(f"  Loaded checkpoint: {ckpt_path.name}\n")
    else:
        print("  No checkpoint found — using untrained model (demo mode)\n")

    results = []
    for art, (clean, noisy) in data.items():
        dwt_out    = apply_dwt(noisy)
        hybrid_out = apply_hybrid(noisy, model)

        row = {
            "artifact":       art.capitalize(),
            "snr_raw":        round(snr_db(clean, noisy), 2),
            "snr_dwt":        round(snr_db(clean, dwt_out), 2),
            "snr_hybrid":     round(snr_db(clean, hybrid_out), 2),
            "rmse_raw":       round(rmse(clean, noisy), 4),
            "rmse_dwt":       round(rmse(clean, dwt_out), 4),
            "rmse_hybrid":    round(rmse(clean, hybrid_out), 4),
        }
        results.append(row)
        print(f"  [{art.upper()}]")
        print(f"    SNR  — Raw: {row['snr_raw']:+.2f} dB  |  DWT: {row['snr_dwt']:+.2f} dB  |  Hybrid: {row['snr_hybrid']:+.2f} dB")
        print(f"    RMSE — Raw: {row['rmse_raw']:.4f}     |  DWT: {row['rmse_dwt']:.4f}     |  Hybrid: {row['rmse_hybrid']:.4f}")

    # Save CSV
    csv_path = RESULTS_DIR / "evaluation_metrics.csv"
    fields = ["artifact","snr_raw","snr_dwt","snr_hybrid","rmse_raw","rmse_dwt","rmse_hybrid"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(results)
    print(f"\n  [✓] Saved: evaluation_metrics.csv")

    # Plots
    plot_snr_comparison(results)
    plot_rmse_comparison(results)


def plot_snr_comparison(results: list) -> None:
    arts   = [r["artifact"] for r in results]
    x      = np.arange(len(arts))
    width  = 0.26
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#1e1e2e")

    b1 = ax.bar(x - width, [r["snr_raw"]    for r in results], width, label="Raw Input",   color="#ef4444", alpha=0.9)
    b2 = ax.bar(x,          [r["snr_dwt"]    for r in results], width, label="DWT-Only",    color="#60a5fa", alpha=0.9)
    b3 = ax.bar(x + width,  [r["snr_hybrid"] for r in results], width, label="Hybrid (Ours)", color="#22c55e", alpha=0.9)

    for bar_group in [b1, b2, b3]:
        for bar in bar_group:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                    f"{h:.1f}", ha="center", va="bottom", color="white", fontsize=8)

    ax.axhline(0, color="white", lw=0.6, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(arts, color="white", fontsize=11)
    ax.set_ylabel("SNR (dB)", color="white")
    ax.set_title("SNR Comparison Across Artifact Types and Pipelines",
                 color="white", fontsize=12, fontweight="bold")
    ax.legend(facecolor="#2e2e3e", labelcolor="white")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "snr_comparison.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  [✓] Saved: snr_comparison.png")


def plot_rmse_comparison(results: list) -> None:
    arts   = [r["artifact"] for r in results]
    x      = np.arange(len(arts))
    width  = 0.26
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#1e1e2e")

    ax.bar(x - width, [r["rmse_raw"]    for r in results], width, label="Raw Input",    color="#ef4444", alpha=0.9)
    ax.bar(x,          [r["rmse_dwt"]    for r in results], width, label="DWT-Only",     color="#60a5fa", alpha=0.9)
    ax.bar(x + width,  [r["rmse_hybrid"] for r in results], width, label="Hybrid (Ours)", color="#22c55e", alpha=0.9)

    ax.set_xticks(x); ax.set_xticklabels(arts, color="white", fontsize=11)
    ax.set_ylabel("RMSE (µV)", color="white")
    ax.set_title("RMSE Comparison Across Artifact Types and Pipelines",
                 color="white", fontsize=12, fontweight="bold")
    ax.legend(facecolor="#2e2e3e", labelcolor="white")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "rmse_comparison.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  [✓] Saved: rmse_comparison.png")


if __name__ == "__main__":
    main()
