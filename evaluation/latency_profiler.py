"""
latency_profiler.py — Phase 2 Evaluation
==========================================
Profiles per-window inference latency (ms) for three pipeline stages:
    Stage A:  DWT soft-threshold denoising only
    Stage B:  DAE forward pass only
    Stage C:  Full Hybrid (DWT → DAE)

Repeats 200 inference trials and reports:
    - Mean ± SD latency
    - 95th percentile
    - Real-time factor (RTF = latency / window_duration)

Target budget: < 55 ms total (meeting real-time EEG constraint at 256 Hz).

Usage:
    python latency_profiler.py

Outputs (in ../../results/):
    latency_profile.png    — violin + box plots per stage
    latency_results.csv    — per-trial latency table
"""

import csv
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from dae_model import HybridDAE

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))
from dwt_preprocessing import dwt_denoise

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
FS          = 256
WINDOW_LEN  = 256
N_TRIALS    = 200
SEED        = 42
WINDOW_DUR  = WINDOW_LEN / FS       # seconds per window


def time_dwt(signal: np.ndarray) -> float:
    t0 = time.perf_counter()
    dwt_denoise(signal)
    return (time.perf_counter() - t0) * 1000   # ms


def time_dae(signal: np.ndarray, model: HybridDAE) -> float:
    scale = max(np.percentile(np.abs(signal), 99), 1e-6)
    x     = torch.tensor((signal / scale)[None, None, :].astype(np.float32))
    model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        _  = model(x)
        return (time.perf_counter() - t0) * 1000


def time_hybrid(signal: np.ndarray, model: HybridDAE) -> float:
    t0     = time.perf_counter()
    dwt_out = dwt_denoise(signal)
    scale  = max(np.percentile(np.abs(dwt_out), 99), 1e-6)
    x      = torch.tensor((dwt_out / scale)[None, None, :].astype(np.float32))
    model.eval()
    with torch.no_grad():
        _ = model(x)
    return (time.perf_counter() - t0) * 1000


def summarise(latencies: np.ndarray, label: str) -> dict:
    return {
        "stage":   label,
        "mean_ms": round(float(np.mean(latencies)), 3),
        "std_ms":  round(float(np.std(latencies)), 3),
        "p95_ms":  round(float(np.percentile(latencies, 95)), 3),
        "rtf":     round(float(np.mean(latencies)) / (WINDOW_DUR * 1000), 4),
    }


def main():
    print("Phase 2 — latency_profiler.py\n")
    print(f"  Window: {WINDOW_LEN} samples ({WINDOW_DUR*1000:.1f} ms)  |  Trials: {N_TRIALS}\n")

    rng   = np.random.default_rng(SEED)
    model = HybridDAE(window_len=WINDOW_LEN)
    ckpt  = RESULTS_DIR / "hybrid_dae_best.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state["state_dict"])
        print("  Loaded checkpoint for latency profiling\n")

    # Warm-up run
    dummy = rng.standard_normal(WINDOW_LEN).astype(np.float32)
    for _ in range(5):
        time_hybrid(dummy, model)

    lat_dwt    = np.zeros(N_TRIALS)
    lat_dae    = np.zeros(N_TRIALS)
    lat_hybrid = np.zeros(N_TRIALS)

    for i in range(N_TRIALS):
        sig = rng.standard_normal(WINDOW_LEN).astype(np.float32) * 20
        lat_dwt[i]    = time_dwt(sig)
        lat_dae[i]    = time_dae(sig, model)
        lat_hybrid[i] = time_hybrid(sig, model)

    stages = [
        summarise(lat_dwt,    "DWT-Only"),
        summarise(lat_dae,    "DAE-Only"),
        summarise(lat_hybrid, "Hybrid (DWT+DAE)"),
    ]

    print(f"  {'Stage':<22} {'Mean (ms)':>10} {'±SD':>8} {'P95':>8} {'RTF':>8}")
    print("  " + "-" * 60)
    for s in stages:
        budget_ok = "✓" if s["p95_ms"] < 55 else "✗"
        print(f"  {s['stage']:<22} {s['mean_ms']:>10.3f} {s['std_ms']:>8.3f} "
              f"{s['p95_ms']:>8.3f} {s['rtf']:>8.4f}  {budget_ok}")

    # Save CSV
    csv_path = RESULTS_DIR / "latency_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stage","mean_ms","std_ms","p95_ms","rtf"])
        w.writeheader(); w.writerows(stages)

    # Per-trial CSV
    trial_path = RESULTS_DIR / "latency_per_trial.csv"
    with open(trial_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial","dwt_ms","dae_ms","hybrid_ms"])
        for i in range(N_TRIALS):
            w.writerow([i+1, f"{lat_dwt[i]:.4f}", f"{lat_dae[i]:.4f}", f"{lat_hybrid[i]:.4f}"])

    print(f"\n  [✓] Saved: latency_results.csv, latency_per_trial.csv")

    plot_latency(lat_dwt, lat_dae, lat_hybrid)


def plot_latency(lat_dwt: np.ndarray, lat_dae: np.ndarray,
                 lat_hybrid: np.ndarray) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor("#1e1e2e")
    fig.suptitle("Inference Latency Profiling — Hybrid DWT + DAE Pipeline",
                 color="white", fontsize=12, fontweight="bold")

    labels = ["DWT-Only", "DAE-Only", "Hybrid"]
    data   = [lat_dwt, lat_dae, lat_hybrid]
    colors = ["#60a5fa", "#f472b6", "#22c55e"]

    for ax in (ax1, ax2):
        ax.set_facecolor("#1e1e2e")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")

    # Violin plot
    parts = ax1.violinplot(data, positions=range(len(labels)), showmedians=True)
    for pc, col in zip(parts["bodies"], colors):
        pc.set_facecolor(col)
        pc.set_alpha(0.7)
    parts["cmedians"].set_colors("white")
    parts["cbars"].set_colors("white")
    parts["cmins"].set_colors("white")
    parts["cmaxes"].set_colors("white")
    ax1.axhline(55, color="#fbbf24", lw=1.5, ls="--", label="55 ms budget")
    ax1.set_xticks(range(len(labels))); ax1.set_xticklabels(labels, color="white")
    ax1.set_ylabel("Latency (ms)", color="white")
    ax1.set_title("Distribution (Violin)", color="white")
    ax1.legend(facecolor="#2e2e3e", labelcolor="white")

    # CDF
    for d, col, lbl in zip(data, colors, labels):
        sorted_d = np.sort(d)
        cdf      = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
        ax2.plot(sorted_d, cdf * 100, color=col, lw=1.8, label=lbl)
    ax2.axvline(55, color="#fbbf24", lw=1.5, ls="--", label="55 ms budget")
    ax2.set_xlabel("Latency (ms)", color="white")
    ax2.set_ylabel("Cumulative %", color="white")
    ax2.set_title("CDF", color="white")
    ax2.legend(facecolor="#2e2e3e", labelcolor="white")

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "latency_profile.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  [✓] Saved: latency_profile.png")


if __name__ == "__main__":
    main()
