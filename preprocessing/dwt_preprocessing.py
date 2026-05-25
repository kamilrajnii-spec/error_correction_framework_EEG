"""
dwt_preprocessing.py — Phase 2 Preprocessing
=============================================
Discrete Wavelet Transform (DWT) based artifact suppression.
Implements soft-thresholding on detail coefficients to attenuate
artifact energy before the DAE stage.

Pipeline:
    raw EEG  →  DWT decompose (db4, level 5)
             →  soft-threshold detail coefficients
             →  IDWT reconstruct  →  DWT-cleaned EEG

Usage:
    python dwt_preprocessing.py

Outputs (in ../../results/):
    dwt_decomposition.png   — wavelet decomposition figure
    dwt_reconstruction_comparison.png — before / after comparison
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FS       = 256
SEED     = 42
WAVELET  = "db4"
LEVEL    = 5

# ── Pure-Python DWT (no PyWavelets dependency needed) ────────────────────────

def _db4_filters():
    """Daubechies-4 decomposition low-pass and high-pass filter coefficients."""
    h = np.array([
        0.48296291314469025,  0.8365163037378079,
        0.22414386804185735, -0.12940952255092145,
       -0.04294133854651015,  0.05012421021949272,
        0.01768711408720280, -0.00782105480240671
    ])
    g = h[::-1] * np.array([1, -1, 1, -1, 1, -1, 1, -1])
    return h, g


def _dwt_level(signal: np.ndarray, h: np.ndarray, g: np.ndarray):
    """Single-level DWT: returns (approx, detail)."""
    from numpy import convolve
    n   = len(signal)
    ext = np.concatenate([signal[-len(h) + 1:], signal, signal[:len(h) - 1]])
    ca  = convolve(ext, h[::-1], mode="valid")[::2][:n // 2]
    cd  = convolve(ext, g[::-1], mode="valid")[::2][:n // 2]
    return ca, cd


def dwt_decompose(signal: np.ndarray, level: int = LEVEL):
    """Multi-level DWT decomposition. Returns list: [cA_n, cD_n, ..., cD_1]."""
    h, g    = _db4_filters()
    coeffs  = []
    approx  = signal.copy()
    for _ in range(level):
        approx, detail = _dwt_level(approx, h, g)
        coeffs.append(detail)
    coeffs.append(approx)
    coeffs.reverse()           # [cA_5, cD_5, cD_4, ..., cD_1]
    return coeffs


def _idwt_level(ca: np.ndarray, cd: np.ndarray,
                h: np.ndarray, g: np.ndarray, target_len: int) -> np.ndarray:
    """Single-level IDWT: upsample + filter + add."""
    def upsample(x: np.ndarray) -> np.ndarray:
        out = np.zeros(2 * len(x))
        out[::2] = x
        return out
    from numpy import convolve
    rec_a = convolve(upsample(ca), h, mode="full")
    rec_d = convolve(upsample(cd), g, mode="full")
    length = min(len(rec_a), len(rec_d), target_len + len(h) - 1)
    rec    = (rec_a[:length] + rec_d[:length])
    offset = len(h) - 2
    return rec[offset: offset + target_len]


def idwt_reconstruct(coeffs: list, orig_len: int) -> np.ndarray:
    """Multi-level IDWT from coefficient list [cA_n, cD_n, ..., cD_1]."""
    h, g   = _db4_filters()
    approx = coeffs[0]
    levels = len(coeffs) - 1
    for lv in range(1, levels + 1):
        detail     = coeffs[lv]
        target_len = len(detail) * 2 if lv < levels else orig_len
        approx     = _idwt_level(approx, detail, h, g, target_len)
    return approx[:orig_len]


def soft_threshold(arr: np.ndarray, lam: float) -> np.ndarray:
    return np.sign(arr) * np.maximum(np.abs(arr) - lam, 0)


def universal_threshold(signal: np.ndarray) -> float:
    """VisuShrink universal threshold: σ * sqrt(2 * ln(N))."""
    n     = len(signal)
    sigma = np.median(np.abs(signal)) / 0.6745
    return sigma * np.sqrt(2 * np.log(n))


def dwt_denoise(signal: np.ndarray, level: int = LEVEL) -> np.ndarray:
    """DWT soft-threshold denoising for a 1-D signal."""
    coeffs = dwt_decompose(signal, level)
    # Threshold detail coefficients only (skip approx at index 0)
    thresholded = [coeffs[0]]
    for cd in coeffs[1:]:
        lam = universal_threshold(cd)
        thresholded.append(soft_threshold(cd, lam))
    return idwt_reconstruct(thresholded, len(signal))


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_dwt_decomposition(signal: np.ndarray, coeffs: list,
                           channel: str = "Fp1") -> None:
    t       = np.linspace(0, len(signal) / FS, len(signal))
    n_plots = len(coeffs) + 1
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 2.2 * n_plots))
    fig.patch.set_facecolor("#1e1e2e")
    fig.suptitle(f"DWT Decomposition — {WAVELET} Level {LEVEL}  |  Channel: {channel}",
                 color="white", fontsize=12, fontweight="bold", y=1.01)

    palette = ["#60a5fa", "#f472b6", "#fb923c", "#34d399", "#a78bfa", "#fbbf24"]

    # Original signal
    ax = axes[0]
    ax.set_facecolor("#1e1e2e")
    ax.plot(t, signal, color="#60a5fa", lw=0.8)
    ax.set_ylabel("Raw EEG\n(µV)", color="white", fontsize=8)
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
    ax.set_xlim([t[0], t[-1]])

    labels = [f"cA{LEVEL}"] + [f"cD{LEVEL - i}" for i in range(len(coeffs) - 1)]
    for idx, (cd, label) in enumerate(zip(coeffs, labels)):
        ax = axes[idx + 1]
        ax.set_facecolor("#1e1e2e")
        t_cd = np.linspace(0, len(signal) / FS, len(cd))
        ax.plot(t_cd, cd, color=palette[idx % len(palette)], lw=0.8)
        ax.set_ylabel(label, color="white", fontsize=8)
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")
        ax.set_xlim([0, len(signal) / FS])

    axes[-1].set_xlabel("Time (s)", color="white")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "dwt_decomposition.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  [✓] Saved: dwt_decomposition.png")


def plot_reconstruction_comparison(raw: np.ndarray, clean_ref: np.ndarray,
                                   dwt_out: np.ndarray, channel: str = "Fp1") -> None:
    t = np.linspace(0, len(raw) / FS, len(raw))
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.patch.set_facecolor("#1e1e2e")
    fig.suptitle(f"DWT Soft-Threshold Reconstruction  |  Channel: {channel}",
                 color="white", fontsize=12, fontweight="bold")

    pairs = [
        (raw,      "#ef4444", "Noisy Input (Ocular + EEG)"),
        (dwt_out,  "#22c55e", "DWT-Cleaned Output"),
        (clean_ref,"#60a5fa", "Ground-Truth Clean EEG"),
    ]
    for ax, (sig, col, lbl) in zip(axes, pairs):
        ax.set_facecolor("#1e1e2e")
        ax.plot(t, sig, color=col, lw=0.8, label=lbl)
        ax.set_ylabel("µV", color="white")
        ax.legend(loc="upper right", fontsize=8,
                  facecolor="#2e2e3e", labelcolor="white")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")

    axes[-1].set_xlabel("Time (s)", color="white")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "dwt_reconstruction_comparison.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  [✓] Saved: dwt_reconstruction_comparison.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Phase 2 — dwt_preprocessing.py\n")

    # Load or synthesise data
    clean_path = RESULTS_DIR / "clean_eeg_array.npy"
    noisy_path = RESULTS_DIR / "ocular_artifact_array.npy"

    if clean_path.exists() and noisy_path.exists():
        clean = np.load(clean_path)
        noisy = np.load(noisy_path)
        print("  Loaded arrays from results/")
    else:
        print("  Arrays not found — run inject_artifacts.py first.")
        return

    ch_idx  = 0          # Fp1 — highest ocular contamination
    raw_ch  = noisy[ch_idx]
    clean_ch = clean[ch_idx]

    print(f"  DWT decomposing channel Fp1 ({WAVELET}, level {LEVEL}) …")
    coeffs  = dwt_decompose(raw_ch, LEVEL)
    dwt_out = dwt_denoise(raw_ch, LEVEL)

    rmse_raw = np.sqrt(np.mean((raw_ch   - clean_ch) ** 2))
    rmse_dwt = np.sqrt(np.mean((dwt_out  - clean_ch) ** 2))
    print(f"  RMSE (raw vs clean) : {rmse_raw:.4f} µV")
    print(f"  RMSE (DWT vs clean) : {rmse_dwt:.4f} µV")
    print(f"  Improvement         : {100*(rmse_raw - rmse_dwt)/rmse_raw:.1f}%\n")

    plot_dwt_decomposition(raw_ch, coeffs, channel="Fp1")
    plot_reconstruction_comparison(raw_ch, clean_ch, dwt_out, channel="Fp1")

    # Save DWT-preprocessed full array
    dwt_all = np.stack([dwt_denoise(noisy[ch]) for ch in range(noisy.shape[0])])
    np.save(RESULTS_DIR / "dwt_preprocessed_array.npy", dwt_all)
    print("  [✓] Saved: dwt_preprocessed_array.npy")


if __name__ == "__main__":
    main()
