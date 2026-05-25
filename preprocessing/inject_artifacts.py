"""
inject_artifacts.py — Phase 2 Preprocessing
============================================
Generates synthetic EEG segments with clean, ocular (eye-blink),
EMG (myogenic), and mixed artifact contamination.
Saves NumPy arrays and PNG plots to results/.

Usage:
    python inject_artifacts.py

Outputs (in ../../results/):
    clean_eeg_segment.png
    ocular_artifact_segment.png
    emg_artifact_segment.png
    mixed_artifact_segment.png
    clean_eeg_array.npy
    ocular_artifact_array.npy
    emg_artifact_array.npy
    mixed_artifact_array.npy
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
FS        = 256          # sampling rate (Hz)
DURATION  = 4.0          # seconds
N_CH      = 19           # channels (standard 10-20 system)
SEED      = 42

CH_LABELS = [
    "Fp1","Fp2","F7","F3","Fz","F4","F8",
    "T3","C3","Cz","C4","T4",
    "T5","P3","Pz","P4","T6",
    "O1","O2"
]

# Frontal gradient weights for ocular artifact (Fp highest, occipital lowest)
FRONTAL_W = np.array([1.00,1.00,0.80,0.70,0.65,0.70,0.80,
                       0.40,0.30,0.25,0.30,0.40,
                       0.15,0.10,0.08,0.10,0.15,
                       0.05,0.05])

# Temporal gradient weights for EMG artifact (T3/T4 highest)
TEMPORAL_W = np.array([0.10,0.10,0.50,0.30,0.20,0.30,0.50,
                        1.00,0.60,0.40,0.60,1.00,
                        0.80,0.40,0.30,0.40,0.80,
                        0.20,0.20])

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Signal synthesis ──────────────────────────────────────────────────────────

def make_clean_eeg(rng: np.random.Generator, n_samples: int) -> np.ndarray:
    """Return (N_CH × n_samples) physiological EEG (μV)."""
    t = np.linspace(0, DURATION, n_samples)
    eeg = np.zeros((N_CH, n_samples))
    bands = [
        (0.5, 4.0,  15.0),   # delta
        (4.0, 8.0,  12.0),   # theta
        (8.0, 13.0, 20.0),   # alpha
        (13.0,30.0,  8.0),   # beta
    ]
    for ch in range(N_CH):
        for f_lo, f_hi, amp in bands:
            fc = rng.uniform(f_lo, f_hi)
            phase = rng.uniform(0, 2 * np.pi)
            eeg[ch] += amp * np.sin(2 * np.pi * fc * t + phase)
        # 1/f pink noise floor
        freqs = np.fft.rfftfreq(n_samples, 1 / FS)
        freqs[0] = 1e-6
        pink = rng.standard_normal(len(freqs)) / np.sqrt(freqs)
        eeg[ch] += 5.0 * np.real(np.fft.irfft(pink, n=n_samples))
    return eeg


def inject_ocular(eeg: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Inject a half-cosine eye-blink at ~1.5 s (165 µV peak, 300 ms)."""
    blink_dur  = int(0.30 * FS)          # 300 ms
    blink_amp  = 165.0                   # µV
    onset      = int(1.5 * FS)
    t_blink    = np.arange(blink_dur)
    blink_wave = blink_amp * np.cos(np.pi * t_blink / blink_dur - np.pi / 2) ** 2
    noisy = eeg.copy()
    end   = min(onset + blink_dur, n_samples)
    noisy[:, onset:end] += np.outer(FRONTAL_W, blink_wave[: end - onset])
    return noisy


def inject_emg(eeg: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Inject Hann-windowed band-limited (20–100 Hz) EMG burst at ~2.0 s."""
    burst_dur  = int(0.80 * FS)          # 800 ms
    burst_amp  = 42.0                    # µV
    onset      = int(2.0 * FS)
    noise      = rng.standard_normal((burst_dur,)) * burst_amp
    # Band-pass 20-100 Hz via FFT masking
    freqs = np.fft.rfftfreq(burst_dur, 1 / FS)
    mask  = ((freqs >= 20) & (freqs <= 100)).astype(float)
    noise_bp = np.real(np.fft.irfft(np.fft.rfft(noise) * mask, n=burst_dur))
    # Hann window
    noise_bp *= np.hanning(burst_dur)
    noisy = eeg.copy()
    end   = min(onset + burst_dur, n_samples)
    noisy[:, onset:end] += np.outer(TEMPORAL_W, noise_bp[: end - onset])
    return noisy


def inject_mixed(eeg: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Inject both ocular (1.2 s) and EMG (2.5 s) artifacts simultaneously."""
    noisy = eeg.copy()
    # Blink at 1.2 s
    blink_dur = int(0.30 * FS)
    blink_amp = 145.0
    onset_b   = int(1.2 * FS)
    t_b       = np.arange(blink_dur)
    blink_wave = blink_amp * np.cos(np.pi * t_b / blink_dur - np.pi / 2) ** 2
    end_b = min(onset_b + blink_dur, n_samples)
    noisy[:, onset_b:end_b] += np.outer(FRONTAL_W, blink_wave[: end_b - onset_b])
    # EMG burst at 2.5 s
    burst_dur = int(0.60 * FS)
    burst_amp = 38.0
    onset_e   = int(2.5 * FS)
    noise     = rng.standard_normal(burst_dur) * burst_amp
    freqs     = np.fft.rfftfreq(burst_dur, 1 / FS)
    mask      = ((freqs >= 20) & (freqs <= 100)).astype(float)
    noise_bp  = np.real(np.fft.irfft(np.fft.rfft(noise) * mask, n=burst_dur))
    noise_bp *= np.hanning(burst_dur)
    end_e = min(onset_e + burst_dur, n_samples)
    noisy[:, onset_e:end_e] += np.outer(TEMPORAL_W, noise_bp[: end_e - onset_e])
    return noisy

# ── Plotting ──────────────────────────────────────────────────────────────────

def compute_snr(clean: np.ndarray, noisy: np.ndarray) -> np.ndarray:
    """Per-channel SNR (dB)."""
    signal_power   = np.mean(clean ** 2, axis=1)
    artifact_power = np.mean((noisy - clean) ** 2, axis=1)
    artifact_power = np.where(artifact_power < 1e-12, 1e-12, artifact_power)
    return 10 * np.log10(signal_power / artifact_power)


def plot_segment(data: np.ndarray, title: str, out_path: Path,
                 clean: np.ndarray | None = None) -> None:
    t = np.linspace(0, DURATION, data.shape[1])
    spacing = 150
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 10),
                                      gridspec_kw={"width_ratios": [3, 1]})
    fig.patch.set_facecolor("#1e1e2e")
    for ax in (ax_l, ax_r):
        ax.set_facecolor("#1e1e2e")

    colors = plt.cm.plasma(np.linspace(0.2, 0.9, N_CH))
    for i, (ch, col) in enumerate(zip(range(N_CH), colors)):
        offset = (N_CH - 1 - i) * spacing
        ax_l.plot(t, data[ch] + offset, color=col, lw=0.8, alpha=0.9)

    ax_l.set_yticks([(N_CH - 1 - i) * spacing for i in range(N_CH)])
    ax_l.set_yticklabels(CH_LABELS, color="white", fontsize=7)
    ax_l.set_xlabel("Time (s)", color="white")
    ax_l.set_title(title, color="white", fontsize=12, fontweight="bold")
    ax_l.tick_params(colors="white")
    for spine in ax_l.spines.values():
        spine.set_edgecolor("#444")

    # Per-channel SNR panel
    if clean is not None:
        snr_vals = compute_snr(clean, data)
        bar_colors = ["#ef4444" if s < 0 else "#22c55e" for s in snr_vals]
        ax_r.barh(range(N_CH), snr_vals[::-1], color=bar_colors[::-1], height=0.7)
        ax_r.axvline(0, color="white", lw=0.8, ls="--")
        ax_r.set_yticks(range(N_CH))
        ax_r.set_yticklabels(CH_LABELS[::-1], color="white", fontsize=7)
        ax_r.set_xlabel("SNR (dB)", color="white")
        ax_r.set_title("Per-channel SNR", color="white", fontsize=10)
        ax_r.tick_params(colors="white")
        for spine in ax_r.spines.values():
            spine.set_edgecolor("#444")
    else:
        ax_r.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [✓] Saved: {out_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rng       = np.random.default_rng(SEED)
    n_samples = int(FS * DURATION)

    print("Phase 2 — inject_artifacts.py")
    print(f"  Channels : {N_CH}  |  FS : {FS} Hz  |  Duration : {DURATION} s\n")

    clean = make_clean_eeg(rng, n_samples)
    ocular = inject_ocular(clean, n_samples, rng)
    emg    = inject_emg(clean, n_samples, rng)
    mixed  = inject_mixed(clean, n_samples, rng)

    # Save arrays
    np.save(RESULTS_DIR / "clean_eeg_array.npy",          clean)
    np.save(RESULTS_DIR / "ocular_artifact_array.npy",     ocular)
    np.save(RESULTS_DIR / "emg_artifact_array.npy",        emg)
    np.save(RESULTS_DIR / "mixed_artifact_array.npy",      mixed)

    # Save plots
    plot_segment(clean,  "Clean EEG Baseline (19-ch, 256 Hz)",
                 RESULTS_DIR / "clean_eeg_segment.png")
    plot_segment(ocular, "Ocular (Eye-Blink) Artifact — 165 µV peak",
                 RESULTS_DIR / "ocular_artifact_segment.png", clean)
    plot_segment(emg,    "EMG (Myogenic) Artifact — 42 µV burst",
                 RESULTS_DIR / "emg_artifact_segment.png",    clean)
    plot_segment(mixed,  "Mixed Artifact (Ocular + EMG simultaneous)",
                 RESULTS_DIR / "mixed_artifact_segment.png",  clean)

    print("\nAll arrays and plots saved to results/")


if __name__ == "__main__":
    main()
