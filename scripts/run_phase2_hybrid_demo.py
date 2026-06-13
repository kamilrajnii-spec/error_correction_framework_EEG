"""Create the Phase 2 clean/noisy/wavelet/hybrid comparison plot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_denoising.data_loading.load_eegdenoisenet import load_eegdenoisenet  # noqa: E402
from eeg_denoising.evaluation.metrics import rmse, rrmse, snr, snr_gain  # noqa: E402
from eeg_denoising.pipeline.hybrid_pipeline import run_hybrid_pipeline  # noqa: E402
from eeg_denoising.preprocessing.artifact_mixing import (  # noqa: E402
    _as_2d_epochs,
    create_artifact_pairs,
)


OUTPUT_DIR = PROJECT_ROOT / "results" / "phase2"
PLOT_PATH = OUTPUT_DIR / "hybrid_comparison_plot.png"
METRICS_PATH = OUTPUT_DIR / "hybrid_demo_metrics.csv"
CHECKPOINT_PATH = OUTPUT_DIR / "dae_best_model.pt"


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.checkpoint.exists():
        print("DAE checkpoint not found. Run scripts/train_phase2_dae.py first.")
        return 1

    pair = load_demo_pair(args)
    outputs = run_hybrid_pipeline(
        pair.noisy,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    save_comparison_plot(
        clean=pair.clean[0],
        noisy=pair.noisy[0],
        wavelet=outputs["wavelet"][0],
        hybrid=outputs["hybrid"][0],
    )
    save_metrics(pair.clean, pair.noisy, outputs["wavelet"], outputs["hybrid"])

    print(f"Created {PLOT_PATH}")
    print(f"Created {METRICS_PATH}")

    return 0


def load_demo_pair(args: argparse.Namespace):
    eegdenoisenet = load_eegdenoisenet(PROJECT_ROOT / "data" / "eegdenoisenet")
    clean = _as_2d_epochs(eegdenoisenet["clean_eeg"])[:1]
    eog = _as_2d_epochs(eegdenoisenet["eog"])[:1]
    emg = _as_2d_epochs(eegdenoisenet["emg"])[:1]

    pairs = create_artifact_pairs(clean, eog, emg, snr_levels_db=[args.snr_db])
    for pair in pairs:
        if pair.artifact_type == args.artifact_type:
            return pair

    raise ValueError(f"Unknown artifact type: {args.artifact_type}")


def save_comparison_plot(clean, noisy, wavelet, hybrid) -> None:
    signals = [
        ("Clean reference", clean),
        ("Noisy input", noisy),
        ("Wavelet-only output", wavelet),
        ("Hybrid output", hybrid),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for axis, (title, signal) in zip(axes, signals):
        axis.plot(signal, linewidth=1.0)
        axis.set_title(title)
        axis.set_ylabel("Amplitude")

    axes[-1].set_xlabel("Sample")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)


def save_metrics(clean, noisy, wavelet, hybrid) -> None:
    rows = [
        {
            "method": "noisy_input",
            "snr_db": snr(clean, noisy),
            "snr_gain_db": 0.0,
            "rmse": rmse(clean, noisy),
            "rrmse": rrmse(clean, noisy),
        },
        {
            "method": "wavelet_only",
            "snr_db": snr(clean, wavelet),
            "snr_gain_db": snr_gain(clean, noisy, wavelet),
            "rmse": rmse(clean, wavelet),
            "rrmse": rrmse(clean, wavelet),
        },
        {
            "method": "hybrid",
            "snr_db": snr(clean, hybrid),
            "snr_gain_db": snr_gain(clean, noisy, hybrid),
            "rmse": rmse(clean, hybrid),
            "rrmse": rrmse(clean, hybrid),
        },
    ]

    pd.DataFrame(rows).to_csv(METRICS_PATH, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--artifact-type", default="mixed", choices=["blink", "muscle", "mixed"])
    parser.add_argument("--snr-db", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())

