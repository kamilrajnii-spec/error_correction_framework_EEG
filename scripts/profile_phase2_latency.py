"""Measure Phase 2 per-segment inference time on real artifact pairs."""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_denoising.data_loading.load_eegdenoisenet import load_eegdenoisenet  # noqa: E402
from eeg_denoising.models.dae import load_dae_checkpoint  # noqa: E402
from eeg_denoising.pipeline.hybrid_pipeline import apply_dae_to_epochs  # noqa: E402
from eeg_denoising.preprocessing.artifact_mixing import (  # noqa: E402
    _as_2d_epochs,
    create_artifact_pairs,
)
from eeg_denoising.wavelet.dwt_denoising import denoise_epochs_dwt  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "results" / "phase2"
CHECKPOINT_PATH = OUTPUT_DIR / "dae_best_model.pt"
LATENCY_PATH = OUTPUT_DIR / "inference_time.csv"
SUMMARY_PATH = OUTPUT_DIR / "inference_time_summary.csv"


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.checkpoint.exists():
        print("DAE checkpoint not found. Run scripts/train_phase2_dae.py first.")
        return 1

    noisy = load_profile_noisy_epochs(args)
    model = load_dae_checkpoint(args.checkpoint, device=args.device)
    run_warm_up_passes(noisy[0], model, args)

    rows = []
    for segment_id, epoch in enumerate(noisy, start=1):
        segment = epoch.reshape(1, -1)

        start = time.perf_counter()
        wavelet = denoise_epochs_dwt(segment)
        wavelet_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        _ = apply_dae_to_epochs(wavelet, model, device=args.device)
        dae_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        wavelet_for_hybrid = denoise_epochs_dwt(segment)
        _ = apply_dae_to_epochs(wavelet_for_hybrid, model, device=args.device)
        hybrid_ms = (time.perf_counter() - start) * 1000.0

        rows.extend(
            [
                latency_row(segment_id, "wavelet_only", wavelet_ms),
                latency_row(segment_id, "dae_after_wavelet", dae_ms),
                latency_row(segment_id, "hybrid_wavelet_dae", hybrid_ms),
            ]
        )

    latency_frame = pd.DataFrame(rows)
    latency_frame.to_csv(LATENCY_PATH, index=False)
    make_latency_summary(latency_frame, args).to_csv(SUMMARY_PATH, index=False)

    print(f"Created {LATENCY_PATH}")
    print(f"Created {SUMMARY_PATH}")
    print("Latency values are measured by this script, not manually claimed.")

    return 0


def load_profile_noisy_epochs(args: argparse.Namespace):
    eegdenoisenet = load_eegdenoisenet(PROJECT_ROOT / "data" / "eegdenoisenet")
    clean = _as_2d_epochs(eegdenoisenet["clean_eeg"])[: args.segments]
    eog = _as_2d_epochs(eegdenoisenet["eog"])[: args.segments]
    emg = _as_2d_epochs(eegdenoisenet["emg"])[: args.segments]

    pairs = create_artifact_pairs(clean, eog, emg, snr_levels_db=[args.snr_db])
    for pair in pairs:
        if pair.artifact_type == args.artifact_type:
            return pair.noisy

    raise ValueError(f"Unknown artifact type: {args.artifact_type}")


def latency_row(segment_id: int, method: str, milliseconds: float) -> dict[str, object]:
    return {
        "segment_id": segment_id,
        "method": method,
        "milliseconds": round(milliseconds, 6),
    }


def run_warm_up_passes(epoch, model, args: argparse.Namespace) -> None:
    """Run untimed warm-up passes before measuring latency."""
    segment = epoch.reshape(1, -1)

    for _ in range(args.warm_up_passes):
        wavelet = denoise_epochs_dwt(segment)
        _ = apply_dae_to_epochs(wavelet, model, device=args.device)


def make_latency_summary(
    latency_frame: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Create summary latency statistics with hardware context."""
    cpu = platform.processor() or platform.machine()
    rows = []

    for method, group in latency_frame.groupby("method"):
        values = group["milliseconds"].to_numpy(dtype=float)
        std_ms = np.std(values, ddof=1) if values.size > 1 else 0.0
        rows.append(
            {
                "method": method,
                "mean_ms": round(float(np.mean(values)), 6),
                "std_ms": round(float(std_ms), 6),
                "median_ms": round(float(np.median(values)), 6),
                "p95_ms": round(float(np.percentile(values, 95)), 6),
                "n_segments": int(values.size),
                "device": args.device,
                "cpu": cpu,
            }
        )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--segments", type=int, default=20)
    parser.add_argument("--artifact-type", default="mixed", choices=["blink", "muscle", "mixed"])
    parser.add_argument("--snr-db", type=float, default=0.0)
    parser.add_argument("--warm-up-passes", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
