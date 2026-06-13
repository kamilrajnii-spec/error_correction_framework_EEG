"""Create small Phase 1 artifact-pair outputs from real datasets."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_denoising.data_loading.load_eegdenoisenet import (  # noqa: E402
    DatasetNotFoundError as EEGDatasetNotFoundError,
)
from eeg_denoising.data_loading.load_eegdenoisenet import load_eegdenoisenet  # noqa: E402
from eeg_denoising.data_loading.load_physionet_mi import (  # noqa: E402
    DatasetNotFoundError as PhysioNetDatasetNotFoundError,
)
from eeg_denoising.data_loading.load_physionet_mi import load_first_physionet_edf  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "results" / "phase1"
SNR_LEVELS_DB = (-5.0, 0.0, 5.0)
N_EXAMPLE_EPOCHS = 5


def _write_example_plot(pairs, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    selected_pairs = [
        pair
        for pair in pairs
        if pair.target_snr_db == 0.0
        and pair.artifact_type in {"blink", "muscle", "mixed"}
    ]

    fig, axes = plt.subplots(len(selected_pairs), 1, figsize=(10, 7), sharex=True)
    if len(selected_pairs) == 1:
        axes = [axes]

    for axis, pair in zip(axes, selected_pairs):
        axis.plot(pair.clean[0], label="clean", linewidth=1.2)
        axis.plot(pair.noisy[0], label="noisy", linewidth=1.0, alpha=0.8)
        axis.set_title(f"{pair.artifact_type} artifact at {pair.target_snr_db:g} dB")
        axis.set_ylabel("Amplitude")
        axis.legend(loc="upper right")

    axes[-1].set_xlabel("Sample")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        eegdenoisenet = load_eegdenoisenet(PROJECT_ROOT / "data" / "eegdenoisenet")
    except EEGDatasetNotFoundError as error:
        print(error)
        return 1

    try:
        physionet = load_first_physionet_edf(PROJECT_ROOT / "data" / "physionet_mi")
    except PhysioNetDatasetNotFoundError as error:
        print(error)
        return 1

    import pandas as pd

    from eeg_denoising.evaluation.metrics import rmse, rrmse, snr
    from eeg_denoising.preprocessing.artifact_mixing import (
        _as_2d_epochs,
        create_artifact_pairs,
    )

    clean = _as_2d_epochs(eegdenoisenet["clean_eeg"])[:N_EXAMPLE_EPOCHS]
    eog = _as_2d_epochs(eegdenoisenet["eog"])[:N_EXAMPLE_EPOCHS]
    emg = _as_2d_epochs(eegdenoisenet["emg"])[:N_EXAMPLE_EPOCHS]

    pairs = create_artifact_pairs(clean, eog, emg, snr_levels_db=SNR_LEVELS_DB)

    dataset_rows: list[dict] = [
        {
            "dataset": "EEGdenoiseNet",
            "status": "loaded",
            "clean_eeg_shape": str(eegdenoisenet["clean_eeg"].shape),
            "eog_shape": str(eegdenoisenet["eog"].shape),
            "emg_shape": str(eegdenoisenet["emg"].shape),
            "file": "",
            "channels": "",
            "sampling_rate_hz": "",
            "duration_seconds": "",
        },
        {
            "dataset": "PhysioNet Motor Imagery",
            "status": "loaded",
            "clean_eeg_shape": "",
            "eog_shape": "",
            "emg_shape": "",
            "file": str(Path(physionet["file"]).relative_to(PROJECT_ROOT)),
            "channels": physionet["channels"],
            "sampling_rate_hz": physionet["sampling_rate"],
            "duration_seconds": round(float(physionet["duration_seconds"]), 2),
        },
    ]

    pair_rows = []
    for pair_id, pair in enumerate(pairs, start=1):
        pair_rows.append(
            {
                "pair_id": pair_id,
                "artifact_type": pair.artifact_type,
                "target_snr_db": pair.target_snr_db,
                "measured_snr_db": round(snr(pair.clean, pair.noisy), 6),
                "rmse_clean_vs_noisy": round(rmse(pair.clean, pair.noisy), 6),
                "rrmse_clean_vs_noisy": round(rrmse(pair.clean, pair.noisy), 6),
                "epochs_used": pair.clean.shape[0],
                "samples_per_epoch": pair.clean.shape[1],
            }
        )

    dataset_check_path = OUTPUT_DIR / "phase1_dataset_check.csv"
    artifact_pairs_path = OUTPUT_DIR / "phase1_artifact_pairs.csv"
    plot_path = OUTPUT_DIR / "example_clean_noisy_plot.png"

    pd.DataFrame(dataset_rows).to_csv(dataset_check_path, index=False)
    pd.DataFrame(pair_rows).to_csv(artifact_pairs_path, index=False)
    _write_example_plot(pairs, plot_path)

    print(f"Created {dataset_check_path}")
    print(f"Created {artifact_pairs_path}")
    print(f"Created {plot_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
