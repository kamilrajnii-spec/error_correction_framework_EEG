"""Run Phase 2 DWT baseline and honest ICA compatibility check."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_denoising.data_loading.load_eegdenoisenet import load_eegdenoisenet  # noqa: E402
from eeg_denoising.data_loading.load_physionet_mi import (  # noqa: E402
    DatasetNotFoundError as PhysioNetDatasetNotFoundError,
)
from eeg_denoising.data_loading.load_physionet_mi import load_first_physionet_edf  # noqa: E402
from eeg_denoising.evaluation.metrics import rmse, rrmse, snr, snr_gain  # noqa: E402
from eeg_denoising.ica.ica_baseline import (  # noqa: E402
    run_fastica_baseline,
    run_fastica_on_raw,
)
from eeg_denoising.preprocessing.artifact_mixing import (  # noqa: E402
    _as_2d_epochs,
    create_artifact_pairs,
)
from eeg_denoising.wavelet.dwt_denoising import denoise_epochs_dwt  # noqa: E402


OUTPUT_PATH = PROJECT_ROOT / "results" / "phase2" / "wavelet_ica_baseline_table.csv"
PHYSIONET_ICA_PATH = PROJECT_ROOT / "results" / "phase2" / "physionet_ica_baseline_table.csv"


def main() -> int:
    args = parse_args()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    eegdenoisenet = load_eegdenoisenet(PROJECT_ROOT / "data" / "eegdenoisenet")
    clean = _as_2d_epochs(eegdenoisenet["clean_eeg"])[: args.max_clean_epochs]
    eog = _as_2d_epochs(eegdenoisenet["eog"])[: args.max_clean_epochs]
    emg = _as_2d_epochs(eegdenoisenet["emg"])[: args.max_clean_epochs]

    pairs = create_artifact_pairs(clean, eog, emg, snr_levels_db=args.snr_levels)

    rows = []
    for pair in pairs:
        wavelet_output = denoise_epochs_dwt(pair.noisy)
        rows.append(
            {
                "dataset": "EEGdenoiseNet",
                "method": "DWT db4 level 4 soft threshold",
                "artifact_type": pair.artifact_type,
                "target_snr_db": pair.target_snr_db,
                "input_snr_db": round(snr(pair.clean, pair.noisy), 6),
                "output_snr_db": round(snr(pair.clean, wavelet_output), 6),
                "snr_gain_db": round(snr_gain(pair.clean, pair.noisy, wavelet_output), 6),
                "rmse": round(rmse(pair.clean, wavelet_output), 6),
                "rrmse": round(rrmse(pair.clean, wavelet_output), 6),
                "epochs_used": pair.clean.shape[0],
                "status": "applied",
                "notes": "Wavelet baseline uses real EEGdenoiseNet artifact pairs.",
            }
        )

    ica_skip = run_fastica_baseline(clean[:1], sfreq=args.eegdenoisenet_sfreq)
    rows.append(
        {
            "dataset": "EEGdenoiseNet",
            "method": "ICA FastICA",
            "artifact_type": "not_applicable",
            "target_snr_db": "",
            "input_snr_db": "",
            "output_snr_db": "",
            "snr_gain_db": "",
            "rmse": "",
            "rrmse": "",
            "epochs_used": clean.shape[0],
            "status": ica_skip.status,
            "notes": ica_skip.message,
        }
    )

    physionet_rows = []
    if not args.skip_physionet_ica:
        physionet_row = run_physionet_ica_row(args)
        rows.append(physionet_row)
        physionet_rows.append(physionet_row)

    pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)
    if physionet_rows:
        pd.DataFrame(physionet_rows).to_csv(PHYSIONET_ICA_PATH, index=False)
    print(f"Created {OUTPUT_PATH}")
    if physionet_rows:
        print(f"Created {PHYSIONET_ICA_PATH}")

    return 0


def run_physionet_ica_row(args: argparse.Namespace) -> dict[str, object]:
    """Run ICA on PhysioNet if an EDF is present."""
    try:
        physionet = load_first_physionet_edf(PROJECT_ROOT / "data" / "physionet_mi")
    except PhysioNetDatasetNotFoundError as error:
        return {
            "dataset": "PhysioNet Motor Imagery",
            "method": "ICA FastICA",
            "artifact_type": "not_applicable",
            "target_snr_db": "",
            "input_snr_db": "",
            "output_snr_db": "",
            "snr_gain_db": "",
            "rmse": "",
            "rrmse": "",
            "epochs_used": "",
            "subjects_processed": 0,
            "subject_ids": "",
            "channels": "",
            "samples_processed": "",
            "ica_components_estimated": "",
            "artifact_components_identified": "",
            "excluded_components": "",
            "processing_time_seconds": "",
            "status": "skipped",
            "notes": str(error),
        }

    result = run_fastica_on_raw(
        physionet["raw"],
        duration_seconds=args.physionet_ica_seconds,
    )
    edf_file = Path(physionet["file"])
    subject_id = edf_file.parent.name

    return {
        "dataset": "PhysioNet Motor Imagery",
        "method": "ICA FastICA",
        "artifact_type": "real_raw_eeg_no_clean_reference",
        "target_snr_db": "",
        "input_snr_db": "",
        "output_snr_db": "",
        "snr_gain_db": "",
        "rmse": "",
        "rrmse": "",
        "epochs_used": "",
        "subjects_processed": 1,
        "subject_ids": subject_id,
        "channels": physionet["channels"],
        "samples_processed": result.n_samples,
        "ica_components_estimated": result.n_components,
        "artifact_components_identified": len(result.excluded_components),
        "excluded_components": str(result.excluded_components),
        "processing_time_seconds": round(result.processing_time_seconds, 6),
        "status": result.status,
        "notes": (
            f"{result.message} Excluded components: "
            f"{result.excluded_components}. Clean-reference metrics are not "
            "reported because PhysioNet MI does not provide clean/noisy pairs."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-clean-epochs", type=int, default=300)
    parser.add_argument("--snr-levels", type=float, nargs="+", default=[-5.0, 0.0, 5.0])
    parser.add_argument("--eegdenoisenet-sfreq", type=float, default=256.0)
    parser.add_argument("--physionet-ica-seconds", type=float, default=10.0)
    parser.add_argument("--skip-physionet-ica", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
