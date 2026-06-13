"""Verify that Phase 1 datasets are present and loadable."""

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


def main() -> int:
    try:
        eegdenoisenet = load_eegdenoisenet(PROJECT_ROOT / "data" / "eegdenoisenet")
    except EEGDatasetNotFoundError as error:
        print("EEGdenoiseNet found: no")
        print(error)
        return 1

    print("EEGdenoiseNet found: yes")
    print(f"Clean EEG shape: {eegdenoisenet['clean_eeg'].shape}")
    print(f"EOG shape: {eegdenoisenet['eog'].shape}")
    print(f"EMG shape: {eegdenoisenet['emg'].shape}")
    print()

    try:
        physionet = load_first_physionet_edf(PROJECT_ROOT / "data" / "physionet_mi")
    except PhysioNetDatasetNotFoundError as error:
        print("PhysioNet found: no")
        print(error)
        return 1

    print("PhysioNet found: yes")
    print("First EDF loaded: yes")
    print(f"Channels: {physionet['channels']}")
    print(f"Sampling rate: {physionet['sampling_rate']} Hz")
    print(f"Duration: {physionet['duration_seconds']:.2f} seconds")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

