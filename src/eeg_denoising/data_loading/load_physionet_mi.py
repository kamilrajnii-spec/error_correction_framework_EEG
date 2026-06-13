"""Load one PhysioNet EEG Motor Movement/Imagery EDF file."""

from __future__ import annotations

from pathlib import Path


DEFAULT_PHYSIONET_MI_DIR = Path("data/physionet_mi")


class DatasetNotFoundError(FileNotFoundError):
    """Raised when required Phase 1 PhysioNet files are missing."""


def _dataset_missing_message(root: Path) -> str:
    return f"Dataset not found. Please download PhysioNet Motor Imagery EDF files into {root}/"


def find_first_edf(root: str | Path = DEFAULT_PHYSIONET_MI_DIR) -> Path:
    """Return the first EDF file found recursively."""
    root_path = Path(root)

    if not root_path.exists():
        raise DatasetNotFoundError(_dataset_missing_message(root_path))

    edf_files = sorted(root_path.rglob("*.edf"))
    if not edf_files:
        raise DatasetNotFoundError(_dataset_missing_message(root_path))

    return edf_files[0]


def load_first_physionet_edf(root: str | Path = DEFAULT_PHYSIONET_MI_DIR) -> dict[str, object]:
    """Load one EDF file and return simple metadata."""
    import mne

    edf_file = find_first_edf(root)
    raw = mne.io.read_raw_edf(edf_file, preload=False, verbose=False)

    sampling_rate = float(raw.info["sfreq"])
    duration_seconds = float(raw.n_times / sampling_rate)

    return {
        "file": edf_file,
        "raw": raw,
        "channels": len(raw.ch_names),
        "sampling_rate": sampling_rate,
        "duration_seconds": duration_seconds,
    }


def main() -> None:
    dataset = load_first_physionet_edf()

    print("PhysioNet found: yes")
    print("First EDF loaded: yes")
    print(f"EDF file: {dataset['file']}")
    print(f"Channels: {dataset['channels']}")
    print(f"Sampling rate: {dataset['sampling_rate']} Hz")
    print(f"Duration: {dataset['duration_seconds']:.2f} seconds")


if __name__ == "__main__":
    main()

