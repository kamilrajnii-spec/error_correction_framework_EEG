"""Load EEGdenoiseNet clean EEG, EOG, and EMG epochs."""

from __future__ import annotations

from pathlib import Path


DEFAULT_EEGDENOISENET_DIR = Path("data/eegdenoisenet")


class DatasetNotFoundError(FileNotFoundError):
    """Raised when required Phase 1 dataset files are missing."""


def _dataset_missing_message(root: Path) -> str:
    return f"Dataset not found. Please download EEGdenoiseNet into {root}/"


def _candidate_files(root: Path) -> list[Path]:
    allowed_suffixes = {".npy", ".mat", ".csv", ".txt"}
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed_suffixes
    ]


def _find_file(root: Path, include_words: tuple[str, ...], exclude_words: tuple[str, ...]) -> Path:
    files = _candidate_files(root)
    matches: list[Path] = []

    for path in files:
        name = path.name.lower()
        if all(word in name for word in include_words) and not any(
            word in name for word in exclude_words
        ):
            matches.append(path)

    if not matches:
        raise DatasetNotFoundError(_dataset_missing_message(root))

    return sorted(matches)[0]


def _load_mat_array(path: Path, preferred_word: str) -> np.ndarray:
    import numpy as np
    from scipy.io import loadmat

    mat_data = loadmat(path)
    arrays = []

    for name, value in mat_data.items():
        if name.startswith("__"):
            continue
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
            arrays.append((name.lower(), value))

    if not arrays:
        raise ValueError(f"No numeric arrays found in {path}")

    preferred = [item for item in arrays if preferred_word in item[0]]
    selected = preferred[0] if preferred else max(arrays, key=lambda item: item[1].size)
    return np.asarray(selected[1], dtype=float)


def _load_array(path: Path, preferred_word: str) -> np.ndarray:
    import numpy as np

    suffix = path.suffix.lower()

    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix == ".mat":
        return _load_mat_array(path, preferred_word)
    if suffix in {".csv", ".txt"}:
        delimiter = "," if suffix == ".csv" else None
        return np.loadtxt(path, delimiter=delimiter)

    raise ValueError(f"Unsupported file type: {path}")


def load_eegdenoisenet(root: str | Path = DEFAULT_EEGDENOISENET_DIR) -> dict[str, object]:
    """Load clean EEG, EOG blink, and EMG muscle epochs from EEGdenoiseNet."""
    root_path = Path(root)

    if not root_path.exists():
        raise DatasetNotFoundError(_dataset_missing_message(root_path))

    clean_file = _find_file(root_path, include_words=("eeg",), exclude_words=("eog", "emg"))
    eog_file = _find_file(root_path, include_words=("eog",), exclude_words=())
    emg_file = _find_file(root_path, include_words=("emg",), exclude_words=())

    return {
        "clean_eeg": _load_array(clean_file, "eeg"),
        "eog": _load_array(eog_file, "eog"),
        "emg": _load_array(emg_file, "emg"),
        "files": {
            "clean_eeg": clean_file,
            "eog": eog_file,
            "emg": emg_file,
        },
    }


def main() -> None:
    dataset = load_eegdenoisenet()

    print("EEGdenoiseNet found: yes")
    print(f"Clean EEG shape: {dataset['clean_eeg'].shape}")
    print(f"EOG shape: {dataset['eog'].shape}")
    print(f"EMG shape: {dataset['emg'].shape}")
    print(f"Clean EEG file: {dataset['files']['clean_eeg']}")
    print(f"EOG file: {dataset['files']['eog']}")
    print(f"EMG file: {dataset['files']['emg']}")


if __name__ == "__main__":
    main()
