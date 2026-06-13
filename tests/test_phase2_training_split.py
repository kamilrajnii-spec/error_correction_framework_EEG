from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "train_phase2_dae.py"


def load_training_script():
    spec = importlib.util.spec_from_file_location("train_phase2_dae", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_clean_epoch_split_has_no_overlap() -> None:
    training_script = load_training_script()

    splits = training_script.split_clean_epoch_indices(
        n_epochs=100,
        seed=42,
        train_fraction=0.70,
        validation_fraction=0.15,
    )

    train = set(splits["train"])
    validation = set(splits["validation"])
    test = set(splits["test"])

    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)
    assert len(train | validation | test) == 100


def test_split_pairs_use_wavelet_output_as_dae_input() -> None:
    training_script = load_training_script()

    samples = np.linspace(0.0, 2.0 * np.pi, 512)
    clean = np.vstack([np.sin(samples), np.sin(samples + 0.25)])
    eog = np.vstack([np.cos(samples), np.cos(samples + 0.25)])
    emg = np.vstack([np.sin(samples * 20.0), np.sin(samples * 25.0)])

    split_data, manifest = training_script.create_split_pairs(
        split_name="train",
        clean=clean,
        eog=eog,
        emg=emg,
        clean_indices=np.array([0, 1]),
        snr_levels=[0.0],
    )

    assert split_data["raw_noisy"].shape == split_data["dae_input"].shape
    assert split_data["clean"].shape == split_data["dae_input"].shape
    assert not np.allclose(split_data["raw_noisy"], split_data["dae_input"])
    assert {row["dae_input"] for row in manifest} == {"dwt_wavelet_output"}

