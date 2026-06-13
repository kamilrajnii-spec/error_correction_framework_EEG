"""Dataset helpers for the Phase 2 DAE."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


EPSILON = 1e-8


class EEGPairDataset(Dataset):
    """PyTorch dataset for noisy EEG input and clean EEG target pairs."""

    def __init__(self, noisy: np.ndarray, clean: np.ndarray) -> None:
        noisy_array = _to_model_array(noisy)
        clean_array = _to_model_array(clean)

        if noisy_array.shape != clean_array.shape:
            raise ValueError("Noisy and clean arrays must have the same shape.")

        self.noisy = torch.tensor(noisy_array, dtype=torch.float32)
        self.clean = torch.tensor(clean_array, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.noisy.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.noisy[index], self.clean[index]


def normalize_pairs(
    noisy: np.ndarray,
    clean: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize clean/noisy pairs using noisy epoch statistics."""
    noisy_epochs = _to_2d(noisy)
    clean_epochs = _to_2d(clean)

    if noisy_epochs.shape != clean_epochs.shape:
        raise ValueError("Noisy and clean arrays must have the same shape.")

    means = noisy_epochs.mean(axis=1, keepdims=True)
    stds = noisy_epochs.std(axis=1, keepdims=True) + EPSILON

    normalized_noisy = (noisy_epochs - means) / stds
    normalized_clean = (clean_epochs - means) / stds

    return normalized_noisy, normalized_clean, means, stds


def normalize_epochs(epochs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize epochs and return normalized data, means, and standard deviations."""
    epoch_array = _to_2d(epochs)
    means = epoch_array.mean(axis=1, keepdims=True)
    stds = epoch_array.std(axis=1, keepdims=True) + EPSILON

    return (epoch_array - means) / stds, means, stds


def denormalize_epochs(
    normalized_epochs: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
) -> np.ndarray:
    """Invert normalize_epochs."""
    return normalized_epochs * stds + means


def split_arrays(
    noisy: np.ndarray,
    clean: np.ndarray,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Split pair arrays into train, validation, and test sets."""
    noisy_array = _to_2d(noisy)
    clean_array = _to_2d(clean)

    if noisy_array.shape != clean_array.shape:
        raise ValueError("Noisy and clean arrays must have the same shape.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(noisy_array.shape[0])

    train_end = int(train_fraction * len(indices))
    validation_end = train_end + int(validation_fraction * len(indices))

    train_indices = indices[:train_end]
    validation_indices = indices[train_end:validation_end]
    test_indices = indices[validation_end:]

    return {
        "train": (noisy_array[train_indices], clean_array[train_indices]),
        "validation": (
            noisy_array[validation_indices],
            clean_array[validation_indices],
        ),
        "test": (noisy_array[test_indices], clean_array[test_indices]),
    }


def _to_2d(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    array = np.squeeze(array)

    if array.size == 0:
        raise ValueError("Array cannot be empty.")
    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.ndim == 2:
        return array

    return array.reshape(array.shape[0], -1)


def _to_model_array(values: np.ndarray) -> np.ndarray:
    array = _to_2d(values)
    return array[:, np.newaxis, :]

