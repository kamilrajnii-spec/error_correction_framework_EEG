"""Simple EEG denoising metrics used in Phase 1."""

from __future__ import annotations

import numpy as np


EPSILON = 1e-12


def _as_float_array(values: np.ndarray | list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("Metric input cannot be empty.")
    return array


def snr(clean: np.ndarray, observed: np.ndarray) -> float:
    """Return SNR in dB between clean EEG and an observed/noisy signal."""
    clean_array = _as_float_array(clean)
    observed_array = _as_float_array(observed)

    if clean_array.shape != observed_array.shape:
        raise ValueError("clean and observed must have the same shape.")

    noise = observed_array - clean_array
    signal_power = np.mean(clean_array**2)
    noise_power = np.mean(noise**2)

    if noise_power <= EPSILON:
        return float("inf")

    if signal_power <= EPSILON:
        return float("-inf")

    return float(10.0 * np.log10(signal_power / noise_power))


def snr_gain(clean: np.ndarray, noisy: np.ndarray, denoised: np.ndarray) -> float:
    """Return improvement in SNR from noisy input to denoised output."""
    return snr(clean, denoised) - snr(clean, noisy)


def rmse(clean: np.ndarray, observed: np.ndarray) -> float:
    """Return root mean squared error."""
    clean_array = _as_float_array(clean)
    observed_array = _as_float_array(observed)

    if clean_array.shape != observed_array.shape:
        raise ValueError("clean and observed must have the same shape.")

    return float(np.sqrt(np.mean((observed_array - clean_array) ** 2)))


def rrmse(clean: np.ndarray, observed: np.ndarray) -> float:
    """Return relative RMSE using the clean signal energy as denominator."""
    clean_array = _as_float_array(clean)
    observed_array = _as_float_array(observed)

    if clean_array.shape != observed_array.shape:
        raise ValueError("clean and observed must have the same shape.")

    denominator = np.sqrt(np.mean(clean_array**2))
    if denominator <= EPSILON:
        raise ValueError("RRMSE is undefined when clean signal energy is zero.")

    return float(rmse(clean_array, observed_array) / denominator)
