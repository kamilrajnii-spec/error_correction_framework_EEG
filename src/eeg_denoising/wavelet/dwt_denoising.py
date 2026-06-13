"""DWT wavelet denoising baseline for Phase 2.

The supervision plan asks for db4, level 4, and soft thresholding. This
module keeps those choices explicit so they are easy to explain.
"""

from __future__ import annotations

import numpy as np
import pywt


DEFAULT_WAVELET = "db4"
# For 512-sample EEGdenoiseNet epochs treated at 256 Hz, level 4 gives
# approximate DWT bands of D1 64-128 Hz, D2 32-64 Hz, D3 16-32 Hz,
# D4 8-16 Hz, and A4 0-8 Hz. This keeps high-frequency muscle activity
# in detail bands while preserving low-frequency EEG trends in A4.
DEFAULT_LEVEL = 4
DEFAULT_MODE = "soft"
EPSILON = 1e-12


def _as_2d_epochs(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    array = np.squeeze(array)

    if array.size == 0:
        raise ValueError("Input signal cannot be empty.")
    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.ndim == 2:
        return array

    return array.reshape(array.shape[0], -1)


def _safe_level(n_samples: int, wavelet_name: str, requested_level: int) -> int:
    wavelet = pywt.Wavelet(wavelet_name)
    max_level = pywt.dwt_max_level(n_samples, wavelet.dec_len)
    return max(1, min(requested_level, max_level))


def _universal_threshold(detail_coefficients: np.ndarray) -> float:
    sigma = np.median(np.abs(detail_coefficients)) / 0.6745
    n_values = max(detail_coefficients.size, 2)
    return float(sigma * np.sqrt(2.0 * np.log(n_values)))


def denoise_epoch_dwt(
    epoch: np.ndarray,
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
    mode: str = DEFAULT_MODE,
) -> np.ndarray:
    """Denoise one EEG epoch with DWT soft thresholding."""
    signal = np.asarray(epoch, dtype=float).reshape(-1)
    safe_level = _safe_level(signal.size, wavelet, level)
    coefficients = pywt.wavedec(signal, wavelet, level=safe_level)

    threshold = _universal_threshold(coefficients[-1])
    denoised_coefficients = [coefficients[0]]

    for detail in coefficients[1:]:
        denoised_coefficients.append(pywt.threshold(detail, threshold, mode=mode))

    reconstructed = pywt.waverec(denoised_coefficients, wavelet)
    return reconstructed[: signal.size]


def denoise_epochs_dwt(
    epochs: np.ndarray,
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
    mode: str = DEFAULT_MODE,
) -> np.ndarray:
    """Denoise one or more EEG epochs with DWT soft thresholding."""
    epoch_array = _as_2d_epochs(epochs)
    denoised = [
        denoise_epoch_dwt(epoch, wavelet=wavelet, level=level, mode=mode)
        for epoch in epoch_array
    ]

    return np.asarray(denoised, dtype=float)
