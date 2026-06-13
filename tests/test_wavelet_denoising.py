from __future__ import annotations

import numpy as np

from eeg_denoising.wavelet.dwt_denoising import denoise_epochs_dwt


def test_dwt_denoising_preserves_epoch_shape() -> None:
    samples = np.linspace(0.0, 2.0 * np.pi, 512)
    noisy = np.sin(samples) + 0.2 * np.sin(samples * 30.0)
    epochs = noisy.reshape(1, -1)

    denoised = denoise_epochs_dwt(epochs)

    assert denoised.shape == epochs.shape


def test_dwt_denoising_changes_noisy_signal() -> None:
    samples = np.linspace(0.0, 2.0 * np.pi, 512)
    noisy = np.sin(samples) + 0.3 * np.sin(samples * 40.0)

    denoised = denoise_epochs_dwt(noisy)

    assert not np.allclose(denoised[0], noisy)

