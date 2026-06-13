from __future__ import annotations

import numpy as np
import pytest

from eeg_denoising.evaluation.metrics import rmse, rrmse, snr, snr_gain


def test_snr_is_zero_db_when_signal_and_noise_power_match() -> None:
    clean = np.array([1.0, 1.0, 1.0, 1.0])
    noisy = np.array([2.0, 0.0, 2.0, 0.0])

    assert snr(clean, noisy) == pytest.approx(0.0)


def test_snr_gain_uses_code_calculated_values() -> None:
    clean = np.array([1.0, 1.0, 1.0, 1.0])
    noisy = np.array([2.0, 0.0, 2.0, 0.0])
    denoised = np.array([1.5, 0.5, 1.5, 0.5])

    assert snr_gain(clean, noisy, denoised) == pytest.approx(6.020599913279624)


def test_rmse_and_rrmse() -> None:
    clean = np.array([1.0, 1.0, 1.0, 1.0])
    observed = np.array([2.0, 0.0, 2.0, 0.0])

    assert rmse(clean, observed) == pytest.approx(1.0)
    assert rrmse(clean, observed) == pytest.approx(1.0)


def test_perfect_observation_has_infinite_snr() -> None:
    clean = np.array([1.0, 2.0, 3.0])

    assert snr(clean, clean) == float("inf")

