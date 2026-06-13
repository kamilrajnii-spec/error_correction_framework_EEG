"""Honest ICA baseline implementation for compatible multi-channel EEG.

EEGdenoiseNet is usually stored as single-channel epochs. ICA needs
multi-channel EEG, so the baseline is skipped for incompatible epoch data
instead of producing fake metrics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.stats import kurtosis


@dataclass(frozen=True)
class ICAResult:
    """Result returned by the ICA baseline."""

    cleaned_data: np.ndarray | None
    excluded_components: list[int]
    n_components: int
    n_channels: int
    n_samples: int
    processing_time_seconds: float
    status: str
    message: str


def is_ica_compatible(data: np.ndarray, min_channels: int = 2) -> bool:
    """Return True when data looks like channels x samples."""
    array = np.asarray(data)

    if array.ndim != 2:
        return False

    n_channels, n_samples = array.shape
    return n_channels >= min_channels and n_samples > n_channels


def choose_artifact_components(
    source_data: np.ndarray,
    z_threshold: float = 3.0,
    max_components: int = 3,
) -> list[int]:
    """Choose ICA components using simple variance and kurtosis heuristics."""
    if source_data.ndim != 2:
        raise ValueError("source_data must have shape components x samples.")

    component_variance = np.var(source_data, axis=1)
    component_kurtosis = np.abs(kurtosis(source_data, axis=1, fisher=True))

    variance_z = _zscore(component_variance)
    kurtosis_z = _zscore(component_kurtosis)
    score = np.maximum(variance_z, kurtosis_z)

    candidates = np.where(score >= z_threshold)[0]
    ordered = sorted(candidates, key=lambda index: score[index], reverse=True)

    return [int(index) for index in ordered[:max_components]]


def run_fastica_baseline(
    data: np.ndarray,
    sfreq: float,
    ch_names: list[str] | None = None,
    n_components: int | None = None,
    random_state: int = 42,
    max_iter: int = 1000,
) -> ICAResult:
    """Apply MNE FastICA when data is compatible multi-channel EEG."""
    if not is_ica_compatible(data):
        array = np.asarray(data)
        return ICAResult(
            cleaned_data=None,
            excluded_components=[],
            n_components=0,
            n_channels=int(array.shape[0]) if array.ndim == 2 else 0,
            n_samples=int(array.shape[1]) if array.ndim == 2 else 0,
            processing_time_seconds=0.0,
            status="skipped",
            message=(
                "ICA requires multi-channel data. EEGdenoiseNet epoch pairs are "
                "single-channel, so ICA is not used as the primary baseline."
            ),
        )

    import mne
    from mne.preprocessing import ICA

    data_array = np.asarray(data, dtype=float)
    n_channels = data_array.shape[0]

    if ch_names is None:
        ch_names = [f"EEG{index + 1:03d}" for index in range(n_channels)]

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data_array, info, verbose=False)
    raw_for_ica = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)

    component_count = n_components or min(5, n_channels - 1)
    if component_count < 1:
        return ICAResult(
            cleaned_data=None,
            excluded_components=[],
            n_components=0,
            n_channels=n_channels,
            n_samples=int(data_array.shape[1]),
            processing_time_seconds=0.0,
            status="skipped",
            message="Not enough channels for ICA component estimation.",
        )

    start_time = time.perf_counter()
    ica = ICA(
        n_components=component_count,
        method="fastica",
        random_state=random_state,
        fit_params={"tol": 0.01},
        max_iter=max_iter,
    )
    ica.fit(raw_for_ica, verbose=False)

    sources = ica.get_sources(raw_for_ica).get_data()
    excluded = choose_artifact_components(sources)

    cleaned_raw = raw.copy()
    if excluded:
        ica.exclude = excluded
        ica.apply(cleaned_raw, verbose=False)
    processing_time = time.perf_counter() - start_time

    return ICAResult(
        cleaned_data=cleaned_raw.get_data(),
        excluded_components=excluded,
        n_components=int(component_count),
        n_channels=n_channels,
        n_samples=int(data_array.shape[1]),
        processing_time_seconds=float(processing_time),
        status="applied",
        message="ICA applied with FastICA and variance/kurtosis heuristics.",
    )


def run_fastica_on_raw(
    raw,
    duration_seconds: float = 20.0,
    random_state: int = 42,
) -> ICAResult:
    """Run the ICA baseline on an MNE Raw object after cropping for speed."""
    raw_copy = raw.copy().pick("eeg")

    if duration_seconds > 0:
        max_time = min(duration_seconds, raw_copy.times[-1])
        raw_copy.crop(tmin=0.0, tmax=max_time)

    data = raw_copy.get_data()
    return run_fastica_baseline(
        data,
        sfreq=float(raw_copy.info["sfreq"]),
        ch_names=raw_copy.ch_names,
        random_state=random_state,
    )


def _zscore(values: np.ndarray) -> np.ndarray:
    std = float(np.std(values))
    if std <= 1e-12:
        return np.zeros_like(values, dtype=float)

    return (values - np.mean(values)) / std
