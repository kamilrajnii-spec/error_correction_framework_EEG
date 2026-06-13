from __future__ import annotations

import numpy as np
import pytest

from eeg_denoising.evaluation.metrics import snr
from eeg_denoising.preprocessing.artifact_mixing import (
    align_artifact_to_clean,
    create_artifact_pairs,
    mix_artifact,
)


def test_mix_artifact_hits_target_snr() -> None:
    samples = np.linspace(0.0, 2.0 * np.pi, 512)
    clean = np.sin(samples).reshape(1, -1)
    artifact = np.cos(samples).reshape(1, -1)

    noisy, added_artifact = mix_artifact(clean, artifact, target_snr_db=5.0)

    assert noisy.shape == clean.shape
    assert added_artifact.shape == clean.shape
    assert snr(clean, noisy) == pytest.approx(5.0, abs=1e-6)


def test_create_artifact_pairs_creates_all_phase1_conditions() -> None:
    samples = np.linspace(0.0, 2.0 * np.pi, 512)
    clean = np.vstack([np.sin(samples), np.sin(samples + 0.5)])
    eog = np.vstack([np.cos(samples), np.cos(samples + 0.5)])
    emg = np.vstack([np.sin(samples * 8.0), np.sin(samples * 9.0)])

    pairs = create_artifact_pairs(clean, eog, emg, snr_levels_db=(-5.0, 0.0, 5.0))

    assert len(pairs) == 9
    assert {pair.artifact_type for pair in pairs} == {"blink", "muscle", "mixed"}
    assert {pair.target_snr_db for pair in pairs} == {-5.0, 0.0, 5.0}


def test_align_artifact_to_clean_repeats_or_crops_shape_only() -> None:
    clean = np.ones((3, 512))
    artifact = np.arange(128).reshape(1, -1)

    aligned = align_artifact_to_clean(clean, artifact)

    assert aligned.shape == clean.shape

