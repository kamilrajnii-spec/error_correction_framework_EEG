from __future__ import annotations

import numpy as np

from eeg_denoising.ica.ica_baseline import is_ica_compatible, run_fastica_baseline


def test_ica_skips_single_channel_epoch_data() -> None:
    single_channel_epoch = np.ones((1, 512))

    result = run_fastica_baseline(single_channel_epoch, sfreq=256.0)

    assert not is_ica_compatible(single_channel_epoch)
    assert result.status == "skipped"
    assert result.cleaned_data is None

