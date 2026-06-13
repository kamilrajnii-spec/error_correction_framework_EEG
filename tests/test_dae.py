from __future__ import annotations

import torch

from eeg_denoising.models.dae import ConvDAE, count_parameters


def test_dae_keeps_expected_input_output_shape() -> None:
    model = ConvDAE()
    x = torch.zeros(2, 1, 512)

    y = model(x)

    assert y.shape == x.shape


def test_dae_parameter_count_is_computed_from_model() -> None:
    model = ConvDAE()

    assert count_parameters(model) > 0

