"""Tests for the DPDFNet stable-whisper adapter."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
import torch

from halospeak.audio import dpdfnet_denoiser as denoiser


class _FakeModel:
    """Minimal DPDFNet model double for adapter tests."""

    samplerate = 16_000

    def __init__(self) -> None:
        """Initialize the fake model."""
        self.calls: list[tuple[int, int]] = []

    def run(self, samples: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        """Return a constant denoised waveform."""
        self.calls.append((samples.shape[0], sample_rate))
        return np.full(samples.shape[0], 0.5, dtype=np.float32), sample_rate


def test_dpdfnet_audio__denoises_stereo_tensor_to_mono() -> None:
    """dpdfnet_audio should downmix and run the configured model."""
    model = _FakeModel()

    output = denoiser.dpdfnet_audio(
        torch.zeros(2, 16_000),
        input_sr=16_000,
        model=cast(denoiser.DPDFNetDenoiser, model),
    )

    assert output.shape == (16_000,)
    assert output.min().item() == pytest.approx(0.5)
    assert output.max().item() == pytest.approx(0.5)
    assert model.calls == [(16_000, 16_000)]


def test_dpdfnet_audio__requires_input_sample_rate() -> None:
    """Tensor inputs must specify their sample rate."""
    model = _FakeModel()

    with pytest.raises(ValueError, match="input_sr"):
        denoiser.dpdfnet_audio(
            torch.zeros(16_000),
            model=cast(denoiser.DPDFNetDenoiser, model),
        )


def test_register_dpdfnet_denoiser__adds_stable_whisper_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registration should add the dpdfnet backend exactly once."""
    stable_audio = pytest.importorskip("stable_whisper.audio")

    monkeypatch.delitem(stable_audio.SUPPORTED_DENOISERS, "dpdfnet", raising=False)

    denoiser.register_dpdfnet_denoiser()
    denoiser.register_dpdfnet_denoiser()

    assert "dpdfnet" in stable_audio.SUPPORTED_DENOISERS
