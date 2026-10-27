"""DPDFNet speech-enhancement adapter for stable-whisper.

DPDFNet is an Apache-2.0 speech denoiser. This module exposes a
stable-whisper-compatible denoiser so the existing ``transcribe_any`` path can
use it without vendoring or patching stable-whisper itself.
"""

from __future__ import annotations

import importlib.util
import logging
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from insanely_fast_whisper_rocm.utils import constants

logger = logging.getLogger(__name__)

_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: DPDFNetDenoiser | None = None


def is_dpdfnet_available() -> bool:
    """Return whether DPDFNet can be used in this environment.

    Returns:
        True when ``sherpa_onnx`` is installed and the configured model exists.
    """
    if constants.DENOISE_BACKEND.lower() != "dpdfnet":
        return False
    return (
        importlib.util.find_spec("sherpa_onnx") is not None
        and Path(constants.DENOISE_MODEL_PATH).is_file()
    )


class DPDFNetDenoiser:
    """Thin wrapper around sherpa-onnx's offline DPDFNet denoiser."""

    samplerate = 16_000

    def __init__(self, model_path: Path) -> None:
        """Initialize the DPDFNet denoiser.

        Args:
            model_path: Path to the DPDFNet ONNX model.

        Raises:
            RuntimeError: If sherpa-onnx rejects the model configuration.
        """
        import sherpa_onnx

        config = sherpa_onnx.OfflineSpeechDenoiserConfig(
            model=sherpa_onnx.OfflineSpeechDenoiserModelConfig(
                dpdfnet=sherpa_onnx.OfflineSpeechDenoiserDpdfNetModelConfig(
                    model=str(model_path),
                    attenuation_limit_db=constants.DENOISE_ATTENUATION_LIMIT_DB,
                ),
                num_threads=constants.DENOISE_NUM_THREADS,
                debug=False,
                provider="cpu",
            )
        )
        if not config.validate():
            raise RuntimeError(
                "Invalid DPDFNet configuration. Check DPDFNET_MODEL_PATH and "
                "sherpa-onnx installation."
            )
        self._denoiser = sherpa_onnx.OfflineSpeechDenoiser(config)

    def run(self, samples: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        """Denoise a mono waveform.

        Args:
            samples: One-dimensional float32 audio samples.
            sample_rate: Input sample rate in Hz.

        Returns:
            Tuple of denoised samples and output sample rate.
        """
        contiguous = np.ascontiguousarray(samples, dtype=np.float32)
        with _MODEL_LOCK:
            output = self._denoiser.run(contiguous, int(sample_rate))
        return (
            np.asarray(output.samples, dtype=np.float32),
            int(output.sample_rate),
        )


def load_dpdfnet_model(cache: bool = True, **kwargs: object) -> DPDFNetDenoiser:
    """Load and optionally cache the DPDFNet model.

    Args:
        cache: Reuse the process-wide model instance when True.
        **kwargs: Optional ``model_path`` override.

    Returns:
        Loaded DPDFNet denoiser.

    Raises:
        ModuleNotFoundError: If sherpa-onnx is not installed.
        FileNotFoundError: If the configured model file does not exist.
    """
    global _MODEL_CACHE  # noqa: PLW0603

    if importlib.util.find_spec("sherpa_onnx") is None:
        raise ModuleNotFoundError(
            "sherpa-onnx is required for DPDFNet noise reduction. Install the "
            "GPU/CPU runtime that includes sherpa-onnx and DPDFNet."
        )

    model_path_value = kwargs.get("model_path")
    model_path = Path(
        str(model_path_value) if model_path_value else constants.DENOISE_MODEL_PATH
    )
    if not model_path.is_file():
        raise FileNotFoundError(
            f"DPDFNet model not found at {model_path}. Set DPDFNET_MODEL_PATH "
            "to a valid dpdfnet8.onnx file."
        )

    if cache and _MODEL_CACHE is not None:
        return _MODEL_CACHE

    model = DPDFNetDenoiser(model_path)
    if cache:
        _MODEL_CACHE = model
    return model


def dpdfnet_audio(
    audio: torch.Tensor | str | bytes,
    input_sr: int | None = None,
    output_sr: int | None = None,
    model: DPDFNetDenoiser | None = None,
    device: str | None = None,
    verbose: bool = True,
    save_path: str | Callable[[torch.Tensor], None] | None = None,
    **kwargs: object,
) -> torch.Tensor:
    """Denoise audio using DPDFNet in a stable-whisper-compatible interface.

    Args:
        audio: Audio tensor, file path, or encoded bytes.
        input_sr: Input sample rate for tensor inputs.
        output_sr: Optional output sample rate.
        model: Preloaded DPDFNet model.
        device: Ignored; DPDFNet runs on CPU.
        verbose: Accepted for API compatibility.
        save_path: Optional path or callback for saving denoised audio.
        **kwargs: Ignored compatibility options.

    Returns:
        Denoised mono audio tensor.

    Raises:
        ValueError: If a tensor input has no sample rate.
    """
    from stable_whisper.audio.utils import load_audio, resample

    if model is None:
        model = load_dpdfnet_model()

    if isinstance(audio, (str, bytes)):
        audio = torch.from_numpy(load_audio(audio, model.samplerate, mono=True))
    elif input_sr is None:
        raise ValueError("input_sr is required when denoising a tensor")
    elif input_sr != model.samplerate:
        audio = resample(audio, input_sr, model.samplerate)

    if audio.dim() > 1:
        audio = audio.mean(dim=-2)

    samples = audio.detach().cpu().numpy().astype(np.float32, copy=False)
    denoised, sample_rate = model.run(samples, model.samplerate)
    if denoised.ndim != 1:
        denoised = denoised.reshape(-1)
    output = torch.from_numpy(denoised)

    if output_sr is not None and sample_rate != output_sr:
        output = resample(output, sample_rate, output_sr)

    if save_path is not None:
        if isinstance(save_path, str):
            from stable_whisper.audio.output import save_audio_tensor

            save_audio_tensor(
                output,
                save_path,
                output_sr or sample_rate,
                verbose=verbose,
            )
        else:
            save_path(output)

    return output


def register_dpdfnet_denoiser() -> None:
    """Register DPDFNet with stable-whisper's denoiser registry.

    Registration is idempotent and safe to call on every import.
    """
    try:
        from stable_whisper.audio import SUPPORTED_DENOISERS

        SUPPORTED_DENOISERS["dpdfnet"] = {
            "run": dpdfnet_audio,
            "load": load_dpdfnet_model,
            "access": is_dpdfnet_available,
        }
    except Exception as exc:  # pragma: no cover - optional dependency guard
        logger.debug("Could not register DPDFNet denoiser: %s", exc)
