"""Speaker diarization integration via pyannote.audio.

Provides ``diarize`` to assign speaker labels to Whisper transcription chunks
as an optional post-processing step.  The module follows the same integration
pattern as ``stable_ts.py``: guarded import, pipeline caching, and graceful
degradation when the optional dependency is absent.
"""

from __future__ import annotations

import hashlib
import logging
import re
import subprocess
import tempfile
import threading
import time
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import numpy as np
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from huggingface_hub.utils import HFValidationError

from halospeak.core.backend_cache import invalidate_gpu_cache
from halospeak.core.errors import DiarizationError
from halospeak.utils.constants import (
    DEFAULT_DIARIZATION_MODEL,
    DIARIZATION_ALLOW_CPU_FALLBACK,
    DIARIZATION_DISABLE_MIOPEN,
    DIARIZATION_EMBEDDING_CLUSTERING,
    DIARIZATION_FFMPEG_TIMEOUT_SECONDS,
    DIARIZATION_PRELOAD_AUDIO,
)

# Check if torchcodec is available (required by pyannote.audio v4 for
# built-in audio decoding).  On ROCm, torchcodec is incompatible with
# the custom PyTorch build, so we preload audio via torchaudio instead.
try:
    import torchcodec as _torchcodec  # noqa: F401  # type: ignore[import-untyped]

    _TORCHCODEC_AVAILABLE = True
except (ImportError, OSError):
    # ImportError: package not installed.
    # OSError: package installed but native libs fail to load (e.g. ROCm
    # PyTorch build has incompatible symbols, or FFmpeg .so missing).
    _TORCHCODEC_AVAILABLE = False

logger = logging.getLogger(__name__)

# Optional dependency guard
# Suppress the misleading torchcodec warning from pyannote.audio.core.io.
# On ROCm, torchcodec is intentionally excluded; the torchaudio fallback
# in diarize() handles audio loading correctly.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r"\s*torchcodec is not installed correctly.*",
        category=UserWarning,
    )
    try:
        from pyannote.audio import Pipeline  # type: ignore[import-untyped]
    except ImportError:
        Pipeline = None  # type: ignore[assignment, misc]

# Pipeline cache and active-use tracking.
# Key includes a hash of hf_token so different tokens get separate entries.
_CACHE: dict[tuple[str, str, str], Pipeline] = {}  # type: ignore[type-arg]
_ACTIVE_PIPELINE_USES: dict[int, int] = {}
_PENDING_GPU_EVICTIONS: dict[int, Pipeline] = {}  # type: ignore[type-arg]
_LOCK = threading.RLock()
_KERNEL_MODE_LOCK = threading.RLock()


@contextmanager
def _rocm_native_diarization_kernels(enabled: bool) -> Iterator[None]:
    """Temporarily use native PyTorch kernels for ROCm diarization.

    MIOpen JIT compilation is broken for some ROCm/gfx combinations (notably
    gfx1151 with ROCm 7.0). Pyannote's BatchNorm and LSTM dropout paths then
    fail during inference. Native PyTorch kernels avoid MIOpen entirely and
    produce the same diarization result.

    Args:
        enabled: Whether the workaround may be applied. Pass True for GPU
            diarization requests.

    Yields:
        None while MIOpen/cuDNN is disabled if applicable.
    """
    if not enabled or not DIARIZATION_DISABLE_MIOPEN:
        yield
        return

    try:
        import torch
    except Exception:  # pragma: no cover - defensive import guard
        yield
        return

    if getattr(torch.version, "hip", None) is None:
        yield
        return

    with _KERNEL_MODE_LOCK:
        previous = torch.backends.cudnn.enabled
        if previous:
            logger.info(
                "Disabling MIOpen/cuDNN for ROCm diarization to avoid MIOpen "
                "JIT kernel failures"
            )
        torch.backends.cudnn.enabled = False
        try:
            yield
        finally:
            torch.backends.cudnn.enabled = previous


def _resolve_local_pipeline_dir(model_name: str) -> Path | None:
    """Return a local pyannote pipeline path when one is already available.

    Args:
        model_name: HuggingFace model identifier or local path.

    Returns:
        The local pipeline directory or checkpoint file, or ``None`` when the
        pipeline is neither a local path nor present in the Hugging Face cache.
    """
    candidate = Path(model_name).expanduser()
    if candidate.is_dir() or candidate.is_file():
        return candidate

    try:
        snapshot_dir = snapshot_download(repo_id=model_name, local_files_only=True)
    except (LocalEntryNotFoundError, HFValidationError, FileNotFoundError) as exc:
        logger.debug("Diarization pipeline %s is not cached: %s", model_name, exc)
        return None

    cached_path = Path(snapshot_dir)
    return cached_path if cached_path.exists() else None


def _get_or_create_pipeline(
    model_name: str,
    device: str,
    hf_token: str | None,
    *,
    reserve: bool = False,
) -> Pipeline:  # type: ignore[valid-type]
    """Return a cached pyannote Pipeline or create a new one.

    Args:
        model_name: HuggingFace model identifier (e.g.
            ``pyannote/speaker-diarization-community-1``).
        device: Torch device string (``"cpu"`` or ``"cuda"``).
        hf_token: Optional HuggingFace access token. When set, it is used to
            download or authenticate the model from the Hub. When empty, an
            already-cached pipeline is loaded locally.
        reserve: Track the returned pipeline as actively in use.

    Returns:
        A ``pyannote.audio.Pipeline`` instance.

    Raises:
        DiarizationError: If the pipeline is not cached and no token is set, or
            if the pipeline cannot be loaded.
    """
    device = _normalize_diarization_device(device)

    # Normalize hf_token to bytes before hashing; handle None/bytes gracefully.
    if isinstance(hf_token, bytes):
        token_bytes = hf_token
    else:
        token_bytes = (hf_token or "").encode()
    token_hash = hashlib.sha256(token_bytes).hexdigest()[:16]
    key = (model_name, device, token_hash)
    with _LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            if reserve:
                _reserve_pipeline(cached)
            logger.info(
                "Diarization timing: pipeline_cache_hit model=%s device=%s",
                model_name,
                device,
            )
            return cached

    # Create outside the lock to avoid blocking other callers during download.
    pipeline: Pipeline | None = None  # type: ignore[type-arg]
    started_at = time.perf_counter()
    try:
        if hf_token:
            pipeline = Pipeline.from_pretrained(  # type: ignore[union-attr]
                model_name,
                token=hf_token,
                revision="main",
            )
        else:
            local_checkpoint = _resolve_local_pipeline_dir(model_name)
            if local_checkpoint is None:
                raise DiarizationError(
                    f"Diarization model '{model_name}' is not available in the "
                    + "local Hugging Face cache, and no HF_TOKEN was provided. "
                    + "Set HF_TOKEN to download it once, or pre-populate the cache.",
                    model=model_name,
                    reason="model_not_cached",
                )
            logger.info(
                "Loading diarization pipeline from local cache: %s",
                local_checkpoint,
            )
            pipeline = Pipeline.from_pretrained(  # type: ignore[union-attr]
                str(local_checkpoint)
            )
    except DiarizationError:
        raise
    except Exception as exc:
        _raise_load_error(model_name, exc)

    # Move to requested device.
    try:
        import torch

        pipeline = pipeline.to(torch.device(device))  # type: ignore[union-attr]
    except Exception as exc:
        if (
            device == "cuda"
            and DIARIZATION_ALLOW_CPU_FALLBACK
            and _is_cuda_unavailable_error(exc)
        ):
            logger.warning(
                "CUDA is unavailable for diarization (%s); using CPU instead",
                exc,
            )
            try:
                pipeline = pipeline.to(torch.device("cpu"))  # type: ignore[union-attr]
                device = "cpu"
            except Exception as cpu_exc:  # pragma: no cover - defensive
                _raise_load_error(model_name, cpu_exc)
        else:
            _raise_load_error(model_name, exc)
    load_elapsed = time.perf_counter() - started_at
    logger.info(
        "Diarization timing: pipeline_load_seconds=%.3f model=%s device=%s",
        load_elapsed,
        model_name,
        device,
    )

    with _LOCK:
        # Another thread may have created the same entry meanwhile.
        if key not in _CACHE:
            _CACHE[key] = pipeline
        else:
            # Another thread won the race; move orphaned pipeline off GPU
            # to avoid leaked device memory.
            try:
                import torch

                if pipeline is not None:
                    pipeline.to(torch.device("cpu"))
                    pipeline = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.warning(
                    "Duplicate pipeline created for key=%s; orphan freed", key
                )
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "Failed to clean up orphaned pipeline for key=%s",
                    key,
                    exc_info=exc,
                )
        cached = _CACHE[key]
        if reserve:
            _reserve_pipeline(cached)
        return cached


def _reserve_pipeline(pipeline: Pipeline) -> None:  # type: ignore[valid-type]
    """Record an active caller for a cached pipeline.

    Args:
        pipeline: Pipeline that is about to run inference.
    """
    pipeline_id = id(pipeline)
    _ACTIVE_PIPELINE_USES[pipeline_id] = _ACTIVE_PIPELINE_USES.get(pipeline_id, 0) + 1


def _release_pipeline(pipeline: Pipeline) -> None:  # type: ignore[valid-type]
    """Release an active pipeline and complete any deferred GPU eviction.

    Args:
        pipeline: Pipeline whose inference call has completed.
    """
    pipeline_to_evict: Pipeline | None = None  # type: ignore[type-arg]
    pipeline_id = id(pipeline)
    with _LOCK:
        active_uses = _ACTIVE_PIPELINE_USES.get(pipeline_id, 0)
        if active_uses <= 1:
            _ACTIVE_PIPELINE_USES.pop(pipeline_id, None)
            pipeline_to_evict = _PENDING_GPU_EVICTIONS.pop(pipeline_id, None)
        else:
            _ACTIVE_PIPELINE_USES[pipeline_id] = active_uses - 1

    if pipeline_to_evict is not None:
        _move_pipeline_to_cpu(pipeline_to_evict)
        _empty_gpu_cache()
        logger.info("Completed deferred GPU diarization pipeline eviction")


def _normalize_diarization_device(device: str) -> str:
    """Return a torch-compatible diarization device string.

    Args:
        device: User-facing device string.

    Returns:
        Torch device string for pyannote inference.
    """
    normalized = device.strip().lower()
    if normalized in {"gpu", "0"}:
        return "cuda"
    return normalized


def _resolve_diarization_device(device: str) -> str:
    """Resolve a requested diarization device against the CPU-only runtime.

    CPU builds may still receive a stale ``DIARIZATION_DEVICE=cuda`` value from
    an existing user ``.env``. Fall back to CPU when CUDA is unavailable and
    the configured fallback is enabled.

    Args:
        device: User-facing device string.

    Returns:
        ``"cpu"`` or ``"cuda"`` depending on availability.

    Raises:
        DiarizationError: If CUDA is requested, unavailable, and fallback is
            disabled.
    """
    normalized = _normalize_diarization_device(device)
    if normalized != "cuda":
        return normalized

    try:
        import torch

        cuda_available = torch.cuda.is_available()
    except Exception:  # pragma: no cover - defensive import guard
        cuda_available = False

    if cuda_available:
        return "cuda"

    if DIARIZATION_ALLOW_CPU_FALLBACK:
        logger.warning(
            "Diarization device 'cuda' requested but CUDA is unavailable; "
            "falling back to CPU"
        )
        return "cpu"

    raise DiarizationError(
        "CUDA was requested for diarization but is not available in this "
        "runtime. Set DIARIZATION_DEVICE=cpu or enable "
        "DIARIZATION_ALLOW_CPU_FALLBACK=true.",
        model=DEFAULT_DIARIZATION_MODEL,
        reason="device_unavailable",
    )


def _log_timing(label: str, started_at: float, **fields: object) -> None:
    """Log a concise diarization timing event.

    Args:
        label: Event label.
        started_at: ``time.perf_counter()`` value captured before the work.
        **fields: Additional structured fields to append to the log line.
    """
    elapsed = time.perf_counter() - started_at
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    if details:
        logger.info("Diarization timing: %s_seconds=%.3f %s", label, elapsed, details)
    else:
        logger.info("Diarization timing: %s_seconds=%.3f", label, elapsed)


def _required_diarization_repos(model_name: str) -> list[str]:
    """Return gated Hugging Face repos required for a diarization model.

    pyannote.audio 4.x eagerly downloads a PLDA from the community-1 repo even
    when loading the legacy 3.1 pipeline. The community-1 pipeline only needs
    its own repository.

    Args:
        model_name: Diarization pipeline identifier.

    Returns:
        Ordered list of gated repository identifiers.
    """
    if "community-1" in model_name:
        return [model_name]
    return [
        model_name,
        "pyannote/segmentation-3.0",
        "pyannote/speaker-diarization-community-1",
    ]


def _extract_gated_repo_id(exc: Exception, fallback: str) -> str:
    """Return the Hugging Face repo named by a gated-model exception.

    Args:
        exc: Original pipeline-load exception.
        fallback: Repo to report when the exception does not expose a repo.

    Returns:
        Repository identifier such as ``pyannote/speaker-diarization-community-1``.
    """
    candidates = [
        getattr(exc, "repo_id", None),
        getattr(exc, "repository", None),
        getattr(exc, "model_name", None),
    ]
    response = getattr(exc, "response", None)
    if response is not None:
        candidates.extend([
            getattr(response, "repo_id", None),
            getattr(response, "model_name", None),
        ])

    message = str(exc)
    match = re.search(
        r"(?:https?://huggingface\.co/)?([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)",
        message,
    )
    if match:
        candidates.append(match.group(1))

    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        repo = candidate.strip().rstrip(".,)")
        for prefix in ("https://huggingface.co/", "http://huggingface.co/", "hf.co/"):
            if repo.startswith(prefix):
                repo = repo[len(prefix) :]
                break
        if "/" in repo:
            return repo

    return fallback


def _raise_load_error(model_name: str, exc: Exception) -> None:
    """Translate a pipeline-load exception into a ``DiarizationError``.

    Detects HTTP 401/403 (missing authentication or gated-model license not
    accepted) and provides an
    actionable message naming the actual gated repository.  All other errors
    are wrapped generically.

    Args:
        model_name: The model that was being loaded.
        exc: The original exception.

    Raises:
        DiarizationError: Always.
    """
    status_code = getattr(exc, "status_code", None)
    # Also check for requests-style .response.status_code
    if status_code is None and hasattr(exc, "response"):
        status_code = getattr(exc.response, "status_code", None)  # type: ignore[union-attr]

    if status_code in {401, 403}:
        gated_repo = _extract_gated_repo_id(exc, model_name)
        required_repos = _required_diarization_repos(model_name)
        if gated_repo not in required_repos:
            required_repos.insert(0, gated_repo)
        license_list = "\n".join(
            f"  - https://huggingface.co/{repo}" for repo in required_repos
        )
        raise DiarizationError(
            "Authentication or license acceptance is required for gated "
            + f"diarization model '{gated_repo}'. Accept the license terms for "
            + "every required repository:\n"
            + license_list
            + "\nThen ensure HF_TOKEN is set to a valid HuggingFace access token "
            + "with access to those gated repositories.",
            model=model_name,
            reason="license_not_accepted",
        ) from exc

    raise DiarizationError(
        f"Failed to load diarization model '{model_name}': {exc}",
        model=model_name,
        reason="load_error",
    ) from exc


def _is_retriable_gpu_runtime_error(exc: Exception) -> bool:
    """Return whether an exception is a ROCm/MIOpen or GPU OOM failure.

    Args:
        exc: Runtime exception raised by pyannote/PyTorch.

    Returns:
        True when the error matches a known ROCm/MIOpen runtime failure
        or GPU out-of-memory condition.
    """
    message = str(exc).lower()
    return (
        "miopenstatusunknownerror" in message
        or "miopen" in message
        or "rocrand" in message
        or "out of memory" in message
    )


def _is_cuda_unavailable_error(exc: Exception) -> bool:
    """Return whether an exception indicates CUDA is not available.

    Args:
        exc: Exception raised while moving a pipeline to CUDA.

    Returns:
        True for known CPU-only PyTorch/CUDA availability messages.
    """
    message = str(exc).lower()
    return (
        "torch not compiled with cuda" in message
        or "no cuda gpus are available" in message
        or "cuda is not available" in message
        or "no cuda-capable device" in message
        or "found no nvidia driver" in message
    )


def clear_diarization_cache() -> None:
    """Invalidate all cached diarization pipelines.

    Useful for OOM recovery or when switching HuggingFace tokens.
    """
    with _LOCK:
        _CACHE.clear()
    logger.info("Diarization pipeline cache cleared")


def _move_pipeline_to_cpu(pipeline: Pipeline) -> None:  # type: ignore[valid-type]
    """Move a diarization pipeline to CPU without allowing cleanup to raise.

    Args:
        pipeline: Pipeline to migrate off the GPU.
    """
    try:
        import torch

        pipeline.to(torch.device("cpu"))
    except Exception as exc:  # noqa: BLE001  # defensive: cleanup must not raise
        logger.debug("Failed to move cached pipeline to CPU: %s", exc)


def _empty_gpu_cache() -> None:
    """Release unused GPU memory without allowing cleanup to raise."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001  # defensive: cleanup must not raise
        logger.debug("Failed to empty CUDA cache after GPU cleanup: %s", exc)


def _clear_gpu_diarization_cache() -> None:
    """Remove GPU-based diarization pipelines from the cache.

    Moves cached GPU pipelines to CPU and clears the cache entries so
    VRAM is freed for the ASR backend on the next transcription request.
    """
    pipelines_to_evict: list[Pipeline] = []  # type: ignore[type-arg]
    deferred_count = 0
    with _LOCK:
        keys_to_remove: list[tuple[str, str, str]] = []
        for key, cached_pipeline in _CACHE.items():
            device = key[1]
            if isinstance(device, str) and device == "cuda":
                keys_to_remove.append(key)
                pipeline_id = id(cached_pipeline)
                if _ACTIVE_PIPELINE_USES.get(pipeline_id, 0):
                    _PENDING_GPU_EVICTIONS[pipeline_id] = cached_pipeline
                    deferred_count += 1
                else:
                    pipelines_to_evict.append(cached_pipeline)
        for key in keys_to_remove:
            _CACHE.pop(key, None)

    for cached_pipeline in pipelines_to_evict:
        _move_pipeline_to_cpu(cached_pipeline)
    if pipelines_to_evict:
        _empty_gpu_cache()
    if keys_to_remove:
        logger.info(
            "Cleared %d GPU diarization pipeline(s) from cache (%d deferred)",
            len(keys_to_remove),
            deferred_count,
        )


# --- Speaker-to-segment alignment -------------------------------------------


def _align_speakers_to_segments(
    chunks: list[dict[str, Any]],
    speaker_turns: list[tuple[float, float, str]],
) -> list[dict[str, Any]]:
    """Assign speaker labels to transcription chunks by time overlap.

    For each Whisper chunk, the speaker with the maximum time overlap is
    assigned.  If a chunk has no overlap with any speaker turn, ``speaker``
    is set to ``None``.

    Args:
        chunks: Whisper output chunks, each with ``start`` and ``end`` keys.
        speaker_turns: List of ``(start, end, speaker_label)`` tuples from
            pyannote.

    Returns:
        A new list of chunk dicts with an added ``speaker`` key.
    """
    aligned: list[dict[str, Any]] = []
    normalized_turns = _normalize_speaker_turns(speaker_turns)
    for chunk in chunks:
        chunk_start, chunk_end = _chunk_time_bounds(chunk)

        best_speaker: str | None = None
        best_overlap = 0.0

        for turn_start, turn_end, speaker in normalized_turns:
            overlap = _time_overlap(chunk_start, chunk_end, turn_start, turn_end)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker

        aligned.append({**chunk, "speaker": best_speaker})
    return aligned


def _chunk_time_bounds(chunk: dict[str, Any]) -> tuple[float, float]:
    """Return stable start and end times for an ASR chunk or segment.

    Args:
        chunk: Whisper chunk or segment dictionary.

    Returns:
        Normalized ``(start, end)`` seconds.
    """
    ts = chunk.get("timestamp")
    if ts is not None:
        chunk_start = ts[0]
        chunk_end = ts[1] if len(ts) > 1 and ts[1] is not None else chunk_start
    else:
        chunk_start = chunk.get("start", 0.0)
        chunk_end = chunk.get("end", chunk_start)

    start = _coerce_time(chunk_start)
    end = _coerce_time(chunk_end, default=start)
    if end < start:
        return end, start
    return start, end


def _coerce_time(value: object, *, default: float = 0.0) -> float:
    """Return ``value`` as a finite float timestamp.

    Args:
        value: Timestamp-like value.
        default: Fallback when ``value`` is missing or invalid.

    Returns:
        Finite float timestamp.
    """
    try:
        coerced = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if coerced != coerced:
        return default
    return coerced


def _normalize_speaker_turns(
    speaker_turns: list[tuple[float, float, str]],
) -> list[tuple[float, float, str]]:
    """Return valid speaker turns sorted by start time.

    Args:
        speaker_turns: Raw pyannote speaker turns.

    Returns:
        Sorted ``(start, end, speaker)`` tuples with invalid spans removed.
    """
    normalized: list[tuple[float, float, str]] = []
    for start, end, speaker in speaker_turns:
        turn_start = _coerce_time(start)
        turn_end = _coerce_time(end, default=turn_start)
        if turn_end <= turn_start:
            continue
        normalized.append((turn_start, turn_end, speaker))
    return sorted(normalized, key=lambda item: (item[0], item[1], item[2]))


def _time_overlap(
    chunk_start: float,
    chunk_end: float,
    turn_start: float,
    turn_end: float,
) -> float:
    """Return overlap seconds, including point-in-turn timestamp matches.

    Args:
        chunk_start: Chunk start time in seconds.
        chunk_end: Chunk end time in seconds.
        turn_start: Speaker turn start time in seconds.
        turn_end: Speaker turn end time in seconds.

    Returns:
        Positive overlap weight for speaker selection.
    """
    if chunk_end == chunk_start and turn_start <= chunk_start < turn_end:
        return 1e-6
    overlap_start = max(chunk_start, turn_start)
    overlap_end = min(chunk_end, turn_end)
    return max(0.0, overlap_end - overlap_start)


def _embedding_waveform(audio_input: dict[str, Any]) -> tuple[Any, int]:
    """Return a mono 16 kHz waveform for speaker embedding extraction.

    Args:
        audio_input: Pyannote preloaded audio dictionary.

    Returns:
        Tuple of waveform tensor shaped ``(1, time)`` and sample rate.

    Raises:
        ValueError: If the preloaded waveform is missing or invalid.
    """
    import torch
    import torchaudio

    waveform = audio_input.get("waveform")
    sample_rate = int(audio_input.get("sample_rate", 16_000))
    if not isinstance(waveform, torch.Tensor):
        raise ValueError("preloaded waveform is missing")
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16_000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16_000)
        sample_rate = 16_000
    return waveform, sample_rate


def _compute_item_embeddings(
    pipeline: object,
    audio_input: dict[str, Any],
    items: list[dict[str, Any]],
) -> list[list[float] | None] | None:
    """Compute normalized speaker embeddings for each ASR item.

    Args:
        pipeline: Loaded pyannote pipeline with an ``_embedding`` model.
        audio_input: Pyannote preloaded audio dictionary.
        items: ASR chunks or segments.

    Returns:
        One embedding per item, or ``None`` when embeddings cannot be computed.
    """
    import torch

    embedding_model_raw = getattr(pipeline, "_embedding", None)
    if embedding_model_raw is None or not items:
        return None
    embedding_model = cast(Callable[[Any], Any], embedding_model_raw)

    try:
        waveform, sample_rate = _embedding_waveform(audio_input)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Speaker embedding extraction unavailable: %s", exc)
        return None

    vectors: list[list[float] | None] = []
    for item in items:
        start, end = _chunk_time_bounds(item)
        start_sample = max(0, int(start * sample_rate))
        end_sample = min(waveform.shape[1], int(end * sample_rate))
        if end_sample <= start_sample:
            vectors.append(None)
            continue

        clip = waveform[:, start_sample:end_sample]
        if clip.shape[1] < 400:
            clip = torch.nn.functional.pad(clip, (0, 400 - clip.shape[1]))
        try:
            with torch.inference_mode():
                raw = embedding_model(clip.unsqueeze(0))
            vector = np.asarray(raw[0], dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(vector))
            vectors.append((vector / norm).tolist() if norm > 0 else None)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Speaker embedding failed for item: %s", exc)
            vectors.append(None)

    if not any(vector is not None for vector in vectors):
        return None
    return vectors


def _cluster_embedding_vectors(
    vectors: list[list[float] | None],
    n_clusters: int,
) -> tuple[list[int], dict[int, list[float]]] | None:
    """Cluster normalized speaker embeddings with KMeans.

    Args:
        vectors: One normalized embedding per item, or ``None``.
        n_clusters: Automatic speaker count from pyannote.

    Returns:
        Cluster id per item and normalized centroid per cluster, or ``None``
        when clustering is not possible.
    """
    valid = [
        (index, np.asarray(vector, dtype=np.float32))
        for index, vector in enumerate(vectors)
        if vector is not None
    ]
    n_clusters = max(1, int(n_clusters))
    if not valid or len(valid) < n_clusters:
        return None

    matrix = np.stack([vector for _, vector in valid])
    if n_clusters == 1:
        centroid = np.mean(matrix, axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm == 0:
            return None
        return [0 for _ in vectors], {0: (centroid / norm).tolist()}

    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(matrix)
    cluster_ids = [0 for _ in vectors]
    for (index, _), cluster_id in zip(valid, labels):
        cluster_ids[index] = int(cluster_id)

    centroids: dict[int, list[float]] = {}
    for cluster_id in range(n_clusters):
        members = matrix[labels == cluster_id]
        if len(members) == 0:
            continue
        centroid = np.mean(members, axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm > 0:
            centroids[cluster_id] = (centroid / norm).tolist()

    if not centroids:
        return None
    return cluster_ids, centroids


def _labels_for_clusters(cluster_ids: list[int]) -> dict[int, str]:
    """Map KMeans cluster ids to stable first-appearance speaker labels.

    Args:
        cluster_ids: Cluster id per primary item.

    Returns:
        Mapping from cluster id to ``SPEAKER_XX``.
    """
    order: list[int] = []
    for cluster_id in cluster_ids:
        if cluster_id not in order:
            order.append(cluster_id)
    return {
        cluster_id: f"SPEAKER_{index:02d}" for index, cluster_id in enumerate(order)
    }


def _preload_audio_as_waveform(audio_path: str) -> dict[str, Any]:
    """Load audio as a waveform dict for pyannote when torchcodec is absent.

    Uses the direct torchaudio path for WAV/FLAC. Compressed formats are
    decoded through ffmpeg first so headerless MP3s and remuxed MP3s produce
    the same waveform. Soundfile/torchaudio MP3 decoding can differ depending
    on the presence of a Xing/Info header, which shifts the waveform and can
    destabilize diarization.

    Args:
        audio_path: Path to the audio file on disk.

    Returns:
        A dict ``{"waveform": Tensor, "sample_rate": int}`` suitable for
        pyannote.

    Raises:
        DiarizationError: If torchaudio cannot be imported, or if both
            direct torchaudio loading and ffmpeg→torchaudio conversion
            fail (reason ``audio_decode_unavailable``).
    """
    try:
        import torchaudio  # pyright: ignore[reportMissingTypeStubs]
    except Exception as exc:
        raise DiarizationError(
            "Audio decode unavailable; torchaudio could not be imported."
            + " Install torchaudio or torchcodec.",
            model=DEFAULT_DIARIZATION_MODEL,
            reason="audio_decode_unavailable",
        ) from exc

    suffix = Path(audio_path).suffix.lower()
    fast_extensions = {".wav", ".flac", ".aif", ".aiff"}

    if suffix in fast_extensions:
        try:
            started_at = time.perf_counter()
            waveform, sample_rate = torchaudio.load(audio_path)
            # Ensure mono — pyannote expects single-channel audio.
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            logger.debug(
                "Preloaded audio via torchaudio: shape=%s, sr=%d",
                waveform.shape,
                sample_rate,
            )
            _log_timing("audio_preload", started_at, route="torchaudio")
            return {"waveform": waveform, "sample_rate": sample_rate}
        except Exception as direct_exc:
            logger.debug(
                "torchaudio.load failed for %s: %s - trying ffmpeg conversion",
                audio_path,
                direct_exc,
            )

    # Slow path: convert to WAV via ffmpeg, then load. Also the forced path for
    # compressed formats, where decoder-version differences matter.
    started_at = time.perf_counter()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="diarize_")
    tmp_path = tmp.name
    tmp.close()
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            audio_path,
            "-ar",
            "16000",
            "-ac",
            "1",
            tmp_path,
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=DIARIZATION_FFMPEG_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as timeout_exc:
            raise DiarizationError(
                "Audio decode unavailable; ffmpeg conversion timed out"
                + f" for {audio_path}. Install torchcodec or provide"
                + " WAV/FLAC input.",
                model=DEFAULT_DIARIZATION_MODEL,
                reason="audio_decode_unavailable",
            ) from timeout_exc

        if result.returncode != 0:
            raise DiarizationError(
                "Audio decode unavailable; ffmpeg conversion failed"
                + f" for {audio_path}: {result.stderr[:300]}."
                + " Install torchcodec or provide WAV/FLAC input.",
                model=DEFAULT_DIARIZATION_MODEL,
                reason="audio_decode_unavailable",
            )

        waveform, sample_rate = torchaudio.load(tmp_path)  # type: ignore[possibly-undefined]
        logger.debug(
            "Preloaded audio via ffmpeg→torchaudio: shape=%s, sr=%d, src=%s",
            waveform.shape,
            sample_rate,
            suffix,
        )
        _log_timing("audio_preload", started_at, route="ffmpeg_torchaudio")
        return {"waveform": waveform, "sample_rate": sample_rate}
    except DiarizationError:
        raise
    except Exception as exc:
        raise DiarizationError(
            f"Audio decode unavailable; failed to preload audio: {exc}."
            + " Install torchcodec or provide WAV/FLAC input.",
            model=DEFAULT_DIARIZATION_MODEL,
            reason="audio_decode_unavailable",
        ) from exc
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except OSError:
            pass


# --- Public API --------------------------------------------------------------


def validate_speaker_config(
    *,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
) -> None:
    """Validate speaker-count parameters before diarization inference.

    Args:
        num_speakers: Exact number of speakers (``None`` = auto-detect).
        min_speakers: Minimum number of speakers (``None`` = unspecified).
        max_speakers: Maximum number of speakers (``None`` = unspecified).

    Raises:
        DiarizationError: If any count is less than 1, if
            ``num_speakers`` is combined with ``min_speakers`` or
            ``max_speakers``, or if ``min_speakers > max_speakers``
            (reason ``invalid_speaker_config``).
    """
    invalid_count = next(
        (
            count
            for count in (num_speakers, min_speakers, max_speakers)
            if count is not None and count < 1
        ),
        None,
    )
    if invalid_count is not None:
        raise DiarizationError(
            "Speaker counts must be greater than zero.",
            model=DEFAULT_DIARIZATION_MODEL,
            reason="invalid_speaker_config",
        )
    if num_speakers is not None and (
        min_speakers is not None or max_speakers is not None
    ):
        raise DiarizationError(
            "num_speakers cannot be combined with min_speakers or max_speakers.",
            model=DEFAULT_DIARIZATION_MODEL,
            reason="invalid_speaker_config",
        )
    if (
        min_speakers is not None
        and max_speakers is not None
        and min_speakers > max_speakers
    ):
        raise DiarizationError(
            "min_speakers cannot be greater than max_speakers.",
            model=DEFAULT_DIARIZATION_MODEL,
            reason="invalid_speaker_config",
        )


def diarize(
    result: dict[str, Any],
    audio_path: str,
    *,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    hf_token: str | None = None,
    device: str = "cpu",
    return_embeddings: bool = False,
) -> dict[str, Any]:
    """Apply speaker diarization to a transcription result.

    Uses ``pyannote.audio`` to identify speaker turns in the audio and
    assigns speaker labels to each chunk in the result.

    Args:
        result: Whisper transcription result dict (must contain ``chunks``).
        audio_path: Path to the audio file on disk.
        num_speakers: Exact number of speakers (``None`` = auto-detect).
        min_speakers: Minimum number of speakers.
        max_speakers: Maximum number of speakers.
        hf_token: Optional HuggingFace access token. Required only to download
            a gated model that is not already cached.
        device: Device for diarization inference (``"cpu"``, ``"cuda"``, or
            ``"gpu"``).
        return_embeddings: Include one 256-dimensional embedding per speaker
            in ``speaker_embeddings``.

    Returns:
        The result dict with ``speaker`` added to each chunk, ``diarized`` set
        to ``True``, and optionally ``speaker_embeddings``.

    Raises:
        DiarizationError: If speaker count parameters are invalid or
            conflicting, pyannote is not installed, the model is not cached
            and no HuggingFace token is available, or the model cannot be
            loaded.
    """
    validate_speaker_config(
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    if Pipeline is None:
        raise DiarizationError(
            "Speaker diarization was requested but pyannote.audio is not installed. "
            + "Install the diarization extras to enable --diarize support.",
            model=DEFAULT_DIARIZATION_MODEL,
            reason="pyannote_not_installed",
        )

    device = _resolve_diarization_device(device)

    # Load pipeline (cached).
    logger.info(
        "Loading diarization pipeline (model=%s, device=%s)",
        DEFAULT_DIARIZATION_MODEL,
        device,
    )
    started_at = time.perf_counter()
    pipeline = _get_or_create_pipeline(
        DEFAULT_DIARIZATION_MODEL,
        device,
        hf_token,
        reserve=True,
    )
    _log_timing("pipeline_ready", started_at, device=device)
    logger.info("Diarization pipeline ready")

    # Prepare audio input for the pipeline.
    # When torchcodec is unavailable (common on ROCm), pyannote's built-in
    # audio decoding fails.  We preload the audio as a tensor dict instead.
    audio_input: str | dict[str, Any] = audio_path
    if DIARIZATION_PRELOAD_AUDIO or not _TORCHCODEC_AVAILABLE:
        logger.info(
            "Preloading audio for diarization (preload=%s, torchcodec_available=%s)",
            DIARIZATION_PRELOAD_AUDIO,
            _TORCHCODEC_AVAILABLE,
        )
        audio_input = _preload_audio_as_waveform(audio_path)
        logger.info("Audio preloaded for diarization")

    chunks = result.get("chunks") or []
    segments = result.get("segments") or []
    primary_items = chunks if chunks else segments
    embedding_vectors: list[list[float] | None] | None = None

    # Run diarization.
    # Free GPU memory held by the ASR backend cache before inference —
    # on 8 GB cards the Whisper model and pyannote cannot coexist on GPU.
    if device == "cuda":
        invalidate_gpu_cache()

    logger.info(
        "Starting diarization inference (device=%s, num_speakers=%s, min=%s, max=%s)",
        device,
        num_speakers,
        min_speakers,
        max_speakers,
    )
    try:
        kwargs: dict[str, Any] = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

        try:
            started_at = time.perf_counter()
            with _rocm_native_diarization_kernels(device == "cuda"):
                diarization_result = pipeline(audio_input, **kwargs)
            _log_timing("inference", started_at, device=device)

            if (
                DIARIZATION_EMBEDDING_CLUSTERING
                and isinstance(audio_input, dict)
                and primary_items
            ):
                started_at = time.perf_counter()
                embedding_vectors = _compute_item_embeddings(
                    pipeline,
                    audio_input,
                    primary_items,
                )
                _log_timing(
                    "speaker_embedding_extraction",
                    started_at,
                    items=len(primary_items),
                    available=embedding_vectors is not None,
                )
        except Exception as exc:
            is_gpu_device = device == "cuda"
            can_retry_cpu = (
                DIARIZATION_ALLOW_CPU_FALLBACK
                and is_gpu_device
                and _is_retriable_gpu_runtime_error(exc)
            )
            if not can_retry_cpu:
                raise

            logger.warning(
                "Diarization failed on GPU with %s; retrying on CPU",
                exc,
            )
            cpu_pipeline = _get_or_create_pipeline(
                DEFAULT_DIARIZATION_MODEL,
                "cpu",
                hf_token,
                reserve=True,
            )
            try:
                started_at = time.perf_counter()
                with _rocm_native_diarization_kernels(False):
                    diarization_result = cpu_pipeline(audio_input, **kwargs)
                _log_timing("inference", started_at, device="cpu", fallback=True)

                if (
                    DIARIZATION_EMBEDDING_CLUSTERING
                    and isinstance(audio_input, dict)
                    and primary_items
                ):
                    started_at = time.perf_counter()
                    embedding_vectors = _compute_item_embeddings(
                        cpu_pipeline,
                        audio_input,
                        primary_items,
                    )
                    _log_timing(
                        "speaker_embedding_extraction",
                        started_at,
                        items=len(primary_items),
                        available=embedding_vectors is not None,
                    )
            finally:
                _release_pipeline(cpu_pipeline)
    except Exception as exc:
        logger.error("Diarization inference failed: %s", exc, exc_info=True)
        raise DiarizationError(
            f"Diarization inference failed: {exc}",
            model=DEFAULT_DIARIZATION_MODEL,
            reason="inference_error",
        ) from exc

    finally:
        _release_pipeline(pipeline)
        # Release VRAM even when inference or the CPU retry fails. Eviction is
        # deferred when another request still uses the shared GPU pipeline.
        if device == "cuda":
            _clear_gpu_diarization_cache()

    # Extract speaker turns as (start, end, label) tuples.
    # pyannote.audio v4 returns a DiarizeOutput dataclass; v3 returns
    # an Annotation directly. Handle both. This is the fallback path when
    # segment-embedding clustering is unavailable.
    annotation = diarization_result  # default: v3-style Annotation
    embedding_labels: list[str] = []
    speaker_embeddings: dict[str, list[float]] = {}
    if hasattr(diarization_result, "speaker_diarization"):
        # pyannote.audio v4 — DiarizeOutput dataclass
        regular = diarization_result.speaker_diarization  # type: ignore[union-attr]
        embedding_labels = list(regular.labels())
        exclusive = getattr(diarization_result, "exclusive_speaker_diarization", None)
        if exclusive is not None:
            exclusive_labels = list(exclusive.labels())
            if exclusive_labels:
                annotation = exclusive
                logger.info(
                    "Fallback diarization alignment annotation uses exclusive turns "
                    "(labels=%s)",
                    exclusive_labels,
                )
            else:
                annotation = regular
        else:
            annotation = regular

        if return_embeddings:
            raw_embeddings = getattr(diarization_result, "speaker_embeddings", None)
            if raw_embeddings is not None:
                for index, label in enumerate(embedding_labels):
                    try:
                        vector = raw_embeddings[index]
                    except (IndexError, TypeError):
                        continue
                    tolist = getattr(vector, "tolist", None)
                    try:
                        values = tolist() if callable(tolist) else vector
                        speaker_embeddings[label] = [
                            float(value) for value in cast("list[float]", values)
                        ]
                    except (TypeError, ValueError):
                        continue
    else:
        try:
            embedding_labels = list(annotation.labels())
        except Exception:  # pragma: no cover - legacy annotation guard
            embedding_labels = []

    if not chunks and not segments:
        logger.warning("No chunks or segments in result - nothing to diarize")
        return {
            **result,
            "diarized": False,
            "diarization_error": (
                "No chunks or segments available for speaker alignment."
            ),
        }

    # Preferred path: cluster per-segment speaker embeddings. pyannote's turn
    # boundaries can miss short speaker changes; embeddings of the actual ASR
    # segments separate speakers more reliably.
    if (
        DIARIZATION_EMBEDDING_CLUSTERING
        and embedding_vectors is not None
        and len(embedding_vectors) == len(primary_items)
        and embedding_labels
    ):
        started_at = time.perf_counter()
        cluster_result = _cluster_embedding_vectors(
            embedding_vectors,
            len(embedding_labels),
        )
        if cluster_result is not None:
            cluster_ids, centroids = cluster_result
            label_map = _labels_for_clusters(cluster_ids)
            aligned_primary = [
                {**item, "speaker": label_map.get(cluster_id)}
                for item, cluster_id in zip(primary_items, cluster_ids)
            ]
            if centroids:
                speaker_embeddings = {
                    label_map[cluster_id]: vector
                    for cluster_id, vector in centroids.items()
                    if cluster_id in label_map
                }
            _log_timing(
                "speaker_embedding_clustering",
                started_at,
                items=len(aligned_primary),
                speakers=len(label_map),
            )
            logger.info(
                "Using segment embedding clustering for speaker alignment "
                "(speakers=%s)",
                sorted(label_map.values()),
            )

            out = {
                **result,
                "chunks": aligned_primary,
                "diarized": True,
            }
            if chunks:
                if segments and segments is not chunks:
                    primary_turns: list[tuple[float, float, str]] = []
                    for item in aligned_primary:
                        speaker = item.get("speaker")
                        if not speaker:
                            continue
                        start, end = _chunk_time_bounds(item)
                        primary_turns.append((start, end, str(speaker)))
                    out["segments"] = _align_speakers_to_segments(
                        segments,
                        primary_turns,
                    )
                else:
                    out["segments"] = aligned_primary
            else:
                out["segments"] = aligned_primary

            if return_embeddings:
                out["speaker_embeddings"] = speaker_embeddings
            return out

        logger.warning(
            "Segment embedding clustering failed; falling back to pyannote turns"
        )

    # Fallback: align ASR items to exclusive/regular pyannote turns.
    speaker_turns: list[tuple[float, float, str]] = []
    started_at = time.perf_counter()
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        speaker_turns.append((turn.start, turn.end, speaker))
    _log_timing("speaker_turn_extraction", started_at, turns=len(speaker_turns))

    if not speaker_turns:
        logger.warning("Diarization produced no speaker turns")
        return {
            **result,
            "diarized": False,
            "diarization_error": "Diarization produced no speaker turns.",
        }

    primary = chunks if chunks else segments
    started_at = time.perf_counter()
    aligned_primary = _align_speakers_to_segments(primary, speaker_turns)
    _log_timing("speaker_alignment", started_at, items=len(primary))

    out = {**result, "chunks": aligned_primary, "diarized": True}

    if chunks:
        if segments and segments is not chunks:
            out["segments"] = _align_speakers_to_segments(segments, speaker_turns)
        else:
            out["segments"] = aligned_primary
    else:
        out["segments"] = aligned_primary

    if return_embeddings:
        out["speaker_embeddings"] = speaker_embeddings

    return out
