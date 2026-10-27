"""API route definitions for the Insanely Fast Whisper API.

This module contains clean, focused route definitions that use dependency
injection for ASR pipeline instances and file handling.
"""

import json
import logging
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from insanely_fast_whisper_rocm.api.dependencies import (
    get_asr_pipeline,
    get_file_handler,
)
from insanely_fast_whisper_rocm.api.responses import ResponseFormatter
from insanely_fast_whisper_rocm.core.errors import (
    DiarizationError,
    OutOfMemoryError,
)
from insanely_fast_whisper_rocm.core.integrations.stable_ts import (
    stabilize_timestamps,
)
from insanely_fast_whisper_rocm.core.orchestrator import create_orchestrator
from insanely_fast_whisper_rocm.core.pipeline import WhisperPipeline
from insanely_fast_whisper_rocm.utils import (
    DEFAULT_DEMUCS,
    DEFAULT_DIARIZATION_DEVICE,
    DEFAULT_DIARIZE,
    DEFAULT_STABILIZE,
    DEFAULT_TIMESTAMP_TYPE,
    DEFAULT_VAD,
    DEFAULT_VAD_THRESHOLD,
    DENOISE_AVAILABLE,
    HF_TOKEN,
    RESPONSE_FORMAT_JSON,
    SUPPORTED_RESPONSE_FORMATS,
    FileHandler,
)

logger = logging.getLogger(__name__)

router = APIRouter()
VALID_DIARIZATION_DEVICES = {"cpu", "cuda", "gpu"}


def _json_default(value: object) -> object:
    """Convert non-standard JSON scalar values for persisted API results.

    Handles numpy/torch scalar wrappers (``.item()``) and array-like objects
    (``.tolist()``) by dispatching through ``getattr`` so the ``object``
    parameter stays statically typed.

    Args:
        value: Value passed by ``json.dumps`` when default encoding fails.

    Returns:
        JSON-compatible scalar or container value.

    Raises:
        TypeError: If the value cannot be converted.
    """
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _persist_post_processed_result(result: dict[str, Any]) -> None:
    """Persist the final API result when the pipeline already created a JSON file.

    The pipeline saves before API-level stabilization and diarization run. Rewrite
    that file after post-processing so disk artifacts match the response.

    Args:
        result: Final post-processed transcription or translation result.
    """
    output_file_path = result.get("output_file_path")
    if not isinstance(output_file_path, str) or not output_file_path:
        return

    output_path = Path(output_file_path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            ),
            encoding="utf-8",
        )
        logger.info("Post-processed API result saved to %s", output_path)
    except (OSError, TypeError, ValueError) as exc:
        logger.warning(
            "Failed to update post-processed API result at %s: %s",
            output_path,
            exc,
            exc_info=True,
        )


def _apply_post_processing(
    result: dict[str, Any],
    *,
    stabilize: bool,
    demucs: bool,
    vad: bool,
    vad_threshold: float,
    diarize: bool,
    diarization_device: str,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
    audio_path: str,
    return_speaker_embeddings: bool = False,
) -> dict[str, Any]:
    """Apply optional stabilization and diarization post-processing.

    Args:
        result: Raw Whisper transcription result.
        stabilize: Whether to run timestamp stabilization.
        demucs: Enable AI noise reduction (DPDFNet) in stabilization.
        vad: Enable VAD in stabilization.
        vad_threshold: VAD sensitivity threshold.
        diarize: Whether to run speaker diarization.
        diarization_device: Device for diarization pipeline.
        num_speakers: Exact speaker count (None = auto).
        min_speakers: Minimum speaker count.
        max_speakers: Maximum speaker count.
        audio_path: Path to the audio file on disk.
        return_speaker_embeddings: Include per-speaker embeddings in the result.

    Returns:
        The result dict, potentially enriched with stabilization and/or
        speaker labels.  When ``stabilize=True``, a ``"stabilized"`` key
        is added (``True`` on success, ``False`` if stabilization failed).

    Raises:
        HTTPException: If noise reduction is requested in a build that excludes
            it, if diarization_device is invalid, or if diarization fails.
            Diarization failures are recorded on ``result`` before raising.
    """
    if demucs and not DENOISE_AVAILABLE:
        raise HTTPException(
            status_code=400,
            detail=(
                "AI noise reduction is not available in this build. Install the "
                "runtime with sherpa-onnx and DPDFNet, or disable noise "
                "reduction."
            ),
        )

    if stabilize:
        try:
            result = stabilize_timestamps(
                result, demucs=demucs, vad=vad, vad_threshold=vad_threshold
            )
            result["stabilized"] = True
        except Exception as stab_exc:  # noqa: BLE001
            logger.error("Stabilization failed: %s", stab_exc, exc_info=True)
            result["stabilized"] = False

    if diarize:
        if diarization_device not in VALID_DIARIZATION_DEVICES:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid diarization_device "
                    f"'{diarization_device}'. Must be one of: "
                    f"{sorted(VALID_DIARIZATION_DEVICES)}"
                ),
            )

        normalized_device = (
            "cuda" if diarization_device == "gpu" else diarization_device
        )

        try:
            from insanely_fast_whisper_rocm.core.integrations.diarization import (
                diarize as diarize_result,
            )
            from insanely_fast_whisper_rocm.core.integrations.diarization import (
                validate_speaker_config,
            )

            # Validate early so invalid speaker config returns HTTP 400
            # before any diarization pipeline work (diarize() also validates).
            validate_speaker_config(
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
            result = diarize_result(
                result,
                audio_path=audio_path,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                device=normalized_device,
                hf_token=HF_TOKEN,
                return_embeddings=return_speaker_embeddings,
            )
        except DiarizationError as e:
            result["diarized"] = False
            result["diarization_error"] = str(e)
            _persist_post_processed_result(result)
            raise HTTPException(status_code=400, detail=str(e)) from e

    return result


@router.post(
    "/v1/audio/transcriptions",
    tags=["Transcription"],
    summary="Transcribe Audio",
    description="Convert speech in an audio file to text using the Whisper model",
    response_model=None,
    responses={
        200: {
            "description": "Successful transcription",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/TranscriptionResponse"}
                },
                "text/plain": {"schema": {"type": "string"}},
            },
        },
        400: {"description": "Invalid request parameters"},
        422: {"description": "Validation error (e.g., unsupported file format)"},
        500: {"description": "Internal server error"},
        503: {"description": "Model not loaded or unavailable"},
    },
)
async def create_transcription(
    file: UploadFile = File(..., description="The audio file to transcribe"),  # noqa: B008
    response_format: str = Form(
        RESPONSE_FORMAT_JSON,
        description="Response format (json, verbose_json, text, srt, vtt)",
    ),
    timestamp_type: str = Form(
        DEFAULT_TIMESTAMP_TYPE,
        description="Type of timestamp to generate ('chunk' or 'word')",
    ),
    language: str | None = Form(
        None, description="Source language code (auto-detect if None)"
    ),
    task: Literal["transcribe"] = Form("transcribe", description="ASR task type"),
    stabilize: bool = Form(
        DEFAULT_STABILIZE, description="Enable timestamp stabilization"
    ),
    demucs: bool = Form(
        DEFAULT_DEMUCS, description="Enable AI noise reduction (DPDFNet)"
    ),
    vad: bool = Form(DEFAULT_VAD, description="Enable Voice Activity Detection"),
    vad_threshold: float = Form(
        DEFAULT_VAD_THRESHOLD, description="VAD threshold for speech detection"
    ),
    diarize: bool = Form(DEFAULT_DIARIZE, description="Enable speaker diarization"),
    num_speakers: int | None = Form(None, description="Exact number of speakers"),
    min_speakers: int | None = Form(None, description="Minimum number of speakers"),
    max_speakers: int | None = Form(None, description="Maximum number of speakers"),
    diarization_device: str = Form(
        DEFAULT_DIARIZATION_DEVICE,
        description="Device for diarization (cpu, cuda, or gpu)",
    ),
    asr_pipeline: WhisperPipeline = Depends(get_asr_pipeline),  # noqa: B008
    file_handler: FileHandler = Depends(get_file_handler),  # noqa: B008
) -> JSONResponse | PlainTextResponse:
    """Transcribe speech in an audio file to text.

    This endpoint processes an audio file and returns its transcription using the
    specified Whisper model. It supports various configuration options including
    timestamp generation.

    Args:
        file: The audio file to transcribe (supported formats: mp3, wav, etc.)
        response_format: Desired response format ("json", "verbose_json",
            "text", "srt", or "vtt").
        timestamp_type: Type of timestamp to generate ("chunk" or "word")
        language: Optional source language code (auto-detect if None)
        task: ASR task type (must be "transcribe")
        stabilize: Enable timestamp stabilization if True.
        demucs: Enable AI noise reduction (DPDFNet) if True.
        vad: Enable Voice Activity Detection if True.
        vad_threshold: VAD sensitivity threshold (0.0 - 1.0).
        diarize: Enable speaker diarization if True.
        num_speakers: Exact number of speakers (auto-detect if None).
        min_speakers: Minimum number of speakers for diarization.
        max_speakers: Maximum number of speakers for diarization.
        diarization_device: Device for diarization ("cpu", "cuda", or
            "gpu"; "gpu" is an alias for "cuda").
        asr_pipeline: Injected ASR pipeline instance
        file_handler: Injected file handler instance

    Returns:
        JSONResponse | PlainTextResponse: Transcription result as plain text
            or JSON with metadata.

    Raises:
        HTTPException: If file validation fails or processing errors occur
    """
    logger.info("-" * 50)
    logger.info("Received transcription request:")
    logger.info("  File: %s", file.filename)
    logger.debug("  Timestamp type: %s", timestamp_type)
    logger.debug("  Language: %s", language)
    logger.debug("  Task: %s", task)

    # Validate and save file
    file_handler.validate_audio_file(file)
    temp_filepath = file_handler.save_upload(file)

    try:
        logger.info("Starting transcription process...")

        # Use orchestrator for transcription with OOM recovery
        orchestrator = create_orchestrator()

        # We need to construct a backend config.
        # Since we use dependency injection for asr_pipeline,
        # we can get the config from it.
        # However, the orchestrator handles pipeline acquisition
        # via borrow_pipeline.
        # We'll use the config from the injected pipeline as
        # the starting point.
        base_config = asr_pipeline.asr_backend.config

        try:
            result = orchestrator.run_transcription(
                audio_path=temp_filepath,
                backend_config=base_config,
                language=language,
                task=task,
                timestamp_type=timestamp_type,
            )
        except OutOfMemoryError as oom:
            raise HTTPException(
                status_code=507,
                detail=f"Insufficient GPU memory for transcription: {str(oom)}",
            ) from oom
        except Exception as e:
            if isinstance(e, HTTPException):
                raise
            raise HTTPException(status_code=500, detail=str(e)) from e

        # Optional post-processing (stabilization + diarization)
        result = _apply_post_processing(
            result,
            stabilize=stabilize,
            demucs=demucs,
            vad=vad,
            vad_threshold=vad_threshold,
            diarize=diarize,
            diarization_device=diarization_device,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            audio_path=temp_filepath,
        )
        _persist_post_processed_result(result)

        logger.info("Transcription completed successfully")

        # Validate response_format
        if response_format not in SUPPORTED_RESPONSE_FORMATS:
            raise HTTPException(status_code=400, detail="Unsupported response_format")
        logger.debug("Transcription result: %s", result)

        # Format response according to requested response_format
        return ResponseFormatter.format_transcription(result, response_format)

    finally:
        file_handler.cleanup(temp_filepath)


@router.post(
    "/v1/audio/translations",
    tags=["Translation"],
    summary="Translate Audio",
    description="Translate speech in an audio file to English using the Whisper model",
    response_model=None,
    responses={
        200: {
            "description": "Successful translation",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/TranscriptionResponse"}
                },
                "text/plain": {"schema": {"type": "string"}},
            },
        },
        400: {"description": "Invalid request parameters"},
        422: {"description": "Validation error (e.g., unsupported file format)"},
        500: {"description": "Internal server error"},
        503: {"description": "Model not loaded or unavailable"},
    },
)
async def create_translation(
    file: UploadFile = File(..., description="The audio file to translate"),  # noqa: B008
    response_format: str = Form(
        RESPONSE_FORMAT_JSON,
        description="Response format (json, verbose_json, text, srt, vtt)",
    ),
    timestamp_type: str = Form(
        DEFAULT_TIMESTAMP_TYPE,
        description="Type of timestamp to generate ('chunk' or 'word')",
    ),
    language: str | None = Form(
        None, description="Source language code (auto-detect if None)"
    ),
    stabilize: bool = Form(
        DEFAULT_STABILIZE, description="Enable timestamp stabilization"
    ),
    demucs: bool = Form(
        DEFAULT_DEMUCS, description="Enable AI noise reduction (DPDFNet)"
    ),
    vad: bool = Form(DEFAULT_VAD, description="Enable Voice Activity Detection"),
    vad_threshold: float = Form(
        DEFAULT_VAD_THRESHOLD, description="VAD threshold for speech detection"
    ),
    diarize: bool = Form(DEFAULT_DIARIZE, description="Enable speaker diarization"),
    num_speakers: int | None = Form(None, description="Exact number of speakers"),
    min_speakers: int | None = Form(None, description="Minimum number of speakers"),
    max_speakers: int | None = Form(None, description="Maximum number of speakers"),
    diarization_device: str = Form(
        DEFAULT_DIARIZATION_DEVICE,
        description="Device for diarization (cpu, cuda, or gpu)",
    ),
    asr_pipeline: WhisperPipeline = Depends(get_asr_pipeline),  # noqa: B008
    file_handler: FileHandler = Depends(get_file_handler),  # noqa: B008
) -> JSONResponse | PlainTextResponse:
    """Translate speech in an audio file to English.

    This endpoint processes an audio file in any supported language and translates
    the speech to English using the specified Whisper model. It supports various
    configuration options.

    Args:
        file: The audio file to translate (supported formats: mp3, wav, etc.)
        response_format: Desired response format ("json" or "text")
        timestamp_type: Type of timestamp to generate ("chunk" or "word")
        language: Optional source language code (auto-detect if None)
        stabilize: Enable timestamp stabilization if True.
        demucs: Enable AI noise reduction (DPDFNet) if True.
        vad: Enable Voice Activity Detection if True.
        vad_threshold: VAD sensitivity threshold (0.0 - 1.0).
        diarize: Enable speaker diarization if True.
        num_speakers: Exact number of speakers (auto-detect if None).
        min_speakers: Minimum number of speakers for diarization.
        max_speakers: Maximum number of speakers for diarization.
        diarization_device: Device for diarization ("cpu", "cuda", or
            "gpu"; "gpu" is an alias for "cuda").
        asr_pipeline: Injected ASR pipeline instance
        file_handler: Injected file handler instance

    Returns:
        JSONResponse | PlainTextResponse: Translation result as plain text or
            JSON with metadata.

    Raises:
        HTTPException: If file validation fails or processing errors occur
    """
    logger.info("-" * 50)
    logger.info("Received translation request:")
    logger.info("  File: %s", file.filename)
    logger.debug("  Timestamp type: %s", timestamp_type)
    logger.debug("  Language: %s", language)
    logger.debug("  Response format: %s", response_format)

    # Validate and save file
    file_handler.validate_audio_file(file)
    temp_filepath = file_handler.save_upload(file)

    try:
        logger.info("Starting translation process...")

        # Use orchestrator for translation with OOM recovery
        orchestrator = create_orchestrator()
        base_config = asr_pipeline.asr_backend.config

        try:
            result = orchestrator.run_transcription(
                audio_path=temp_filepath,
                backend_config=base_config,
                language=language,
                task="translate",
                timestamp_type=timestamp_type,
            )
        except OutOfMemoryError as oom:
            raise HTTPException(
                status_code=507,
                detail=f"Insufficient GPU memory for translation: {str(oom)}",
            ) from oom
        except Exception as e:
            if isinstance(e, HTTPException):
                raise
            raise HTTPException(status_code=500, detail=str(e)) from e

        # Optional post-processing (stabilization + diarization)
        result = _apply_post_processing(
            result,
            stabilize=stabilize,
            demucs=demucs,
            vad=vad,
            vad_threshold=vad_threshold,
            diarize=diarize,
            diarization_device=diarization_device,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            audio_path=temp_filepath,
        )
        _persist_post_processed_result(result)

        logger.info("Translation completed successfully")
        logger.debug("Translation result: %s", result)

        # Validate response_format
        if response_format not in SUPPORTED_RESPONSE_FORMATS:
            raise HTTPException(status_code=400, detail="Unsupported response_format")

        # Format response
        return ResponseFormatter.format_translation(result, response_format)

    finally:
        file_handler.cleanup(temp_filepath)


def _segment_start_end(segment: dict[str, Any]) -> tuple[float, float]:
    """Return start/end seconds from a chunk-style or segment-style dict.

    Args:
        segment: Whisper chunk or diarized segment dictionary.

    Returns:
        ``(start, end)`` in seconds. Invalid or missing values fall back to
        ``0.0``.
    """
    from insanely_fast_whisper_rocm.core.integrations.diarization import (
        _chunk_time_bounds,
    )

    return _chunk_time_bounds(segment)


@router.post(
    "/asr",
    tags=["Compatibility"],
    summary="WhisperX-compatible ASR endpoint",
    description=(
        "Compatibility endpoint for whisper-asr-webservice/WhisperX clients "
        "such as Speakr. Accepts the audio_file multipart field and query "
        "parameters used by those clients."
    ),
    response_model=None,
)
async def create_asr_compatibility(
    audio_file: UploadFile = File(..., description="The audio file to transcribe"),  # noqa: B008
    encode: bool = Query(True, description="Accepted for compatibility"),
    task: str = Query("transcribe", description="transcribe or translate"),
    output: str = Query("json", description="json or verbose_json"),
    language: str | None = Query(None, description="Source language code"),
    diarize: bool = Query(DEFAULT_DIARIZE, description="Enable diarization"),
    enable_diarization: bool | None = Query(
        None, description="WhisperX-compatible diarization flag"
    ),
    return_speaker_embeddings: bool = Query(
        False, description="Return one embedding per speaker"
    ),
    num_speakers: int | None = Query(None, description="Exact speaker count"),
    min_speakers: int | None = Query(None, description="Minimum speaker count"),
    max_speakers: int | None = Query(None, description="Maximum speaker count"),
    initial_prompt: str | None = Query(None, description="Accepted for compatibility"),
    hotwords: str | None = Query(None, description="Accepted for compatibility"),
    model: str | None = Query(None, description="Accepted for compatibility"),
    diarization_device: str = Query(
        DEFAULT_DIARIZATION_DEVICE,
        description="Device for diarization (cpu, cuda, or gpu)",
    ),
    asr_pipeline: WhisperPipeline = Depends(get_asr_pipeline),  # noqa: B008
    file_handler: FileHandler = Depends(get_file_handler),  # noqa: B008
) -> JSONResponse:
    """Transcribe audio using the whisper-asr-webservice-compatible contract.

    Args:
        audio_file: Audio upload named ``audio_file`` (not ``file``).
        encode: Compatibility flag; ignored.
        task: ``transcribe`` or ``translate``.
        output: ``json`` or ``verbose_json``.
        language: Optional source language code.
        diarize: Enable speaker diarization.
        enable_diarization: WhisperX-compatible alias for ``diarize``.
        return_speaker_embeddings: Include per-speaker 256-dimensional vectors.
        num_speakers: Exact speaker count.
        min_speakers: Minimum speaker count.
        max_speakers: Maximum speaker count.
        initial_prompt: Compatibility flag; ignored.
        hotwords: Compatibility flag; ignored.
        model: Compatibility model hint; the service model remains authoritative.
        diarization_device: Diarization device.
        asr_pipeline: Injected ASR pipeline.
        file_handler: Injected file handler.

    Returns:
        WhisperX-compatible JSON containing text, segments, speakers, and
        optional speaker embeddings.

    Raises:
        HTTPException: If the file is invalid or transcription fails.
    """
    if task not in {"transcribe", "translate"}:
        raise HTTPException(status_code=400, detail="Unsupported task")
    if output.lower() not in {"json", "verbose_json"}:
        raise HTTPException(status_code=400, detail="Unsupported output format")

    should_diarize = diarize or bool(enable_diarization)
    logger.info(
        "Received /asr compatibility request: file=%s diarize=%s "
        "min=%s max=%s embeddings=%s",
        audio_file.filename,
        should_diarize,
        min_speakers,
        max_speakers,
        return_speaker_embeddings,
    )
    logger.debug(
        "ASR compatibility passthrough: encode=%s model=%s prompt=%s hotwords=%s",
        encode,
        model,
        initial_prompt,
        hotwords,
    )

    file_handler.validate_audio_file(audio_file)
    temp_filepath = file_handler.save_upload(audio_file)

    try:
        orchestrator = create_orchestrator()
        base_config = asr_pipeline.asr_backend.config
        try:
            result = orchestrator.run_transcription(
                audio_path=temp_filepath,
                backend_config=base_config,
                language=language,
                task=task,
                timestamp_type=DEFAULT_TIMESTAMP_TYPE,
            )
        except OutOfMemoryError as oom:
            raise HTTPException(
                status_code=507,
                detail=f"Insufficient GPU memory for transcription: {oom}",
            ) from oom
        except Exception as exc:
            if isinstance(exc, HTTPException):
                raise
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        result = _apply_post_processing(
            result,
            stabilize=False,
            demucs=False,
            vad=False,
            vad_threshold=DEFAULT_VAD_THRESHOLD,
            diarize=should_diarize,
            diarization_device=diarization_device,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            audio_path=temp_filepath,
            return_speaker_embeddings=return_speaker_embeddings,
        )
        _persist_post_processed_result(result)

        config_used = result.get("config_used") or {}
        payload: dict[str, Any] = {
            "text": result.get("text", ""),
            "language": result.get("language") or language,
            "provider": "insanely-fast-whisper",
            "model": config_used.get("model") or model,
        }

        if should_diarize:
            segment_payload: list[dict[str, Any]] = []
            for segment in result.get("segments") or []:
                start, end = _segment_start_end(segment)
                segment_payload.append({
                    "start": start,
                    "end": end,
                    "text": segment.get("text", ""),
                    "speaker": segment.get("speaker"),
                })
            if segment_payload:
                payload["segments"] = segment_payload
                payload["speakers"] = sorted({
                    item["speaker"] for item in segment_payload if item.get("speaker")
                })

        if return_speaker_embeddings:
            payload["speaker_embeddings"] = result.get("speaker_embeddings") or {}

        logger.info(
            "Completed /asr compatibility request: segments=%s speakers=%s",
            len(payload.get("segments", [])),
            payload.get("speakers", []),
        )
        return JSONResponse(content=payload)

    finally:
        file_handler.cleanup(temp_filepath)
