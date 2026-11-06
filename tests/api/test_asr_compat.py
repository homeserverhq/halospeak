"""Tests for the WhisperX-compatible /asr endpoint."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

from fastapi.responses import JSONResponse

from halospeak.api import routes


def _call_asr_compat(
    *,
    diarize: bool = True,
    return_speaker_embeddings: bool = True,
) -> JSONResponse:
    """Invoke the /asr route directly with mocked collaborators.

    Returns:
        The route JSON response.
    """
    file_handler = MagicMock()
    file_handler.save_upload.return_value = "/tmp/audio.mp3"

    asr_pipeline = MagicMock()
    asr_pipeline.asr_backend.config = MagicMock()

    orchestrator = MagicMock()
    orchestrator.run_transcription.return_value = {
        "text": "hello world",
        "config_used": {"model": "whisper-test"},
    }
    processed = {
        "text": "hello world",
        "config_used": {"model": "whisper-test"},
        "segments": [
            {
                "timestamp": [0.0, 1.0],
                "text": "hello",
                "speaker": "SPEAKER_00",
            },
            {
                "timestamp": [1.0, 2.0],
                "text": "world",
                "speaker": "SPEAKER_01",
            },
        ],
        "speaker_embeddings": {
            "SPEAKER_00": [1.0, 0.0],
            "SPEAKER_01": [0.0, 1.0],
        },
    }

    with (
        patch.object(routes, "create_orchestrator", return_value=orchestrator),
        patch.object(
            routes,
            "_apply_post_processing",
            return_value=processed,
        ) as postprocess,
        patch.object(routes, "_persist_post_processed_result"),
    ):
        response = asyncio.run(
            routes.create_asr_compatibility(
                audio_file=MagicMock(filename="audio.mp3"),
                encode=True,
                task="transcribe",
                output="json",
                language="en",
                diarize=diarize,
                enable_diarization=None,
                return_speaker_embeddings=return_speaker_embeddings,
                num_speakers=None,
                min_speakers=None,
                max_speakers=None,
                initial_prompt=None,
                hotwords=None,
                model="whisper-test",
                diarization_device="cpu",
                asr_pipeline=asr_pipeline,
                file_handler=file_handler,
            )
        )

    assert postprocess.call_args.kwargs["diarize"] is diarize
    assert (
        postprocess.call_args.kwargs["return_speaker_embeddings"]
        is return_speaker_embeddings
    )
    file_handler.cleanup.assert_called_once_with("/tmp/audio.mp3")
    return response


def test_asr_compatibility__returns_whisperx_segments_and_embeddings() -> None:
    """The /asr response contains start/end/speaker segments and embeddings."""
    response = _call_asr_compat()

    payload = json.loads(response.body)
    assert payload["text"] == "hello world"
    assert payload["language"] == "en"
    assert payload["segments"] == [
        {
            "start": 0.0,
            "end": 1.0,
            "text": "hello",
            "speaker": "SPEAKER_00",
        },
        {
            "start": 1.0,
            "end": 2.0,
            "text": "world",
            "speaker": "SPEAKER_01",
        },
    ]
    assert payload["speakers"] == ["SPEAKER_00", "SPEAKER_01"]
    assert payload["speaker_embeddings"] == {
        "SPEAKER_00": [1.0, 0.0],
        "SPEAKER_01": [0.0, 1.0],
    }


def test_asr_compatibility__omits_segments_when_diarization_is_disabled() -> None:
    """Without diarization the response falls back to plain text."""
    response = _call_asr_compat(diarize=False, return_speaker_embeddings=False)

    payload = json.loads(response.body)
    assert "segments" not in payload
    assert "speakers" not in payload
    assert "speaker_embeddings" not in payload
