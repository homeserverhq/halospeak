"""Audio processing utilities for the HaloSpeak API."""

from __future__ import annotations

from halospeak.audio.processing import get_audio_duration, split_audio
from halospeak.audio.results import merge_chunk_results
from halospeak.utils import cleanup_temp_files

__all__ = [
    "cleanup_temp_files",
    "get_audio_duration",
    "merge_chunk_results",
    "split_audio",
]
