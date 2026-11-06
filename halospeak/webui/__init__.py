"""WebUI module for the HaloSpeak API.

This module provides a Gradio-based web interface for audio transcription
with support for file uploads, real-time progress updates, and result downloads.
"""

from halospeak.webui.app import launch_webui
from halospeak.webui.merge_handler import (
    MergeConfiguration,
    MergeHandler,
    MergeResult,
    SrtMerger,
    TxtMerger,
    VttMerger,
    get_merge_handler,
    merge_files,
)

# Import main components for external use
from halospeak.webui.ui import create_ui_components

# Import ZIP and merge utilities
from halospeak.webui.zip_creator import (
    BatchZipBuilder,
    ZipConfiguration,
    ZipStats,
    create_batch_zip,
)

__all__ = [
    "create_ui_components",
    "launch_webui",
    # ZIP creation
    "BatchZipBuilder",
    "ZipConfiguration",
    "ZipStats",
    "create_batch_zip",
    # File merging
    "MergeHandler",
    "TxtMerger",
    "SrtMerger",
    "VttMerger",
    "MergeConfiguration",
    "MergeResult",
    "get_merge_handler",
    "merge_files",
]
