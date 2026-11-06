"""Backward-compatibility layer for CLI error types.

This module preserves the legacy import paths used throughout the CLI tests
and third-party scripts by re-exporting the core exception types defined in
``halospeak.core.errors``.
"""

from __future__ import annotations

from halospeak.core.errors import (  # noqa: F401
    DeviceNotFoundError,
    TranscriptionError,
)
