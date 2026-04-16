"""Visual module stub — LLaVA integration point for camera input.

Currently disabled. When SOMA_CAMERA_ENABLED=true, captures frames
and describes them via LLaVA on Ollama.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

ENABLED = os.getenv("SOMA_CAMERA_ENABLED", "false").lower() == "true"


async def describe() -> str:
    """Return visual scene description. Empty string when camera disabled."""
    if not ENABLED:
        return ""

    # Future: cv2.VideoCapture -> base64 -> LLaVA via Ollama
    logger.info("Visual module: camera not yet implemented")
    return ""
