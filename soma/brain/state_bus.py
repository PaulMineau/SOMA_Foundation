"""Async state bus — shared state between brain modules and UI.

Modules publish embeddings, dashboard reads them.
Simple dict-based pub/sub with timestamp tracking.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class StateBus:
    """Simple async state bus for brain module communication."""

    def __init__(self) -> None:
        self._state: dict[str, Any] = {}
        self._timestamps: dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def publish(self, key: str, value: Any) -> None:
        """Publish a value to the state bus."""
        async with self._lock:
            self._state[key] = value
            self._timestamps[key] = datetime.now().isoformat()
        logger.debug("Bus publish: %s", key)

    async def get(self, key: str) -> Any | None:
        """Read a value from the state bus."""
        async with self._lock:
            return self._state.get(key)

    async def get_all(self) -> dict[str, Any]:
        """Get all current state."""
        async with self._lock:
            return dict(self._state)

    async def get_timestamps(self) -> dict[str, str]:
        """Get timestamps for all published values."""
        async with self._lock:
            return dict(self._timestamps)

    def get_sync(self, key: str) -> Any | None:
        """Synchronous read (for Streamlit)."""
        return self._state.get(key)

    def get_all_sync(self) -> dict[str, Any]:
        """Synchronous read all (for Streamlit)."""
        return dict(self._state)
