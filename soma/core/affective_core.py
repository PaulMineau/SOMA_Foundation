"""
Affective Core — Functional emotional states based on Panksepp's seven primary systems.

These are numerical drive states that decay over time, are activated by inputs,
and critically affect downstream behavior (retrieval bias, attention, expression).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class Drive(str, Enum):
    SEEKING = "seeking"
    CARE = "care"
    PLAY = "play"
    GRIEF = "grief"
    FEAR = "fear"
    RAGE = "rage"


# Default activation parameters per drive
DRIVE_DEFAULTS = {
    Drive.SEEKING: {"decay_rate": 0.05, "baseline": 0.3},
    Drive.CARE: {"decay_rate": 0.02, "baseline": 0.2},
    Drive.PLAY: {"decay_rate": 0.08, "baseline": 0.2},
    Drive.GRIEF: {"decay_rate": 0.03, "baseline": 0.0},
    Drive.FEAR: {"decay_rate": 0.06, "baseline": 0.0},
    Drive.RAGE: {"decay_rate": 0.07, "baseline": 0.0},
}


@dataclass
class AffectiveState:
    """A snapshot of all drive activations at a moment in time."""

    drives: dict[Drive, float] = field(default_factory=lambda: {
        d: DRIVE_DEFAULTS[d]["baseline"] for d in Drive
    })
    timestamp: float = field(default_factory=time.time)

    def valence(self) -> float:
        """Net valence: positive drives minus negative drives, normalized to [-1, 1]."""
        positive = self.drives[Drive.SEEKING] + self.drives[Drive.CARE] + self.drives[Drive.PLAY]
        negative = self.drives[Drive.GRIEF] + self.drives[Drive.FEAR] + self.drives[Drive.RAGE]
        total = positive + negative
        if total == 0:
            return 0.0
        return (positive - negative) / total

    def arousal(self) -> float:
        """Total activation magnitude, normalized to [0, 1]."""
        return min(1.0, sum(self.drives.values()) / len(self.drives))

    def intensity(self) -> float:
        """Overall emotional intensity — max drive activation."""
        return max(self.drives.values())

    def to_vector(self) -> list[float]:
        """Return drive states as a fixed-order vector for correlation analysis."""
        return [self.drives[d] for d in Drive]


class AffectiveCore:
    """Manages drive states over time with activation, decay, and state queries."""

    def __init__(self) -> None:
        self.state = AffectiveState()

    def activate(self, drive: Drive, amount: float) -> None:
        """Increase a drive's activation, clamped to [0, 1]."""
        current = self.state.drives[drive]
        self.state.drives[drive] = min(1.0, max(0.0, current + amount))
        self.state.timestamp = time.time()

    def decay(self, elapsed_seconds: float) -> None:
        """Apply exponential decay toward baseline for all drives."""
        for drive in Drive:
            current = self.state.drives[drive]
            baseline = DRIVE_DEFAULTS[drive]["baseline"]
            rate = DRIVE_DEFAULTS[drive]["decay_rate"]
            diff = current - baseline
            decayed = baseline + diff * (1.0 - rate) ** elapsed_seconds
            self.state.drives[drive] = max(0.0, decayed)
        self.state.timestamp = time.time()

    def snapshot(self) -> AffectiveState:
        """Return a copy of the current state."""
        return AffectiveState(
            drives=dict(self.state.drives),
            timestamp=self.state.timestamp,
        )

    def affect_intensity_for_event(
        self,
        seeking: float = 0.0,
        care: float = 0.0,
        play: float = 0.0,
        grief: float = 0.0,
        fear: float = 0.0,
        rage: float = 0.0,
    ) -> float:
        """Activate drives for an event and return the resulting intensity."""
        activations = {
            Drive.SEEKING: seeking,
            Drive.CARE: care,
            Drive.PLAY: play,
            Drive.GRIEF: grief,
            Drive.FEAR: fear,
            Drive.RAGE: rage,
        }
        for drive, amount in activations.items():
            if amount > 0:
                self.activate(drive, amount)
        return self.state.intensity()
