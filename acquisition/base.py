"""Abstract EEG acquisition interfaces shared by all devices."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

EEGChunk: TypeAlias = tuple[np.ndarray, np.ndarray]
GOOD_IMPEDANCE_THRESHOLD_KOHM = 5.0
OK_IMPEDANCE_THRESHOLD_KOHM = 10.0


@dataclass(slots=True)
class AcquirerMetadata:
    """Static metadata exposed by an acquisition backend."""

    name: str
    sfreq: float
    n_channels: int
    timestamp_domain: str = "relative"
    channel_names: tuple[str, ...] = ()
    channel_types: tuple[str, ...] = ()


@dataclass(slots=True)
class ElectrodeImpedance:
    """Normalized impedance/contact-quality result for one electrode."""

    channel: int
    name: str | None
    impedance_kohm: float | None
    status: str
    message: str | None = None


def classify_impedance_kohm(impedance_kohm: float | None) -> str:
    """Map a numeric impedance in kOhm into the GUI status buckets."""

    if impedance_kohm is None:
        return "unknown"
    if impedance_kohm < GOOD_IMPEDANCE_THRESHOLD_KOHM:
        return "good"
    if impedance_kohm <= OK_IMPEDANCE_THRESHOLD_KOHM:
        return "ok"
    return "poor"


class AbstractAcquirer(ABC):
    """Unified interface for all EEG sources."""

    metadata: AcquirerMetadata

    @abstractmethod
    def start_stream(self) -> None:
        """Start the underlying device or stream connection."""

    @abstractmethod
    def stop_stream(self) -> None:
        """Stop the underlying device or stream connection."""

    @abstractmethod
    def get_chunk(self, window_sec: float) -> EEGChunk:
        """Return the latest EEG window and timestamps."""

    @abstractmethod
    def get_new_samples(self) -> EEGChunk:
        """Return newly arrived EEG samples since the previous incremental read."""

    def get_continuous_chunk(self, min_window_sec: float) -> EEGChunk:
        """Return continuous history for preprocessing before model windowing.

        Backends without a source-rate rolling buffer fall back to their latest
        model-rate window. Hardware backends should override this method and
        expose as much chronological source history as is currently retained.
        """

        return self.get_chunk(min_window_sec)

    @property
    def continuous_sfreq(self) -> float:
        """Sampling rate returned by :meth:`get_continuous_chunk`."""

        return float(self.metadata.sfreq)

    def supports_impedance_check(self) -> bool:
        """Return whether this backend can query real hardware impedance/lead-off data."""

        return False

    def check_impedance(self, timeout_sec: float = 10.0) -> list[ElectrodeImpedance]:
        """Return impedance/contact-quality results from the underlying hardware SDK."""

        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement hardware impedance checking."
        )
