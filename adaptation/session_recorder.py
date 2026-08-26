"""Continuous EEG and sample-aligned event recording for protocol sessions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from acquisition.base import AbstractAcquirer


@dataclass(slots=True)
class SessionEvent:
    name: str
    sample_index: int
    payload: dict[str, Any]


class SessionRecorder:
    """Collect continuous EEG and aligned events during one protocol session."""

    def __init__(self, acquirer: AbstractAcquirer, *, sfreq: float, n_channels: int) -> None:
        self._acquirer = acquirer
        self._sfreq = float(sfreq)
        self._n_channels = int(n_channels)
        self._chunks: list[np.ndarray] = []
        self._events: list[SessionEvent] = []
        self._sample_count = 0
        self._latest_sample_end_monotonic: float | None = None

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def events(self) -> list[SessionEvent]:
        return list(self._events)

    def pull(self) -> np.ndarray:
        samples, timestamps = self._acquirer.get_new_samples()
        if samples.size == 0:
            return np.empty((self._n_channels, 0), dtype=np.float32)
        if samples.ndim != 2 or samples.shape[0] < self._n_channels:
            raise RuntimeError(f"Unexpected incremental EEG shape: {samples.shape}")
        eeg = np.asarray(samples[: self._n_channels], dtype=np.float32)
        self._chunks.append(eeg)
        self._sample_count += int(eeg.shape[1])
        values = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if (
            str(getattr(self._acquirer.metadata, "timestamp_domain", "relative")).lower()
            == "monotonic"
            and values.size == eeg.shape[1]
            and values.size
            and np.all(np.isfinite(values))
        ):
            self._latest_sample_end_monotonic = float(values[-1]) + (1.0 / self._sfreq)
        return eeg

    def add_event(
        self,
        name: str,
        *,
        timestamp_monotonic: float | None = None,
        **payload: Any,
    ) -> SessionEvent:
        event_time = (
            time.monotonic()
            if timestamp_monotonic is None
            else float(timestamp_monotonic)
        )
        sample_index = self._sample_count
        alignment_method = "latest-received-sample"
        if self._latest_sample_end_monotonic is not None:
            sample_index += int(
                round((event_time - self._latest_sample_end_monotonic) * self._sfreq)
            )
            sample_index = max(sample_index, 0)
            alignment_method = "source-clock-projection"
        event_payload = dict(payload)
        event_payload["alignment_method"] = alignment_method
        event = SessionEvent(
            name=name,
            sample_index=sample_index,
            payload=event_payload,
        )
        self._events.append(event)
        return event

    def export(self, output_dir: Path, *, metadata: dict[str, Any]) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        eeg = self.to_array()
        self._save_npy_atomic(output_dir / "continuous_eeg.npy", eeg)
        events_path = output_dir / "events.json"
        events_temporary = output_dir / ".events.json.tmp"
        with events_temporary.open("w", encoding="utf-8") as handle:
            json.dump([asdict(event) for event in self._events], handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(events_temporary, events_path)
        metadata_path = output_dir / "metadata.json"
        metadata_temporary = output_dir / ".metadata.json.tmp"
        with metadata_temporary.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(metadata_temporary, metadata_path)
        return output_dir

    def to_array(self) -> np.ndarray:
        try:
            self.pull()
        except RuntimeError as exc:
            # Calibration stops the stream before export; keep buffered chunks in that case.
            if not self._is_stream_not_started_error(exc):
                raise
        if not self._chunks:
            return np.empty((self._n_channels, 0), dtype=np.float32)
        return np.concatenate(self._chunks, axis=1).astype(np.float32)

    @staticmethod
    def _is_stream_not_started_error(exc: RuntimeError) -> bool:
        message = str(exc).lower()
        return "not started" in message and "stream" in message

    @staticmethod
    def _save_npy_atomic(path: Path, array: np.ndarray) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.save(handle, array)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
