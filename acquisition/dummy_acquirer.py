"""Hardware-free dummy EEG acquisition backend.

This module provides a drop-in replacement for physical EEG devices by
implementing the same AbstractAcquirer interface.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from acquisition.base import AbstractAcquirer, AcquirerMetadata, EEGChunk
from utils.preprocessing import resample_eeg

LOGGER = logging.getLogger(__name__)


class DummyAcquirer(AbstractAcquirer):
    """Generate synthetic EEG samples with realistic timing and a ring buffer."""

    def __init__(
        self,
        sfreq: float = 200.0,
        n_channels: int = 59,
        buffer_sec: float = 60.0,
        *,
        source_sfreq: float | None = 250.0,
        channel_names: Sequence[str] | None = None,
        startup_delay_sec: float = 0.2,
        seed: int | None = 17,
        noise_std_uv: float = 8.0,
        drift_std_uv: float = 0.15,
        alpha_hz: float = 10.0,
        alpha_uv: float = 6.0,
        beta_hz: float = 20.0,
        beta_uv: float = 3.5,
        chunk_ms: float = 20.0,
        label_aware: bool = False,
    ) -> None:
        configured_names = tuple(str(name).strip() for name in (channel_names or ()))
        if configured_names and len(configured_names) != int(n_channels):
            raise ValueError(
                "Configured dummy EEG channel-name count does not match "
                f"n_channels: {len(configured_names)} != {n_channels}."
            )
        if len({name.upper() for name in configured_names}) != len(configured_names):
            raise ValueError("Configured dummy EEG channel names contain duplicates.")
        self.source_sfreq = float(sfreq if source_sfreq is None else source_sfreq)
        self.metadata = AcquirerMetadata(
            name="dummy",
            sfreq=float(sfreq),
            n_channels=int(n_channels),
            timestamp_domain="monotonic",
            channel_names=configured_names,
            channel_types=tuple("EEG" for _ in configured_names),
        )
        self._buffer_sec = float(buffer_sec)
        self._startup_delay_sec = float(startup_delay_sec)
        self._rng = np.random.default_rng(seed)
        self._noise_std = float(noise_std_uv)
        self._drift_std = float(drift_std_uv)
        self._alpha_hz = float(alpha_hz)
        self._alpha_uv = float(alpha_uv)
        self._beta_hz = float(beta_hz)
        self._beta_uv = float(beta_uv)
        self._chunk_ms = float(chunk_ms)
        self._label_aware = bool(label_aware)
        self._intent_label = 0

        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

        self._buffer: np.ndarray | None = None
        self._buffer_samples = 0
        self._write_index = 0
        self._total_samples = 0
        self._last_read_sample = 0

        self._phase = self._rng.random(int(n_channels)).astype(np.float64) * (2.0 * np.pi)
        self._channel_gain = (0.8 + 0.4 * self._rng.random(int(n_channels))).astype(np.float64)
        self._drift_state = np.zeros((int(n_channels),), dtype=np.float64)

    @property
    def label_aware(self) -> bool:
        return self._label_aware

    def set_intent(self, label: int) -> None:
        """Select the synthetic left/right pattern used by test simulations."""

        normalized = int(label)
        if normalized not in {0, 1}:
            raise ValueError(f"Unsupported dummy intent label: {label}")
        with self._lock:
            self._intent_label = normalized

    def start_stream(self) -> None:
        if self._running:
            self.stop_stream()

        if self.metadata.sfreq <= 0 or self.source_sfreq <= 0:
            raise RuntimeError("DummyAcquirer requires positive source and target sampling rates.")
        if self.metadata.n_channels <= 0:
            raise RuntimeError("DummyAcquirer requires a positive channel count.")

        self._buffer_samples = max(1, int(round(self._buffer_sec * self.source_sfreq)))
        self._buffer = np.zeros((self.metadata.n_channels, self._buffer_samples), dtype=np.float32)
        self._write_index = 0
        self._total_samples = 0
        self._last_read_sample = 0

        self._running = True
        self._thread = threading.Thread(target=self._producer_loop, name="dummy-eeg-producer", daemon=True)
        self._thread.start()
        LOGGER.info(
            "Dummy acquisition started (hardware_dummy_mode=true): channels=%s source=%.1fHz target=%.1fHz buffer_sec=%.1f",
            self.metadata.n_channels,
            self.source_sfreq,
            self.metadata.sfreq,
            self._buffer_sec,
        )

    def stop_stream(self) -> None:
        if not self._running:
            return
        self._running = False
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=1.5)
        LOGGER.info("Dummy acquisition stopped")

    def get_chunk(self, window_sec: float) -> EEGChunk:
        if not self._running or self._buffer is None:
            raise RuntimeError("Dummy stream is not started")
        required_source = int(round(float(window_sec) * self.source_sfreq))
        if required_source <= 0:
            raise RuntimeError(f"window_sec must be positive, got {window_sec}")

        with self._lock:
            total = int(self._total_samples)
            if total < required_source:
                raise RuntimeError(
                    f"Not enough source-rate data in ring buffer: {total} < {required_source}"
                )
            end = int(self._write_index)
            start = end - required_source
            raw_eeg = self._read_ring(self._buffer, start, end)

        eeg = resample_eeg(
            raw_eeg,
            source_sfreq=self.source_sfreq,
            target_sfreq=self.metadata.sfreq,
        )
        required_target = int(round(float(window_sec) * self.metadata.sfreq))
        if eeg.shape[1] != required_target:
            raise RuntimeError(
                f"Resampled dummy window has {eeg.shape[1]} points; expected {required_target}."
            )

        window_end = time.monotonic()
        timestamps = window_end - (
            np.arange(required_target, 0, -1, dtype=np.float64) / self.metadata.sfreq
        )
        return eeg.astype(np.float32, copy=False), timestamps

    def get_continuous_chunk(self, min_window_sec: float) -> EEGChunk:
        """Return retained source-rate data, matching physical-device backends."""

        if not self._running or self._buffer is None:
            raise RuntimeError("Dummy stream is not started")
        required_source = int(round(float(min_window_sec) * self.source_sfreq))
        with self._lock:
            total = int(self._total_samples)
            if total < required_source:
                raise RuntimeError(
                    f"Not enough source-rate data in ring buffer: {total} < {required_source}"
                )
            retained = min(total, int(self._buffer_samples))
            end = int(self._write_index)
            eeg = self._read_ring(self._buffer, end - retained, end)
        window_end = time.monotonic()
        timestamps = window_end - (
            np.arange(eeg.shape[1], 0, -1, dtype=np.float64) / self.source_sfreq
        )
        return eeg.astype(np.float32, copy=False), timestamps

    @property
    def continuous_sfreq(self) -> float:
        return float(self.source_sfreq)

    def get_new_samples(self) -> EEGChunk:
        if not self._running or self._buffer is None:
            raise RuntimeError("Dummy stream is not started")

        with self._lock:
            total = int(self._total_samples)
            new_count = total - int(self._last_read_sample)
            if new_count <= 0:
                return (
                    np.empty((self.metadata.n_channels, 0), dtype=np.float32),
                    np.empty((0,), dtype=np.float64),
                )

            if new_count > self._buffer_samples:
                LOGGER.warning(
                    "DummyAcquirer update overflow: requested=%s exceeds buffer=%s; returning latest buffer",
                    new_count,
                    self._buffer_samples,
                )
                new_count = int(self._buffer_samples)
                self._last_read_sample = total - new_count

            end = int(self._write_index)
            start = end - int(new_count)
            eeg = self._read_ring(self._buffer, start, end)
            self._last_read_sample = total

        window_end = time.monotonic()
        timestamps = window_end - (
            np.arange(eeg.shape[1], 0, -1, dtype=np.float64) / self.source_sfreq
        )
        return eeg.astype(np.float32, copy=False), timestamps

    def save_full_buffer_npy(self, path: Path) -> Path:
        if not self._running or self._buffer is None:
            raise RuntimeError("Dummy stream is not started")
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            np.save(path, self._buffer.astype(np.float32, copy=True))
        return path

    @staticmethod
    def _read_ring(buffer: np.ndarray, start: int, end: int) -> np.ndarray:
        length = int(end - start)
        if length <= 0:
            return np.empty((buffer.shape[0], 0), dtype=np.float32)
        n = int(buffer.shape[1])
        start_mod = int(start % n)
        end_mod = int(end % n)
        if start_mod < end_mod and length == end_mod - start_mod:
            return buffer[:, start_mod:end_mod]
        if start_mod < end_mod:
            return buffer[:, start_mod:end_mod]
        head = buffer[:, start_mod:]
        tail = buffer[:, :end_mod]
        if head.size == 0:
            return tail
        if tail.size == 0:
            return head
        return np.concatenate([head, tail], axis=1)

    def _producer_loop(self) -> None:
        if self._startup_delay_sec > 0:
            time.sleep(self._startup_delay_sec)

        sfreq = float(self.source_sfreq)
        chunk_samples = max(1, int(round(sfreq * (self._chunk_ms / 1000.0))))
        dt = 1.0 / sfreq
        next_tick = time.perf_counter()
        local_sample_index = 0

        while self._running:
            now = time.perf_counter()
            if now < next_tick:
                time.sleep(min(0.01, max(0.0, next_tick - now)))
                continue

            samples = self._generate_block(chunk_samples, start_index=local_sample_index, dt=dt)
            local_sample_index += int(samples.shape[1])

            with self._lock:
                if self._buffer is None:
                    continue
                self._write_ring(samples)

            next_tick += chunk_samples * dt

    def _write_ring(self, samples: np.ndarray) -> None:
        if self._buffer is None:
            return
        n = int(self._buffer.shape[1])
        count = int(samples.shape[1])
        if count >= n:
            self._buffer[:] = samples[:, -n:].astype(np.float32, copy=False)
            self._write_index = 0
            self._total_samples += count
            return

        start = int(self._write_index)
        end = start + count
        if end <= n:
            self._buffer[:, start:end] = samples
        else:
            first = n - start
            self._buffer[:, start:] = samples[:, :first]
            self._buffer[:, : end - n] = samples[:, first:]
        self._write_index = int(end % n)
        self._total_samples += count

    def _generate_block(self, n_samples: int, *, start_index: int, dt: float) -> np.ndarray:
        t = (np.arange(n_samples, dtype=np.float64) + float(start_index)) * dt
        with self._lock:
            intent_label = self._intent_label

        if self._label_aware:
            alpha_hz = 8.0 + (2.0 * intent_label)
            beta_hz = 18.0 + (1.5 * intent_label)
            spatial_gain = np.ones((self.metadata.n_channels,), dtype=np.float64)
            if intent_label == 0:
                spatial_gain[0::2] = 1.45
                spatial_gain[1::2] = 0.75
            elif intent_label == 1:
                spatial_gain[0::2] = 0.75
                spatial_gain[1::2] = 1.45
        else:
            alpha_hz = self._alpha_hz
            beta_hz = self._beta_hz
            spatial_gain = np.ones((self.metadata.n_channels,), dtype=np.float64)

        alpha = np.sin(2.0 * np.pi * alpha_hz * t[None, :] + self._phase[:, None])
        beta = np.sin(2.0 * np.pi * beta_hz * t[None, :] + (self._phase[:, None] * 0.5))

        drift = self._rng.normal(0.0, self._drift_std, size=(self.metadata.n_channels, n_samples)).astype(np.float64)
        self._drift_state = (0.995 * self._drift_state) + drift.mean(axis=1)
        drift_component = self._drift_state[:, None]

        noise = self._rng.normal(0.0, self._noise_std, size=(self.metadata.n_channels, n_samples)).astype(np.float64)
        signal = (
            (self._alpha_uv * alpha + self._beta_uv * beta)
            * self._channel_gain[:, None]
            * spatial_gain[:, None]
            + noise
            + drift_component
        )
        return signal.astype(np.float32, copy=False)
