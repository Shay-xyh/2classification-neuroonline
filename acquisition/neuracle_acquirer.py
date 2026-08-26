"""Neuracle/JellyFish acquisition backend based on the legacy collect code."""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from acquisition.base import AbstractAcquirer, AcquirerMetadata, EEGChunk
from utils.preprocessing import resample_eeg

LOGGER = logging.getLogger(__name__)

# Estimate the device-to-PC clock offset from the least-delayed packet in a
# recent window. A bounded window keeps following slow oscillator drift during
# a multi-hour recording; a lifetime minimum would freeze one side of that
# drift into an ever-growing alignment error.
_CLOCK_OFFSET_WINDOW_SEC = 120.0

# Verified against the native 64-channel Neuracle BDF recordings from
# 2026-07-25 and 2026-07-27. Channels 60-64 are ECG/HEOR/HEOL/VEOU/VEOL.
NEURACLE_59_EEG_CHANNEL_NAMES: tuple[str, ...] = (
    "Fpz", "Fp1", "Fp2", "AF3", "AF4", "AF7", "AF8", "Fz",
    "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "FCz",
    "FC1", "FC2", "FC3", "FC4", "FC5", "FC6", "FT7", "FT8",
    "Cz", "C1", "C2", "C3", "C4", "C5", "C6", "T7", "T8",
    "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "TP7", "TP8",
    "Pz", "P3", "P4", "P5", "P6", "P7", "P8", "POz", "PO3",
    "PO4", "PO5", "PO6", "PO7", "PO8", "Oz", "O1", "O2",
)


class NeuracleAcquirer(AbstractAcquirer):
    """Wrap `collect.neuracle_api.DataServerThread` behind the unified acquirer API."""

    def __init__(
        self,
        sfreq: float = 200.0,
        n_channels: int = 59,
        buffer_sec: float = 60.0,
        neuracle_host: str = "127.0.0.1",
        neuracle_port: int = 8712,
        ready_timeout_sec: float = 15.0,
        source_sfreq: float = 250.0,
        transport_delay_sec: float = 0.0,
        eeg_channel_names: Sequence[str] | None = None,
    ) -> None:
        from collect.neuracle_api import DataServerThread

        configured_names = tuple(
            str(name).strip() for name in (eeg_channel_names or ())
        )
        if configured_names and len(configured_names) != int(n_channels):
            raise ValueError(
                "Configured Neuracle EEG channel-name count does not match "
                f"n_channels: {len(configured_names)} != {n_channels}."
            )
        normalized_names = [self._normalize_channel_name(name) for name in configured_names]
        if len(set(normalized_names)) != len(normalized_names):
            raise ValueError("Configured Neuracle EEG channel names contain duplicates.")
        self._expected_eeg_channel_names = configured_names
        self._eeg_channel_indices = tuple(range(int(n_channels)))
        self._source_channel_names: tuple[str, ...] = ()
        self._source_channel_types: tuple[str, ...] = ()
        self.metadata = AcquirerMetadata(
            name="neuracle",
            sfreq=sfreq,
            n_channels=n_channels,
            timestamp_domain="monotonic",
            channel_names=configured_names,
            channel_types=(),
        )
        self._host = neuracle_host
        self._port = neuracle_port
        self._ready_timeout_sec = ready_timeout_sec
        self.source_sfreq = float(source_sfreq)
        self._sample_rate = int(round(self.source_sfreq))
        self._buffer_sec = buffer_sec
        self._transport_delay_sec = max(float(transport_delay_sec), 0.0)
        self._server: DataServerThread | None = None
        self._device_clock_offset_sec: float | None = None
        self._clock_offset_observations: deque[tuple[float, float]] = deque()
        self._last_device_end_ms: float | None = None
        self._device_timestamp_wraps = 0
        self._last_timing_diagnostics: dict[str, float] = {}

    def start_stream(self) -> None:
        from collect.neuracle_api import DataServerThread

        if self._server is not None:
            # Defensive cleanup to avoid leaking a previous connection state.
            self.stop_stream()

        self._server = DataServerThread(sample_rate=self._sample_rate, t_buffer=self._buffer_sec)
        self._device_clock_offset_sec = None
        self._clock_offset_observations.clear()
        self._last_device_end_ms = None
        self._device_timestamp_wraps = 0
        self._last_timing_diagnostics = {}
        not_connected = self._server.connect(hostname=self._host, port=self._port)
        if not_connected:
            self._server = None
            raise RuntimeError("Could not connect to JellyFish/Neuracle forwarder")
        started = time.monotonic()
        while not self._server.isReady():
            if time.monotonic() - started > self._ready_timeout_sec:
                self.stop_stream()
                raise RuntimeError(
                    "Timed out waiting for Neuracle stream metadata. "
                    "Check JellyFish forwarding status and sample-rate settings."
                )
            time.sleep(0.1)
        self._server.start()
        detected_channels = int(getattr(self._server, "n_chan", 0))
        module_name = str(getattr(self._server, "moduleName", "unknown"))
        detected_rates = np.asarray(getattr(self._server, "srates", []), dtype=np.float64).reshape(-1)
        try:
            self._configure_eeg_channel_selection(self._server)
        except (RuntimeError, ValueError):
            self.stop_stream()
            raise
        selected_rates = (
            detected_rates[np.asarray(self._eeg_channel_indices, dtype=np.int64)]
            if detected_rates.size > max(self._eeg_channel_indices, default=-1)
            else np.empty((0,), dtype=np.float64)
        )
        detected_sfreq = (
            float(selected_rates[0])
            if selected_rates.size
            else float(getattr(self._server, "sample_rate", self.source_sfreq))
        )
        LOGGER.info(
            "Neuracle metadata ready: module=%s forwarded_channels=%s selected_eeg_channels=%s sfreq=%.1fHz",
            module_name,
            detected_channels if detected_channels else self.metadata.n_channels,
            self.metadata.n_channels,
            detected_sfreq,
        )
        if detected_channels and self.metadata.n_channels > detected_channels:
            self.stop_stream()
            raise RuntimeError(
                f"Configured channels={self.metadata.n_channels} exceeds forwarded channels={detected_channels}"
            )
        if selected_rates.size != self.metadata.n_channels:
            self.stop_stream()
            raise RuntimeError(
                "Neuracle did not provide a sampling rate for every selected EEG channel."
            )
        if not np.allclose(selected_rates, self.source_sfreq):
            self.stop_stream()
            unique_rates = sorted({float(value) for value in selected_rates})
            raise RuntimeError(
                "Neuracle source sampling rate does not match configuration: "
                f"detected={unique_rates}, configured={self.source_sfreq:.1f}Hz. "
                "Set device.neuracle_source_sfreq to the hardware forwarding rate."
            )
        LOGGER.info("Neuracle acquisition started at %s:%s", self._host, self._port)

    def stop_stream(self) -> None:
        if self._server is None:
            return

        server = self._server
        self._last_timing_diagnostics.update(
            {
                "received_packets": float(getattr(server, "packet_count", 0)),
                "packet_loss_count": float(
                    getattr(server, "packet_loss_count", 0)
                ),
                "total_source_samples": float(
                    getattr(server, "totalSamplesReceived", 0)
                ),
            }
        )
        self._server = None
        try:
            server.stop()
        finally:
            # Give the underlying socket thread a short window to exit fully
            # before the next reconnect attempt.
            time.sleep(0.1)
        LOGGER.info("Neuracle acquisition stopped")

    def get_chunk(self, window_sec: float) -> EEGChunk:
        if self._server is None:
            raise RuntimeError("Neuracle stream is not started")
        get_timed_buffer = getattr(self._server, "GetBufferDataWithTiming", None)
        if callable(get_timed_buffer):
            data, timing = get_timed_buffer()
        else:
            data = self._server.GetBufferData()
            timing = None
        if data.ndim != 2:
            raise RuntimeError(f"Unexpected Neuracle buffer shape: {data.shape}")
        required_channel_index = max(self._eeg_channel_indices, default=-1)
        if data.shape[0] <= required_channel_index:
            raise RuntimeError(
                f"Forwarded channel count {data.shape[0]} does not contain selected channel "
                f"index {required_channel_index}."
            )
        required_source = int(round(window_sec * self.source_sfreq))
        available_source = (
            int(timing.get("total_samples", 0))
            if isinstance(timing, dict)
            else int(data.shape[1])
        )
        if available_source < required_source:
            raise RuntimeError(
                f"Not enough source-rate data in ring buffer: {available_source} < {required_source}"
            )
        raw_eeg = np.asarray(
            data[np.asarray(self._eeg_channel_indices), -required_source:],
            dtype=np.float32,
        )
        eeg = resample_eeg(
            raw_eeg,
            source_sfreq=self.source_sfreq,
            target_sfreq=self.metadata.sfreq,
        )
        required_target = int(round(window_sec * self.metadata.sfreq))
        if eeg.shape[1] != required_target:
            raise RuntimeError(
                f"Resampled Neuracle window has {eeg.shape[1]} points; expected {required_target}."
            )
        window_end = self._resolve_window_end_monotonic(timing)
        timestamps = window_end - (
            np.arange(required_target, 0, -1, dtype=np.float64) / self.metadata.sfreq
        )
        return eeg, timestamps

    def get_continuous_chunk(self, min_window_sec: float) -> EEGChunk:
        """Return the retained source-rate history without window preprocessing."""

        if self._server is None:
            raise RuntimeError("Neuracle stream is not started")
        get_timed_buffer = getattr(self._server, "GetBufferDataWithTiming", None)
        if callable(get_timed_buffer):
            data, timing = get_timed_buffer()
        else:
            data = self._server.GetBufferData()
            timing = None
        if data.ndim != 2:
            raise RuntimeError(f"Unexpected Neuracle buffer shape: {data.shape}")
        required_channel_index = max(self._eeg_channel_indices, default=-1)
        if data.shape[0] <= required_channel_index:
            raise RuntimeError(
                f"Forwarded channel count {data.shape[0]} does not contain selected channel "
                f"index {required_channel_index}."
            )
        required_source = int(round(float(min_window_sec) * self.source_sfreq))
        if data.shape[1] < required_source:
            raise RuntimeError(
                f"Not enough source-rate data in ring buffer: {data.shape[1]} < {required_source}"
            )
        eeg = np.asarray(
            data[np.asarray(self._eeg_channel_indices)],
            dtype=np.float32,
        )
        window_end = self._resolve_window_end_monotonic(timing)
        timestamps = window_end - (
            np.arange(eeg.shape[1], 0, -1, dtype=np.float64) / self.source_sfreq
        )
        return eeg, timestamps

    @property
    def continuous_sfreq(self) -> float:
        return float(self.source_sfreq)

    def get_new_samples(self) -> EEGChunk:
        if self._server is None:
            raise RuntimeError("Neuracle stream is not started")
        get_timed_update = getattr(self._server, "GetBufferUpdateWithTiming", None)
        if callable(get_timed_update):
            data, timing = get_timed_update()
        else:
            data = self._server.buffer.getUpdate()
            timing = None
        if data.ndim != 2:
            raise RuntimeError(f"Unexpected Neuracle update shape: {data.shape}")
        if data.size == 0:
            return (
                np.empty((self.metadata.n_channels, 0), dtype=np.float32),
                np.empty((0,), dtype=np.float64),
            )
        required_channel_index = max(self._eeg_channel_indices, default=-1)
        if data.shape[0] <= required_channel_index:
            raise RuntimeError(
                f"Forwarded channel count {data.shape[0]} does not contain selected channel "
                f"index {required_channel_index}."
            )
        eeg = np.asarray(data[np.asarray(self._eeg_channel_indices)], dtype=np.float32)
        # Incremental reads stay at the hardware rate so calibration events remain
        # aligned to the unmodified continuous recording. Calibrator rescales them
        # when constructing target-rate model windows.
        window_end = self._resolve_window_end_monotonic(timing)
        timestamps = window_end - (
            np.arange(eeg.shape[1], 0, -1, dtype=np.float64) / self.source_sfreq
        )
        return eeg, timestamps

    @property
    def timing_diagnostics(self) -> dict[str, float]:
        """Return the latest source-clock alignment diagnostics."""

        payload = dict(self._last_timing_diagnostics)
        server = self._server
        if server is not None:
            payload["received_packets"] = float(
                getattr(server, "packet_count", 0)
            )
            payload["packet_loss_count"] = float(
                getattr(server, "packet_loss_count", 0)
            )
            payload["total_source_samples"] = float(
                getattr(server, "totalSamplesReceived", 0)
            )
        return payload

    @property
    def channel_diagnostics(self) -> dict[str, object]:
        """Return the auditable source-to-model channel selection."""

        selected = set(self._eeg_channel_indices)
        return {
            "selection_method": (
                "configured_names" if self._expected_eeg_channel_names else "leading_indices"
            ),
            "source_channel_count": len(self._source_channel_names),
            "source_channel_names": list(self._source_channel_names),
            "source_channel_types": list(self._source_channel_types),
            "selected_source_indices_zero_based": list(self._eeg_channel_indices),
            "selected_channel_names": list(self.metadata.channel_names),
            "selected_channel_types": list(self.metadata.channel_types),
            "excluded_channel_names": [
                name
                for index, name in enumerate(self._source_channel_names)
                if index not in selected
            ],
        }

    def _configure_eeg_channel_selection(self, server: object) -> None:
        source_names = tuple(
            str(name).strip() for name in getattr(server, "channelNames", ())
        )
        source_types = tuple(
            str(channel_type).strip() for channel_type in getattr(server, "channelTypes", ())
        )
        self._source_channel_names = source_names
        self._source_channel_types = source_types

        if not self._expected_eeg_channel_names:
            if len(source_names) < self.metadata.n_channels:
                raise RuntimeError(
                    f"Forwarded channel metadata contains only {len(source_names)} channels; "
                    f"expected at least {self.metadata.n_channels}."
                )
            selected_names = source_names[: self.metadata.n_channels]
            selected_types = source_types[: self.metadata.n_channels]
            self._eeg_channel_indices = tuple(range(self.metadata.n_channels))
        else:
            if not source_names:
                raise RuntimeError(
                    "JellyFish did not provide channel names; formal EEG channel identity "
                    "cannot be verified."
                )
            source_lookup: dict[str, int] = {}
            duplicate_names: list[str] = []
            for index, name in enumerate(source_names):
                normalized = self._normalize_channel_name(name)
                if normalized in source_lookup:
                    duplicate_names.append(name)
                source_lookup[normalized] = index
            if duplicate_names:
                raise RuntimeError(
                    "JellyFish channel metadata contains duplicate names: "
                    + ", ".join(duplicate_names)
                )
            missing = [
                name
                for name in self._expected_eeg_channel_names
                if self._normalize_channel_name(name) not in source_lookup
            ]
            if missing:
                raise RuntimeError(
                    "JellyFish is missing required scalp EEG channels: " + ", ".join(missing)
                )
            self._eeg_channel_indices = tuple(
                source_lookup[self._normalize_channel_name(name)]
                for name in self._expected_eeg_channel_names
            )
            selected_names = tuple(source_names[index] for index in self._eeg_channel_indices)
            selected_types = tuple(
                source_types[index] if index < len(source_types) else ""
                for index in self._eeg_channel_indices
            )

        non_eeg = [
            f"{name}({channel_type})"
            for name, channel_type in zip(selected_names, selected_types, strict=False)
            if channel_type and channel_type.upper() != "EEG"
        ]
        if non_eeg:
            raise RuntimeError(
                "Configured scalp channels are not typed as EEG by JellyFish: "
                + ", ".join(non_eeg)
            )
        self.metadata = AcquirerMetadata(
            name="neuracle",
            sfreq=self.metadata.sfreq,
            n_channels=len(selected_names),
            timestamp_domain="monotonic",
            channel_names=tuple(selected_names),
            channel_types=tuple(selected_types),
        )

    @staticmethod
    def _normalize_channel_name(name: str) -> str:
        return "".join(str(name).split()).upper()

    def _resolve_window_end_monotonic(self, timing: object) -> float:
        if not isinstance(timing, dict):
            return time.monotonic() - self._transport_delay_sec

        try:
            device_end_ms = float(timing["device_end_ms"])
            arrival_monotonic = float(timing["arrival_monotonic"])
        except (KeyError, TypeError, ValueError):
            return time.monotonic() - self._transport_delay_sec

        unwrapped_end_ms = self._unwrap_device_timestamp_ms(device_end_ms)
        device_end_sec = unwrapped_end_ms / 1000.0
        observed_offset = arrival_monotonic - device_end_sec
        self._clock_offset_observations.append((device_end_sec, observed_offset))
        oldest_device_sec = device_end_sec - _CLOCK_OFFSET_WINDOW_SEC
        while (
            self._clock_offset_observations
            and self._clock_offset_observations[0][0] < oldest_device_sec
        ):
            self._clock_offset_observations.popleft()

        # Queueing and scheduler delay can only make a packet arrive later, so
        # the recent lower envelope is the best software-only clock-offset
        # estimate. Keeping only a recent window lets the estimate follow slow
        # device/PC oscillator drift in either direction.
        self._device_clock_offset_sec = min(
            offset for _, offset in self._clock_offset_observations
        )

        mapped_end = (
            device_end_sec
            + self._device_clock_offset_sec
            - self._transport_delay_sec
        )
        self._last_timing_diagnostics = {
            "packet_arrival_monotonic": arrival_monotonic,
            "window_end_monotonic": mapped_end,
            "queueing_jitter_sec": max(observed_offset - self._device_clock_offset_sec, 0.0),
            "transport_delay_compensation_sec": self._transport_delay_sec,
            "clock_offset_sec": self._device_clock_offset_sec,
            "clock_offset_window_sec": _CLOCK_OFFSET_WINDOW_SEC,
            "clock_offset_observation_count": float(
                len(self._clock_offset_observations)
            ),
        }
        return mapped_end

    def _unwrap_device_timestamp_ms(self, timestamp_ms: float) -> float:
        raw = float(timestamp_ms)
        modulus = float(2**32)
        normalized = raw % modulus
        if (
            self._last_device_end_ms is not None
            and normalized < self._last_device_end_ms - (modulus / 2.0)
        ):
            self._device_timestamp_wraps += 1
        self._last_device_end_ms = normalized
        return normalized + (self._device_timestamp_wraps * modulus)

    def save_full_buffer_npy(self, path: Path) -> Path:
        """Persist the current full forwarded buffer for diagnostics."""

        if self._server is None:
            raise RuntimeError("Neuracle stream is not started")

        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self._server.GetBufferData().astype(np.float32))
        return path
