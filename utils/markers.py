"""Marker and command-stream helpers."""

from __future__ import annotations

import json
import logging
import select
import socket
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Any

LOGGER = logging.getLogger(__name__)

PROTOCOL_EVENT_CODES = {
    "session_start": 101,
    "session_end": 102,
    "block_start": 120,
    "block_end": 121,
    "automatic_break_start": 122,
    "automatic_break_end": 123,
    "fixation_on": 130,
    "cue_left_on": 131,
    "cue_right_on": 132,
    "motor_imagery_left_on": 134,
    "motor_imagery_right_on": 135,
    "motor_imagery_off": 136,
    "manual_pause_start": 140,
    "manual_pause_end": 141,
    "trial_discarded": 142,
}


def _encode_command_payload(command: str) -> bytes:
    return json.dumps(
        {
            "command": command,
            "ts_ms": int(time.time() * 1000),
        },
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"


class MarkerBackend(ABC):
    """Abstract marker sink."""

    @abstractmethod
    def send(self, label: int, timestamp: float | None = None) -> None:
        """Emit a marker value."""

    def send_event(self, event_name: str, timestamp: float | None = None) -> None:
        """Emit a named protocol marker when supported."""

        code = PROTOCOL_EVENT_CODES.get(event_name)
        if code is None:
            raise ValueError(f"Unknown protocol event: {event_name}")
        self.send(code, timestamp=timestamp)


class NoOpMarkerBackend(MarkerBackend):
    """Marker sink used by legacy decoder tests without external hardware."""

    def send(self, label: int, timestamp: float | None = None) -> None:
        LOGGER.debug("No-op marker emitted label=%s timestamp=%s", label, timestamp)

    def send_event(self, event_name: str, timestamp: float | None = None) -> None:
        LOGGER.debug("No-op protocol marker emitted event=%s timestamp=%s", event_name, timestamp)

class LSLCommandOutlet:
    """LSL stream used to publish decoded MI commands."""

    def __init__(self, stream_name: str, stream_type: str) -> None:
        from pylsl import StreamInfo, StreamOutlet

        info = StreamInfo(stream_name, stream_type, 1, 0.0, "string", "oi-mi-command-stream")
        self._outlet = StreamOutlet(info)

    def push(self, command: str) -> None:
        self._outlet.push_sample([command], time.time())


class ArTcpCommandSender:
    """TCP client for the AR game command server."""

    def __init__(self, host: str, port: int, *, timeout_sec: float = 3.0) -> None:
        self._host = host
        self._port = port
        self._timeout_sec = timeout_sec
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()
        self._receive_buffer = bytearray()
        self._events: deque[dict[str, Any]] = deque()

    def push(self, command: str) -> None:
        payload = _encode_command_payload(command)
        with self._lock:
            try:
                self._ensure_connected()
                assert self._sock is not None
                self._sock.sendall(payload)
            except OSError as exc:
                self._close_locked()
                raise RuntimeError(
                    f"Failed to send AR command to {self._host}:{self._port}: {exc}"
                ) from exc

    def push_with_ack(self, command: str) -> dict[str, Any]:
        """Send a scene command and return Unity's structured confirmation."""

        payload = _encode_command_payload(command)
        with self._lock:
            try:
                self._ensure_connected()
                assert self._sock is not None
                sent_at = time.monotonic()
                self._sock.sendall(payload)
                deadline = time.monotonic() + self._timeout_sec
                while time.monotonic() < deadline:
                    response = self._drain_messages_locked(expected_ack=command)
                    if response is not None:
                        received_at = float(
                            response.get("_received_at_monotonic", time.monotonic())
                        )
                        response.setdefault("_sent_at_monotonic", sent_at)
                        response.setdefault(
                            "_ack_round_trip_sec",
                            max(received_at - sent_at, 0.0),
                        )
                        return response
                    chunk = self._sock.recv(4096)
                    if not chunk:
                        raise OSError("Unity closed the connection before scene ACK")
                    self._receive_buffer.extend(chunk)
                raise TimeoutError(f"Unity did not ACK {command}")
            except (OSError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
                self._close_locked()
                raise RuntimeError(
                    f"Unity scene protocol mismatch or connection failure for {command}: {exc}"
                ) from exc

    def poll_events(self) -> list[dict[str, Any]]:
        """Return Unity events already available without blocking decoding."""

        with self._lock:
            if self._sock is None:
                return []
            try:
                self._drain_messages_locked()
                while select.select([self._sock], [], [], 0.0)[0]:
                    chunk = self._sock.recv(4096)
                    if not chunk:
                        raise OSError("Unity closed the scene event connection")
                    self._receive_buffer.extend(chunk)
                    self._drain_messages_locked()
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self._close_locked()
                raise RuntimeError(f"Failed to receive Unity scene event: {exc}") from exc

            events = list(self._events)
            self._events.clear()
            return events

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def _ensure_connected(self) -> None:
        if self._sock is not None:
            return
        self._sock = socket.create_connection(
            (self._host, self._port),
            timeout=self._timeout_sec,
        )
        self._sock.settimeout(self._timeout_sec)

    def _drain_messages_locked(
        self,
        *,
        expected_ack: str | None = None,
    ) -> dict[str, Any] | None:
        ack_response: dict[str, Any] | None = None
        while True:
            line_break = self._receive_buffer.find(b"\n")
            if line_break < 0:
                return ack_response
            raw = bytes(self._receive_buffer[:line_break]).strip()
            del self._receive_buffer[: line_break + 1]
            if not raw:
                continue
            response = json.loads(raw.decode("utf-8"))
            response.setdefault("_received_at_monotonic", time.monotonic())
            if str(response.get("event", "")).strip():
                self._events.append(response)
            if (
                expected_ack is not None
                and str(response.get("nack", "")).upper() == expected_ack.upper()
            ):
                raise ValueError(f"Unity rejected {expected_ack}: {response}")
            if (
                expected_ack is not None
                and str(response.get("ack", "")).upper() == expected_ack.upper()
            ):
                ack_response = response

    def _close_locked(self) -> None:
        if self._sock is None:
            return
        try:
            self._sock.close()
        finally:
            self._sock = None
            self._receive_buffer.clear()
            self._events.clear()
