"""Shared AR game command router with manual web override support."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from utils.markers import ArTcpCommandSender

LOGGER = logging.getLogger(__name__)

_ROUTER_LOCK = threading.Lock()
_ROUTER_INSTANCE: "SharedGameCommandRouter | None" = None


def _build_transport(config: dict[str, Any]) -> Any:
    game_output_cfg = config.get("output", {}).get("ar_game", {})
    enabled = bool(game_output_cfg.get("enabled", False))
    if not enabled:
        return None

    host = str(game_output_cfg.get("host", "127.0.0.1")).strip() or "127.0.0.1"
    port = int(game_output_cfg.get("port", 5005))
    timeout_sec = float(game_output_cfg.get("timeout_sec", 3.0))
    return ArTcpCommandSender(host=host, port=port, timeout_sec=timeout_sec)


class _GameCommandProxy:
    def __init__(self, router: "SharedGameCommandRouter", *, source: str) -> None:
        self._router = router
        self._source = source

    def push(self, command: str) -> None:
        self._router.push(command, source=self._source)

    def push_with_ack(self, command: str) -> dict[str, Any]:
        return self._router.push_with_ack(command, source=self._source)

    def poll_events(self) -> list[dict[str, Any]]:
        return self._router.poll_events()

    def close(self) -> None:
        # Shared transport lives for the process lifetime.
        return


class SharedGameCommandRouter:
    """Arbitrate commands from realtime decoding and manual web control."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._transport = _build_transport(config)
        web_cfg = config.get("output", {}).get("web_control", {})
        self._manual_hold_sec = float(web_cfg.get("manual_override_hold_sec", 0.8))
        self._manual_release_sec = float(web_cfg.get("manual_override_release_sec", 0.25))
        self._manual_override_until = 0.0
        self._lock = threading.Lock()

    def push(self, command: str, *, source: str) -> None:
        if self._transport is None:
            raise RuntimeError("AR game output is disabled in config.")

        now = time.monotonic()
        with self._lock:
            if source == "web":
                hold_sec = self._manual_release_sec if command == "STOP" else self._manual_hold_sec
                self._manual_override_until = now + max(hold_sec, 0.0)
            elif source == "decoder" and now < self._manual_override_until:
                LOGGER.debug(
                    "Dropped decoder command '%s' because manual override is active for %.3fs",
                    command,
                    self._manual_override_until - now,
                )
                return

            self._transport.push(command)

    def push_with_ack(self, command: str, *, source: str) -> dict[str, Any]:
        """Forward a scene command and require the Unity runtime to confirm it.

        Scene-layout commands are independent from steering, so a temporary
        manual steering override must not suppress them. Otherwise the decoder
        could label EEG against a layout that Unity never applied.
        """

        if self._transport is None:
            raise RuntimeError("AR game output is disabled in config.")

        push_with_ack = getattr(self._transport, "push_with_ack", None)
        if not callable(push_with_ack):
            raise RuntimeError(
                "Configured AR game transport does not support scene ACK. "
                "Use the direct Unity TCP transport for the continuous-scene protocol."
            )

        with self._lock:
            response = push_with_ack(command)
        if not isinstance(response, dict):
            raise RuntimeError(
                f"Unity returned an invalid structured ACK for {command}: {response!r}"
            )
        return response

    def poll_events(self) -> list[dict[str, Any]]:
        if self._transport is None:
            return []
        poll_events = getattr(self._transport, "poll_events", None)
        if not callable(poll_events):
            return []
        with self._lock:
            return list(poll_events())

    def build_proxy(self, *, source: str) -> _GameCommandProxy:
        return _GameCommandProxy(self, source=source)

def get_shared_game_command_router(config: dict[str, Any]) -> SharedGameCommandRouter:
    global _ROUTER_INSTANCE
    with _ROUTER_LOCK:
        if _ROUTER_INSTANCE is None:
            _ROUTER_INSTANCE = SharedGameCommandRouter(config)
        return _ROUTER_INSTANCE
