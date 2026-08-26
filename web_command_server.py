"""HTTP bridge that accepts simple web commands and forwards them to the AR game."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from game_command_router import get_shared_game_command_router

LOGGER = logging.getLogger(__name__)
_SERVER_LOCK = threading.Lock()
_SERVER_INSTANCE: "WebCommandServer | None" = None

_ALLOWED_COMMANDS = {"START", "STOP", "RESTART", "LEFT", "RIGHT"}


@dataclass(slots=True)
class _CommandTask:
    command: str
    done: threading.Event = field(default_factory=threading.Event)
    error: str | None = None
    enqueued_at: float = field(default_factory=time.time)


class _WebCommandHandler(BaseHTTPRequestHandler):
    server: "_CommandHttpServer"

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/api/health":
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})
            return
        self._send_json(
            HTTPStatus.OK,
            {
                "ok": True,
                "listen_host": self.server.app.listen_host,
                "listen_port": self.server.app.listen_port,
                "queue_depth": self.server.app.queue_depth,
            },
        )

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/command":
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Invalid Content-Length"})
            return

        raw_body = self.rfile.read(max(length, 0))
        try:
            payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        except json.JSONDecodeError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Invalid JSON body"})
            return

        command = str(payload.get("command", "")).strip().upper()
        if command not in _ALLOWED_COMMANDS:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": f"Unsupported command: {command or '<empty>'}"},
            )
            return

        try:
            self.server.app.push(command)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Web command forwarding failed for %s: %s", command, exc)
            self._send_json(HTTPStatus.BAD_GATEWAY, {"ok": False, "error": str(exc), "command": command})
            return

        self._send_json(HTTPStatus.OK, {"ok": True, "command": command})

    def log_message(self, format: str, *args: Any) -> None:
        LOGGER.debug("web-command-server " + format, *args)

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")


class _CommandHttpServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], handler_class: type[BaseHTTPRequestHandler], app: "WebCommandServer") -> None:
        super().__init__(server_address, handler_class)
        self.daemon_threads = True
        self.app = app


class WebCommandServer:
    """Background HTTP server that exposes a simple command bridge for web UIs."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        web_cfg = config.get("output", {}).get("web_control", {})
        self.listen_host = str(web_cfg.get("host", "0.0.0.0")).strip() or "0.0.0.0"
        self.listen_port = int(web_cfg.get("port", 8765))
        self._router = get_shared_game_command_router(config)
        self._queue: queue.Queue[_CommandTask | None] = queue.Queue()
        self._server = _CommandHttpServer(
            (self.listen_host, self.listen_port),
            partial(_WebCommandHandler),
            self,
        )
        self._thread = threading.Thread(target=self._server.serve_forever, name="oi-mi-web-command-server", daemon=True)
        self._worker = threading.Thread(target=self._run_sender_loop, name="oi-mi-web-command-sender", daemon=True)
        self._running = True

    def start(self) -> None:
        self._worker.start()
        self._thread.start()
        LOGGER.info("Web command server listening on http://%s:%s", self.listen_host, self.listen_port)

    def push(self, command: str) -> None:
        task = _CommandTask(command=command)
        self._queue.put(task)
        if not task.done.wait(timeout=5.0):
            raise RuntimeError(f"Timed out while forwarding command: {command}")
        if task.error is not None:
            raise RuntimeError(task.error)

    def close(self) -> None:
        self._running = False
        self._server.shutdown()
        self._server.server_close()
        self._queue.put(None)
        self._thread.join(timeout=1.0)
        self._worker.join(timeout=1.0)

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    def _run_sender_loop(self) -> None:
        while self._running:
            task = self._queue.get()
            if task is None:
                break
            try:
                self._router.push(task.command, source="web")
            except Exception as exc:  # noqa: BLE001
                task.error = str(exc)
            finally:
                task.done.set()


def start_web_command_server(config: dict[str, Any]) -> WebCommandServer | None:
    """Start the singleton web command server when enabled."""

    web_cfg = config.get("output", {}).get("web_control", {})
    if not bool(web_cfg.get("enabled", True)):
        return None

    global _SERVER_INSTANCE
    with _SERVER_LOCK:
        if _SERVER_INSTANCE is not None:
            return _SERVER_INSTANCE
        try:
            server = WebCommandServer(config)
            server.start()
        except OSError as exc:
            LOGGER.warning("Failed to start web command server on %s:%s: %s", web_cfg.get("host", "127.0.0.1"), web_cfg.get("port", 8765), exc)
            return None
        _SERVER_INSTANCE = server
        return _SERVER_INSTANCE
