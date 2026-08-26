"""Helpers for locating and launching the bundled Unity driving task."""

from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import os
import socket
import subprocess
import time
from ctypes import wintypes
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILD_DIR = Path("..") / "oi-car-unity-src" / "Car_game" / "Builds" / "Windows"
DEFAULT_EXECUTABLE = DEFAULT_BUILD_DIR / "ARPrototype3D.exe"
RUNTIME_MANIFEST_FILENAME = "oi-mi-runtime.json"
REQUIRED_RUNTIME_PROTOCOL = "continuous-scene-v5-centered-single-decision"
DEFAULT_UNITY_WINDOW_TITLE = "ARPrototype3D"
REQUIRED_RUNTIME_FEATURES = frozenset(
    {
        "continuous_control",
        "obstacle_truth",
        "scene_ack",
        "scene_failure_event",
        "lane_state_ack",
        "relative_action_truth",
        "dynamic_action_truth",
        "lane_settled_event",
        "centered_scene_start",
        "obstacles_visible_during_primary_window",
        "single_decision_control",
    }
)
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}

# Win32 window-style flags used to compensate for Unity builds made with
# PlayerSettings.resizableWindow disabled.  Importing ctypes is portable; the
# WinDLL calls below are guarded by os.name == "nt".
_GWL_STYLE = -16
_WS_CAPTION = 0x00C00000
_WS_THICKFRAME = 0x00040000
_WS_MAXIMIZEBOX = 0x00010000
_SWP_NOSIZE = 0x0001
_SWP_NOMOVE = 0x0002
_SWP_NOZORDER = 0x0004
_SWP_FRAMECHANGED = 0x0020


def resolve_project_path(path_value: str | os.PathLike[str], *, project_root: Path | None = None) -> Path:
    """Resolve an absolute or project-relative path."""

    root = (project_root or PROJECT_ROOT).resolve()
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def configured_unity_executable(config: dict[str, Any], *, project_root: Path | None = None) -> Path:
    """Return the configured Unity executable path."""

    ar_game_cfg = config.get("output", {}).get("ar_game", {})
    executable_path = str(ar_game_cfg.get("executable_path") or DEFAULT_EXECUTABLE)
    return resolve_project_path(executable_path, project_root=project_root)


def validate_unity_runtime(executable: Path) -> dict[str, Any]:
    """Validate that a bundled Unity player implements the required protocol."""

    executable = executable.resolve()
    if not executable.is_file():
        raise RuntimeError(f"Unity game executable was not found: {executable}")

    build_dir = executable.parent
    managed_dir = build_dir / f"{executable.stem}_Data" / "Managed"
    required_paths = (
        build_dir / "UnityPlayer.dll",
        managed_dir / "ARPrototype3D.Runtime.dll",
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise RuntimeError(
            "Unity runtime is incomplete; missing: " + ", ".join(missing)
        )

    manifest_path = build_dir / RUNTIME_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise RuntimeError(
            f"Unity runtime manifest was not found: {manifest_path}. "
            "Install the current oi-mi Unity release; older unversioned builds "
            "cannot be used for the continuous-scene experiment."
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid Unity runtime manifest: {manifest_path}: {exc}") from exc

    protocol = str(manifest.get("protocol_version", "")).strip()
    if protocol != REQUIRED_RUNTIME_PROTOCOL:
        raise RuntimeError(
            "Unity runtime protocol mismatch: "
            f"expected {REQUIRED_RUNTIME_PROTOCOL!r}, got {protocol or '<missing>'!r}. "
            "Reinstall the current oi-mi Unity release."
        )

    features = {
        str(feature).strip()
        for feature in manifest.get("features", [])
        if str(feature).strip()
    }
    missing_features = sorted(REQUIRED_RUNTIME_FEATURES - features)
    if missing_features:
        raise RuntimeError(
            "Unity runtime is missing required features: "
            + ", ".join(missing_features)
        )

    declared_files = manifest.get("files", {})
    if not isinstance(declared_files, dict) or not declared_files:
        raise RuntimeError("Unity runtime manifest must declare file SHA-256 hashes.")
    required_declared_files = {
        executable.name,
        "UnityPlayer.dll",
        f"{executable.stem}_Data/Managed/ARPrototype3D.Runtime.dll",
    }
    normalized_declared_files = {
        str(relative_name).replace("\\", "/") for relative_name in declared_files
    }
    missing_hashes = sorted(required_declared_files - normalized_declared_files)
    if missing_hashes:
        raise RuntimeError(
            "Unity runtime manifest is missing critical file hashes: "
            + ", ".join(missing_hashes)
        )
    for relative_name, expected_hash in declared_files.items():
        relative_path = Path(str(relative_name))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError(f"Unsafe path in Unity runtime manifest: {relative_name}")
        file_path = (build_dir / relative_path).resolve()
        try:
            file_path.relative_to(build_dir)
        except ValueError as exc:
            raise RuntimeError(
                f"Unsafe path in Unity runtime manifest: {relative_name}"
            ) from exc
        if not file_path.is_file():
            raise RuntimeError(f"Unity runtime file is missing: {file_path}")
        actual_hash = _sha256(file_path)
        if actual_hash.casefold() != str(expected_hash).strip().casefold():
            raise RuntimeError(
                f"Unity runtime file hash mismatch: {relative_name}. "
                "Reinstall the current oi-mi Unity release."
            )

    return manifest


def write_unity_runtime_manifest(executable: Path, *, build_id: str) -> Path:
    """Create the protocol manifest shipped beside a Unity Windows player."""

    executable = executable.resolve()
    build_id = str(build_id).strip()
    if not build_id:
        raise ValueError("build_id must not be empty.")

    build_dir = executable.parent
    unity_player = build_dir / "UnityPlayer.dll"
    managed_dir = build_dir / f"{executable.stem}_Data" / "Managed"
    managed_dlls = (managed_dir / "ARPrototype3D.Runtime.dll",)
    for required_path in (executable, unity_player, *managed_dlls):
        if not required_path.is_file():
            raise RuntimeError(f"Unity runtime file is missing: {required_path}")

    manifest = {
        "schema_version": 1,
        "build_id": build_id,
        "protocol_version": REQUIRED_RUNTIME_PROTOCOL,
        "features": sorted(REQUIRED_RUNTIME_FEATURES),
        "files": {
            executable.name: _sha256(executable),
            unity_player.name: _sha256(unity_player),
            **{
                managed_dll.relative_to(build_dir).as_posix(): _sha256(managed_dll)
                for managed_dll in managed_dlls
            },
        },
    }
    manifest_path = build_dir / RUNTIME_MANIFEST_FILENAME
    temporary_path = manifest_path.with_suffix(f"{manifest_path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, manifest_path)
    return manifest_path


def ensure_unity_game_running(
    config: dict[str, Any],
    *,
    console: Any | None = None,
    project_root: Path | None = None,
) -> subprocess.Popen[Any] | None:
    """Launch the Unity game executable when local auto-launch is enabled."""

    ar_game_cfg = config.get("output", {}).get("ar_game", {})
    if not bool(ar_game_cfg.get("enabled", False)):
        return None
    if not bool(ar_game_cfg.get("auto_launch", False)):
        return None

    host = str(ar_game_cfg.get("host", "127.0.0.1")).strip() or "127.0.0.1"
    port = int(ar_game_cfg.get("port", 5005))
    timeout_sec = float(ar_game_cfg.get("startup_timeout_sec", 15.0))

    if not _is_local_host(host):
        _notify(
            console,
            f"Unity auto-launch skipped because AR game host is not local: {host}:{port}",
        )
        return None

    executable = configured_unity_executable(config, project_root=project_root)
    try:
        manifest = validate_unity_runtime(executable)
    except RuntimeError as exc:
        raise RuntimeError(
            f"{exc} Rebuild the standalone `oi-car-unity-src/Car_game` project "
            "before realtime decoding."
        ) from exc

    _notify(
        console,
        "Unity runtime verified: "
        f"{manifest.get('build_id', 'unknown')} ({REQUIRED_RUNTIME_PROTOCOL})",
    )

    if _is_tcp_open(host, port, timeout_sec=0.25):
        _enable_existing_window_resize(config, project_root=project_root, console=console)
        return None

    _notify(console, f"Launching Unity game: {executable}")
    process = _launch_process(executable, ar_game_cfg)
    if timeout_sec <= 0:
        return process

    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Unity game exited before opening TCP {host}:{port} "
                f"(exit code {process.returncode})."
            )
        if _is_tcp_open(host, port, timeout_sec=0.25):
            _enable_launched_window_resize(process, ar_game_cfg, console=console)
            return process
        time.sleep(0.25)

    raise RuntimeError(
        f"Unity game did not open TCP {host}:{port} within {timeout_sec:.1f}s. "
        "Start the executable manually once and check Unity logs if this repeats."
    )


def wait_for_unity_scene_ready(
    outlet: Any,
    *,
    timeout_sec: float,
    retry_interval_sec: float = 0.25,
    console: Any | None = None,
) -> dict[str, Any]:
    """Wait for the driving scene's authoritative, non-mutating state ACK.

    Opening the Unity TCP port only proves that the Game Hub is alive. The
    driving controller registers ``SCENE_STATE`` after the 3D scene has loaded,
    so this handshake removes the fixed-sleep startup race without creating an
    obstacle layout or advancing Unity's scene counter.
    """

    push_with_ack = getattr(outlet, "push_with_ack", None)
    if not callable(push_with_ack):
        raise RuntimeError("Unity command outlet does not support scene readiness ACK.")

    timeout = max(float(timeout_sec), 0.1)
    retry_interval = max(float(retry_interval_sec), 0.0)
    deadline = time.monotonic() + timeout
    attempts = 0
    last_error: Exception | None = None
    while True:
        attempts += 1
        try:
            response = push_with_ack("SCENE_STATE")
            if not isinstance(response, dict):
                raise ValueError(f"invalid SCENE_STATE response: {response!r}")
            if str(response.get("ack", "")).strip().upper() != "SCENE_STATE":
                raise ValueError(f"unexpected SCENE_STATE ACK: {response!r}")
            if (
                str(response.get("protocol_version", "")).strip()
                != REQUIRED_RUNTIME_PROTOCOL
            ):
                raise ValueError(
                    "Unity driving scene returned an incompatible protocol: "
                    f"{response!r}"
                )
            scene_number = int(response.get("scene_number", -1))
            current_lane = int(response.get("current_lane", -9))
            if scene_number < 1 or current_lane not in {-1, 0, 1}:
                raise ValueError(f"invalid Unity lane-state ACK: {response!r}")
            _notify(
                console,
                "Unity driving scene ready: "
                f"next_scene={scene_number}, current_lane={current_lane}, "
                f"attempts={attempts}",
            )
            return response
        except Exception as exc:  # noqa: BLE001
            last_error = exc

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        if retry_interval > 0:
            time.sleep(min(retry_interval, remaining))

    raise RuntimeError(
        "Unity TCP opened, but the driving scene did not become ACK-ready within "
        f"{timeout:.1f}s after {attempts} attempt(s): {last_error}"
    ) from last_error


def _launch_process(executable: Path, ar_game_cfg: dict[str, Any]) -> subprocess.Popen[Any]:
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

    args = [str(executable)]
    if bool(ar_game_cfg.get("windowed", True)):
        width = int(ar_game_cfg.get("window_width", 1280))
        height = int(ar_game_cfg.get("window_height", 720))
        args.extend([
            "-screen-fullscreen",
            "0",
            "-screen-width",
            str(max(width, 320)),
            "-screen-height",
            str(max(height, 240)),
        ])

    return subprocess.Popen(
        args,
        cwd=str(executable.parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )


def _enable_existing_window_resize(
    config: dict[str, Any],
    *,
    project_root: Path | None,
    console: Any | None,
) -> None:
    """Make an already-running local Unity window resizable when configured."""

    ar_game_cfg = config.get("output", {}).get("ar_game", {})
    if os.name != "nt" or not bool(ar_game_cfg.get("resizable_window", True)):
        return

    executable = configured_unity_executable(config, project_root=project_root)
    configured_title = str(ar_game_cfg.get("window_title") or "").strip()
    titles = [configured_title] if configured_title else [executable.stem]
    # The packaged player may use its product name as the
    # top-level window title, which is different from the executable stem.
    if DEFAULT_UNITY_WINDOW_TITLE.casefold() not in {
        title.casefold() for title in titles
    }:
        titles.append(DEFAULT_UNITY_WINDOW_TITLE)
    for title in titles:
        if _make_windows_resizable(window_title=title):
            _notify(console, f"Enabled resizing for Unity window: {title}")
            return


def _enable_launched_window_resize(
    process: subprocess.Popen[Any],
    ar_game_cfg: dict[str, Any],
    *,
    console: Any | None,
) -> None:
    """Make the newly launched Unity window resizable on Windows."""

    if os.name != "nt" or not bool(ar_game_cfg.get("resizable_window", True)):
        return

    if _make_windows_resizable(process_id=process.pid):
        _notify(console, "Enabled resizing for the Unity game window.")
    else:
        LOGGER.warning(
            "Unity opened TCP successfully, but its top-level window was not found; "
            "the window resize workaround was not applied."
        )


def _make_windows_resizable(
    *,
    process_id: int | None = None,
    window_title: str | None = None,
) -> bool:
    """Add resize/maximize styles to matching top-level Windows windows.

    This remains a compatibility fallback for older packaged players. Builds
    with ``PlayerSettings.resizableWindow`` enabled can disable it in config.
    """

    if os.name != "nt" or (process_id is None and not window_title):
        return False

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    enum_windows_callback = ctypes.WINFUNCTYPE(
        wintypes.BOOL, wintypes.HWND, wintypes.LPARAM
    )

    user32.EnumWindows.argtypes = [enum_windows_callback, wintypes.LPARAM]
    user32.EnumWindows.restype = wintypes.BOOL
    user32.GetWindowThreadProcessId.argtypes = [
        wintypes.HWND,
        ctypes.POINTER(wintypes.DWORD),
    ]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.GetWindowLongW.restype = ctypes.c_long
    user32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
    user32.SetWindowLongW.restype = ctypes.c_long
    user32.SetWindowPos.argtypes = [
        wintypes.HWND,
        wintypes.HWND,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.UINT,
    ]
    user32.SetWindowPos.restype = wintypes.BOOL
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int

    expected_title = window_title.casefold() if window_title else None
    changed = False

    @enum_windows_callback
    def visit_window(hwnd: int, _lparam: int) -> bool:
        nonlocal changed
        if not user32.IsWindowVisible(hwnd):
            return True

        if process_id is not None:
            owner_pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner_pid))
            if owner_pid.value != int(process_id):
                return True

        if expected_title is not None:
            length = user32.GetWindowTextLengthW(hwnd)
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, len(buffer))
            if buffer.value.strip().casefold() != expected_title:
                return True

        style = user32.GetWindowLongW(hwnd, _GWL_STYLE)
        if not style & _WS_CAPTION:
            return True
        new_style = style | _WS_THICKFRAME | _WS_MAXIMIZEBOX
        if new_style != style:
            previous = user32.SetWindowLongW(hwnd, _GWL_STYLE, new_style)
            if previous == 0 and ctypes.get_last_error() != 0:
                LOGGER.warning(
                    "Could not update Unity window style: WinError %s",
                    ctypes.get_last_error(),
                )
                return True

        if not user32.SetWindowPos(
            hwnd,
            0,
            0,
            0,
            0,
            0,
            _SWP_NOMOVE | _SWP_NOSIZE | _SWP_NOZORDER | _SWP_FRAMECHANGED,
        ):
            LOGGER.warning(
                "Could not refresh Unity window frame: WinError %s",
                ctypes.get_last_error(),
            )
            return True
        changed = True
        return True

    user32.EnumWindows(visit_window, 0)
    return changed


def _is_local_host(host: str) -> bool:
    normalized = host.strip().lower()
    if normalized in LOCAL_HOSTS:
        return True
    try:
        return socket.gethostbyname(normalized).startswith("127.")
    except OSError:
        return False


def _is_tcp_open(host: str, port: int, *, timeout_sec: float) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout_sec):
            return True
    except OSError:
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _notify(console: Any | None, message: str) -> None:
    LOGGER.info(message)
    if console is not None and hasattr(console, "print"):
        console.print(f"[bold cyan]{message}[/bold cyan]")
