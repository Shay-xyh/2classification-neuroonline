from __future__ import annotations

import json
from types import SimpleNamespace

import utils.unity_runtime as unity_runtime


def test_launched_unity_window_is_made_resizable_by_default(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(unity_runtime.os, "name", "nt")
    monkeypatch.setattr(
        unity_runtime,
        "_make_windows_resizable",
        lambda **kwargs: calls.append(kwargs) or True,
    )

    unity_runtime._enable_launched_window_resize(
        SimpleNamespace(pid=1234),
        {},
        console=None,
    )

    assert calls == [{"process_id": 1234}]


def test_launched_unity_window_resize_workaround_can_be_disabled(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(unity_runtime.os, "name", "nt")
    monkeypatch.setattr(
        unity_runtime,
        "_make_windows_resizable",
        lambda **kwargs: calls.append(kwargs) or True,
    )

    unity_runtime._enable_launched_window_resize(
        SimpleNamespace(pid=1234),
        {"resizable_window": False},
        console=None,
    )

    assert calls == []


def test_existing_unity_window_matches_configured_executable_title(
    monkeypatch, tmp_path
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(unity_runtime.os, "name", "nt")
    monkeypatch.setattr(
        unity_runtime,
        "_make_windows_resizable",
        lambda **kwargs: calls.append(kwargs) or True,
    )
    config = {
        "output": {
            "ar_game": {
                "executable_path": "build/MyDrivingGame.exe",
                "resizable_window": True,
            }
        }
    }

    unity_runtime._enable_existing_window_resize(
        config,
        project_root=tmp_path,
        console=None,
    )

    assert calls == [{"window_title": "MyDrivingGame"}]


def test_existing_packaged_unity_window_uses_default_product_title_once(
    monkeypatch, tmp_path
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(unity_runtime.os, "name", "nt")

    def make_resizable(**kwargs: object) -> bool:
        calls.append(kwargs)
        return kwargs.get("window_title") == "ARPrototype3D"

    monkeypatch.setattr(unity_runtime, "_make_windows_resizable", make_resizable)
    config = {
        "output": {
            "ar_game": {
                "executable_path": "build/ARPrototype3D.exe",
                "resizable_window": True,
            }
        }
    }

    unity_runtime._enable_existing_window_resize(
        config,
        project_root=tmp_path,
        console=None,
    )

    assert calls == [{"window_title": "ARPrototype3D"}]


def test_runtime_manifest_round_trip_and_hash_validation(tmp_path) -> None:
    build_dir = tmp_path / "ARPrototype3D-windows-x64"
    managed_dir = build_dir / "ARPrototype3D_Data" / "Managed"
    managed_dir.mkdir(parents=True)
    executable = build_dir / "ARPrototype3D.exe"
    executable.write_bytes(b"player")
    (build_dir / "UnityPlayer.dll").write_bytes(b"unity")
    runtime_dll = managed_dir / "ARPrototype3D.Runtime.dll"
    runtime_dll.write_bytes(b"runtime")

    manifest_path = unity_runtime.write_unity_runtime_manifest(
        executable,
        build_id="test-build",
    )
    manifest = unity_runtime.validate_unity_runtime(executable)

    assert manifest_path.name == unity_runtime.RUNTIME_MANIFEST_FILENAME
    assert manifest["build_id"] == "test-build"
    assert manifest["protocol_version"] == unity_runtime.REQUIRED_RUNTIME_PROTOCOL
    assert "ARPrototype3D_Data/Managed/ARPrototype3D.Runtime.dll" in manifest["files"]

    runtime_dll.write_bytes(b"tampered")
    try:
        unity_runtime.validate_unity_runtime(executable)
    except RuntimeError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("Tampered Unity runtime unexpectedly validated.")


def test_runtime_manifest_rejects_old_unversioned_build(tmp_path) -> None:
    build_dir = tmp_path / "ARPrototype3D-windows-x64"
    managed_dir = build_dir / "ARPrototype3D_Data" / "Managed"
    managed_dir.mkdir(parents=True)
    executable = build_dir / "ARPrototype3D.exe"
    executable.write_bytes(b"player")
    (build_dir / "UnityPlayer.dll").write_bytes(b"unity")
    (managed_dir / "ARPrototype3D.Runtime.dll").write_bytes(b"runtime")

    try:
        unity_runtime.validate_unity_runtime(executable)
    except RuntimeError as exc:
        assert "manifest was not found" in str(exc)
    else:
        raise AssertionError("Unversioned Unity runtime unexpectedly validated.")


def test_runtime_manifest_requires_runtime_code_hash(tmp_path) -> None:
    build_dir = tmp_path / "ARPrototype3D-windows-x64"
    managed_dir = build_dir / "ARPrototype3D_Data" / "Managed"
    managed_dir.mkdir(parents=True)
    executable = build_dir / "ARPrototype3D.exe"
    executable.write_bytes(b"player")
    (build_dir / "UnityPlayer.dll").write_bytes(b"unity")
    (managed_dir / "ARPrototype3D.Runtime.dll").write_bytes(b"runtime")

    manifest_path = unity_runtime.write_unity_runtime_manifest(
        executable,
        build_id="test-build",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["files"]["ARPrototype3D_Data/Managed/ARPrototype3D.Runtime.dll"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    try:
        unity_runtime.validate_unity_runtime(executable)
    except RuntimeError as exc:
        assert "missing critical file hashes" in str(exc)
    else:
        raise AssertionError("Runtime manifest without code hash unexpectedly validated.")


def test_scene_readiness_retries_until_authoritative_ack(monkeypatch) -> None:
    class _Outlet:
        def __init__(self) -> None:
            self.calls = 0

        def push_with_ack(self, command: str) -> dict[str, object]:
            assert command == "SCENE_STATE"
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError("driving scene is still loading")
            return {
                "ack": "SCENE_STATE",
                "protocol_version": unity_runtime.REQUIRED_RUNTIME_PROTOCOL,
                "scene_number": 4,
                "current_lane": -1,
            }

    monkeypatch.setattr(unity_runtime.time, "sleep", lambda _seconds: None)
    outlet = _Outlet()

    ack = unity_runtime.wait_for_unity_scene_ready(
        outlet,
        timeout_sec=1.0,
        retry_interval_sec=0.01,
    )

    assert outlet.calls == 3
    assert ack["scene_number"] == 4


def test_scene_readiness_rejects_invalid_lane_state(monkeypatch) -> None:
    class _Outlet:
        def push_with_ack(self, _command: str) -> dict[str, object]:
            return {
                "ack": "SCENE_STATE",
                "protocol_version": unity_runtime.REQUIRED_RUNTIME_PROTOCOL,
                "scene_number": 1,
                "current_lane": 9,
            }

    clock = iter([0.0, 0.0, 1.0])
    monkeypatch.setattr(unity_runtime.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(unity_runtime.time, "sleep", lambda _seconds: None)

    try:
        unity_runtime.wait_for_unity_scene_ready(
            _Outlet(),
            timeout_sec=0.5,
            retry_interval_sec=0.0,
        )
    except RuntimeError as exc:
        assert "did not become ACK-ready" in str(exc)
    else:
        raise AssertionError("Invalid Unity lane state unexpectedly passed readiness.")
