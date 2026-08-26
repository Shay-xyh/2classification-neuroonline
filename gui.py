"""Streamlit web interface for oi-mi."""

from __future__ import annotations

import argparse
import base64
import copy
from dataclasses import dataclass, field
import html
import json
import os
import re
import secrets
import sys
import threading
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from acquisition.base import AbstractAcquirer
from acquisition.factory import AcquirerFactory, register_default_acquirers
from adaptation.calibrator import Calibrator, CollectionPauseControl
from adaptation.mi_protocol import (
    LABEL_DISPLAY,
    LABEL_SYMBOL,
    TASK_LABELS,
    ProtocolConfig,
    TrialTiming,
)
from cli import (
    build_acquirer,
    build_game_command_outlet,
    build_model_path,
    resolve_model_path,
    load_config as load_app_config,
    resolve_config_path,
    write_config,
)
from game_command_router import get_shared_game_command_router
from utils.markers import LSLCommandOutlet, NoOpMarkerBackend
from utils.online_adaptation_dashboard import render_online_adaptation_panel, render_online_cue_panel
from utils.online_labels import (
    CuedOnlineLabelSource,
    ManualLabelHttpServer,
    ManualOnlineLabelSource,
    OnlineLabelSource,
    SimulatedOnlineLabelSource,
    build_cued_online_label_source,
)
from utils.binary_mi_gui import (
    GUIDANCE_STEPS,
    STIMULUS_CSS,
    MiVisualFrame,
    MiVisualStage,
    frame_html as mi_frame_html,
    resolve_mi_visual,
)
from utils.timebase import seconds_to_windows
from web_command_server import start_web_command_server

TEST_MODE_PROMPTS = {0: "想象左手", 1: "想象右手"}

_GUI_ROOT = Path(__file__).resolve().parent
_PAGE_ICON_FILENAME = "OMNI_ICON.svg"
_LOGO_FILENAME = "OMNI_LOGO_ENG_double_line.svg"
_COLLECTION_STATUS_SCHEMA = 3
_STIMULUS_COMPONENT_DIR = _GUI_ROOT / "components" / "stimulus_surface"
_stimulus_surface_component = components.declare_component(
    "oi_stimulus_surface",
    path=str(_STIMULUS_COMPONENT_DIR),
)


def _hardware_free_rehearsal_config(config: dict) -> dict:
    """Build an isolated short run that exercises the formal collection path."""

    rehearsal = copy.deepcopy(config)
    rehearsal["hardware_dummy_mode"] = True
    rehearsal["device_type"] = "dummy"
    rehearsal["collection_rehearsal"] = True
    rehearsal["subject_id"] = f"{config.get('subject_id', 'unknown')}-rehearsal"
    protocol = rehearsal.setdefault("protocol", {})
    protocol["collection_blocks"] = 2
    protocol["collection_trials_per_class_per_block"] = 2
    protocol["rest_between_blocks_sec"] = 3.0
    return rehearsal


def _resolve_asset_path(filename: str) -> Path | None:
    """Resolve asset path across source and installed-package launch modes."""

    candidates = (
        _GUI_ROOT / "assets" / filename,
        Path.cwd() / "assets" / filename,
        Path.cwd() / "oi-mi" / "assets" / filename,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _collection_status_path(config: dict) -> Path:
    """Return the durable GUI status file for one subject's collection."""

    runtime_dir = Path(
        str(config.get("storage", {}).get("runtime_dir", ".runtime"))
    ).expanduser()
    if not runtime_dir.is_absolute():
        runtime_dir = _GUI_ROOT / runtime_dir
    subject_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(config.get("subject_id", "unknown")))
    return runtime_dir / f"collection-{subject_id}.json"


def _write_collection_status(config: dict, payload: dict[str, object]) -> None:
    """Atomically persist collection state so a reconnect can recover it."""

    path = _collection_status_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": _COLLECTION_STATUS_SCHEMA,
        "subject_id": str(config.get("subject_id", "")),
        **payload,
    }
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_collection_status(config: dict) -> dict[str, object] | None:
    path = _collection_status_path(config)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if int(payload.get("schema_version", -1)) != _COLLECTION_STATUS_SCHEMA:
        return None
    if str(payload.get("subject_id", "")) != str(config.get("subject_id", "")):
        return None
    return payload


_ACTIVE_CHECKPOINT_STATES = frozenset(
    {"collecting", "raw_exporting", "raw_exported", "processing_complete", "finalizing"}
)


def _read_latest_active_collection_checkpoint(
    config: dict,
    *,
    subject_id: str,
) -> dict[str, object] | None:
    """Find the newest independently running collection without controlling it."""

    records_root = Path(
        str(config.get("storage", {}).get("records_dir", "records_storage"))
    ).expanduser()
    if not records_root.is_absolute():
        records_root = _GUI_ROOT / records_root
    candidates: list[tuple[int, Path, dict[str, object]]] = []
    try:
        checkpoint_paths = records_root.glob("*/collection/*/checkpoint.json")
        for path in checkpoint_paths:
            try:
                if path.parents[2].name != subject_id:
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    continue
                state = str(payload.get("state", ""))
                if state not in _ACTIVE_CHECKPOINT_STATES:
                    continue
                candidates.append((path.stat().st_mtime_ns, path, payload))
            except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
                continue
    except OSError:
        return None
    if not candidates:
        return None
    _modified_ns, path, payload = max(candidates, key=lambda item: item[0])
    result = dict(payload)
    result["checkpoint_path"] = str(path)
    result["subject_id"] = path.parents[2].name
    return result


def _requested_monitor_subject() -> str:
    try:
        return str(st.query_params.get("monitor_subject", "")).strip()
    except Exception:  # noqa: BLE001
        return ""


@st.fragment(run_every=2.0)
def render_external_collection_progress(config: dict, subject_id: str) -> None:
    """Render a read-only progress panel for a CLI or other GUI process."""

    checkpoint = _read_latest_active_collection_checkpoint(
        config,
        subject_id=subject_id,
    )
    if checkpoint is None:
        return
    completed = max(int(checkpoint.get("completed_trials", 0)), 0)
    total = max(int(checkpoint.get("total_trials", 0)), 0)
    fraction = min(completed / total, 1.0) if total else 0.0
    st.warning(
        "检测到另一个进程正在采集。本页只读显示进度，请勿再次开始采集。"
    )
    st.progress(
        fraction,
        text=f"有效 trial：{completed}/{total}（{fraction * 100:.1f}%）",
    )
    block = int(checkpoint.get("last_completed_block", 0))
    trial_in_block = int(checkpoint.get("last_completed_trial_in_block", 0))
    st.caption(
        f"被试：{checkpoint.get('subject_id', '-')} · "
        f"已完成 Block {block} / Trial {trial_in_block} · "
        f"连续样本：{int(checkpoint.get('sample_count', 0))} · "
        f"事件：{int(checkpoint.get('event_count', 0))}"
    )
    st.caption(f"只读检查点：`{checkpoint.get('checkpoint_path', '')}`")


def _validate_collection_outcome(outcome: dict[str, object]) -> None:
    """Refuse to report success until every experiment-critical artifact exists."""

    required_artifacts = [
        ("continuous_eeg_path", "连续 EEG"),
        ("events_path", "事件文件"),
        ("windows_path", "4 秒窗口文件"),
        ("session_dir", "采集 session"),
    ]
    required_paths: list[tuple[str, Path]] = []
    for key, label in required_artifacts:
        value = str(outcome.get(key, "") or "").strip()
        if not value:
            raise RuntimeError(f"采集完成结果缺少{label}路径")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = _GUI_ROOT / path
        required_paths.append((label, path))

    missing = [
        f"{label}: {path}"
        for label, path in required_paths
        if not path.exists() or (path.is_file() and path.stat().st_size <= 0)
    ]
    session_dir = Path(str(outcome["session_dir"])).expanduser()
    if not session_dir.is_absolute():
        session_dir = _GUI_ROOT / session_dir
    session_metadata = session_dir / "metadata.json"
    if not session_metadata.is_file() or session_metadata.stat().st_size <= 0:
        missing.append(f"session 元数据: {session_metadata}")
    if missing:
        raise RuntimeError("采集产物校验失败；" + "；".join(missing))


def _recover_completed_collection(config: dict) -> dict[str, object] | None:
    """Recover a completed run whose Streamlit page disconnected before rerun."""

    status = _read_collection_status(config)
    if not isinstance(status, dict):
        return None
    persisted_outcome = status.get("outcome")
    if status.get("state") == "completed" and isinstance(persisted_outcome, dict):
        try:
            _validate_collection_outcome(persisted_outcome)
        except RuntimeError:
            return None
        recovered = dict(persisted_outcome)
        recovered["recovered_after_reconnect"] = True
        return recovered
    if status.get("state") == "failed" and isinstance(persisted_outcome, dict):
        recovered = dict(persisted_outcome)
        recovered["recovered_after_reconnect"] = True
        return recovered

    if status.get("state") != "running":
        return None
    session_id = str(status.get("session_id", ""))
    if not session_id or Path(session_id).name != session_id:
        return None
    subject_id = str(config.get("subject_id", ""))
    records_root = Path(
        str(config.get("storage", {}).get("records_dir", "records_storage"))
    ).expanduser()
    if not records_root.is_absolute():
        records_root = _GUI_ROOT / records_root
    collection_root = records_root / subject_id / "collection"
    if not collection_root.is_dir():
        return None

    session_dir = collection_root / session_id
    if session_dir.is_dir():
        metadata_path = session_dir / "metadata.json"
        if not metadata_path.is_file():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            outcome = {
                "ok": True,
                "trials_collected": int(metadata["trials_collected"]),
                "windows_collected": int(metadata["windows_collected"]),
                "continuous_eeg_path": str(session_dir / "continuous_eeg.npy"),
                "events_path": str(session_dir / "events.json"),
                "windows_path": str(session_dir / "mi_windows.npz"),
                "session_dir": str(session_dir),
                "recovered_after_reconnect": True,
            }
            _validate_collection_outcome(outcome)
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError, RuntimeError):
            return None
        _write_collection_status(
            config,
            {
                "state": "completed",
                "session_id": session_id,
                "outcome": outcome,
            },
        )
        return outcome
    return None


_PAGE_ICON_PATH = _resolve_asset_path(_PAGE_ICON_FILENAME)
st.set_page_config(
    page_title="oi-mi Control Panel",
    page_icon=str(_PAGE_ICON_PATH) if _PAGE_ICON_PATH is not None else None,
    layout="wide",
)


def parse_config_path(argv: list[str] | None = None) -> Path:
    """Parse the optional config path passed after `streamlit run ... --`."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", dest="config_path", type=Path, default=None)
    args, _ = parser.parse_known_args(argv)
    return resolve_config_path(args.config_path)


CONFIG_PATH = parse_config_path(sys.argv[1:])
_DISPLAY_SYMBOLS = {
    "LEFT": "←",
    "RIGHT": "→",
    "BLANK": "",
}

_AR_TEST_COMMANDS = ("START", "LEFT", "RIGHT", "STOP")


def _ar_game_mode(ar_game_cfg: dict) -> str:
    del ar_game_cfg
    return "direct TCP"


def _ar_game_target(ar_game_cfg: dict) -> str:
    host = str(ar_game_cfg.get("host", "127.0.0.1"))
    port = int(ar_game_cfg.get("port", 5005))
    return f"{host}:{port}"


def _get_ar_forward_status() -> dict:
    return dict(st.session_state.get("ar_forward_status", {}))


def _set_ar_forward_status(**updates: object) -> None:
    status = _get_ar_forward_status()
    status.update(updates)
    status["updated_at"] = time.time()
    st.session_state.ar_forward_status = status


def _update_ar_decoder_status(payload: dict) -> None:
    _set_ar_forward_status(
        last_prediction=payload.get("prediction", "-"),
        confidence=payload.get("confidence"),
        mapped_command=payload.get("mapped_command", "-"),
        last_transport_command=payload.get("last_transport_command"),
        last_send_success=payload.get("last_send_success"),
        last_send_error=payload.get("last_send_error"),
        online_adaptation=payload.get("online_adaptation"),
        online_label_source=payload.get("online_label_source"),
        timing_alignment=payload.get("timing_alignment"),
    )


def _format_send_state(
    status: dict,
    *,
    now: float | None = None,
    stale_after_sec: float = 3.0,
) -> str:
    updated_at = status.get("updated_at")
    if updated_at is not None:
        current_time = time.time() if now is None else float(now)
        if current_time - float(updated_at) > max(float(stale_after_sec), 0.0):
            return "stale"
    success = status.get("last_send_success")
    if success is True:
        return "success"
    if success is False:
        return "failed"
    return "-"


def _current_streamlit_context() -> object | None:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:  # noqa: BLE001
        return None
    return get_script_run_ctx()


def _missing_model_guidance(config: dict) -> str:
    if bool(config.get("hardware_dummy_mode", False)) or str(config.get("device_type", "")) == "dummy":
        return "数据采集本身不生成模型；请在采集结束后独立训练，或运行 `oi-mi seed-dummy-decoders` 生成 dummy 测试权重。"
    return "数据采集本身不生成模型；请先完成采集，再通过独立的采后训练流程生成测试或实时解码权重。"


def render_ar_forwarding_panel(config: dict, *, render_adaptation: bool = True) -> None:
    output_cfg = config.get("output", {})
    ar_game_cfg = output_cfg.get("ar_game", {})
    enabled = bool(ar_game_cfg.get("enabled", False))
    status = _get_ar_forward_status()

    st.markdown("### AR 转发状态")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AR output", "enabled" if enabled else "disabled")
    col2.metric("Mode", _ar_game_mode(ar_game_cfg))
    col3.metric("Target", _ar_game_target(ar_game_cfg))
    col4.metric("Last send", _format_send_state(status))
    st.caption("Last send 仅表示最近一次 TCP 写入结果；stale 表示当前没有持续心跳，不能视为 Unity 仍在线。")

    pred_col, command_col, transport_col = st.columns(3)
    confidence = status.get("confidence")
    confidence_text = "-" if confidence is None else f"{float(confidence):.2f}"
    pred_col.metric("Last prediction", str(status.get("last_prediction", "-")), confidence_text)
    command_col.metric("Mapped command", str(status.get("mapped_command", "-")))
    transport_col.metric("Transport command", str(status.get("last_transport_command", "-")))

    error = status.get("last_send_error")
    if error:
        st.warning(f"最近一次 AR 转发失败: {error}")

    if render_adaptation:
        _render_online_adaptation_panel(status.get("online_adaptation"))

    st.markdown("### 小车连接测试")
    st.caption("这些按钮只测试 AR/Unity 转发链路，不依赖 EEG、模型或实时解码。")
    if st.button("启动/重置并进入小车", key="ar_test_open_car"):
        if not enabled:
            _set_ar_forward_status(
                mapped_command="OPEN_3D_GAME + LAUNCHER_SELECT",
                last_transport_command=None,
                last_send_success=False,
                last_send_error="output.ar_game.enabled is false.",
            )
            st.error("AR 游戏 TCP 控制未启用。请先在配置页启用后保存。")
        else:
            try:
                build_game_command_outlet(config)
            except Exception as exc:  # noqa: BLE001
                _set_ar_forward_status(
                    mapped_command="OPEN_3D_GAME + LAUNCHER_SELECT",
                    last_transport_command="LAUNCHER_SELECT",
                    last_send_success=False,
                    last_send_error=str(exc),
                )
                st.error(f"小车启动失败: {exc}")
            else:
                _set_ar_forward_status(
                    mapped_command="OPEN_3D_GAME + LAUNCHER_SELECT",
                    last_transport_command="LAUNCHER_SELECT",
                    last_send_success=True,
                    last_send_error=None,
                )
                st.success("Unity 已启动并进入 Fixed Speed 小车模式。")
    cols = st.columns(len(_AR_TEST_COMMANDS))
    for column, command in zip(cols, _AR_TEST_COMMANDS, strict=True):
        if column.button(f"Send {command}", key=f"ar_test_{command}"):
            if not enabled:
                _set_ar_forward_status(
                    mapped_command=command,
                    last_transport_command=None,
                    last_send_success=False,
                    last_send_error="output.ar_game.enabled is false.",
                )
                st.error("AR 游戏 TCP 控制未启用。请先在配置页启用后保存。")
                continue
            try:
                get_shared_game_command_router(config).push(command, source="web")
            except Exception as exc:  # noqa: BLE001
                _set_ar_forward_status(
                    mapped_command=command,
                    last_transport_command=command,
                    last_send_success=False,
                    last_send_error=str(exc),
                )
                st.error(f"{command} 发送失败: {exc}")
            else:
                _set_ar_forward_status(
                    mapped_command=command,
                    last_transport_command=command,
                    last_send_success=True,
                    last_send_error=None,
                )
                st.success(f"{command} 已发送。")

_CUE_COLORS = {
    "action": "#15803D",
    "rest": "#2563EB",
    "default": "#C2410C",
}

def _resolve_display_color(symbol: str, message: str) -> str:
    upper_message = message.upper()
    if symbol in {"←", "→"} or "LEFT" in upper_message or "RIGHT" in upper_message or "左手" in message or "右手" in message:
        return _CUE_COLORS["action"]
    if "休息" in message:
        return _CUE_COLORS["rest"]
    return _CUE_COLORS["default"]


def _resolve_cue_symbol(message: str, *, event_type: str) -> tuple[str, bool] | None:
    normalized = re.sub(r"\s+", " ", message.strip())
    upper_message = normalized.upper()
    mi_frame = resolve_mi_visual(normalized)
    if mi_frame is not None:
        return mi_frame.fallback_symbol, event_type == "prediction"
    if event_type == "prediction":
        if "LEFT" in upper_message:
            return _DISPLAY_SYMBOLS["LEFT"], True
        if "RIGHT" in upper_message:
            return _DISPLAY_SYMBOLS["RIGHT"], True
    if event_type == "cue":
        if normalized == TEST_MODE_PROMPTS[0]:
            return _DISPLAY_SYMBOLS["LEFT"], False
        if normalized == TEST_MODE_PROMPTS[1]:
            return _DISPLAY_SYMBOLS["RIGHT"], False
    return None


def _subject_facing_message(message: str, *, prediction: bool) -> str:
    """Return concise text for the subject-facing fullscreen view."""

    if prediction:
        return ""
    if resolve_mi_visual(message) is not None:
        return ""
    if "测试模式启动" in message:
        return "准备开始"
    return ""

SIDEBAR_NAV_PAGES = ("首页", "设置", "连通检测", "数据采集", "测试模式", "实时解码")
_COLLECTION_VIEWS = frozenset({"guidance", "ready", "trial_test", "run"})


def _resolve_logo_svg_path() -> Path | None:
    """Resolve sidebar logo path."""
    return _resolve_asset_path(_LOGO_FILENAME)


def render_sidebar_logo(path: Path) -> None:
    """Render logo without Streamlit's image fullscreen control."""

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    st.markdown(
        (
            "<img "
            f"src='data:image/svg+xml;base64,{encoded}' "
            "alt='Omni-Intelligence' "
            "style='width: 280px; max-width: 100%; height: auto; display: block;'"
            ">"
        ),
        unsafe_allow_html=True,
    )


def load_config() -> dict:
    try:
        return load_app_config(CONFIG_PATH)
    except Exception as exc:  # noqa: BLE001
        st.error(f"加载配置文件失败: {exc}")
        return {}


def save_config(cfg: dict) -> None:
    try:
        write_config(CONFIG_PATH, cfg)
    except Exception as exc:  # noqa: BLE001
        st.error(f"保存配置文件失败: {exc}")


class StreamlitConsole:
    """Minimal Rich Console substitute that writes into Streamlit placeholders."""

    def __init__(
        self,
        cue_placeholder,
        log_placeholder,
        *,
        fullscreen: bool = False,
        show_debug: bool = False,
        stable_surface: bool = False,
    ) -> None:
        self.cue_placeholder = cue_placeholder
        self.log_placeholder = log_placeholder
        self.fullscreen = fullscreen
        self.show_debug = show_debug
        self.stable_surface = stable_surface
        self.logs: list[str] = []
        self._lock = threading.Lock()
        self._pending_events: list[tuple[str, str]] = []
        self._ui_thread_id = threading.get_ident()
        self._last_stage_label = ""
        self._fullscreen_symbol_html = ""
        self._fullscreen_message_html = ""
        self._fullscreen_frame = MiVisualFrame(MiVisualStage.BLANK)
        self._progress_label = "等待阶段"
        self._progress_elapsed = 0.0
        self._progress_duration = 0.0
        self._progress_started_at = time.monotonic()
        self._last_progress_render_at = 0.0
        self._completed_trials = 0
        self._total_trials = 0

    def print(self, message, *args, **kwargs) -> None:
        raw_message = str(message)
        msg = re.sub(r"\[.*?\]", "", raw_message).strip()
        if not msg:
            return

        event_type = "log"
        if "[cue]" in raw_message.lower():
            event_type = "cue"
        elif "confidence:" in msg:
            event_type = "prediction"
        elif _resolve_cue_symbol(msg, event_type="log") is not None:
            event_type = "cue"

        with self._lock:
            self._pending_events.append((event_type, msg))

        if threading.get_ident() == self._ui_thread_id:
            self.render_pending()

    def attach(
        self,
        cue_placeholder,
        log_placeholder,
        *,
        stable_surface: bool | None = None,
    ) -> None:
        """Attach fresh placeholders after a Streamlit rerun."""

        with self._lock:
            self.cue_placeholder = cue_placeholder
            self.log_placeholder = log_placeholder
            self._ui_thread_id = threading.get_ident()
            self._last_progress_render_at = 0.0
            if stable_surface is not None:
                self.stable_surface = stable_surface

    def render_pending(self) -> None:
        with self._lock:
            pending = list(self._pending_events)
            self._pending_events.clear()

        log_updated = False
        latest_visual: tuple[str, str] | None = None
        for event_type, msg in pending:
            if self.fullscreen and event_type == "prediction":
                self._append_log(msg)
                log_updated = True
                continue
            if event_type in {"cue", "prediction"}:
                latest_visual = (event_type, msg)
                self._append_log(msg)
                log_updated = True
            else:
                self._append_log(msg)
                log_updated = True

        # Collection runs in a worker while Streamlit redraws periodically.
        # Several stage changes can therefore arrive in one batch. Rendering
        # every intermediate cue sends multiple browser deltas and can flash a
        # stale arrow during fixation. Only the newest stage is visual; all
        # messages remain preserved in the operator log above.
        if latest_visual is not None:
            event_type, msg = latest_visual
            self._render_cue(msg, prediction=(event_type == "prediction"))

        if log_updated and not self.fullscreen:
            self.log_placeholder.code("\n".join(self.logs))
        if self.fullscreen:
            self._render_fullscreen_surface()

    def _append_log(self, msg: str) -> None:
        self.logs.append(msg)
        if len(self.logs) > 18:
            self.logs.pop(0)

    def set_stage_progress(
        self,
        *,
        stage_name: str,
        elapsed_sec: float,
        duration_sec: float,
        render: bool = True,
    ) -> None:
        if not self.fullscreen:
            return
        with self._lock:
            total = max(float(duration_sec), 0.0)
            elapsed = min(max(float(elapsed_sec), 0.0), total) if total > 0 else 0.0
            label = stage_name.strip() or self._last_stage_label or "阶段"
            self._last_stage_label = label
            self._progress_label = label
            self._progress_elapsed = elapsed
            self._progress_duration = total
            if elapsed <= 0.0:
                self._progress_started_at = time.monotonic()
            now = time.monotonic()
            should_render = (
                render
                and
                threading.get_ident() == self._ui_thread_id
                and (elapsed >= total or now - self._last_progress_render_at >= 0.25)
            )
            if should_render:
                self._last_progress_render_at = now
        if should_render:
            self._render_fullscreen_surface()

    def set_trial_progress(self, *, completed_trials: int, total_trials: int) -> None:
        """Publish valid-trial progress for the operator area below the fold."""

        total = max(int(total_trials), 0)
        completed = min(max(int(completed_trials), 0), total) if total else 0
        with self._lock:
            self._completed_trials = completed
            self._total_trials = total

    def trial_progress(self) -> tuple[int, int]:
        """Return a thread-safe snapshot of completed and planned trials."""

        with self._lock:
            return self._completed_trials, self._total_trials

    def _render_cue(self, msg: str, *, prediction: bool) -> None:
        mi_frame = resolve_mi_visual(msg)
        resolved = _resolve_cue_symbol(msg, event_type="prediction" if prediction else "cue")
        symbol = resolved[0] if resolved is not None else "·"
        is_prediction = resolved[1] if resolved is not None else prediction
        bg = "#F0FFF4" if is_prediction else "#F8FAFC"
        color = _resolve_display_color(symbol, msg)
        prompt_class = " oi-prompt-animation" if "PROMPT" in msg.upper() else ""
        if self.fullscreen:
            subject_message = _subject_facing_message(msg, prediction=is_prediction)
            self._fullscreen_symbol_html = ""
            if mi_frame is not None:
                self._fullscreen_frame = mi_frame
                self._fullscreen_symbol_html = mi_frame_html(mi_frame)
            elif symbol:
                self._fullscreen_frame = MiVisualFrame(MiVisualStage.BLANK)
                self._fullscreen_symbol_html = f"<div class='oi-experiment-symbol{prompt_class}' style='color: {color};'>{symbol}</div>"
            self._fullscreen_message_html = ""
            if subject_message:
                safe_msg = html.escape(subject_message)
                message_class = "oi-experiment-message" if symbol else "oi-experiment-center-message"
                self._fullscreen_message_html = f"<div class='{message_class}'>{safe_msg}</div>"
            self._render_fullscreen_surface()
            return
        self.cue_placeholder.markdown(
            (
                "<div style='padding: 1.25rem; min-height: 8rem; border-radius: 12px; "
                "display: flex; align-items: center; justify-content: center; "
                f"background-color: {bg}; border: 1px solid #E2E8F0;'>"
                f"<div class='{prompt_class.strip()}' style='font-size: 4.5rem; line-height: 1; font-weight: 700; color: {color};'>{symbol}</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

    def _render_fullscreen_surface(self) -> None:
        if self.stable_surface:
            return

        total = max(float(self._progress_duration), 0.0)
        if total > 0:
            elapsed = min(max(time.monotonic() - self._progress_started_at, float(self._progress_elapsed)), total)
            self._progress_elapsed = elapsed
        else:
            elapsed = max(float(self._progress_elapsed), 0.0)
        ratio = 1.0 if total == 0 else elapsed / total
        remaining = max(total - elapsed, 0.0)
        progress_html = (
            "<div class='oi-debug-progress-card'>"
            "<div class='oi-debug-progress-title'>调试进度</div>"
            "<div class='oi-debug-progress-row'>"
            f"<span>{html.escape(self._progress_label)}</span>"
            f"<span>本阶段 {total:.1f}s</span>"
            "</div>"
            "<div class='oi-debug-progress-track'>"
            "<div class='oi-debug-progress-fill' "
            f"style='width: {ratio * 100:.1f}%; animation-duration: {remaining:.3f}s;'></div>"
            "</div>"
            "</div>"
        )
        debug_html = (
            "<section class='oi-debug-progress-section'>"
            f"{progress_html}"
            "</section>"
            if self.show_debug
            else ""
        )
        self.cue_placeholder.markdown(
            (
                "<div class='oi-experiment-scroll-shell'>"
                "<section class='oi-experiment-stage'>"
                f"{self._fullscreen_symbol_html}"
                f"{self._fullscreen_message_html}"
                "</section>"
                f"{debug_html}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

    def stimulus_frame(self) -> MiVisualFrame:
        """Return the latest protocol frame for the persistent browser surface."""

        return self._fullscreen_frame


def _render_collection_stimulus_surface(
    frame: MiVisualFrame,
    *,
    completed_trials: int,
    total_trials: int,
) -> bool:
    """Render the persistent collection surface at one stable Streamlit path."""

    surface_epoch = str(
        st.session_state.get("collection_stimulus_surface_epoch", "initial")
    )
    ready = _stimulus_surface_component(
        stage=frame.stage.value,
        label=frame.label or "",
        message=frame.message,
        completed_trials=int(completed_trials),
        total_trials=int(total_trials),
        # A new collection must not reuse an iframe whose last DOM state may
        # still contain the preceding run's arrow. The epoch stays unchanged
        # throughout one run, so ordinary 50 ms polling does not remount it.
        key=f"collection_stimulus_surface_{surface_epoch}",
        default=False,
    )
    return ready is True


def _render_computer_fullscreen_control() -> None:
    """Render the user-gesture button required by the browser Fullscreen API."""

    _stimulus_surface_component(
        control_only=True,
        stage=MiVisualStage.BLANK.value,
        label="",
        message="",
        key="collection_computer_fullscreen_control",
        default=None,
    )


def enter_experiment_view() -> None:
    """Switch Streamlit chrome into a subject-facing experiment view."""

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"],
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"] {
          display: none !important;
        }
        html,
        body,
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"],
        .block-container {
          height: 100dvh !important;
          max-height: 100dvh !important;
          overflow: hidden !important;
        }
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        .stApp {
          background: #ffffff !important;
        }
        [data-testid="stMainBlockContainer"],
        .block-container {
          max-width: none !important;
          padding: 0 !important;
        }
        .oi-experiment-scroll-shell {
          width: 100vw;
          height: 100dvh;
          position: fixed;
          inset: 0;
          z-index: 9990;
          background: #f8fafc;
          overflow-x: hidden;
          overflow-y: scroll;
          overscroll-behavior: contain;
          scroll-behavior: auto;
          pointer-events: auto;
          touch-action: pan-y;
          scrollbar-width: thin;
        }
        .oi-experiment-stage {
          width: 100vw;
          height: 100dvh;
          position: relative;
          background: #f8fafc;
          border: none;
          box-sizing: border-box;
          overflow: hidden;
        }
        .oi-experiment-symbol {
          position: absolute;
          top: 45%;
          left: 50%;
          transform: translate(-50%, -50%);
          font-size: clamp(8rem, 24vw, 22rem);
          line-height: 1;
          font-weight: 800;
          text-align: center;
        }
        @keyframes oi-mi-grasp-prompt {
          0%, 100% { transform: translate(-50%, -50%) scale(0.92); }
          50% { transform: translate(-50%, -50%) scale(1.08); }
        }
        .oi-experiment-symbol.oi-prompt-animation {
          animation: oi-mi-grasp-prompt 0.8s ease-in-out infinite;
        }
        .oi-experiment-message {
          position: absolute;
          bottom: 16vh;
          left: 8vw;
          right: 8vw;
          text-align: center;
          font-size: clamp(1.3rem, 2.2vw, 2.6rem);
          line-height: 1.35;
          font-weight: 700;
          color: #0f172a;
        }
        .oi-experiment-center-message {
          position: absolute;
          top: 50%;
          left: 10vw;
          right: 10vw;
          transform: translateY(-50%);
          text-align: center;
          font-size: clamp(2.2rem, 5vw, 5rem);
          line-height: 1.18;
          font-weight: 800;
          color: #0f172a;
        }
        .oi-debug-progress-section {
          width: 100vw;
          min-height: 28dvh;
          box-sizing: border-box;
          display: flex;
          align-items: flex-start;
          justify-content: center;
          padding: 2rem 2rem 4rem;
          background: #f8fafc;
        }
        .oi-debug-progress-card {
          width: min(760px, calc(100vw - 4rem));
          margin: 0 auto;
          padding: 0.55rem 0.7rem;
          border: 1px solid #cbd5e1;
          border-radius: 8px;
          background: rgba(255, 255, 255, 0.96);
          color: #0f172a;
          box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
        }
        .oi-debug-progress-title {
          margin-bottom: 0.3rem;
          font-size: 0.72rem;
          line-height: 1.2;
          font-weight: 800;
          color: #334155;
        }
        .oi-debug-progress-row {
          display: flex;
          justify-content: space-between;
          gap: 1rem;
          margin-bottom: 0.3rem;
          font-size: 0.72rem;
          line-height: 1.2;
          font-weight: 600;
        }
        .oi-debug-progress-track {
          height: 0.32rem;
          overflow: hidden;
          border-radius: 999px;
          background: #e2e8f0;
        }
        .oi-debug-progress-fill {
          height: 100%;
          border-radius: inherit;
          background: #2563eb;
          animation-name: oi-debug-progress-fill;
          animation-timing-function: linear;
          animation-fill-mode: forwards;
        }
        @keyframes oi-debug-progress-fill {
          to { width: 100%; }
        }
        .st-key-collection_return_from_experiment,
        .st-key-test_mode_return_from_experiment {
          position: fixed;
          top: 1.15rem;
          left: 1.35rem;
          z-index: 10000;
          width: auto !important;
        }
        .st-key-collection_return_from_experiment > button,
        .st-key-collection_return_from_experiment .stButton > button,
        .st-key-test_mode_return_from_experiment > button,
        .st-key-test_mode_return_from_experiment .stButton > button {
          width: auto !important;
          min-width: 0 !important;
          min-height: 0 !important;
          padding: 0 !important;
          border: none !important;
          background: transparent !important;
          color: #0f172a !important;
          font-size: 1.85rem;
          line-height: 1;
          font-weight: 800;
          box-shadow: none !important;
          opacity: 1;
        }
        .st-key-collection_return_from_experiment > button:hover,
        .st-key-collection_return_from_experiment .stButton > button:hover,
        .st-key-test_mode_return_from_experiment > button:hover,
        .st-key-test_mode_return_from_experiment .stButton > button:hover {
          background: transparent !important;
          color: #0f172a !important;
          opacity: 1;
        }
        .oi-guidance-panel {
          position: fixed;
          inset: 0;
          z-index: 9990;
          display: flex;
          align-items: center;
          justify-content: center;
          background: #f8fafc;
          box-sizing: border-box;
          padding: 8vh 10vw;
        }
        .oi-guidance-content {
          width: min(1100px, 100%);
          text-align: center;
          display: flex;
          flex-direction: column;
          align-items: center;
        }
        .oi-guidance-kicker {
          margin-bottom: 1.2rem;
          font-size: clamp(1.4rem, 2vw, 2.2rem);
          font-weight: 700;
          color: #64748b;
          text-align: center;
        }
        .oi-guidance-symbol {
          margin: 0 auto 1.25rem;
          font-size: clamp(9rem, 20vw, 18rem);
          line-height: 1;
          font-weight: 800;
          color: #C2410C;
          text-align: center;
        }
        .oi-guidance-title {
          margin: 0 0 1.7rem;
          font-size: clamp(4rem, 7vw, 7.5rem);
          line-height: 1.12;
          font-weight: 800;
          color: #0f172a;
          text-align: center;
        }
        .oi-guidance-body {
          display: block;
          width: min(980px, 78vw);
          margin: 0 auto;
          max-width: none;
          font-size: clamp(2.3rem, 3.4vw, 3.9rem);
          line-height: 1.35;
          font-weight: 400;
          color: #1e293b;
          text-align: center !important;
          text-wrap: balance;
        }
        .st-key-collection_guidance_next {
          position: fixed;
          right: 4vw;
          bottom: 4vh;
          z-index: 10000;
          width: min(16rem, 44vw) !important;
        }
        .st-key-collection_guidance_next > button,
        .st-key-collection_guidance_next .stButton > button {
          width: 100% !important;
        }
        .st-key-collection_start_formal {
          position: fixed;
          right: 4vw;
          bottom: 4vh;
          z-index: 10000;
          width: min(20rem, 52vw) !important;
        }
        .st-key-collection_start_formal > button,
        .st-key-collection_start_formal .stButton > button {
          width: 100% !important;
          min-height: 3.5rem !important;
          font-size: 1.2rem !important;
          font-weight: 800 !important;
        }
        .st-key-collection_request_pause,
        .st-key-collection_pause_pending,
        .st-key-collection_resume,
        .st-key-collection_automatic_break,
        .st-key-collection_return_from_running {
          position: fixed;
          z-index: 2147483647 !important;
          width: min(16rem, 42vw) !important;
        }
        .st-key-collection_request_pause,
        .st-key-collection_pause_pending,
        .st-key-collection_resume,
        .st-key-collection_automatic_break {
          right: 3vw;
          bottom: 3vh;
        }
        .st-key-collection_return_from_running {
          top: 2vh;
          left: 2vw;
          width: 4rem !important;
        }
        .st-key-collection_request_pause > button,
        .st-key-collection_pause_pending > button,
        .st-key-collection_resume > button,
        .st-key-collection_automatic_break > button {
          width: 100% !important;
          min-height: 3.5rem !important;
          font-size: 1.1rem !important;
          font-weight: 800 !important;
        }
        [class*="st-key-collection_stimulus_surface_"] {
          position: fixed !important;
          inset: 0 !important;
          width: 100vw !important;
          height: 100dvh !important;
          z-index: 1 !important;
          background: #000000 !important;
          pointer-events: auto !important;
        }
        [class*="st-key-collection_stimulus_surface_"] iframe {
          display: block !important;
          width: 100vw !important;
          height: 100dvh !important;
          border: 0 !important;
          background: #000000 !important;
          pointer-events: auto !important;
        }
        .st-key-collection_computer_fullscreen_control {
          position: fixed !important;
          top: 2vh !important;
          right: 2vw !important;
          width: min(22rem, 46vw) !important;
          height: 4.5rem !important;
          z-index: 2147483647 !important;
        }
        .st-key-collection_computer_fullscreen_control iframe {
          display: block !important;
          width: 100% !important;
          height: 4.5rem !important;
          border: 0 !important;
          background: transparent !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"<style>{STIMULUS_CSS}</style>", unsafe_allow_html=True)


def render_experiment_return_button(
    *,
    target_page: str,
    state_keys: tuple[str, ...],
    key: str,
) -> None:
    """Leave one experiment view without mutating another page's state."""

    if st.button("≪", key=key):
        for state_key in state_keys:
            st.session_state.pop(state_key, None)
        st.session_state.gui_nav_mode = target_page
        st.rerun()


def render_collection_guidance() -> None:
    """Render pre-collection subject instructions."""

    step_index = int(st.session_state.get("collection_guidance_step", 0))
    step_index = max(0, min(step_index, len(GUIDANCE_STEPS) - 1))
    frame, title, body = GUIDANCE_STEPS[step_index]
    symbol_html = (
        "<div class='oi-guidance-stimulus'>"
        f"{mi_frame_html(frame)}"
        "</div>"
    )
    st.markdown(
        (
            "<div class='oi-guidance-panel'>"
            "<div class='oi-guidance-content'>"
            f"<div class='oi-guidance-kicker'>步骤 {step_index + 1} / {len(GUIDANCE_STEPS)}</div>"
            f"{symbol_html}"
            f"<h1 class='oi-guidance-title'>{html.escape(title)}</h1>"
            f"<div class='oi-guidance-body'>{html.escape(body)}</div>"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    is_last_step = step_index >= len(GUIDANCE_STEPS) - 1
    next_label = "开始" if is_last_step else "下一步"
    if st.button(next_label, key="collection_guidance_next", type="primary"):
        if is_last_step:
            next_view = str(st.session_state.get("collection_after_guidance", "return"))
            if next_view not in {"return", "ready"}:
                next_view = "return"
            st.session_state.pop("collection_guidance_step", None)
            st.session_state.pop("collection_after_guidance", None)
            if next_view == "return":
                st.session_state.pop("collection_view", None)
                st.session_state.gui_nav_mode = "数据采集"
            else:
                st.session_state.collection_view = next_view
        else:
            st.session_state.collection_guidance_step = step_index + 1
        st.rerun()


def init_live_view(
    *,
    fullscreen: bool = False,
    show_debug: bool = False,
    existing_console: StreamlitConsole | None = None,
    stable_surface: bool = False,
    render_initial: bool = True,
) -> tuple[StreamlitConsole, callable]:
    """Create cue/log placeholders for a running EEG page."""

    cue_box = st.empty()
    log_box = st.empty()
    if existing_console is None:
        console = StreamlitConsole(
            cue_box,
            log_box,
            fullscreen=fullscreen,
            show_debug=show_debug,
            stable_surface=stable_surface,
        )
    else:
        console = existing_console
        console.show_debug = show_debug
        console.attach(
            cue_box,
            log_box,
            stable_surface=stable_surface,
        )

    def refresh() -> None:
        console.render_pending()
        return

    if render_initial:
        refresh()
    return console, refresh


def run_collection_trial_test(protocol: ProtocolConfig) -> None:
    """Show one left and one right trial without EEG or file output.

    This deliberately follows ``Calibrator._run_trial``: fixation, animated
    hand prompt, then the labeled arrow imagery interval. The preview has no
    extra trial, so each hand appears exactly once.
    """

    console, refresh = init_live_view(fullscreen=True, show_debug=True)
    timing = protocol.trial_timing
    for label in TASK_LABELS:
        _run_visual_trial(
            console,
            refresh,
            label=label,
            timing=timing,
            trial_number=f"正式 {LABEL_DISPLAY[label]} trial",
        )


def _run_visual_trial(
    console: StreamlitConsole,
    refresh: callable,
    *,
    label: str,
    timing: TrialTiming,
    trial_number: str,
) -> None:
    """Render a hardware-free trial with the production timing and symbols."""

    cue_message = f"{LABEL_SYMBOL[label]} {LABEL_DISPLAY[label]}"
    _run_preview_event(
        console,
        refresh,
        message="FIXATION",
        stage_name=f"{trial_number}: fixation",
        duration_sec=timing.fixation_sec,
    )
    prompt_message = f"PROMPT HAND {LABEL_DISPLAY[label]}"
    _run_preview_event(
        console,
        refresh,
        message=prompt_message,
        stage_name=f"{trial_number}: movement prompt ({timing.cue_sec:.1f}s)",
        duration_sec=timing.cue_sec,
    )
    _run_preview_event(
        console,
        refresh,
        message=cue_message,
        stage_name=f"{trial_number}: motor imagery ({timing.control_sec:.1f}s)",
        duration_sec=timing.control_sec,
    )


def _run_preview_event(
    console: StreamlitConsole,
    refresh: callable,
    *,
    message: str,
    stage_name: str,
    duration_sec: float,
) -> None:
    console.set_stage_progress(
        stage_name=stage_name,
        elapsed_sec=0.0,
        duration_sec=duration_sec,
        render=False,
    )
    console.print(message)
    _sleep_preview_stage(console, duration_sec=duration_sec)


def _sleep_preview_stage(
    console: StreamlitConsole,
    *,
    duration_sec: float,
) -> None:
    total = max(float(duration_sec), 0.0)
    started_at = time.monotonic()
    deadline = started_at + total
    while time.monotonic() < deadline:
        time.sleep(min(0.05, max(deadline - time.monotonic(), 0.0)))
    console.set_stage_progress(
        stage_name="",
        elapsed_sec=total,
        duration_sec=total,
        render=False,
    )


@dataclass
class CollectionWorkerHandle:
    """One background collection task retained across Streamlit reruns."""

    console: StreamlitConsole
    session_id: str
    pause_control: CollectionPauseControl
    thread: threading.Thread | None = None
    _outcome: dict[str, object] | None = None
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def set_outcome(self, outcome: dict[str, object]) -> None:
        with self._lock:
            self._outcome = dict(outcome)

    def outcome(self) -> dict[str, object] | None:
        with self._lock:
            return None if self._outcome is None else dict(self._outcome)


@st.cache_resource(show_spinner=False)
def _collection_worker_registry() -> tuple[dict[str, CollectionWorkerHandle], threading.RLock]:
    return {}, threading.RLock()


def _collection_worker_key(config: dict) -> str:
    return f"{_collection_status_path(config).resolve()}"


def _get_collection_worker(config: dict) -> CollectionWorkerHandle | None:
    registry, lock = _collection_worker_registry()
    with lock:
        return registry.get(_collection_worker_key(config))


def _remove_collection_worker(config: dict) -> None:
    registry, lock = _collection_worker_registry()
    with lock:
        registry.pop(_collection_worker_key(config), None)


def _start_collection_worker(
    config: dict,
    protocol: ProtocolConfig,
    console: StreamlitConsole,
) -> CollectionWorkerHandle:
    registry, lock = _collection_worker_registry()
    key = _collection_worker_key(config)
    with lock:
        existing = registry.get(key)
        if existing is not None and existing.outcome() is None:
            return existing
        session_id = f"session_{secrets.token_hex(6)}"
        handle = CollectionWorkerHandle(
            console=console,
            session_id=session_id,
            pause_control=CollectionPauseControl(),
        )

        def worker() -> None:
            outcome = run_collection_session(
                config,
                protocol,
                console=console,
                pause_control=handle.pause_control,
                session_id=session_id,
            )
            handle.set_outcome(outcome)

        handle.thread = threading.Thread(
            target=worker,
            name=f"collection-{config.get('subject_id', 'unknown')}",
            daemon=True,
        )
        registry[key] = handle
        handle.thread.start()
        return handle


def run_collection_session(
    config: dict,
    protocol: ProtocolConfig,
    *,
    console: StreamlitConsole | None = None,
    pause_control: CollectionPauseControl | None = None,
    session_id: str | None = None,
) -> dict[str, object]:
    """Run one fixed collection session in the subject-facing view."""

    refresh = None
    resolved_session_id = session_id or f"session_{secrets.token_hex(6)}"
    _write_collection_status(
        config,
        {
            "state": "running",
            "session_id": resolved_session_id,
        },
    )
    try:
        subject_id = str(config["subject_id"])
        acquirer = build_acquirer(
            device_name=str(config["device_type"]),
            config=config,
        )
        if console is None:
            console, refresh = init_live_view(fullscreen=True, show_debug=False)
        else:
            refresh = lambda: None
        collector = Calibrator(
            acquirer=acquirer,
            console=console,
            sfreq=float(config["sfreq"]),
            window_sec=float(config["window_sec"]),
            step_sec=float(config["step_sec"]),
            session_records_dir=Path(str(config.get("storage", {}).get("records_dir", "records_storage")))
            / subject_id
            / "collection",
            session_id=resolved_session_id,
            protocol_config=protocol,
            experiment_config=config,
        )

        console.set_stage_progress(stage_name="启动 EEG 采集", elapsed_sec=0.0, duration_sec=10.0)
        refresh()
        result = collector.collect(
            heartbeat=refresh,
            pause_control=pause_control,
        )

        refresh()
        outcome: dict[str, object] = {
            "ok": True,
            "rehearsal": bool(config.get("collection_rehearsal", False)),
            "trials_collected": int(result.trials_collected),
            "windows_collected": int(result.windows_collected),
            "continuous_eeg_path": (
                str(result.continuous_eeg_path)
                if result.continuous_eeg_path is not None
                else None
            ),
            "events_path": str(result.events_path) if result.events_path is not None else None,
            "windows_path": str(result.windows_path) if result.windows_path is not None else None,
            "session_dir": str(result.session_dir) if result.session_dir is not None else None,
        }
        _validate_collection_outcome(outcome)
        _write_collection_status(
            config,
            {
                "state": "completed",
                "session_id": resolved_session_id,
                "outcome": outcome,
            },
        )
        return outcome
    except Exception as exc:  # noqa: BLE001
        outcome = {"ok": False, "error": str(exc)}
        _write_collection_status(
            config,
            {
                "state": "failed",
                "session_id": resolved_session_id,
                "outcome": outcome,
            },
        )
        if console is not None and refresh is not None:
            console.print(f"[bold red]执行失败: {exc}[/bold red]")
            refresh()
        return outcome


def render_home() -> None:
    st.title("Omni-Intelligence® 脑机接口系统")
    st.markdown(
        """
        欢迎你，受试者！

        在接下来的任务中，你需要根据屏幕提示，在脑海中想象左手或右手的动作。

        当出现“左”提示时，请在脑海中想象你的左手正在持续做动作（例如握拳、松开），但请不要实际移动手部。

        当出现“右”提示时，请想象你的右手正在做相同的动作，同样不要产生真实动作。

        箭头出现后的 4 秒内，请按照刚才动作动画示范的节奏，持续重复想象对应手完成“握拳—松开”，大约两轮；不要实际运动。

        **请注意：**

        - 想象的是“自己在动”，而不是“看见手在动”
        - 保持身体静止，避免手指、肩膀或面部肌肉的实际运动
        - 尽量减少眨眼和其他多余动作

        如果中途注意力分散，请在下一次提示开始时重新集中即可。
        
        本次实验由 NCCLab 提供。
        """
    )



def render_settings(config: dict) -> None:
    st.title("系统设置")
    st.caption("按实验准备顺序配置；标注为固定的参数不会在此页被修改。")
    register_default_acquirers()

    protocol_cfg = config.setdefault("protocol", {})
    output_cfg = config.setdefault("output", {})
    ar_game_cfg = output_cfg.setdefault("ar_game", {})
    device_cfg = config.setdefault("device", {})
    storage_cfg = config.setdefault("storage", {})

    st.markdown("### 1. 被试与采集设备")
    identity_col, device_col = st.columns(2)
    subject_id = identity_col.text_input(
        "被试 ID",
        value=str(config.get("subject_id", "S001")),
        key="settings_subject_id",
        help="用于创建独立的数据目录；正式开始前必须核对。",
    ).strip()
    subject_id_valid = bool(
        subject_id
        and subject_id not in {".", ".."}
        and re.fullmatch(r"[A-Za-z0-9_.-]+", subject_id)
    )
    if not subject_id_valid:
        identity_col.error("被试 ID 只能包含字母、数字、下划线、短横线和英文句点。")
    available_devices = set(AcquirerFactory.list_devices())
    devices = [name for name in ("neuracle", "dummy") if name in available_devices]
    if not devices:
        st.error("没有可用的采集设备后端。")
        return
    current_device = (
        "dummy"
        if bool(config.get("hardware_dummy_mode", False))
        else str(config.get("device_type", devices[0]))
    )
    device_labels = {
        "dummy": "无硬件（模拟 EEG）",
        "neuracle": "博睿康 Neuracle / JellyFish",
    }
    device_type = device_col.selectbox(
        "采集模式 / 设备",
        devices,
        index=devices.index(current_device) if current_device in devices else 0,
        format_func=lambda value: device_labels.get(value, value),
        key="settings_device_type",
    )

    neuracle_host = str(device_cfg.get("neuracle_host", "127.0.0.1"))
    neuracle_port = int(device_cfg.get("neuracle_port", 8712))
    neuracle_transport_delay_sec = float(
        device_cfg.get("neuracle_transport_delay_sec", 0.0)
    )
    if device_type == "neuracle":
        jellyfish_host_col, jellyfish_port_col = st.columns([2, 1])
        neuracle_host = jellyfish_host_col.text_input(
            "JellyFish 主机地址",
            value=neuracle_host,
            key="settings_neuracle_host",
            help="JellyFish 与 GUI 在同一台电脑时使用 127.0.0.1。",
        )
        neuracle_port = int(
            jellyfish_port_col.number_input(
                "JellyFish TCP 端口",
                min_value=1,
                max_value=65535,
                value=neuracle_port,
                step=1,
                key="settings_neuracle_port",
            )
        )
        st.info(
            "正式输入固定按 250 Hz、59 个头皮 EEG 通道检查；通道缺失、重复或采样率不符时拒绝开始。"
        )
        with st.expander("高级：设备时间对齐", expanded=False):
            neuracle_transport_delay_sec = float(
                st.number_input(
                    "设备到 JellyFish 固定传输延迟（秒）",
                    min_value=0.0,
                    value=neuracle_transport_delay_sec,
                    step=0.001,
                    format="%.3f",
                    key="settings_neuracle_transport_delay",
                    help="没有独立测量结果时保持 0.000；不要凭感觉填写。",
                )
            )
            st.caption("该值只做一次固定偏移补偿，不会按 trial 累计。")
    if device_type == "dummy":
        st.info(
            "无硬件模式会模拟正式采集端的 250 Hz、59 个纯 EEG 通道；"
            "范式、事件、保存和采后处理与正式采集一致。"
        )

    st.markdown("### 2. 采集计划")
    st.caption("单个 trial 固定为 2 秒注视 + 2 秒手部动画 + 4 秒箭头运动想象。")
    fixed_timing = TrialTiming()
    structure_col1, structure_col2, rest_col = st.columns(3)
    collection_blocks = int(
        structure_col1.number_input(
            "Block 数量（个）",
            min_value=1,
            value=int(protocol_cfg.get("collection_blocks", 9)),
            step=1,
            key="settings_collection_blocks",
        )
    )
    collection_trials_per_block = int(
        structure_col2.number_input(
            "每个 block 的有效 trial（个）",
            min_value=2,
            value=int(
                protocol_cfg.get("collection_trials_per_class_per_block", 50)
            )
            * len(TASK_LABELS),
            step=2,
            key="settings_trials_per_block",
            help="必须为偶数，左右手各占一半。",
        )
    )
    rest_between_blocks_sec = float(
        rest_col.number_input(
            "Block 间自动休息（秒）",
            min_value=0.0,
            value=float(protocol_cfg.get("rest_between_blocks_sec", 180.0)),
            step=5.0,
            key="settings_rest_between_blocks",
        )
    )
    structure_valid = collection_trials_per_block % len(TASK_LABELS) == 0
    collection_trials_per_class_per_block = (
        collection_trials_per_block // len(TASK_LABELS)
    )
    if not structure_valid:
        st.error("每个 block 的有效 trial 数必须为偶数，才能保证左右手数量相等。")
    total_trials = collection_blocks * collection_trials_per_block
    trials_per_class = total_trials // len(TASK_LABELS)
    pure_trial_seconds = total_trials * fixed_timing.total_sec
    automatic_rest_seconds = max(collection_blocks - 1, 0) * rest_between_blocks_sec
    planned_seconds = pure_trial_seconds + automatic_rest_seconds
    planned_hours = int(planned_seconds // 3600)
    planned_minutes = int(round((planned_seconds % 3600) / 60.0))
    if planned_minutes == 60:
        planned_hours += 1
        planned_minutes = 0
    planned_duration = (
        f"{planned_hours} 小时 {planned_minutes} 分钟"
        if planned_hours
        else f"{planned_minutes} 分钟"
    )
    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    summary_col1.metric(
        "固定 trial",
        f"{fixed_timing.fixation_sec:g} + {fixed_timing.cue_sec:g} + "
        f"{fixed_timing.control_sec:g} 秒",
    )
    summary_col2.metric("总有效 trial", f"{total_trials}")
    summary_col3.metric("左右手数量", f"各 {trials_per_class}")
    summary_col4.metric("计划总时长", planned_duration)
    st.caption(
        "黑底绿十字 → 手部开合动画 → 黑底绿箭头；箭头结束后直接进入下一 trial。"
        f"纯 trial {pure_trial_seconds / 60.0:.1f} 分钟，自动休息 "
        f"{automatic_rest_seconds / 60.0:.1f} 分钟；手动休息会额外延长。"
    )

    st.markdown("### 3. 数据保存")
    records_dir = st.text_input(
        "数据根目录",
        value=str(storage_cfg.get("records_dir", "records_storage")),
        key="settings_records_dir",
        help="相对路径以项目目录为基准；每个被试和 session 会自动创建子目录。",
    ).strip()
    records_dir_valid = bool(records_dir)
    if not records_dir_valid:
        st.error("数据根目录不能为空。")
    st.caption(
        "正式采集保存连续 250 Hz EEG、样本号事件、元数据、检查点和采后 4 秒窗口；不创建模型文件。"
    )

    step_sec = float(config.get("step_sec", 0.5))
    visual_onset_delay_sec = float(ar_game_cfg.get("visual_onset_delay_sec", 0.0))
    ar_game_enabled = bool(ar_game_cfg.get("enabled", False))
    ar_game_host = str(ar_game_cfg.get("host", "127.0.0.1"))
    ar_game_port = int(ar_game_cfg.get("port", 5005))
    ar_game_timeout_sec = float(ar_game_cfg.get("timeout_sec", 3.0))
    with st.expander("独立功能：实时解码与 AR（不影响正式采集）", expanded=False):
        st.caption("这些参数仅供“测试模式”和“实时解码”页面使用。")
        step_sec = float(
            st.number_input(
                "解码刷新步长（秒）",
                min_value=0.05,
                value=step_sec,
                step=0.05,
                key="settings_decode_step",
            )
        )
        ar_enabled_col, ar_host_col, ar_port_col = st.columns([1, 2, 1])
        ar_game_enabled = ar_enabled_col.checkbox(
            "启用 AR TCP",
            value=ar_game_enabled,
            key="settings_ar_enabled",
        )
        ar_game_host = ar_host_col.text_input(
            "AR 主机",
            value=ar_game_host,
            key="settings_ar_host",
        )
        ar_game_port = int(
            ar_port_col.number_input(
                "AR 端口",
                min_value=1,
                max_value=65535,
                value=ar_game_port,
                step=1,
                key="settings_ar_port",
            )
        )
        ar_timing_col1, ar_timing_col2 = st.columns(2)
        ar_game_timeout_sec = float(
            ar_timing_col1.number_input(
                "AR TCP 超时（秒）",
                min_value=0.1,
                value=ar_game_timeout_sec,
                step=0.1,
                key="settings_ar_timeout",
            )
        )
        visual_onset_delay_sec = float(
            ar_timing_col2.number_input(
                "Unity ACK 到画面显示延迟（秒）",
                min_value=0.0,
                value=visual_onset_delay_sec,
                step=0.001,
                format="%.3f",
                key="settings_visual_onset_delay",
            )
        )

    st.divider()
    if st.button(
        "保存全部设置",
        type="primary",
        disabled=not (structure_valid and subject_id_valid and records_dir_valid),
        key="settings_save",
    ):
        protocol_cfg.pop("trial_timing", None)
        protocol_cfg.pop("collection_stride_sec", None)
        protocol_cfg.pop("motor_imagery_start_offset_sec", None)
        protocol_cfg.pop("motor_imagery_stop_offset_sec", None)
        config.update(
            {
                "subject_id": subject_id,
                "device_type": device_type,
                "hardware_dummy_mode": device_type == "dummy",
                "step_sec": step_sec,
            }
        )
        protocol_cfg.update(
            {
                "collection_blocks": collection_blocks,
                "collection_trials_per_class_per_block": (
                    collection_trials_per_class_per_block
                ),
                "rest_between_blocks_sec": rest_between_blocks_sec,
            }
        )
        device_cfg.update(
            {
                "neuracle_host": neuracle_host,
                "neuracle_port": neuracle_port,
                "neuracle_transport_delay_sec": neuracle_transport_delay_sec,
            }
        )
        storage_cfg["records_dir"] = records_dir
        next_ar_game_cfg = dict(ar_game_cfg)
        next_ar_game_cfg.update(
            {
                "enabled": ar_game_enabled,
                "host": ar_game_host,
                "port": ar_game_port,
                "timeout_sec": ar_game_timeout_sec,
                "visual_onset_delay_sec": visual_onset_delay_sec,
            }
        )
        output_cfg["ar_game"] = next_ar_game_cfg
        save_config(config)
        st.success("设置已保存；下一次启动采集时生效。")


def render_probe(config: dict) -> None:
    st.title("连通检测")
    st.markdown("在正式开始前，先确认采集设备网络可达并能返回 EEG 数据。")
    dur = st.number_input("探测时长 (秒)", min_value=4.0, value=5.0, step=0.5)

    if st.button("开始探测", type="primary"):
        selected_device = str(config.get("device_type", "neuracle"))

        with st.spinner(f"正在尝试连接 {selected_device} ..."):
            try:
                acquirer = build_acquirer(device_name=selected_device, config=config)
                st.info(f"设备对象已创建。尝试读取 {dur:.1f} 秒数据...")
                acquirer.start_stream()
                time.sleep(max(dur, 0.1))
                window, _ = acquirer.get_chunk(float(config.get("window_sec", 4.0)))
                acquirer.stop_stream()

                st.success("设备连通正常。")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Shape", str(window.shape))
                col2.metric("Mean (uV)", f"{window.mean():.3f}")
                col3.metric("Std (uV)", f"{window.std():.3f}")
                col4.metric("Max Abs (uV)", f"{abs(window).max():.3f}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"连通失败: {exc}")


def _render_running_collection(
    config: dict,
    protocol: ProtocolConfig,
) -> dict[str, object] | None:
    handle = _get_collection_worker(config)
    if handle is None:
        console, _ = init_live_view(
            fullscreen=True,
            show_debug=False,
            stable_surface=True,
            render_initial=False,
        )
        completed_trials, total_trials = _collection_trial_progress(
            console,
            protocol,
        )
        surface_ready = _render_collection_stimulus_surface(
            console.stimulus_frame(),
            completed_trials=completed_trials,
            total_trials=total_trials,
        )
        if not surface_ready:
            # The acquisition clock must not start until the persistent iframe
            # has loaded. Otherwise a slow first mount can consume fixation or
            # cue time before the subject sees the stimulus.
            time.sleep(0.05)
            st.rerun()
            return None
        handle = _start_collection_worker(config, protocol, console)
    else:
        init_live_view(
            fullscreen=True,
            show_debug=False,
            existing_console=handle.console,
            stable_surface=True,
            render_initial=False,
        )
        handle.console.render_pending()
        completed_trials, total_trials = _collection_trial_progress(
            handle.console,
            protocol,
        )
        _render_collection_stimulus_surface(
            handle.console.stimulus_frame(),
            completed_trials=completed_trials,
            total_trials=total_trials,
        )
    outcome = handle.outcome()
    if outcome is not None:
        return outcome
    pause_control = handle.pause_control

    def pause_and_leave_collection() -> None:
        pause_control.request_pause()
        st.session_state.collection_worker_hidden = True
        st.session_state.pop("collection_view", None)
        st.session_state.gui_nav_mode = "数据采集"

    st.button(
        "≪",
        key="collection_return_from_running",
        on_click=pause_and_leave_collection,
    )
    if pause_control.automatic_break:
        st.button(
            "组间休息中",
            key="collection_automatic_break",
            disabled=True,
        )
    elif pause_control.paused:
        st.button(
            "继续采集",
            key="collection_resume",
            type="primary",
            on_click=pause_control.resume,
        )
    elif pause_control.pause_requested:
        st.button(
            "正在丢弃当前 trial…",
            key="collection_pause_pending",
            disabled=True,
        )
    else:
        st.button(
            "我要休息",
            key="collection_request_pause",
            on_click=pause_control.request_pause,
        )
    # Keep the browser stimulus close to the acquisition-thread marker onset.
    # Pre-imagery stages last only two seconds, so a 250 ms polling cadence is
    # visibly and scientifically too coarse.
    time.sleep(0.05)
    st.rerun()
    return None


def _collection_trial_progress(
    console: StreamlitConsole,
    protocol: ProtocolConfig,
) -> tuple[int, int]:
    """Return clamped valid-trial progress for the persistent stimulus surface."""

    planned_total = (
        int(getattr(protocol, "collection_blocks", 9))
        * int(getattr(protocol, "collection_trials_per_class_per_block", 50))
        * len(TASK_LABELS)
    )
    completed, reported_total = console.trial_progress()
    total = reported_total if reported_total > 0 else planned_total
    completed = min(max(int(completed), 0), total) if total > 0 else 0
    return completed, total


def _request_collection_start(config: dict) -> None:
    st.session_state.pop("collection_last_outcome", None)
    _remove_collection_worker(config)
    # Force a fresh, initially blank stimulus iframe for each new run. This
    # prevents a stale arrow from the preceding preview/session flashing before
    # the first fixation render arrives.
    st.session_state.collection_stimulus_surface_epoch = str(time.time_ns())
    _write_collection_status(
        config,
        {
            "state": "running",
            "launch_requested": True,
        },
    )
    st.session_state.collection_view = "run"


def render_collection_ready_screen(
    config: dict,
    protocol: ProtocolConfig,
    *,
    hardware_free_rehearsal: bool = False,
) -> None:
    """Final operator-controlled gate before the fixed formal session starts."""

    total_trials = (
        protocol.collection_blocks
        * protocol.collection_trials_per_class_per_block
        * len(TASK_LABELS)
    )
    fixation_preview = mi_frame_html(
        MiVisualFrame(MiVisualStage.FIXATION)
    )
    ready_title = (
        "无硬件演练已就绪"
        if hardware_free_rehearsal
        else "左右手运动想象会话已就绪"
    )
    ready_body = (
        "本次使用模拟 EEG，数据与正式被试记录隔离；"
        "可测试完整刺激、自动休息和手动暂停。"
        if hardware_free_rehearsal
        else "确认 EEG 已连接、被试双手放松且正在注视屏幕后，再开始正式采集。"
    )
    st.markdown(
        (
            "<div class='oi-guidance-panel'>"
            "<div class='oi-guidance-content'>"
            "<div class='oi-guidance-kicker'>正式采集前确认</div>"
            "<div class='oi-guidance-stimulus'>"
            f"{fixation_preview}"
            "</div>"
            f"<h1 class='oi-guidance-title'>{ready_title}</h1>"
            "<div class='oi-guidance-body'>"
            f"{protocol.collection_blocks} 个 block，共 {total_trials} 个 trial。"
            f"{ready_body}"
            "</div>"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    _render_computer_fullscreen_control()
    st.caption("全屏会隐藏浏览器和系统界面；按 Esc 可随时退出。")
    if st.button(
        "开始无硬件演练" if hardware_free_rehearsal else "开始正式采集",
        key="collection_start_formal",
        type="primary",
    ):
        _request_collection_start(config)
        st.rerun()


def render_collection(config: dict) -> None:
    rehearsal_mode = bool(st.session_state.get("collection_hardware_free_rehearsal", False))
    run_config = _hardware_free_rehearsal_config(config) if rehearsal_mode else config
    protocol = ProtocolConfig.from_config(run_config)

    collection_view = st.session_state.get("collection_view")
    if collection_view is not None and collection_view not in _COLLECTION_VIEWS:
        st.session_state.pop("collection_view", None)
        collection_view = None
    active_worker = _get_collection_worker(run_config)
    if collection_view is None and active_worker is not None:
        completed_outcome = active_worker.outcome()
        if completed_outcome is not None:
            st.session_state.collection_last_outcome = completed_outcome
            st.session_state.pop("collection_hardware_free_rehearsal", None)
            st.session_state.pop("collection_worker_hidden", None)
            _remove_collection_worker(run_config)
            active_worker = None
    if (
        collection_view is None
        and active_worker is not None
        and active_worker.outcome() is None
        and not bool(st.session_state.get("collection_worker_hidden", False))
    ):
        st.session_state.collection_view = "run"
        st.rerun()
    if collection_view is not None:
        enter_experiment_view()
        is_running = collection_view == "run"
        if not is_running:
            render_experiment_return_button(
                target_page="数据采集",
                state_keys=(
                    "collection_view",
                    "collection_after_guidance",
                    "collection_guidance_step",
                    "collection_hardware_free_rehearsal",
                ),
                key="collection_return_from_experiment",
            )
        if collection_view == "guidance":
            render_collection_guidance()
        elif collection_view == "ready":
            render_collection_ready_screen(
                run_config,
                protocol,
                hardware_free_rehearsal=rehearsal_mode,
            )
        elif collection_view == "trial_test":
            run_collection_trial_test(protocol)
            st.session_state.pop("collection_view", None)
            st.session_state.gui_nav_mode = "数据采集"
            st.rerun()
        elif collection_view == "run":
            outcome = _render_running_collection(run_config, protocol)
            if outcome is not None:
                st.session_state.collection_last_outcome = outcome
                st.session_state.pop("collection_view", None)
                st.session_state.pop("collection_hardware_free_rehearsal", None)
                st.session_state.gui_nav_mode = "数据采集"
                _remove_collection_worker(run_config)
                st.rerun()
        return

    st.title("运动想象数据采集")
    monitor_subject = _requested_monitor_subject()
    external_collection_active = (
        bool(monitor_subject)
        and _read_latest_active_collection_checkpoint(
            config,
            subject_id=monitor_subject,
        )
        is not None
    )
    if monitor_subject:
        render_external_collection_progress(config, monitor_subject)
    if external_collection_active:
        return
    if (
        active_worker is not None
        and active_worker.outcome() is None
        and bool(st.session_state.get("collection_worker_hidden", False))
    ):
        pause_state = active_worker.pause_control
        if pause_state.paused:
            st.warning("采集已暂停；刚才所在的 trial 已丢弃。")
        else:
            st.warning("正在丢弃当前 trial 并进入暂停，请稍候。")
        if st.button("返回正在进行的采集", type="primary"):
            st.session_state.pop("collection_worker_hidden", None)
            st.session_state.collection_view = "run"
            st.rerun()
        return
    if not isinstance(st.session_state.get("collection_last_outcome"), dict):
        recovered_outcome = _recover_completed_collection(config)
        if recovered_outcome is not None:
            st.session_state.collection_last_outcome = recovered_outcome
    collection_outcome = st.session_state.get("collection_last_outcome")
    if isinstance(collection_outcome, dict):
        if bool(collection_outcome.get("ok", False)):
            if bool(collection_outcome.get("rehearsal", False)):
                st.success("无硬件演练完成；模拟数据已与正式被试记录隔离保存。")
            else:
                st.success("采集完成，原始 EEG、事件、元数据以及采后处理窗口已保存。")
            if bool(collection_outcome.get("recovered_after_reconnect", False)):
                st.info("已从磁盘恢复采集完成状态；此前的页面刷新或断线没有丢失数据。")
            st.write(f"- 有效 trial 数: **{int(collection_outcome.get('trials_collected', 0))}**")
            st.write(f"- 有效4秒窗口数: **{int(collection_outcome.get('windows_collected', 0))}**")
            if collection_outcome.get("continuous_eeg_path"):
                st.write(f"- 连续 EEG: `{collection_outcome['continuous_eeg_path']}`")
            if collection_outcome.get("events_path"):
                st.write(f"- 事件文件: `{collection_outcome['events_path']}`")
            if collection_outcome.get("windows_path"):
                st.write(f"- 4秒窗口文件: `{collection_outcome['windows_path']}`")
            if collection_outcome.get("session_dir"):
                st.write(f"- session 保存位置: `{collection_outcome['session_dir']}`")
        else:
            st.error(
                "本次采集失败："
                f"{collection_outcome.get('error', '未知错误')}。"
                "已写入磁盘的原始 session 会保留，可在修复后离线恢复。"
            )
            if bool(collection_outcome.get("recovered_after_reconnect", False)):
                st.info("已从磁盘恢复失败状态；页面没有卡住，错误信息和原始数据仍然保留。")
    st.markdown("正式采集开始后，页面仅显示被试刺激；运行日志不会覆盖刺激画面。")
    st.info(
        "采集中只记录 250 Hz 连续 EEG 和事件样本点；会话结束后才执行整段预处理、"
        "降采样和 4 秒切窗。此流程不会加载、训练、推理或更新模型。"
    )
    st.caption(
        f"采集范式：绿色注视十字 {protocol.trial_timing.fixation_sec:.1f}s + "
        f"左右手动作提示 {protocol.trial_timing.cue_sec:.1f}s + "
        f"箭头运动想象 {protocol.trial_timing.control_sec:.1f}s；"
        f"只截取箭头出现后的完整 {protocol.window_sec:.1f}s MI 区间；"
        "箭头结束后立即进入下一 trial 的注视十字。"
    )

    tutorial_col, test_col, rehearsal_col, run_col = st.columns([1, 1, 1.15, 1.3])
    tutorial_requested = tutorial_col.button("查看范式", type="secondary", use_container_width=True)
    trial_test_requested = test_col.button("画面测试", type="secondary", use_container_width=True)
    rehearsal_requested = rehearsal_col.button(
        "无硬件演练",
        type="secondary",
        use_container_width=True,
    )
    run_requested = run_col.button("进入正式采集流程", type="primary", use_container_width=True)

    if tutorial_requested:
        st.session_state.collection_view = "guidance"
        st.session_state.collection_after_guidance = "return"
        st.session_state.collection_guidance_step = 0
        st.rerun()

    if trial_test_requested:
        st.session_state.collection_view = "trial_test"
        st.rerun()

    if rehearsal_requested:
        st.session_state.collection_hardware_free_rehearsal = True
        st.session_state.collection_view = "ready"
        st.rerun()

    if run_requested:
        st.session_state.pop("collection_hardware_free_rehearsal", None)
        st.session_state.collection_view = "guidance"
        st.session_state.collection_after_guidance = "ready"
        st.session_state.collection_guidance_step = 0
        st.rerun()


def render_test_mode(config: dict) -> None:
    from decoder.real_time_decoder import RealTimeDecoder
    from models.factory import ModelFactory

    test_view = st.session_state.get("test_mode_view")
    if test_view not in {None, "run"}:
        st.session_state.pop("test_mode_view", None)
        test_view = None
    if test_view == "run":
        enter_experiment_view()
        render_experiment_return_button(
            target_page="测试模式",
            state_keys=("test_mode_view", "test_mode_duration"),
            key="test_mode_return_from_experiment",
        )
        run_test_mode_session(config, duration=int(st.session_state.pop("test_mode_duration", 120)))
        st.session_state.pop("test_mode_view", None)
        return

    st.title("Cue 测试模式")
    st.markdown("运行过程中会展示 cue 和模型输出日志。")
    duration = st.number_input("测试总时长 (秒)", min_value=30, value=120, step=30)

    if st.button("开始测试", type="primary"):
        st.session_state.test_mode_duration = int(duration)
        st.session_state.test_mode_view = "run"
        st.rerun()


def run_test_mode_session(config: dict, *, duration: int) -> None:
    try:
        subject_id = str(config["subject_id"])
        model_name = str(config["model_name"])
        acquirer = build_acquirer(
            device_name=str(config["device_type"]),
            config=config,
        )
        effective_n_channels = int(acquirer.metadata.n_channels)
        console, refresh = init_live_view(fullscreen=True)
        model = ModelFactory.get(
            model_name,
            n_chans=effective_n_channels,
            sfreq=float(config["sfreq"]),
            n_classes=int(config["n_classes"]),
            n_times=int(float(config["sfreq"]) * float(config["window_sec"])),
        )
        model_path = resolve_model_path(
            config,
            subject_id,
            model_name,
            device_name=str(config["device_type"]),
            n_chans=effective_n_channels,
            n_times=int(float(config["sfreq"]) * float(config["window_sec"])),
        )
        if not model_path.exists():
            expected_path = build_model_path(
                config,
                subject_id,
                model_name,
                device_name=str(config["device_type"]),
            )
            st.error(
                f"未找到模型权重文件: {expected_path}。"
                f"{_missing_model_guidance(config)}"
            )
            return
        if model_path.parent.name == "dummy_decoders":
            st.info(f"使用内置 dummy 测试权重: `{model_path}`")
        model.load(model_path)

        command_outlet = LSLCommandOutlet(
            stream_name=str(config["output"]["command_stream_name"]),
            stream_type=str(config["output"]["command_stream_type"]),
        )
        decoder = RealTimeDecoder(
            acquirer=acquirer,
            model=model,
            console=console,
            command_outlet=command_outlet,
            game_command_outlet=build_game_command_outlet(config),
            sfreq=float(config["sfreq"]),
            window_sec=float(config["window_sec"]),
            step_sec=float(config["step_sec"]),
            confidence_threshold=float(config["confidence_threshold"]),
            mc_dropout_passes=int(config["mc_dropout_passes"]),
            status_callback=_update_ar_decoder_status,
            experiment_config=config,
            model_name=model_name,
            model_source_path=model_path,
        )

        test_mode_cfg = config.get("test_mode", {})
        block_sec = float(
            test_mode_cfg.get(
                "block_sec",
                config.get("test_mode_block_sec", config.get("collect_block_sec", 10.0)),
            )
        )
        initial_rest_sec = max(
            float(test_mode_cfg.get("initial_rest_sec", 10.0)),
            0.0,
        )

        def update_test_progress(
            stage_name: str,
            elapsed_sec: float,
            total_sec: float,
        ) -> None:
            console.set_stage_progress(
                stage_name=stage_name,
                elapsed_sec=elapsed_sec,
                duration_sec=total_sec,
            )

        console.set_stage_progress(
            stage_name="启动测试模式",
            elapsed_sec=0.0,
            duration_sec=10.0,
        )
        refresh()
        with st.spinner("测试模式采集中..."):
            result = decoder.run_test_mode(
                subject_id=subject_id,
                marker_backend=NoOpMarkerBackend(),
                duration_sec=int(duration),
                block_sec=block_sec,
                initial_rest_sec=initial_rest_sec,
                save_dir=Path(
                    str(
                        config.get("storage", {}).get(
                            "records_dir",
                            "records_storage",
                        )
                    )
                )
                / subject_id
                / "test_mode",
                heartbeat=refresh,
                stage_progress=update_test_progress,
            )

        console.print("[bold green]测试结束[/bold green]")
        refresh()
        st.success("测试结束。")
        st.write(f"- 记录的窗口数: **{result['windows']}**")
        st.write(f"- 准确率: **{result['accuracy']:.3f}**")
        st.write(f"- 有效准确率: **{result['valid_accuracy']:.3f}**")
    except Exception as exc:  # noqa: BLE001
        st.error(f"执行失败: {exc}")


def _render_online_adaptation_notice(adaptation_cfg: dict) -> None:
    if not bool(adaptation_cfg.get("enabled", False)):
        return
    simulation_cfg = adaptation_cfg.get("simulation", {})
    cued_cfg = adaptation_cfg.get("cued_labels", {})
    if bool(simulation_cfg.get("enabled", False)):
        source_text = "标签驱动 Dummy"
    elif bool(cued_cfg.get("enabled", True)):
        source_text = "连续统一场景"
    else:
        source_text = "HTTP 真值标签"
    neuro_cfg = adaptation_cfg.get("neuroonline", {})
    first_update_seconds = float(neuro_cfg.get("first_update_seconds", 32.0))
    update_stride_seconds = float(neuro_cfg.get("update_stride_seconds", 32.0))
    window_duration_sec = float(neuro_cfg.get("window_duration_sec", 4.0))
    st.info(
        "NeuroOnline 已开启："
        f"累计 {first_update_seconds:g} 个训练窗口秒后，"
        f"每 {update_stride_seconds:g} 个窗口秒更新一次"
        f"（当前 {window_duration_sec:g}s 窗对应 "
        f"{seconds_to_windows(first_update_seconds, window_duration_sec)}/"
        f"{seconds_to_windows(update_stride_seconds, window_duration_sec)} 个窗口）；"
        f"当前标签源为 {source_text}。"
    )


def _initial_online_adaptation_status(adaptation_cfg: dict) -> dict | None:
    """Build a visible pre-run dashboard instead of leaving an empty placeholder."""

    if not bool(adaptation_cfg.get("enabled", False)):
        return None
    neuro_cfg = adaptation_cfg.get("neuroonline", {}) or {}
    window_duration_sec = float(neuro_cfg.get("window_duration_sec", 4.0))
    first_update_seconds = float(neuro_cfg.get("first_update_seconds", 32.0))
    threshold = seconds_to_windows(first_update_seconds, window_duration_sec)
    return {
        "enabled": True,
        "strategy": "neuroonline",
        "state": "等待启动",
        "buffered_windows": 0,
        "buffered_window_seconds": 0.0,
        "seen_labeled_windows": 0,
        "seen_labeled_window_seconds": 0.0,
        "samples_until_update": threshold,
        "window_seconds_until_update": first_update_seconds,
        "next_update_step": threshold,
        "next_update_window_seconds": first_update_seconds,
        "progress": 0.0,
        "class_counts": {"0": 0, "1": 0},
        "prequential": {
            "balanced_accuracy": 0.0,
            "per_class_accuracy": {"0": 0.0, "1": 0.0},
            "confusion_matrix": [[0, 0], [0, 0]],
        },
        "update_history": [],
        "last_result": None,
    }


def _build_online_label_source(
    config: dict,
    adaptation_cfg: dict,
    acquirer: AbstractAcquirer,
) -> tuple[OnlineLabelSource | None, ManualLabelHttpServer | None]:
    if not bool(adaptation_cfg.get("enabled", False)):
        return None, None

    simulation_cfg = adaptation_cfg.get("simulation", {})
    if bool(simulation_cfg.get("enabled", False)) and str(acquirer.metadata.name) == "dummy":
        st.info("在线适配使用标签驱动 Dummy 模拟被试。")
        return (
            SimulatedOnlineLabelSource(
                acquirer,
                trial_sec=float(simulation_cfg.get("trial_sec", 6.0)),
                settle_sec=float(simulation_cfg.get("settle_sec", config["window_sec"])),
                seed=int(adaptation_cfg.get("random_seed", 17)),
            ),
            None,
        )

    cued_cfg = adaptation_cfg.get("cued_labels", {})
    if bool(cued_cfg.get("enabled", True)):
        st.info(
            "在线适配使用与 Unity 障碍布局统一的连续场景真值；每个 Scene 先采集一个"
            "因果主决策窗，再放行模型横向控制。跨场景、换道保护区或质量不合格"
            "的 EEG 窗口不进入训练和准确率。"
        )
        return build_cued_online_label_source(config), None

    source = ManualOnlineLabelSource(default_ttl_sec=2.0)
    server = ManualLabelHttpServer(source, host="127.0.0.1", port=8776)
    server.start()
    st.info("在线标签接口已启动: `http://127.0.0.1:8776/api/label`")
    return source, server


def render_realtime(config: dict) -> None:
    from decoder.real_time_decoder import RealTimeDecoder
    from models.factory import ModelFactory

    st.title("实时解码")
    st.markdown("开始后会持续显示模型输出。")
    render_ar_forwarding_panel(config, render_adaptation=False)
    adaptation_cfg = config.get("online_adaptation", {})
    cue_panel = st.empty()
    adaptation_panel = st.empty()
    online_label_source: OnlineLabelSource | None = None

    def redraw_cue_panel() -> None:
        cue_panel.empty()
        source_status = None
        if isinstance(online_label_source, CuedOnlineLabelSource):
            source_status = online_label_source.status()
            decoder_status = _get_ar_forward_status()
            runtime_label_status = decoder_status.get("online_label_source")
            if isinstance(runtime_label_status, dict):
                source_status.update(runtime_label_status)
            timing_alignment = decoder_status.get("timing_alignment")
            if isinstance(timing_alignment, dict):
                source_status["timing_alignment"] = timing_alignment
        with cue_panel.container():
            render_online_cue_panel(source_status, ui=st)

    def redraw_adaptation_panel() -> None:
        adaptation_panel.empty()
        adaptation_status = _get_ar_forward_status().get("online_adaptation")
        if not isinstance(adaptation_status, dict):
            adaptation_status = _initial_online_adaptation_status(adaptation_cfg)
        with adaptation_panel.container():
            render_online_adaptation_panel(
                adaptation_status,
                ui=st,
            )

    redraw_cue_panel()
    redraw_adaptation_panel()
    record = st.checkbox(
        "保存实时脑波数据至本地记录",
        value=bool(config.get("storage", {}).get("record_realtime_default", False)),
    )
    _render_online_adaptation_notice(adaptation_cfg)

    if st.button("开始实时解码", type="primary"):
        if (
            bool(adaptation_cfg.get("enabled", False))
            and str(adaptation_cfg.get("strategy", "")).strip().lower()
            == "neuroonline"
            and not record
        ):
            st.error("NeuroOnline正式实验必须开启实时记录，请勾选“保存实时脑波数据至本地记录”。")
            return
        try:
            subject_id = str(config["subject_id"])
            model_name = str(config["model_name"])
            acquirer = build_acquirer(
                device_name=str(config["device_type"]),
                config=config,
            )
            effective_n_channels = int(acquirer.metadata.n_channels)
            console, refresh_console = init_live_view()
            last_dashboard_refresh = 0.0

            def refresh() -> None:
                nonlocal last_dashboard_refresh
                refresh_console()
                now = time.monotonic()
                if now - last_dashboard_refresh >= 0.5:
                    redraw_cue_panel()
                    redraw_adaptation_panel()
                    last_dashboard_refresh = now
            model = ModelFactory.get(
                model_name,
                n_chans=effective_n_channels,
                sfreq=float(config["sfreq"]),
                n_classes=int(config["n_classes"]),
                n_times=int(float(config["sfreq"]) * float(config["window_sec"])),
            )
            model_path = resolve_model_path(
                config,
                subject_id,
                model_name,
                device_name=str(config["device_type"]),
                n_chans=effective_n_channels,
                n_times=int(float(config["sfreq"]) * float(config["window_sec"])),
            )
            if not model_path.exists():
                st.error(
                    f"未找到模型权重文件: "
                    f"{build_model_path(config, subject_id, model_name, device_name=str(config['device_type']))}。"
                    f"{_missing_model_guidance(config)}"
                )
                return
            if model_path.parent.name == "dummy_decoders":
                st.info(f"使用内置 dummy 测试权重: `{model_path}`")
            model.load(model_path)

            online_label_source, online_label_server = _build_online_label_source(
                config,
                adaptation_cfg,
                acquirer,
            )

            primary_model_path = build_model_path(
                config,
                subject_id,
                model_name,
                device_name=str(config["device_type"]),
            )
            decoder = RealTimeDecoder(
                acquirer=acquirer,
                model=model,
                console=console,
                command_outlet=LSLCommandOutlet(
                    stream_name=str(config["output"]["command_stream_name"]),
                    stream_type=str(config["output"]["command_stream_type"]),
                ),
                game_command_outlet=build_game_command_outlet(config),
                sfreq=float(config["sfreq"]),
                window_sec=float(config["window_sec"]),
                step_sec=float(config["step_sec"]),
                confidence_threshold=float(config["confidence_threshold"]),
                mc_dropout_passes=int(config["mc_dropout_passes"]),
                status_callback=_update_ar_decoder_status,
                thread_context=_current_streamlit_context(),
                online_label_source=online_label_source,
                model_save_path=primary_model_path,
                batch_update_config=adaptation_cfg,
                n_classes=int(config["n_classes"]),
                experiment_config=config,
                model_name=model_name,
                model_source_path=model_path,
            )

            try:
                with st.spinner("实时解码运行中..."):
                    decoder.run_forever(
                        subject_id=subject_id,
                        record=record,
                        save_dir=Path(str(config.get("storage", {}).get("records_dir", "records_storage")))
                        / subject_id
                        / "realtime",
                        heartbeat=refresh,
                    )
            finally:
                if online_label_server is not None:
                    online_label_server.close()
        except Exception as exc:  # noqa: BLE001
            st.warning(f"解码已停止: {exc}")


def _set_gui_nav_mode(page: str) -> None:
    st.session_state.gui_nav_mode = page


def _inject_gui_nav_styles() -> None:
    st.markdown(
        """
        <style>
        /* Force a light palette so dark-text logo remains readable. */
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMainBlockContainer"] {
          background-color: #ffffff;
          color: #0f172a;
        }
        [data-testid="stHeader"] {
          background-color: #ffffff;
        }
        [data-testid="stToolbar"] {
          color: #334155;
        }
        section[data-testid="stSidebar"] {
          background: linear-gradient(180deg, #fff7ed 0%, #ffffff 70%);
          border-right: 1px solid rgba(15, 23, 42, 0.08);
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
          min-height: 100vh;
          display: flex;
          flex-direction: column;
          padding-bottom: 1rem;
        }
        section[data-testid="stSidebar"] * {
          color: #1e293b;
        }
        .stButton > button {
          color: #0f172a !important;
        }
        .stButton > button * {
          color: inherit !important;
        }
        section[data-testid="stSidebar"] .stButton > button {
          width: 100%;
          border-radius: 10px;
          padding-top: 0.72rem;
          padding-bottom: 0.72rem;
          font-weight: 600;
          font-size: 0.95rem;
          margin-bottom: 0.35rem;
          outline: none;
          transition: background-color 0.12s ease, border-color 0.12s ease, color 0.12s ease;
        }
        section[data-testid="stSidebar"] .stButton > button:focus-visible {
          box-shadow: 0 0 0 2px rgba(255, 90, 1, 0.4);
        }
        section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
          background-color: rgba(248, 250, 252, 0.95);
          border: 1px solid rgba(15, 23, 42, 0.12);
          color: rgb(30, 41, 59);
        }
        section[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
          border-color: rgba(255, 90, 1, 0.4);
          background-color: rgba(255, 90, 1, 0.07);
          color: rgb(15, 23, 42);
        }
        section[data-testid="stSidebar"] .stButton > button[kind="primary"],
        section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
          background-color: #ff4b4b !important;
          border-color: #ff4b4b !important;
          color: #ffffff !important;
        }
        section[data-testid="stSidebar"] .stButton > button[kind="primary"] *,
        section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover * {
          color: #ffffff !important;
        }
        section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"],
        section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"]:hover {
          background-color: #ff4b4b !important;
          border-color: #ff4b4b !important;
          color: #ffffff !important;
        }
        section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] *,
        section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"]:hover * {
          color: #ffffff !important;
        }
        [data-testid="stMain"] .stButton > button[kind="primary"],
        [data-testid="stMain"] .stButton > button[data-testid="stBaseButton-primary"] {
          background-color: #ff4b4b !important;
          border-color: #ff4b4b !important;
          color: #ffffff !important;
        }
        [data-testid="stMain"] .stButton > button[kind="primary"] *,
        [data-testid="stMain"] .stButton > button[data-testid="stBaseButton-primary"] * {
          color: #ffffff !important;
        }
        [data-testid="stMain"] .stButton > button[kind="primary"]:hover,
        [data-testid="stMain"] .stButton > button[data-testid="stBaseButton-primary"]:hover {
          background-color: #e53e3e !important;
          border-color: #e53e3e !important;
          color: #ffffff !important;
        }
        [data-testid="stMain"] .stButton > button[kind="secondary"],
        [data-testid="stMain"] .stButton > button[data-testid="stBaseButton-secondary"] {
          background-color: #ffffff !important;
          border-color: rgba(15, 23, 42, 0.18) !important;
          color: #0f172a !important;
        }
        [data-testid="stMain"] .stButton > button[kind="secondary"] *,
        [data-testid="stMain"] .stButton > button[data-testid="stBaseButton-secondary"] * {
          color: #0f172a !important;
        }
        [data-testid="stMain"] .stButton > button[kind="secondary"]:hover,
        [data-testid="stMain"] .stButton > button[data-testid="stBaseButton-secondary"]:hover {
          border-color: rgba(255, 90, 1, 0.45) !important;
          background-color: rgba(255, 90, 1, 0.06) !important;
          color: #0f172a !important;
        }
        .oi-sidebar-spacer {
          flex: 1 1 auto;
          min-height: 1.5rem;
        }
        .oi-sidebar-copyright {
          margin-top: 1rem;
          padding: 0.25rem 0.1rem 0;
        }
        .oi-sidebar-copyright .oi-company {
          font-size: 0.72rem;
          line-height: 1.45;
          font-weight: 600;
          color: #334155;
        }
        .oi-sidebar-copyright .oi-rights {
          margin-top: 0.35rem;
          font-size: 0.68rem;
          line-height: 1.45;
          color: #64748b;
        }
        .stMarkdown, .stText, p, label, h1, h2, h3, h4, h5, h6 {
          color: #0f172a;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    config = load_config()
    if not config:
        return

    # Streamlit runs in a child process when launched through ``cli.py gui``.
    # Starting the endpoint here keeps manual commands and realtime decoder
    # commands on the same shared Unity transport.
    start_web_command_server(config)
    _inject_gui_nav_styles()
    initial_page = "数据采集" if _requested_monitor_subject() else SIDEBAR_NAV_PAGES[0]
    st.session_state.setdefault("gui_nav_mode", initial_page)

    with st.sidebar:
        logo_path = _resolve_logo_svg_path()
        if logo_path is not None:
            render_sidebar_logo(logo_path)
        st.title("oi-mi 工作台")
        for page in SIDEBAR_NAV_PAGES:
            is_active = st.session_state.gui_nav_mode == page
            st.button(
                page,
                key=f"nav_btn_{page}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
                on_click=_set_gui_nav_mode,
                args=(page,),
            )
        st.markdown("<div class='oi-sidebar-spacer'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="oi-sidebar-copyright">
              <div class="oi-rights">© 2026 Omni-Intelligence. All rights reserved.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        mode = st.session_state.gui_nav_mode

    if mode == "首页":
        render_home()
    elif mode == "设置":
        render_settings(config)
    elif mode == "连通检测":
        render_probe(config)
    elif mode == "数据采集":
        render_collection(config)
    elif mode == "测试模式":
        render_test_mode(config)
    elif mode == "实时解码":
        render_realtime(config)


if __name__ == "__main__":
    main()
