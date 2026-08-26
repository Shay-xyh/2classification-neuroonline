"""Command-line entry point for oi-mi."""

from __future__ import annotations

import logging
import os
import re
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import numpy as np
import yaml
from rich.console import Console
from rich.table import Table

from acquisition.factory import AcquirerFactory, register_default_acquirers
from acquisition.neuracle_acquirer import NEURACLE_59_EEG_CHANNEL_NAMES
from adaptation.mi_protocol import TASK_LABELS, TASK_PARADIGM, ProtocolConfig
from game_command_router import get_shared_game_command_router
from utils.markers import (
    LSLCommandOutlet,
    NoOpMarkerBackend,
)
from utils.online_labels import (
    ManualLabelHttpServer,
    ManualOnlineLabelSource,
    SimulatedOnlineLabelSource,
    build_cued_online_label_source,
)
from utils.preprocessing import filter_and_transform
from utils.reproducibility import seed_experiment
from utils.unity_runtime import ensure_unity_game_running, wait_for_unity_scene_ready
from web_command_server import start_web_command_server

LOGGER = logging.getLogger(__name__)
CONSOLE = Console()
DEFAULT_CONFIG_FILENAME = "config.yaml"
_PROJECT_DEFAULT_CONFIG_PATH = Path(__file__).with_name(DEFAULT_CONFIG_FILENAME)
_DEFAULT_CONFIG_TEMPLATE: dict[str, Any] = {
    "subject_id": "S001",
    "model_name": "cbramod",
    "device_type": "neuracle",
    "hardware_dummy_mode": False,
    "sfreq": 200,
    "task_paradigm": TASK_PARADIGM,
    "n_classes": len(TASK_LABELS),
    "window_sec": 4.0,
    "step_sec": 0.5,
    "confidence_threshold": 0.5,
    "mc_dropout_passes": 8,
    "training_epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001,
    "test_mode_block_sec": 10,
    "buffer_sec": 60,
    "protocol": {
        "collection_blocks": 9,
        "collection_trials_per_class_per_block": 50,
        "rest_between_blocks_sec": 180.0,
        "random_seed": 17,
    },
    "device": {
        "neuracle_host": "127.0.0.1",
        "neuracle_port": 8712,
        "neuracle_source_sfreq": 250,
        "neuracle_transport_delay_sec": 0.0,
        "neuracle_eeg_channels": 59,
        "neuracle_eeg_channel_names": list(NEURACLE_59_EEG_CHANNEL_NAMES),
        # Hardware-free collection emulates the formal Neuracle EEG input.
        "dummy_source_sfreq": 250,
        "dummy_eeg_channels": 59,
        "brainco_addr": "",
        "brainco_port": 0,
        "brainco_source_sfreq": 250,
        "brainco_auto_discover": True,
        "brainco_scan_timeout_sec": 6.0,
        "brainco_ready_timeout_sec": 20.0,
        "brainco_start_retries": 2,
        "brainco_gain": 6,
        "brainco_signal_source": "NORMAL",
        "brainco_device_id": "eeg-cap",
        "dummy_label_aware": False,
    },
    "output": {
        "command_stream_name": "oi_mi_commands",
        "command_stream_type": "Markers",
        "ar_game": {
            "enabled": True,
            "host": "127.0.0.1",
            "port": 5005,
            "timeout_sec": 3.0,
            "visual_onset_delay_sec": 0.0,
            "auto_launch": True,
            "executable_path": "../oi-car-unity-src/Car_game/Builds/Windows/ARPrototype3D.exe",
            "startup_timeout_sec": 15.0,
            "startup_command_delay_sec": 0.75,
            "startup_sequence": [
                {"command": "OPEN_3D_GAME", "delay_after_sec": 2.0},
            ],
            "windowed": True,
            "window_width": 1280,
            "window_height": 720,
            "close_on_stop": False,
        },
        "web_control": {
            "enabled": True,
            "host": "0.0.0.0",
            "port": 8765,
            "manual_override_hold_sec": 0.8,
            "manual_override_release_sec": 0.25,
        },
    },
    "storage": {
        "models_dir": "models_storage",
        "records_dir": "records_storage",
        "record_realtime_default": True,
    },
    "online_adaptation": {
        "enabled": True,
        "strategy": "neuroonline",
        "neuroonline": {
            "window_duration_sec": 4.0,
            "learning_rate": 3e-5,
            "update_batch_seconds": 64.0,
            "epochs": 3,
            "first_update_seconds": 32.0,
            "update_stride_seconds": 32.0,
            "recent_history_seconds": 640.0,
            "weight_decay": 0.05,
            "mask_ratio": 0.5,
            "consistency_weight": 1.0,
            "label_smoothing": 0.1,
            "prompt_count": 32,
            "random_seed": 2026,
            "offline_random_seed": 42,
            "offline_epochs": 50,
            "offline_batch_seconds": 32.0,
            "offline_learning_rate": 1e-4,
            "offline_mask_ratio": 0.1,
            "offline_consistency_weight": 0.1,
            "offline_classification_views": "all",
            "offline_selection_metric": "window_bacc",
            "update_policy": "full",
            "backbone_learning_rate": 3e-5,
            "offline_update_policy": "full",
            "offline_backbone_learning_rate": 1e-4,
        },
        "simulation": {
            "enabled": False,
            "trial_sec": 6.0,
            "settle_sec": 2.0,
        },
        "cued_labels": {
            "enabled": True,
            "scene_duration_sec": 5.0,
            "primary_windows_per_scene": 1,
            "primary_window_spacing_sec": 1.0,
            "boundary_guard_sec": 0.5,
            "balance_pool_window_seconds_per_class": 128.0,
            "start_delay_sec": 5.0,
            "random_seed": None,
        },
    },
}


@dataclass(slots=True)
class AppContext:
    """Shared CLI state."""

    config: dict[str, Any]
    config_path: Path
    console: Console


def default_config() -> dict[str, Any]:
    """Return a writable copy of the bundled default config payload."""

    if _PROJECT_DEFAULT_CONFIG_PATH.exists():
        with _PROJECT_DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as handle:
            project_template = yaml.safe_load(handle) or {}
        if isinstance(project_template, dict):
            return deepcopy(project_template)
        LOGGER.warning(
            "Bundled config template at %s is not a mapping; falling back to static defaults.",
            _PROJECT_DEFAULT_CONFIG_PATH,
        )
    return deepcopy(_DEFAULT_CONFIG_TEMPLATE)


def resolve_config_path(config_path: Path | None = None) -> Path:
    """Resolve the config path, defaulting to the bundled oi-mi config.yaml."""

    if config_path is not None:
        return Path(config_path).expanduser().resolve()

    cwd_config = Path.cwd() / DEFAULT_CONFIG_FILENAME
    if cwd_config.exists():
        return cwd_config.resolve()

    return _PROJECT_DEFAULT_CONFIG_PATH.resolve()


def write_config(path: Path, config: dict[str, Any]) -> None:
    """Persist config as UTF-8 YAML."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)


def ensure_config_exists(path: Path) -> Path:
    """Create a default config file when none exists."""

    if not path.exists():
        write_config(path, default_config())
        LOGGER.info("Created default config at %s", path)
    return path


def get_model_factory() -> Any:
    from models.factory import ModelFactory

    return ModelFactory


def get_calibrator_class() -> Any:
    from adaptation.calibrator import Calibrator

    return Calibrator


def get_realtime_decoder_class() -> Any:
    from decoder.real_time_decoder import RealTimeDecoder

    return RealTimeDecoder


def setup_logging() -> None:
    """Configure app-wide logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    # Keep CLI output focused; hide noisy third-party debug/info logs.
    for noisy_name in ("filelock", "torio", "matplotlib"):
        logging.getLogger(noisy_name).setLevel(logging.WARNING)


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate YAML config."""

    path = ensure_config_exists(resolve_config_path(path))
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    required_keys = {
        "subject_id",
        "model_name",
        "device_type",
        "sfreq",
        "n_classes",
        "window_sec",
        "step_sec",
    }
    missing = sorted(required_keys - set(config))
    if missing:
        raise click.ClickException(f"Missing required config keys: {', '.join(missing)}")
    if config["window_sec"] <= 0 or config["step_sec"] <= 0:
        raise click.ClickException("window_sec and step_sec must be positive.")
    neuroonline_data = (
        ((config.get("online_adaptation", {}) or {}).get("neuroonline", {}) or {})
    )
    configured_duration = neuroonline_data.get("window_duration_sec")
    if configured_duration is not None and not np.isclose(
        float(configured_duration), float(config["window_sec"])
    ):
        raise click.ClickException(
            "online_adaptation.neuroonline.window_duration_sec must match window_sec."
        )
    for name in (
        "update_batch_seconds",
        "first_update_seconds",
        "update_stride_seconds",
        "recent_history_seconds",
        "offline_batch_seconds",
    ):
        value = neuroonline_data.get(name)
        if value is not None and (not np.isfinite(float(value)) or float(value) <= 0.0):
            raise click.ClickException(
                f"online_adaptation.neuroonline.{name} must be positive."
            )
    if str(config.get("task_paradigm", TASK_PARADIGM)) != TASK_PARADIGM:
        raise click.ClickException(
            f"task_paradigm must be {TASK_PARADIGM!r} for the current experiment."
        )
    if int(config["n_classes"]) != len(TASK_LABELS):
        raise click.ClickException(
            "The binary hand-MI paradigm requires n_classes=2 (left/right)."
        )
    if not np.isclose(float(config["sfreq"]), 200.0):
        raise click.ClickException(
            "The NeuroOnline experiment pipeline requires sfreq=200 Hz."
        )
    if str(config.get("device_type", "")).strip().lower() == "neuracle":
        device_config = config.get("device", {}) or {}
        eeg_channel_count = int(device_config.get("neuracle_eeg_channels", 59))
        eeg_channel_names = [
            str(name).strip()
            for name in device_config.get(
                "neuracle_eeg_channel_names",
                NEURACLE_59_EEG_CHANNEL_NAMES,
            )
        ]
        if len(eeg_channel_names) != eeg_channel_count:
            raise click.ClickException(
                "device.neuracle_eeg_channel_names must contain exactly "
                f"{eeg_channel_count} unique names."
            )
        if len({name.upper() for name in eeg_channel_names}) != len(eeg_channel_names):
            raise click.ClickException(
                "device.neuracle_eeg_channel_names contains duplicate names."
            )
        source_sfreq = float(
            device_config.get("neuracle_source_sfreq", 250.0)
        )
        if not np.isclose(source_sfreq, 250.0):
            raise click.ClickException(
                "Neuracle/JellyFish hardware must acquire at 250 Hz: "
                "set device.neuracle_source_sfreq=250. "
                "The pipeline output remains sfreq=200 Hz."
            )
    if str(config.get("device_type", "")).strip().lower() == "brainco":
        source_sfreq = float(
            (config.get("device", {}) or {}).get("brainco_source_sfreq", 250.0)
        )
        if not np.isclose(source_sfreq, 250.0):
            raise click.ClickException(
                "BrainCo hardware must acquire at 250 Hz: "
                "set device.brainco_source_sfreq=250. "
                "The pipeline output remains sfreq=200 Hz."
            )
    if bool(config.get("hardware_dummy_mode", False)) or str(
        config.get("device_type", "")
    ).strip().lower() == "dummy":
        device_config = config.get("device", {}) or {}
        dummy_source_sfreq = float(
            device_config.get(
                "dummy_source_sfreq",
                device_config.get("neuracle_source_sfreq", 250.0),
            )
        )
        dummy_channels = int(
            device_config.get(
                "dummy_eeg_channels",
                device_config.get("neuracle_eeg_channels", 59),
            )
        )
        dummy_names = tuple(
            str(name).strip()
            for name in device_config.get(
                "dummy_eeg_channel_names",
                device_config.get(
                    "neuracle_eeg_channel_names",
                    NEURACLE_59_EEG_CHANNEL_NAMES,
                ),
            )
        )
        if not np.isclose(dummy_source_sfreq, 250.0) or dummy_channels != 59:
            raise click.ClickException(
                "Hardware-free formal collection must emulate the 250 Hz, "
                "59-channel EEG acquisition input."
            )
        if len(dummy_names) != 59 or len({name.upper() for name in dummy_names}) != 59:
            raise click.ClickException(
                "Hardware-free formal collection requires 59 unique EEG channel names."
            )
    return config


def parse_subject_number(subject_id: str) -> int:
    """Extract a numeric subject index, defaulting to 1 when absent."""

    match = re.search(r"(\d+)", subject_id)
    if match is None:
        return 1
    return max(int(match.group(1)), 1)


def default_device_channels(device_name: str) -> int:
    """Return the default channel count for each supported device."""

    return 32 if device_name == "brainco" else 59


def _interactive_menu(ctx: click.Context, app: AppContext) -> None:
    """Interactive parameter input when user runs `oi-mi` only."""

    while True:
        CONSOLE.print("\n[bold cyan]oi-mi 交互菜单[/bold cyan]")
        CONSOLE.print("1) 列出可用模型")
        CONSOLE.print("2) 列出可用采集设备")
        CONSOLE.print("3) 正式数据采集（只采集、预处理和切窗，不训练）")
        CONSOLE.print("4) 实时解码（无 cue 自动输出）")
        CONSOLE.print("5) 测试模式（有 cue + 保存流式 npy + 计算准确率）")
        CONSOLE.print("6) 设备连通性探测（probe-device）")
        CONSOLE.print("7) 配置参数 (窗长, 步长等)")
        CONSOLE.print("8) 启动 GUI (Streamlit)")
        CONSOLE.print("0) 退出")

        choice = click.prompt(
            "选择功能",
            type=click.Choice(["0", "1", "2", "3", "4", "5", "6", "7", "8"]),
            default="3",
        )
        if choice == "0":
            return
        if choice == "1":
            ctx.invoke(ctx.command.get_command(ctx, "list-models"))
            continue
        if choice == "2":
            ctx.invoke(ctx.command.get_command(ctx, "list-devices"))
            continue

        if choice == "3":
            register_default_acquirers()
            protocol = ProtocolConfig.from_config(app.config)
            subject_id = click.prompt(
                "被试ID",
                default=str(app.config.get("subject_id", "S001")),
                type=str,
            )
            formal_trials = (
                protocol.collection_blocks
                * protocol.collection_trials_per_class_per_block
                * len(TASK_LABELS)
            )
            CONSOLE.print(
                f"[bold cyan]当前 protocol[/bold cyan] formal_trials={formal_trials} "
                f"trial_total={protocol.trial_timing.total_sec:.1f}s "
                f"window={protocol.window_sec:.1f}s stride={protocol.stride_sec:.1f}s"
            )
            device_name = click.prompt(
                "采集设备(device_type)",
                type=click.Choice(AcquirerFactory.list_devices()),
                default=effective_device_name(app.config),
            )
            collect_cmd = ctx.command.get_command(ctx, "collect")
            ctx.invoke(
                collect_cmd,
                subject_id=subject_id,
                device_name=device_name,
            )
            continue

        if choice == "4":
            model_factory = get_model_factory()
            subject_id = click.prompt(
                "被试ID",
                default=str(app.config.get("subject_id", "S001")),
                type=str,
            )
            model_name = click.prompt(
                "模型(model registry)",
                type=click.Choice(model_factory.list_models()),
                default=str(app.config.get("model_name", "cbramod")),
            )
            device_name = click.prompt(
                "设备类型(device_type)",
                type=str,
                default=str(app.config.get("device_type", "neuracle")),
                show_default=True,
            )
            record = click.confirm("是否保存实时解码数据？", default=False)
            run_cmd = ctx.command.get_command(ctx, "run")
            ctx.invoke(
                run_cmd,
                subject_id=subject_id,
                model_name=model_name,
                device_name=device_name,
                test_mode=False,
                test_duration=600,
                record=record,
            )
            continue

        if choice == "5":
            model_factory = get_model_factory()
            subject_id = click.prompt(
                "被试ID",
                default=str(app.config.get("subject_id", "S001")),
                type=str,
            )
            model_name = click.prompt(
                "模型(model registry)",
                type=click.Choice(model_factory.list_models()),
                default=str(app.config.get("model_name", "cbramod")),
            )
            device_name = click.prompt(
                "设备类型(device_type)",
                type=str,
                default=str(app.config.get("device_type", "neuracle")),
                show_default=True,
            )
            test_duration = click.prompt("测试时长(秒)", type=int, default=600)
            run_cmd = ctx.command.get_command(ctx, "run")
            ctx.invoke(
                run_cmd,
                subject_id=subject_id,
                model_name=model_name,
                device_name=device_name,
                test_mode=True,
                test_duration=test_duration,
            )
            continue

        if choice == "6":
            device_name = click.prompt(
                "设备类型(device_type)",
                type=str,
                default=str(app.config.get("device_type", "neuracle")),
            )
            duration = click.prompt("等待时长(秒)", type=float, default=5.0)
            probe_cmd = ctx.command.get_command(ctx, "probe-device")
            ctx.invoke(probe_cmd, device_name=device_name, duration=duration, save_buffer=False)
            continue

        if choice == "7":
            protocol_cfg = app.config.setdefault("protocol", {})
            protocol_cfg.pop("trial_timing", None)
            protocol_cfg.pop("collection_stride_sec", None)
            protocol_cfg.pop("motor_imagery_start_offset_sec", None)
            protocol_cfg.pop("motor_imagery_stop_offset_sec", None)
            output_cfg = app.config.setdefault("output", {})
            ar_game_cfg = output_cfg.setdefault("ar_game", {})
            while True:
                CONSOLE.print("\n[bold magenta]-- 配置参数设置 --[/bold magenta]")
                CONSOLE.print(f"1) 采后/解码窗长 (window_sec): [green]{app.config.get('window_sec')}[/green]")
                CONSOLE.print(f"2) 解码刷新步长 (step_sec): [green]{app.config.get('step_sec')}[/green]")
                CONSOLE.print(f"3) 当前被试 (subject_id): [green]{app.config.get('subject_id')}[/green]")
                CONSOLE.print(f"4) 默认模型 (model_name): [green]{app.config.get('model_name')}[/green]")
                CONSOLE.print("固定采集范式: [green]2 秒注视 + 2 秒动作提示 + 4 秒运动想象[/green]")
                CONSOLE.print(f"5) 采集 block 数: [green]{protocol_cfg.get('collection_blocks', 9)}[/green]")
                CONSOLE.print(f"6) 每类每 block trial 数: [green]{protocol_cfg.get('collection_trials_per_class_per_block', 50)}[/green]")
                CONSOLE.print(f"7) 采后训练固定 epoch: [green]{app.config.get('training_epochs', app.config.get('calibration_epochs', 50))}[/green]")
                CONSOLE.print(f"8) block 间休息: [green]{protocol_cfg.get('rest_between_blocks_sec', 180.0)}[/green]")
                CONSOLE.print(f"9) AR游戏控制启用: [green]{ar_game_cfg.get('enabled', False)}[/green]")
                CONSOLE.print(f"10) AR游戏主机: [green]{ar_game_cfg.get('host', '127.0.0.1')}[/green]")
                CONSOLE.print(f"11) AR游戏端口: [green]{ar_game_cfg.get('port', 5005)}[/green]")
                CONSOLE.print(f"12) AR游戏超时: [green]{ar_game_cfg.get('timeout_sec', 1.0)}[/green]")
                CONSOLE.print(f"13) AR game auto_launch: [green]{ar_game_cfg.get('auto_launch', False)}[/green]")
                CONSOLE.print(f"14) AR game executable_path: [green]{ar_game_cfg.get('executable_path', '')}[/green]")
                CONSOLE.print(f"15) AR game startup_timeout_sec: [green]{ar_game_cfg.get('startup_timeout_sec', 15.0)}[/green]")
                CONSOLE.print("0) 返回上级菜单")

                sub_choice = click.prompt("选择要修改的项", type=click.Choice([str(i) for i in range(16)]), default="0")
                if sub_choice == "0":
                    break
                elif sub_choice == "1":
                    val = click.prompt("输入采后/解码窗长 (window_sec)", type=float, default=float(app.config.get("window_sec", 4.0)))
                    app.config["window_sec"] = val
                elif sub_choice == "2":
                    val = click.prompt("输入解码刷新步长 (step_sec)", type=float, default=float(app.config.get("step_sec", 0.5)))
                    app.config["step_sec"] = val
                elif sub_choice == "3":
                    val = click.prompt("输入新的被试ID (subject_id)", type=str, default=str(app.config.get("subject_id", "S001")))
                    app.config["subject_id"] = val
                elif sub_choice == "4":
                    val = click.prompt("输入新的默认模型 (model_name)", type=str, default=str(app.config.get("model_name", "cbramod")))
                    app.config["model_name"] = val
                elif sub_choice == "5":
                    val = click.prompt("采集 block 数", type=int, default=int(protocol_cfg.get("collection_blocks", 9)))
                    protocol_cfg["collection_blocks"] = val
                elif sub_choice == "6":
                    val = click.prompt("每类每 block trial 数", type=int, default=int(protocol_cfg.get("collection_trials_per_class_per_block", 50)))
                    protocol_cfg["collection_trials_per_class_per_block"] = val
                elif sub_choice == "7":
                    val = click.prompt("采后训练固定 epoch", type=int, default=int(app.config.get("training_epochs", app.config.get("calibration_epochs", 50))))
                    app.config["training_epochs"] = val
                    app.config.pop("calibration_epochs", None)
                elif sub_choice == "8":
                    val = click.prompt("block 间休息时长 (秒)", type=float, default=float(protocol_cfg.get("rest_between_blocks_sec", 180.0)))
                    protocol_cfg["rest_between_blocks_sec"] = val
                elif sub_choice == "9":
                    val = click.confirm("是否启用 AR 游戏 TCP 控制", default=bool(ar_game_cfg.get("enabled", False)))
                    ar_game_cfg["enabled"] = val
                elif sub_choice == "10":
                    val = click.prompt("输入 AR 游戏主机地址", type=str, default=str(ar_game_cfg.get("host", "127.0.0.1")))
                    ar_game_cfg["host"] = val
                elif sub_choice == "11":
                    val = click.prompt("输入 AR 游戏端口", type=int, default=int(ar_game_cfg.get("port", 5005)))
                    ar_game_cfg["port"] = val
                elif sub_choice == "12":
                    val = click.prompt("输入 AR 游戏 TCP 超时(秒)", type=float, default=float(ar_game_cfg.get("timeout_sec", 3.0)))
                    ar_game_cfg["timeout_sec"] = val
                elif sub_choice == "13":
                    val = click.confirm("Enable local Unity exe auto-launch", default=bool(ar_game_cfg.get("auto_launch", False)))
                    ar_game_cfg["auto_launch"] = val
                elif sub_choice == "14":
                    val = click.prompt(
                        "Unity executable path",
                        type=str,
                        default=str(
                            ar_game_cfg.get(
                                "executable_path",
                                "../oi-car-unity-src/Car_game/Builds/Windows/ARPrototype3D.exe",
                            )
                        ),
                    )
                    ar_game_cfg["executable_path"] = val
                elif sub_choice == "15":
                    val = click.prompt(
                        "Unity startup timeout seconds",
                        type=float,
                        default=float(ar_game_cfg.get("startup_timeout_sec", 15.0)),
                    )
                    ar_game_cfg["startup_timeout_sec"] = val
                with app.config_path.open("w", encoding="utf-8") as f:
                    yaml.safe_dump(app.config, f, allow_unicode=True, sort_keys=False)
                CONSOLE.print("[bold green]配置已更新！[/bold green]")
            continue

        if choice == "8":
            gui_cmd = ctx.command.get_command(ctx, "gui")
            ctx.invoke(gui_cmd)
            continue


def build_model_path(
    config: dict[str, Any],
    subject_id: str,
    model_name: str,
    *,
    device_name: str | None = None,
) -> Path:
    """Return the persisted weight path for a subject/device/model tuple."""

    models_dir = Path(config["storage"]["models_dir"])
    resolved_device = effective_device_name(config, device_name)
    return models_dir / subject_id / resolved_device / f"{model_name}.pt"


DUMMY_DECODER_ASSET_DIR = Path(__file__).resolve().parent / "assets" / "dummy_decoders"


def effective_device_name(config: dict[str, Any], device_name: str | None = None) -> str:
    """Return the device namespace used for model storage and acquisition."""

    if bool(config.get("hardware_dummy_mode", False)):
        return "dummy"
    return str(device_name or config.get("device_type", "unknown"))


def dummy_decoder_asset_path(
    model_name: str,
    *,
    n_chans: int,
    n_times: int,
) -> Path:
    """Return the bundled decoder asset path for a dummy EEG profile."""

    return DUMMY_DECODER_ASSET_DIR / f"{model_name}_{n_chans}x{n_times}.pt"


def resolve_model_path(
    config: dict[str, Any],
    subject_id: str,
    model_name: str,
    *,
    device_name: str | None = None,
    n_chans: int,
    n_times: int,
) -> Path:
    """Resolve subject weights, falling back to bundled dummy decoder assets."""

    primary = build_model_path(
        config,
        subject_id,
        model_name,
        device_name=device_name,
    )
    if primary.exists():
        return primary
    if effective_device_name(config, device_name) != "dummy":
        return primary
    asset = dummy_decoder_asset_path(model_name, n_chans=n_chans, n_times=n_times)
    if asset.exists():
        return asset
    return primary


def resolve_records_dir(config: dict[str, Any]) -> Path:
    """Return the root directory used for recorded sessions."""

    return Path(str(config.get("storage", {}).get("records_dir", "records_storage")))


def load_collection_windows(
    records_dir: Path,
    subject_id: str,
    *,
    session_ids: tuple[str, ...] = (),
    use_processed: bool = True,
    include_groups: bool = False,
) -> (
    tuple[np.ndarray, np.ndarray, list[Path]]
    | tuple[np.ndarray, np.ndarray, np.ndarray | None, list[Path]]
):
    """Load current collection windows, with legacy calibration fallback."""

    collection_root = records_dir / subject_id / "collection"
    legacy_root = records_dir / subject_id / "calibration"
    available_roots = [root for root in (collection_root, legacy_root) if root.exists()]
    if not available_roots:
        raise click.ClickException(f"Collection directory not found: {collection_root}")

    if session_ids:
        session_dirs: list[Path] = []
        for session_id in session_ids:
            matches = [
                root / session_id
                for root in available_roots
                if (root / session_id).is_dir()
            ]
            if not matches:
                raise click.ClickException(f"Collection session not found: {session_id}")
            session_dirs.append(matches[0])
    else:
        session_dirs = sorted(
            path
            for root in available_roots
            for path in root.iterdir()
            if path.is_dir()
        )
    if not session_dirs:
        raise click.ClickException(f"No collection sessions found in {collection_root}")

    feature_key = "processed_windows" if use_processed else "raw_windows"
    windows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    all_sessions_have_groups = True
    next_group_id = 0
    used_sessions: list[Path] = []
    reference_shape: tuple[int, int] | None = None

    for session_dir in session_dirs:
        dataset_path = session_dir / "mi_windows.npz"
        if not dataset_path.exists():
            dataset_path = session_dir / "training_windows_main.npz"
        if not dataset_path.exists():
            raise click.ClickException(f"Missing collection window dataset: {session_dir}")
        with np.load(dataset_path) as payload:
            if feature_key not in payload or "labels" not in payload:
                raise click.ClickException(f"Collection dataset missing required arrays: {dataset_path}")
            X = payload[feature_key].astype(np.float32)
            y = payload["labels"].astype(np.int64)
            session_groups = (
                payload["trial_ids"].astype(np.int64)
                if "trial_ids" in payload
                else None
            )
        if X.shape[0] != y.shape[0]:
            raise click.ClickException(f"Mismatched window and label counts in {dataset_path}")
        if X.shape[0] == 0:
            continue
        current_shape = (int(X.shape[1]), int(X.shape[2]))
        if reference_shape is None:
            reference_shape = current_shape
        elif current_shape != reference_shape:
            raise click.ClickException(
                f"Inconsistent collection window shape: expected {reference_shape}, got {current_shape} in {dataset_path}"
            )
        windows.append(X)
        labels.append(y)
        if session_groups is None or session_groups.shape != y.shape:
            all_sessions_have_groups = False
        else:
            _, remapped = np.unique(session_groups, return_inverse=True)
            groups.append(remapped.astype(np.int64) + next_group_id)
            next_group_id += int(np.max(remapped)) + 1
        used_sessions.append(session_dir)

    if not windows:
        raise click.ClickException(f"No usable collection windows found in {collection_root}")
    combined_X = np.concatenate(windows, axis=0)
    combined_y = np.concatenate(labels, axis=0)
    if include_groups:
        combined_groups = (
            np.concatenate(groups, axis=0)
            if all_sessions_have_groups and len(groups) == len(windows)
            else None
        )
        return combined_X, combined_y, combined_groups, used_sessions
    return combined_X, combined_y, used_sessions


# Public compatibility alias for older analysis scripts.
load_calibration_windows = load_collection_windows


def iter_test_mode_chunks(test_mode_dir: Path) -> list[Path]:
    """Return sorted test-mode chunk files."""

    chunks_dir = test_mode_dir / "chunks"
    chunk_paths = sorted(chunks_dir.glob("chunk_*.npz"))
    if not chunk_paths:
        raise click.ClickException(f"No test-mode chunks found in {chunks_dir}")
    return chunk_paths


def replay_test_mode(
    *,
    model: Any,
    test_mode_dir: Path,
    sfreq: float,
    mc_dropout_passes: int,
) -> dict[str, Any]:
    """Replay saved test-mode windows through the current decoder model."""

    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    confidence_all: list[np.ndarray] = []
    total_windows = 0

    for chunk_path in iter_test_mode_chunks(test_mode_dir):
        with np.load(chunk_path) as payload:
            if "eeg_windows" not in payload or "labels_true" not in payload:
                raise click.ClickException(f"Invalid test-mode chunk format: {chunk_path}")
            windows = payload["eeg_windows"].astype(np.float32)
            y_true = payload["labels_true"].astype(np.int64)
        processed = np.stack(
            [filter_and_transform(window, sfreq=sfreq) for window in windows],
            axis=0,
        ).astype(np.float32)
        probabilities = model.predict_proba(
            processed,
            mc_dropout_passes=mc_dropout_passes,
        )
        y_pred = np.argmax(probabilities, axis=1).astype(np.int64)
        confidences = np.max(probabilities, axis=1).astype(np.float32)
        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
        confidence_all.append(confidences)
        total_windows += int(windows.shape[0])

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)
    confidences = np.concatenate(confidence_all, axis=0)
    accuracy = float(np.mean(y_pred == y_true)) if y_true.size else 0.0
    return {
        "windows": total_windows,
        "accuracy": accuracy,
        "mean_confidence": float(np.mean(confidences)) if confidences.size else 0.0,
        "y_true": y_true,
        "y_pred": y_pred,
        "confidences": confidences,
    }


def build_acquirer(
    *,
    device_name: str,
    config: dict[str, Any],
) -> Any:
    """Instantiate the selected acquisition backend."""

    register_default_acquirers()
    device_cfg = config.get("device", {})
    device_name = "dummy" if bool(config.get("hardware_dummy_mode", False)) else device_name
    if device_name == "neuracle":
        resolved_channels = int(device_cfg.get("neuracle_eeg_channels", 59))
    elif device_name == "dummy":
        resolved_channels = int(
            device_cfg.get(
                "dummy_eeg_channels",
                device_cfg.get("neuracle_eeg_channels", 59),
            )
        )
    else:
        resolved_channels = default_device_channels(device_name)
    kwargs: dict[str, Any] = {
        "sfreq": float(config["sfreq"]),
        "n_channels": resolved_channels,
        "buffer_sec": float(config["buffer_sec"]),
    }
    if device_name == "neuracle":
        kwargs["neuracle_host"] = str(device_cfg.get("neuracle_host", "127.0.0.1"))
        kwargs["neuracle_port"] = int(device_cfg.get("neuracle_port", 8712))
        kwargs["source_sfreq"] = float(device_cfg.get("neuracle_source_sfreq", 250.0))
        kwargs["transport_delay_sec"] = float(
            device_cfg.get("neuracle_transport_delay_sec", 0.0)
        )
        kwargs["eeg_channel_names"] = device_cfg.get(
            "neuracle_eeg_channel_names",
            NEURACLE_59_EEG_CHANNEL_NAMES,
        )
    if device_name == "brainco":
        kwargs["source_sfreq"] = float(device_cfg.get("brainco_source_sfreq", 250.0))
        kwargs["brainco_addr"] = str(device_cfg.get("brainco_addr", ""))
        kwargs["brainco_port"] = int(device_cfg.get("brainco_port", 0))
        kwargs["auto_discover"] = bool(device_cfg.get("brainco_auto_discover", True))
        kwargs["scan_timeout_sec"] = float(device_cfg.get("brainco_scan_timeout_sec", 6.0))
        kwargs["ready_timeout_sec"] = float(device_cfg.get("brainco_ready_timeout_sec", 20.0))
        kwargs["start_retries"] = int(device_cfg.get("brainco_start_retries", 2))
        kwargs["eeg_gain"] = int(device_cfg.get("brainco_gain", 6))
        kwargs["signal_source"] = str(device_cfg.get("brainco_signal_source", "NORMAL"))
        kwargs["device_id"] = str(device_cfg.get("brainco_device_id", "eeg-cap"))
    if device_name == "dummy":
        kwargs["label_aware"] = bool(device_cfg.get("dummy_label_aware", False))
        kwargs["source_sfreq"] = float(
            device_cfg.get(
                "dummy_source_sfreq",
                device_cfg.get("neuracle_source_sfreq", 250.0),
            )
        )
        kwargs["channel_names"] = device_cfg.get(
            "dummy_eeg_channel_names",
            device_cfg.get(
                "neuracle_eeg_channel_names",
                NEURACLE_59_EEG_CHANNEL_NAMES,
            ),
        )
    return AcquirerFactory.create(device_name, **kwargs)


def build_game_command_outlet(config: dict[str, Any]) -> Any:
    """Build the shared command outlet used to control the AR game."""

    game_output_cfg = config.get("output", {}).get("ar_game", {})
    if not bool(game_output_cfg.get("enabled", False)):
        return None
    ensure_unity_game_running(config, console=CONSOLE)
    outlet = get_shared_game_command_router(config).build_proxy(source="decoder")
    startup_sequence = game_output_cfg.get("startup_sequence", ())
    if not isinstance(startup_sequence, (list, tuple)):
        raise ValueError("output.ar_game.startup_sequence must be a list.")

    driving_scene_requested = False
    if startup_sequence:
        startup_command_delay_sec = max(
            float(game_output_cfg.get("startup_command_delay_sec", 0.75)),
            0.0,
        )
        if startup_command_delay_sec > 0:
            time.sleep(startup_command_delay_sec)
        for step in startup_sequence:
            if isinstance(step, str):
                command = step.strip().upper()
                delay_after_sec = 0.0
            elif isinstance(step, dict):
                command = str(step.get("command", "")).strip().upper()
                delay_after_sec = max(float(step.get("delay_after_sec", 0.0)), 0.0)
            else:
                raise ValueError("Each output.ar_game.startup_sequence item must be a string or mapping.")
            if not command:
                continue
            outlet.push(command)
            driving_scene_requested = driving_scene_requested or command == "OPEN_3D_GAME"
            if delay_after_sec > 0:
                time.sleep(delay_after_sec)
    if driving_scene_requested:
        wait_for_unity_scene_ready(
            outlet,
            timeout_sec=float(
                game_output_cfg.get(
                    "scene_ready_timeout_sec",
                    game_output_cfg.get("startup_timeout_sec", 15.0),
                )
            ),
            console=CONSOLE,
        )
    return outlet


@click.group(invoke_without_command=True)
@click.option(
    "--config",
    "config_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Path to YAML config. Defaults to ./config.yaml and creates it if missing.",
)
@click.pass_context
def cli(ctx: click.Context, config_path: Path | None) -> None:
    """oi-mi command group."""

    setup_logging()
    resolved_config_path = resolve_config_path(config_path)
    config = load_config(resolved_config_path)
    ctx.obj = AppContext(config=config, config_path=resolved_config_path, console=CONSOLE)
    if ctx.invoked_subcommand is None:
        _interactive_menu(ctx, app=ctx.obj)


@cli.command()
@click.pass_obj
def gui(app: AppContext) -> None:
    """Launch the Streamlit graphical user interface."""
    import sys
    import subprocess
    # Always prefer the gui.py next to this cli.py to avoid
    # accidentally importing an unrelated top-level "gui" module.
    gui_script = Path(__file__).with_name("gui.py").resolve()
    if not gui_script.exists():
        app.console.print("[bold red]未找到 gui.py 文件！[/bold red]")
        return
    app.console.print(f"[bold cyan]正在启动 GUI: streamlit run {gui_script}[/bold cyan]")
    environment = os.environ.copy()
    environment["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(gui_script),
            "--browser.gatherUsageStats",
            "false",
            "--",
            "--config",
            str(app.config_path),
        ],
        env=environment,
    )

@cli.command("list-models")
def list_models() -> None:
    """List all registered decoder names."""

    model_factory = get_model_factory()
    table = Table(title="oi-mi Models")
    table.add_column("Model")
    for model_name in model_factory.list_models():
        table.add_row(model_name)
    CONSOLE.print(table)


@cli.command("list-devices")
def list_devices() -> None:
    """List all registered acquisition backends."""

    register_default_acquirers()
    table = Table(title="oi-mi Devices")
    table.add_column("Device")
    for device_name in AcquirerFactory.list_devices():
        table.add_row(device_name)
    CONSOLE.print(table)


@cli.command("probe-device")
@click.option("--device", "device_name", type=str, default=None, help="Acquirer registry name.")
@click.option(
    "--duration",
    type=float,
    default=5.0,
    show_default=True,
    help="How many seconds to wait before grabbing one window.",
)
@click.option(
    "--save-buffer",
    is_flag=True,
    help="Save full forwarded buffer as .npy for diagnostics.",
)
@click.pass_obj
def probe_device(
    app: AppContext,
    device_name: str | None,
    duration: float,
    save_buffer: bool,
) -> None:
    """Quickly verify local device forwarding and data reception."""

    config = app.config
    selected_device = device_name or str(config["device_type"])
    acquirer = build_acquirer(device_name=selected_device, config=config)
    resolved_channels = int(acquirer.metadata.n_channels)

    app.console.print(
        f"[bold cyan]连接设备中...[/bold cyan] device={selected_device} channels={resolved_channels}"
    )
    try:
        acquirer.start_stream()
        time.sleep(max(duration, 0.1))
        window, _ = acquirer.get_chunk(float(config["window_sec"]))
        stats = {
            "shape": tuple(int(d) for d in window.shape),
            "mean_uV": float(np.mean(window)),
            "std_uV": float(np.std(window)),
            "max_abs_uV": float(np.max(np.abs(window))),
        }
        app.console.print(
            "[bold green]设备转发正常[/bold green] "
            f"shape={stats['shape']} mean={stats['mean_uV']:.3f} "
            f"std={stats['std_uV']:.3f} max_abs={stats['max_abs_uV']:.3f}"
        )
        if save_buffer and hasattr(acquirer, "save_full_buffer_npy"):
            stamp = time.strftime("%Y%m%d_%H%M%S")
            out = Path("records_storage") / "device_probe" / f"{selected_device}_{stamp}.npy"
            saved_path = acquirer.save_full_buffer_npy(out)  # type: ignore[attr-defined]
            app.console.print(f"[bold green]已保存完整缓冲区[/bold green] {saved_path}")
    except Exception as exc:
        raise click.ClickException(f"Probe failed for device={selected_device}: {exc}") from exc
    finally:
        acquirer.stop_stream()


@cli.command("collect")
@click.option("--subject", "subject_id", required=True, type=str)
@click.option("--device", "device_name", type=str, default=None, help="Acquirer registry name.")
@click.pass_obj
def collect_data(
    app: AppContext,
    subject_id: str,
    device_name: str | None,
) -> None:
    """Run the formal acquisition-only protocol and save its derived windows."""

    run_config = deepcopy(app.config)
    selected_device = device_name or effective_device_name(run_config)
    run_config["subject_id"] = subject_id
    run_config["device_type"] = selected_device
    run_config["hardware_dummy_mode"] = selected_device == "dummy"
    acquirer = build_acquirer(device_name=selected_device, config=run_config)
    collector = get_calibrator_class()(
        acquirer=acquirer,
        console=app.console,
        sfreq=float(run_config["sfreq"]),
        window_sec=float(run_config["window_sec"]),
        step_sec=float(run_config["step_sec"]),
        session_records_dir=resolve_records_dir(run_config)
        / subject_id
        / "collection",
        protocol_config=ProtocolConfig.from_config(run_config),
        experiment_config=run_config,
    )
    result = collector.collect()
    app.console.print(
        "[bold green]正式数据采集完成[/bold green] "
        f"trials={result.trials_collected} windows={result.windows_collected} "
        f"session={result.session_dir}"
    )
    app.console.print(
        "[cyan]本命令未加载、训练、推理或更新任何模型。[/cyan]"
    )


@cli.command(hidden=True)
@click.option("--subject", "subject_id", required=True, type=str)
@click.option("--model", "model_name", type=str, default=None, help="Model registry name.")
@click.pass_obj
def calibrate(
    app: AppContext,
    subject_id: str,
    model_name: str | None,
) -> None:
    """Legacy combined collection/training workflow; prefer collect then train-from-records."""

    app.console.print(
        "[bold yellow]注意：calibrate 是兼容旧流程的采集并训练命令。"
        "正式数据采集请使用 collect；它不会调用任何模型代码。[/bold yellow]"
    )

    config = app.config
    selected_model = model_name or str(config["model_name"])
    epochs = int(config.get("training_epochs", config.get("calibration_epochs", 50)))
    acquirer = build_acquirer(
        device_name=str(config["device_type"]),
        config=config,
    )
    effective_n_channels = int(acquirer.metadata.n_channels)
    model_path = build_model_path(
        config,
        subject_id,
        selected_model,
        device_name=str(config["device_type"]),
    )
    experiment_seed = int(
        config.get("online_adaptation", {}).get("neuroonline", {}).get(
            "offline_random_seed",
            config.get("online_adaptation", {}).get("neuroonline", {}).get(
                "random_seed",
                config.get("online_adaptation", {}).get("random_seed", 42),
            ),
        )
    )
    seed_experiment(experiment_seed)
    model_factory = get_model_factory()
    model = model_factory.get(
        selected_model,
        n_chans=effective_n_channels,
        sfreq=float(config["sfreq"]),
        n_classes=int(config["n_classes"]),
        n_times=int(float(config["sfreq"]) * float(config["window_sec"])),
    )
    calibrator_class = get_calibrator_class()
    calibrator = calibrator_class(
        acquirer=acquirer,
        model=model,
        console=app.console,
        sfreq=float(config["sfreq"]),
        window_sec=float(config["window_sec"]),
        step_sec=float(config["step_sec"]),
        model_path=model_path,
        session_records_dir=Path(str(config.get("storage", {}).get("records_dir", "records_storage")))
        / subject_id
        / "calibration",
        protocol_config=ProtocolConfig.from_config(config),
        online_adaptation_config=config.get("online_adaptation", {}),
        experiment_config=config,
    )
    result = calibrator.calibrate(
        duration_sec=None,
        epochs=epochs,
        batch_size=int(config["batch_size"]),
        learning_rate=float(config["learning_rate"]),
        head_only=False,
    )
    app.console.print(
        f"[bold green]数据采集与训练完成[/bold green] "
        f"windows={result.windows_collected} "
        f"val_acc={result.metrics.get('val_acc', 0.0):.3f} "
        f"saved={result.model_path}"
    )
    if result.calibration_data_path is not None:
        app.console.print(f"[bold green]采集数据已保存[/bold green] {result.calibration_data_path}")


@cli.command()
@click.option("--subject", "subject_id", required=True, type=str)
@click.option("--model", "model_name", type=str, default=None, help="Model registry name.")
@click.option("--device", "device_name", type=str, default=None, help="Acquirer registry name.")
@click.option("--test-mode", is_flag=True, help="Enable cue-based test mode and save EEG/labels.")
@click.option(
    "--test-duration",
    type=int,
    default=600,
    show_default=True,
    help="Test mode duration in seconds.",
)
@click.option("--record", is_flag=True, help="Record realtime decoding data.")
@click.option(
    "--label-source",
    type=click.Choice(["auto", "none", "cued", "manual-http"]),
    default="auto",
    show_default=True,
    help="Realtime true-label source for online updates during normal run.",
)
@click.option(
    "--label-host",
    type=str,
    default="127.0.0.1",
    show_default=True,
    help="Host for the manual realtime-label HTTP server.",
)
@click.option(
    "--label-port",
    type=int,
    default=8776,
    show_default=True,
    help="Port for the manual realtime-label HTTP server.",
)
@click.option(
    "--label-ttl-sec",
    type=float,
    default=2.0,
    show_default=True,
    help="How long each manual label remains active for window alignment.",
)
@click.option(
    "--max-scenes",
    type=click.IntRange(min=1),
    default=None,
    help="Stop cleanly after N completed cued Unity scenes (diagnostic runs).",
)
@click.pass_obj
def run(
    app: AppContext,
    subject_id: str,
    model_name: str | None,
    device_name: str | None,
    test_mode: bool,
    test_duration: int,
    record: bool = False,
    label_source: str = "none",
    label_host: str = "127.0.0.1",
    label_port: int = 8776,
    label_ttl_sec: float = 2.0,
    max_scenes: int | None = None,
) -> None:
    """Run the realtime decoder."""

    config = app.config
    adaptation_cfg = config.get("online_adaptation", {}) or {}
    selected_device = device_name or str(config["device_type"])
    # Keep the web-control endpoint and decoder in the same process so both
    # share one Unity TCP router/connection.
    start_web_command_server(config)
    selected_model = model_name or str(config["model_name"])
    acquirer = build_acquirer(device_name=selected_device, config=config)
    effective_n_channels = int(acquirer.metadata.n_channels)
    n_times = int(float(config["sfreq"]) * float(config["window_sec"]))
    model_path = resolve_model_path(
        config,
        subject_id,
        selected_model,
        device_name=selected_device,
        n_chans=effective_n_channels,
        n_times=n_times,
    )
    if not model_path.exists():
        raise click.ClickException(
            f"Model not found: {build_model_path(config, subject_id, selected_model, device_name=selected_device)}. "
            "Run calibrate first or generate bundled dummy weights via `oi-mi seed-dummy-decoders`."
        )
    model_factory = get_model_factory()
    model = model_factory.get(
        selected_model,
        n_chans=effective_n_channels,
        sfreq=float(config["sfreq"]),
        n_classes=int(config["n_classes"]),
        n_times=n_times,
    )
    if model_path.parent == DUMMY_DECODER_ASSET_DIR:
        app.console.print(f"[bold yellow]使用内置 dummy 测试权重[/bold yellow] {model_path}")
    model.load(model_path)
    online_label_source = None
    online_label_server = None
    adaptation_cfg = config.get("online_adaptation", {})
    simulation_cfg = adaptation_cfg.get("simulation", {})
    cued_cfg = adaptation_cfg.get("cued_labels", {})
    simulation_enabled = (
        bool(adaptation_cfg.get("enabled", False))
        and bool(simulation_cfg.get("enabled", False))
        and effective_device_name(config, selected_device) == "dummy"
        and not test_mode
    )
    if simulation_enabled:
        online_label_source = SimulatedOnlineLabelSource(
            acquirer,
            trial_sec=float(simulation_cfg.get("trial_sec", 6.0)),
            settle_sec=float(simulation_cfg.get("settle_sec", config["window_sec"])),
            seed=int(adaptation_cfg.get("random_seed", 17)),
        )
        app.console.print("[bold cyan]已启动标签驱动 Dummy 模拟被试[/bold cyan]")
    elif not test_mode and (
        label_source == "cued"
        or (
            label_source == "auto"
            and bool(adaptation_cfg.get("enabled", False))
            and bool(cued_cfg.get("enabled", True))
        )
    ):
        online_label_source = build_cued_online_label_source(config)
        app.console.print("[bold cyan]已启动自动 cue 在线实验协议[/bold cyan]")
    elif label_source == "manual-http" and not test_mode:
        online_label_source = ManualOnlineLabelSource(default_ttl_sec=label_ttl_sec)
        online_label_server = ManualLabelHttpServer(
            online_label_source,
            host=label_host,
            port=label_port,
        )
        online_label_server.start()
        app.console.print(
            f"[bold cyan]实时标签服务器已启动[/bold cyan] "
            f"http://{label_host}:{label_port}/api/label"
        )

    command_outlet = LSLCommandOutlet(
        stream_name=str(config["output"]["command_stream_name"]),
        stream_type=str(config["output"]["command_stream_type"]),
    )
    game_command_outlet = build_game_command_outlet(config)
    realtime_decoder_class = get_realtime_decoder_class()
    decoder = realtime_decoder_class(
        acquirer=acquirer,
        model=model,
        console=app.console,
        command_outlet=command_outlet,
        game_command_outlet=game_command_outlet,
        sfreq=float(config["sfreq"]),
        window_sec=float(config["window_sec"]),
        step_sec=float(config["step_sec"]),
        confidence_threshold=float(config["confidence_threshold"]),
        mc_dropout_passes=int(config["mc_dropout_passes"]),
        model_save_path=build_model_path(
            config,
            subject_id,
            selected_model,
            device_name=selected_device,
        ),
        online_label_source=online_label_source,
        batch_update_config=adaptation_cfg if not test_mode else None,
        n_classes=int(config["n_classes"]),
        experiment_config=config,
        model_name=selected_model,
        model_source_path=model_path,
    )
    if test_mode:
        records_dir = Path(str(config.get("storage", {}).get("records_dir", "records_storage")))
        result = decoder.run_test_mode(
            subject_id=subject_id,
            marker_backend=NoOpMarkerBackend(),
            duration_sec=test_duration,
            block_sec=float(config.get("test_mode_block_sec", config.get("collect_block_sec", 10))),
            save_dir=records_dir / subject_id / "test_mode",
        )
        app.console.print(
            f"[bold green]测试完成[/bold green] windows={result['windows']} "
            f"accuracy={result['accuracy']:.3f} valid_accuracy={result['valid_accuracy']:.3f}"
        )
        return

    if (
        bool(adaptation_cfg.get("enabled", False))
        and str(adaptation_cfg.get("strategy", "")).strip().lower() == "neuroonline"
        and not record
    ):
        record = True
        app.console.print(
            "[bold yellow]NeuroOnline正式运行已自动开启论文级数据记录[/bold yellow]"
        )
    app.console.print("[bold cyan]开始实时解码，按 Ctrl+C 停止[/bold cyan]")
    records_dir = Path(str(config.get("storage", {}).get("records_dir", "records_storage")))
    try:
        decoder.run_forever(
            subject_id=subject_id,
            record=record,
            save_dir=records_dir / subject_id / "realtime",
            max_scenes=max_scenes,
        )
    finally:
        if online_label_server is not None:
            online_label_server.close()


@cli.command("train-from-records")
@click.option("--subject", "subject_id", required=True, type=str)
@click.option("--model", "model_name", type=str, default=None, help="Model registry name.")
@click.option("--device", "device_name", type=str, default=None, help="Acquirer registry name.")
@click.option(
    "--session",
    "session_ids",
    multiple=True,
    type=str,
    help="Collection session ID to include. Defaults to all collection sessions.",
)
@click.option(
    "--use-processed/--use-raw",
    default=True,
    show_default=True,
    help="Train on processed_windows or raw_windows from saved collection records.",
)
@click.option(
    "--head-only",
    is_flag=True,
    help="Load existing weights first and only fine-tune the classifier head.",
)
@click.pass_obj
def train_from_records(
    app: AppContext,
    subject_id: str,
    model_name: str | None,
    device_name: str | None,
    session_ids: tuple[str, ...],
    use_processed: bool,
    head_only: bool,
) -> None:
    """Train a subject model after acquisition, from saved collection sessions."""

    config = app.config
    selected_model = model_name or str(config["model_name"])
    selected_device = device_name or str(config["device_type"])
    records_dir = resolve_records_dir(config)
    X, y, trial_groups, used_sessions = load_collection_windows(
        records_dir,
        subject_id,
        session_ids=session_ids,
        use_processed=use_processed,
        include_groups=True,
    )
    app.console.print(
        f"[bold cyan]加载采集后窗口[/bold cyan] sessions={len(used_sessions)} "
        f"windows={int(X.shape[0])} shape={tuple(int(dim) for dim in X.shape[1:])}"
    )
    if trial_groups is None:
        app.console.print(
            "[bold yellow]这些旧记录没有 trial_ids；无法执行按 trial 分组验证。"
            "建议使用当前版本重新采集。[/bold yellow]"
        )

    model_path = build_model_path(
        config,
        subject_id,
        selected_model,
        device_name=selected_device,
    )
    model = get_model_factory().get(
        selected_model,
        n_chans=int(X.shape[1]),
        sfreq=float(config["sfreq"]),
        n_classes=int(config["n_classes"]),
        n_times=int(X.shape[2]),
    )
    load_path: Path | None = None
    if head_only:
        load_path = resolve_model_path(
            config,
            subject_id,
            selected_model,
            device_name=selected_device,
            n_chans=int(X.shape[1]),
            n_times=int(X.shape[2]),
        )
        if not load_path.exists():
            raise click.ClickException(f"Model not found for head-only adaptation: {load_path}")

    from adaptation.neuroonline import NeuroOnlineConfig, NeuroOnlineModelAdapter
    from models.factory import TorchModelAdapter

    neuroonline_config = NeuroOnlineConfig.from_mapping(
        config.get("online_adaptation", {}),
        window_duration_sec=float(config["window_sec"]),
    )
    if neuroonline_config.enabled:
        if not isinstance(model, TorchModelAdapter):
            raise click.ClickException("NeuroOnline record recovery requires a PyTorch decoder model.")
        model = NeuroOnlineModelAdapter(
            model,
            config=neuroonline_config,
            state_path=load_path,
        )
        app.console.print(
            "[bold cyan]NeuroOnline 记录恢复[/bold cyan] "
            f"offline_epochs={neuroonline_config.offline_epochs}，将同时保存主模型与 CRM"
        )

    if load_path is not None:
        model.load(load_path)

    def report_training_progress(
        current_epoch: int,
        total_epochs: int,
        epoch_metrics: dict[str, float],
    ) -> None:
        summary = " ".join(
            f"{name}={float(value):.4f}"
            for name, value in epoch_metrics.items()
            if name in {"loss", "train_loss", "val_loss", "val_acc", "val_kappa"}
        )
        app.console.print(
            f"[cyan]训练 epoch {current_epoch}/{total_epochs}[/cyan]"
            + (f" {summary}" if summary else "")
        )

    metrics = model.fit(
        X,
        y,
        epochs=int(config.get("training_epochs", config.get("calibration_epochs", 50))),
        batch_size=int(config["batch_size"]),
        learning_rate=float(config["learning_rate"]),
        head_only=head_only,
        groups=trial_groups,
        progress_callback=report_training_progress,
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    metrics_path = model_path.with_suffix(".metrics.yaml")
    temporary_metrics_path = metrics_path.with_suffix(metrics_path.suffix + ".tmp")
    metrics_payload = {
        "model_path": str(model_path),
        "subject_id": subject_id,
        "device_name": selected_device,
        "model_name": selected_model,
        "windows_collected": int(X.shape[0]),
        "sessions": [path.name for path in used_sessions],
        "training_source": "saved_collection_records",
        "metrics": {key: float(value) for key, value in metrics.items()},
    }
    with temporary_metrics_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(metrics_payload, handle, allow_unicode=True, sort_keys=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_metrics_path, metrics_path)

    label_ids, label_counts = np.unique(y, return_counts=True)
    distribution = ", ".join(f"{int(label)}:{int(count)}" for label, count in zip(label_ids, label_counts, strict=False))
    app.console.print(
        f"[bold green]训练完成[/bold green] val_acc={metrics.get('val_acc', 0.0):.3f} "
        f"saved={model_path} labels=[{distribution}]"
    )
    if neuroonline_config.enabled:
        app.console.print(f"[bold green]CRM 已保存[/bold green] {model_path}.neuroonline.pt")
    app.console.print(f"[bold green]训练指标已保存[/bold green] {metrics_path}")


@cli.command("replay-test-mode")
@click.option("--subject", "subject_id", required=True, type=str)
@click.option("--model", "model_name", type=str, default=None, help="Model registry name.")
@click.option("--device", "device_name", type=str, default=None, help="Acquirer registry name.")
@click.option(
    "--test-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Override the default records_storage/<subject>/test_mode directory.",
)
@click.pass_obj
def replay_test_mode_command(
    app: AppContext,
    subject_id: str,
    model_name: str | None,
    device_name: str | None,
    test_dir: Path | None,
) -> None:
    """Replay saved test-mode windows through the current decoder."""

    config = app.config
    selected_model = model_name or str(config["model_name"])
    selected_device = device_name or str(config["device_type"])
    resolved_test_dir = test_dir or (resolve_records_dir(config) / subject_id / "test_mode")
    chunk_paths = iter_test_mode_chunks(resolved_test_dir)

    first_chunk = chunk_paths[0]
    with np.load(first_chunk) as payload:
        if "eeg_windows" not in payload:
            raise click.ClickException(f"Invalid test-mode chunk format: {first_chunk}")
        sample_windows = payload["eeg_windows"]
        n_chans = int(sample_windows.shape[1])
        n_times = int(sample_windows.shape[2])

    model_path = resolve_model_path(
        config,
        subject_id,
        selected_model,
        device_name=selected_device,
        n_chans=n_chans,
        n_times=n_times,
    )
    if not model_path.exists():
        raise click.ClickException(
            f"Model not found: {build_model_path(config, subject_id, selected_model, device_name=selected_device)}. "
            "Train, calibrate, or run `oi-mi seed-dummy-decoders` for dummy mode."
        )

    model = get_model_factory().get(
        selected_model,
        n_chans=n_chans,
        sfreq=float(config["sfreq"]),
        n_classes=int(config["n_classes"]),
        n_times=n_times,
    )
    model.load(model_path)
    result = replay_test_mode(
        model=model,
        test_mode_dir=resolved_test_dir,
        sfreq=float(config["sfreq"]),
        mc_dropout_passes=int(config["mc_dropout_passes"]),
    )

    y_true = result["y_true"]
    y_pred = result["y_pred"]
    class_ids = sorted(set(np.unique(y_true).tolist()) | set(np.unique(y_pred).tolist()))
    class_summary = []
    for class_id in class_ids:
        mask = y_true == class_id
        class_acc = float(np.mean(y_pred[mask] == y_true[mask])) if np.any(mask) else 0.0
        class_summary.append(f"{int(class_id)}:{class_acc:.3f}")
    app.console.print(
        f"[bold green]回放完成[/bold green] windows={result['windows']} "
        f"accuracy={result['accuracy']:.3f} mean_confidence={result['mean_confidence']:.3f} "
        f"class_acc=[{', '.join(class_summary)}]"
    )


@cli.command("seed-dummy-decoders")
@click.option("--output-dir", type=click.Path(path_type=Path), default=None, help="Asset output directory.")
@click.option("--models", multiple=True, type=str, help="Model registry names to export.")
@click.option("--n-chans", type=int, default=59, show_default=True)
@click.option("--sfreq", type=float, default=200.0, show_default=True)
@click.option("--window-sec", type=float, default=4.0, show_default=True)
@click.pass_obj
def seed_dummy_decoders_cmd(
    app: AppContext,
    output_dir: Path | None,
    models: tuple[str, ...],
    n_chans: int,
    sfreq: float,
    window_sec: float,
) -> None:
    """Train bundled decoder weights for hardware-free dummy EEG testing."""

    del app
    from tools.seed_dummy_decoders import DEFAULT_PROFILE, seed_profile, write_manifest

    target_dir = output_dir or DUMMY_DECODER_ASSET_DIR
    model_names = list(models) if models else ["cbramod"]
    profile = dict(DEFAULT_PROFILE)
    profile.update(
        {
            "n_chans": n_chans,
            "sfreq": sfreq,
            "window_sec": window_sec,
        }
    )
    n_times = int(round(sfreq * window_sec))
    saved = seed_profile(output_dir=target_dir, model_names=model_names, profile=profile)
    manifest_path = write_manifest(
        output_dir=target_dir,
        profiles=[
            {
                "n_chans": n_chans,
                "sfreq": sfreq,
                "window_sec": window_sec,
                "n_times": n_times,
                "n_classes": int(profile["n_classes"]),
                "models": saved,
            }
        ],
    )
    CONSOLE.print(f"[bold green]Dummy 测试解码器已生成[/bold green] {target_dir}")
    CONSOLE.print(f"[bold cyan]Manifest[/bold cyan] {manifest_path}")
    for item in saved:
        CONSOLE.print(f"- {item['model_name']}: {item['asset_path']} metrics={item['metrics']}")


if __name__ == "__main__":
    cli()
