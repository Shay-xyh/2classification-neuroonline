"""Protocol-driven motor-imagery collection and optional legacy training."""

from __future__ import annotations

from collections.abc import Callable
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import secrets
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml
from rich.console import Console

from acquisition.base import AbstractAcquirer
from adaptation.mi_protocol import (
    LABEL_DISPLAY,
    LABEL_SYMBOL,
    LABEL_TO_ID,
    RECOMMENDED_INSTRUCTIONS,
    ProtocolConfig,
    SessionPlan,
    build_session_plan,
)
from adaptation.session_recorder import SessionRecorder
from utils.markers import PROTOCOL_EVENT_CODES
from utils.preprocessing import (
    continuous_preprocessing_metadata,
    finalize_preprocessed_window,
    preprocess_eeg_continuous,
)

if TYPE_CHECKING:
    from models.factory import BaseModelAdapter

LABEL_SEQUENCE: list[tuple[int, str]] = list(
    (label_id, label) for label, label_id in LABEL_TO_ID.items()
)
REST_INCREMENTAL_SAVE_INTERVAL_SEC = 10.0


def _offline_parameter_snapshot(config: Any) -> dict[str, Any]:
    """Return the effective offline settings for session provenance/UI output."""

    return {
        "offline_learning_rate": config.offline_learning_rate,
        "offline_backbone_learning_rate": config.offline_backbone_learning_rate,
        "offline_batch_seconds": config.time_budget["offline_batch"][
            "requested_seconds"
        ],
        "offline_batch_size": config.offline_batch_size,
        "mask_ratio": (
            config.mask_ratio
            if config.offline_mask_ratio is None
            else config.offline_mask_ratio
        ),
        "consistency_weight": (
            config.consistency_weight
            if config.offline_consistency_weight is None
            else config.offline_consistency_weight
        ),
        "weight_decay": config.weight_decay,
        "label_smoothing": config.label_smoothing,
        "offline_epochs": config.offline_epochs,
        "offline_update_policy": config.offline_update_policy,
    }


@dataclass(slots=True)
class CollectionResult:
    """Artifacts produced by collection, without any model-training state."""

    trials_collected: int
    windows_collected: int
    continuous_eeg_path: Path | None = None
    events_path: Path | None = None
    windows_path: Path | None = None
    session_dir: Path | None = None


class TrialDiscarded(RuntimeError):
    """Signal that a pause request invalidated the current trial attempt."""


class CollectionPauseControl:
    """Thread-safe subject pause control with no stop or training behavior."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._pause_requested = False
        self._paused = False
        self._automatic_break = False

    def request_pause(self) -> None:
        with self._lock:
            if not self._automatic_break:
                self._pause_requested = True

    def begin_pause(self) -> None:
        with self._lock:
            self._pause_requested = False
            self._paused = True

    def resume(self) -> None:
        with self._lock:
            self._pause_requested = False
            self._paused = False

    def set_automatic_break(self, active: bool) -> None:
        with self._lock:
            self._automatic_break = bool(active)
            if active:
                self._pause_requested = False

    @property
    def pause_requested(self) -> bool:
        with self._lock:
            return self._pause_requested

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    @property
    def automatic_break(self) -> bool:
        with self._lock:
            return self._automatic_break


@dataclass(slots=True)
class CalibrationResult:
    """Result metadata for an explicit offline training run."""

    model_path: Path
    metrics: dict[str, float]
    windows_collected: int
    calibration_data_path: Path | None = None
    session_dir: Path | None = None
    selected_hyperparameters: dict[str, Any] | None = None


class Calibrator:
    """Collect continuous MI protocol data and train or adapt a decoder."""

    def __init__(
        self,
        acquirer: AbstractAcquirer,
        console: Console,
        *,
        model: BaseModelAdapter | None = None,
        sfreq: float,
        window_sec: float,
        step_sec: float,
        model_path: Path | None = None,
        session_records_dir: Path | None = None,
        session_id: str | None = None,
        protocol_config: ProtocolConfig | None = None,
        online_adaptation_config: dict | None = None,
        experiment_config: dict[str, Any] | None = None,
    ) -> None:
        self._acquirer = acquirer
        if model is None:
            self._model = None
            self._neuroonline_config = None
        else:
            # Model and NeuroOnline modules are intentionally imported only for
            # the explicit legacy training path. Acquisition-only collection
            # must not import or initialize model code.
            from adaptation.neuroonline import NeuroOnlineConfig, NeuroOnlineModelAdapter
            from models.factory import TorchModelAdapter

            self._neuroonline_config = NeuroOnlineConfig.from_mapping(
                online_adaptation_config,
                window_duration_sec=window_sec,
            )
        if model is not None and self._neuroonline_config.enabled:
            if not isinstance(model, TorchModelAdapter):
                raise ValueError("NeuroOnline calibration requires a PyTorch decoder model.")
            self._model = NeuroOnlineModelAdapter(
                model,
                config=self._neuroonline_config,
                state_path=None,
            )
        else:
            self._model = model
        self._console = console
        self._sfreq = float(sfreq)
        self._source_sfreq = float(getattr(acquirer, "source_sfreq", self._sfreq))
        self._window_sec = float(window_sec)
        self._step_sec = float(step_sec)
        self._model_path = model_path
        self._session_records_dir = session_records_dir
        if session_id is not None and (
            not session_id or Path(session_id).name != session_id
        ):
            raise ValueError("session_id must be one non-empty path component")
        self._session_id = session_id
        source_experiment_config = experiment_config or {}
        self._experiment_config = (
            self._collection_config_snapshot(source_experiment_config)
            if model is None
            else copy.deepcopy(source_experiment_config)
        )
        self._protocol = protocol_config or ProtocolConfig.from_config(
            {
                "window_sec": float(window_sec),
                "step_sec": float(step_sec),
            }
        )

    def calibrate(
        self,
        *,
        duration_sec: int | None,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        head_only: bool,
        heartbeat: Callable[[], None] | None = None,
    ) -> CalibrationResult:
        del duration_sec
        if head_only:
            raise ValueError(
                "Head-only calibration was removed; each experiment must train "
                "a fresh full decoder."
            )
        if self._model is None:
            raise RuntimeError(
                "Calibration training requires a decoder model. Use collect() "
                "for acquisition-only sessions."
            )
        if self._model_path is None:
            raise RuntimeError("Calibration training requires a model output path.")
        if self._neuroonline_config is None:
            raise RuntimeError("Calibration training configuration was not initialized.")
        selected_hyperparameters: dict[str, Any] | None = (
            _offline_parameter_snapshot(self._neuroonline_config)
            if self._neuroonline_config.enabled
            else None
        )

        plan = build_session_plan(self._protocol)
        (
            session_dir,
            raw_windows,
            processed_windows,
            labels,
            trial_groups,
            session_metadata,
        ) = self._collect_session_data(
            plan=plan,
            heartbeat=heartbeat,
            window_filename="training_windows_main.npz",
        )
        if self._neuroonline_config.enabled:
            session_metadata["hyperparameters"] = {
                "mode": "configured_fixed",
                "source": "config.yaml",
                "parameters": selected_hyperparameters,
            }
        self._console.print("[bold cyan]采集完成，正在保存和训练，请等待工作人员[/bold cyan]")
        if self._neuroonline_config.enabled:
            self._console.print(
                "[bold yellow]正在执行 NeuroOnline 离线训练 "
                f"(固定 {self._neuroonline_config.offline_epochs} epochs，"
                "结束后恢复验证表现最佳的 epoch)。"
                "在出现“采集完成”和保存路径前，请勿返回、刷新或关闭页面。[/bold yellow]"
            )
        if heartbeat is not None:
            heartbeat()
        training_progress = getattr(self._console, "set_stage_progress", None)
        if callable(training_progress):
            training_progress(
                stage_name="模型训练",
                elapsed_sec=0.0,
                duration_sec=float(
                    self._neuroonline_config.offline_epochs
                    if self._neuroonline_config.enabled
                    else epochs
                ),
            )

        def report_training_progress(
            current_epoch: int,
            total_epochs: int,
            epoch_metrics: dict[str, float],
        ) -> None:
            del epoch_metrics
            if callable(training_progress):
                training_progress(
                    stage_name=f"模型训练 epoch {current_epoch}/{total_epochs}",
                    elapsed_sec=float(current_epoch),
                    duration_sec=float(total_epochs),
                )
            if heartbeat is not None:
                heartbeat()

        metrics = self._model.fit(
            processed_windows,
            labels,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            head_only=False,
            groups=trial_groups,
            progress_callback=report_training_progress,
        )
        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(self._model_path)
        self._save_metadata(metrics=metrics, windows_collected=int(processed_windows.shape[0]), head_only=False)
        self._write_session_summary(session_dir, metrics=metrics, windows_collected=int(processed_windows.shape[0]), session_metadata=session_metadata)
        SessionRecorder.prepare_final_bundle(session_dir)
        self._seal_session_bundle(session_dir)
        SessionRecorder.finalize_session(session_dir)
        self._console.print("[bold green]采集完成，请等待工作人员[/bold green]")
        if heartbeat is not None:
            heartbeat()
        return CalibrationResult(
            model_path=self._model_path,
            metrics=metrics,
            windows_collected=int(processed_windows.shape[0]),
            calibration_data_path=(session_dir / "training_windows_main.npz") if session_dir is not None else None,
            session_dir=session_dir,
            selected_hyperparameters=selected_hyperparameters,
        )

    @staticmethod
    def _collection_config_snapshot(config: dict[str, Any]) -> dict[str, Any]:
        """Retain only acquisition-relevant settings in collection metadata."""

        snapshot = {
            key: copy.deepcopy(config[key])
            for key in (
                "subject_id",
                "device_type",
                "hardware_dummy_mode",
                "sfreq",
                "task_paradigm",
                "n_classes",
                "window_sec",
                "buffer_sec",
                "protocol",
                "device",
            )
            if key in config
        }
        storage = config.get("storage", {}) or {}
        if "records_dir" in storage:
            snapshot["storage"] = {
                "records_dir": copy.deepcopy(storage["records_dir"]),
            }
        return snapshot

    def collect(
        self,
        *,
        heartbeat: Callable[[], None] | None = None,
        pause_control: CollectionPauseControl | None = None,
    ) -> CollectionResult:
        """Run acquisition and persist data without invoking model code."""

        plan = build_session_plan(self._protocol)
        (
            session_dir,
            _raw_windows,
            processed_windows,
            _labels,
            _trial_groups,
            session_metadata,
        ) = self._collect_session_data(
            plan=plan,
            heartbeat=heartbeat,
            window_filename="mi_windows.npz",
            pause_control=pause_control,
        )
        if session_dir is None:
            raise RuntimeError("Collection requires a session records directory.")
        trials_collected = int(session_metadata["formal_trial_count"])
        windows_collected = int(processed_windows.shape[0])
        self._console.print("[bold green]采集完成，数据已保存[/bold green]")
        self._write_collection_summary(
            session_dir,
            trials_collected=trials_collected,
            windows_collected=windows_collected,
            session_metadata=session_metadata,
        )
        SessionRecorder.prepare_final_bundle(session_dir)
        self._seal_session_bundle(session_dir, include_model_files=False)
        SessionRecorder.finalize_session(session_dir)
        if heartbeat is not None:
            heartbeat()
        return CollectionResult(
            trials_collected=trials_collected,
            windows_collected=windows_collected,
            continuous_eeg_path=session_dir / "continuous_eeg.npy",
            events_path=session_dir / "events.json",
            windows_path=session_dir / "mi_windows.npz",
            session_dir=session_dir,
        )

    def _collect_session_data(
        self,
        *,
        plan: SessionPlan,
        heartbeat: Callable[[], None] | None = None,
        window_filename: str | None,
        pause_control: CollectionPauseControl | None = None,
    ) -> tuple[
        Path | None,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[str, Any],
    ]:
        self._console.print("[bold cyan]开始左右手二分类运动想象采集[/bold cyan]")
        self._print_instructions(plan)
        session_id = self._session_id or f"session_{secrets.token_hex(6)}"
        session_dir = self._session_records_dir / session_id if self._session_records_dir is not None else None
        self._acquirer.start_stream()
        try:
            recorder = SessionRecorder(
                self._acquirer,
                sfreq=self._source_sfreq,
                n_channels=self._acquirer.metadata.n_channels,
                output_dir=session_dir,
                session_id=session_id,
                total_trials=plan.total_formal_trials,
            )
        except BaseException:
            self._acquirer.stop_stream()
            raise
        trials: list[dict[str, Any]] = []
        try:
            self._wait_for_first_samples(recorder, heartbeat=heartbeat)
            self._emit_event(recorder, "session_start", phase="session", subject_mode=plan.subject_mode)
            self._run_formal_blocks(
                plan,
                recorder=recorder,
                heartbeat=heartbeat,
                trials=trials,
                pause_control=pause_control,
            )
            self._emit_event(recorder, "session_end", phase="session")
            self._flush_recorder(recorder)
            self._acquirer.stop_stream()
            if heartbeat is not None:
                heartbeat()
        except BaseException as exc:
            try:
                self._flush_recorder(recorder)
            except BaseException:
                pass
            try:
                self._acquirer.stop_stream()
            except BaseException:
                pass
            if heartbeat is not None:
                heartbeat()
            try:
                recorder.abort(error=f"{type(exc).__name__}: {exc}")
            except BaseException:
                # Preserve the acquisition exception; the writer error will also
                # be visible in checkpoint state when persistence itself failed.
                pass
            raise

        try:
            session_metadata = self._build_session_metadata(
                plan,
                session_id=session_id,
                trials=trials,
            )
            if session_dir is not None:
                recorder.export(session_dir, metadata=session_metadata)
            if window_filename is None:
                empty_windows = np.empty(
                    (0, self._acquirer.metadata.n_channels, 0),
                    dtype=np.float32,
                )
                recorder.mark_processing_complete()
                return (
                    session_dir,
                    empty_windows,
                    empty_windows.copy(),
                    np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=np.int64),
                    session_metadata,
                )
            eeg = self._get_continuous_eeg(session_dir=session_dir, recorder=recorder)
            raw_windows, processed_windows, labels, trial_groups = self._build_mi_windows(
                eeg=eeg,
                events=recorder.events,
                trials=trials,
                session_dir=session_dir,
                window_filename=window_filename,
            )
            if raw_windows.shape[0] == 0:
                raise RuntimeError("Collection did not yield any valid MI windows.")
            recorder.mark_processing_complete()
            return (
                session_dir,
                raw_windows,
                processed_windows,
                labels,
                trial_groups,
                session_metadata,
            )
        except BaseException as exc:
            try:
                recorder.mark_processing_failed(error=f"{type(exc).__name__}: {exc}")
            except BaseException:
                pass
            raise

    def _run_formal_blocks(
        self,
        plan: SessionPlan,
        *,
        recorder: SessionRecorder,
        heartbeat: Callable[[], None] | None,
        trials: list[dict[str, Any]],
        pause_control: CollectionPauseControl | None,
    ) -> None:
        total_blocks = len(plan.blocks)
        self._update_trial_progress(
            completed_trials=len(trials),
            total_trials=plan.total_formal_trials,
        )
        for block_index, sequence in enumerate(plan.blocks):
            self._console.print(f"[bold cyan]Block {block_index + 1}/{total_blocks}[/bold cyan] 共 {len(sequence)} 个 trial")
            self._emit_event(recorder, "block_start", phase="formal", block_index=block_index)
            for trial_index, label in enumerate(sequence):
                attempt_index = 0
                while True:
                    if pause_control is not None and pause_control.pause_requested:
                        self._wait_for_manual_resume(
                            pause_control,
                            recorder=recorder,
                            heartbeat=heartbeat,
                            block_index=block_index,
                            trial_index=trial_index,
                        )
                    try:
                        trial_info = self._run_trial(
                            label=label,
                            recorder=recorder,
                            heartbeat=heartbeat,
                            trial_index=trial_index,
                            block_index=block_index,
                            attempt_index=attempt_index,
                            pause_control=pause_control,
                        )
                    except TrialDiscarded:
                        self._emit_event(
                            recorder,
                            "trial_discarded",
                            phase="formal",
                            block_index=block_index,
                            trial_index=trial_index,
                            attempt_index=attempt_index,
                            target_hand=label,
                        )
                        self._wait_for_manual_resume(
                            pause_control,
                            recorder=recorder,
                            heartbeat=heartbeat,
                            block_index=block_index,
                            trial_index=trial_index,
                        )
                        attempt_index += 1
                        continue
                    trials.append(trial_info)
                    self._persist_recorder(
                        recorder,
                        completed_trials=len(trials),
                        total_trials=plan.total_formal_trials,
                        last_completed_block=block_index + 1,
                        last_completed_trial_in_block=trial_index + 1,
                    )
                    self._update_trial_progress(
                        completed_trials=len(trials),
                        total_trials=plan.total_formal_trials,
                    )
                    break
            self._emit_event(recorder, "block_end", phase="formal", block_index=block_index)
            if block_index < total_blocks - 1:
                if pause_control is not None:
                    pause_control.set_automatic_break(True)
                self._emit_event(
                    recorder,
                    "automatic_break_start",
                    phase="automatic_break",
                    block_index=block_index,
                    next_block_index=block_index + 1,
                )
                self._persist_recorder(recorder)
                self._console.print(
                    f"[bold yellow]休息 {plan.rest_between_blocks_sec:.0f} 秒，请放松但不要大幅动作[/bold yellow]"
                )
                try:
                    self._sleep_with_recording(
                        plan.rest_between_blocks_sec,
                        recorder=recorder,
                        heartbeat=heartbeat,
                        stage_name=f"Block {block_index + 1} 休息",
                        incremental_save_interval_sec=(
                            REST_INCREMENTAL_SAVE_INTERVAL_SEC
                        ),
                    )
                finally:
                    try:
                        self._emit_event(
                            recorder,
                            "automatic_break_end",
                            phase="automatic_break",
                            block_index=block_index,
                            next_block_index=block_index + 1,
                        )
                        self._persist_recorder(recorder)
                    finally:
                        if pause_control is not None:
                            pause_control.set_automatic_break(False)

    def _run_trial(
        self,
        *,
        label: str,
        recorder: SessionRecorder,
        heartbeat: Callable[[], None] | None,
        trial_index: int,
        block_index: int,
        attempt_index: int = 0,
        pause_control: CollectionPauseControl | None = None,
    ) -> dict[str, Any]:
        trial_timing = self._protocol.trial_timing
        self._console.print("[bold yellow]FIXATION[/bold yellow]")
        self._emit_event(
            recorder,
            "fixation_on",
            phase="formal",
            block_index=block_index,
            trial_index=trial_index,
            attempt_index=attempt_index,
            paradigm_stage="fixation",
        )
        stage_prefix = f"Block {block_index + 1} / Trial {trial_index + 1}"
        self._sleep_with_recording(
            trial_timing.fixation_sec,
            recorder=recorder,
            heartbeat=heartbeat,
            stage_name=f"{stage_prefix}: fixation",
            pause_control=pause_control,
            interruptible=True,
        )

        cue_event = f"cue_{label}_on"
        cue_message = f"PROMPT HAND {LABEL_DISPLAY[label]}"
        self._console.print(f"[bold yellow]{cue_message}[/bold yellow]")
        self._emit_event(
            recorder,
            cue_event,
            phase="formal",
            block_index=block_index,
            trial_index=trial_index,
            attempt_index=attempt_index,
            target_hand=label,
            paradigm_stage="movement_prompt",
        )
        self._sleep_with_recording(
            trial_timing.cue_sec,
            recorder=recorder,
            heartbeat=heartbeat,
            stage_name=f"{stage_prefix}: movement prompt {label}",
            pause_control=pause_control,
            interruptible=True,
        )

        task_message = f"{LABEL_SYMBOL[label]} {LABEL_DISPLAY[label]}"
        self._console.print(f"[bold yellow]{task_message}[/bold yellow]")
        motor_imagery_on_event = self._emit_event(
            recorder,
            f"motor_imagery_{label}_on",
            phase="formal",
            block_index=block_index,
            trial_index=trial_index,
            attempt_index=attempt_index,
            label=label,
            label_id=LABEL_TO_ID[label],
            paradigm_stage="motor_imagery",
        )
        motor_imagery_on_sample = int(motor_imagery_on_event.sample_index)
        self._sleep_with_recording(
            trial_timing.control_sec,
            recorder=recorder,
            heartbeat=heartbeat,
            stage_name=f"{stage_prefix}: motor imagery {label}",
            pause_control=pause_control,
            interruptible=True,
        )
        motor_imagery_off_event = self._emit_event(
            recorder,
            "motor_imagery_off",
            phase="formal",
            block_index=block_index,
            trial_index=trial_index,
            attempt_index=attempt_index,
            label=label,
            label_id=LABEL_TO_ID[label],
            paradigm_stage="motor_imagery",
        )
        motor_imagery_off_sample = int(motor_imagery_off_event.sample_index)
        return {
            "phase": "formal",
            "block_index": block_index,
            "trial_index": trial_index,
            "attempt_index": attempt_index,
            "label": label,
            "label_id": LABEL_TO_ID[label],
            "motor_imagery_on_sample": motor_imagery_on_sample,
            "motor_imagery_off_sample": motor_imagery_off_sample,
        }

    def _build_mi_windows(
        self,
        *,
        eeg: np.ndarray,
        events: list[Any],
        trials: list[dict[str, Any]],
        session_dir: Path | None,
        window_filename: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        del events
        source_window_samples = int(round(self._protocol.window_sec * self._source_sfreq))
        target_window_samples = int(round(self._protocol.window_sec * self._sfreq))
        stride_samples = int(round(self._protocol.stride_sec * self._source_sfreq))
        start_offset = int(round(self._protocol.motor_imagery_start_offset_sec * self._source_sfreq))
        stop_offset = int(round(self._protocol.motor_imagery_stop_offset_sec * self._source_sfreq))
        raw_windows: list[np.ndarray] = []
        processed_windows: list[np.ndarray] = []
        labels: list[int] = []
        trial_groups: list[int] = []
        quality_peak_abs_uv: list[float] = []
        quality_clip_fraction: list[float] = []
        quality_bad_channel_fraction: list[float] = []
        quality_bad_channel_indices: list[str] = []
        rejection_reason_counts: dict[str, int] = {}
        window_start_samples: list[int] = []
        window_stop_samples: list[int] = []
        window_offsets_sec: list[float] = []
        rejected_windows = 0
        continuous = preprocess_eeg_continuous(
            eeg,
            source_sfreq=self._source_sfreq,
            target_sfreq=self._sfreq,
        )

        for trial_group, trial in enumerate(trials):
            motor_imagery_on = int(trial["motor_imagery_on_sample"])
            max_start = motor_imagery_on + stop_offset - source_window_samples
            for offset in range(start_offset, max_start - motor_imagery_on + 1, stride_samples):
                start = motor_imagery_on + offset
                stop = start + source_window_samples
                if stop > eeg.shape[1]:
                    continue
                target_start = int(round(start * self._sfreq / self._source_sfreq))
                target_stop = target_start + target_window_samples
                window = continuous.raw_data[:, target_start:target_stop]
                if window.shape[1] != target_window_samples:
                    raise RuntimeError(
                        f"Continuous motor-imagery window has {window.shape[1]} points; "
                        f"expected {target_window_samples}."
                    )
                filtered_window = continuous.data[:, target_start:target_stop]
                nonfinite_fraction = float(
                    np.mean(continuous.source_nonfinite_mask[:, start:stop])
                )
                result = finalize_preprocessed_window(
                    filtered_window,
                    bad_channel_indices=continuous.bad_channel_indices,
                    nonfinite_fraction=nonfinite_fraction,
                )
                if not result.quality.accepted:
                    rejected_windows += 1
                    for reason in result.quality.reasons:
                        rejection_reason_counts[reason] = (
                            rejection_reason_counts.get(reason, 0) + 1
                        )
                    continue
                raw_windows.append(window)
                processed_windows.append(result.data)
                labels.append(int(trial["label_id"]))
                trial_groups.append(trial_group)
                quality_peak_abs_uv.append(result.quality.peak_abs_uv)
                quality_clip_fraction.append(result.quality.clip_fraction)
                quality_bad_channel_fraction.append(result.quality.bad_channel_fraction)
                quality_bad_channel_indices.append(
                    json.dumps(
                        list(getattr(result.quality, "bad_channel_indices", ())),
                        separators=(",", ":"),
                    )
                )
                window_start_samples.append(start)
                window_stop_samples.append(stop)
                window_offsets_sec.append(offset / self._source_sfreq)

        empty_shape = (0, eeg.shape[0], target_window_samples)
        raw_X = np.stack(raw_windows, axis=0).astype(np.float32) if raw_windows else np.empty(empty_shape, dtype=np.float32)
        X = np.stack(processed_windows, axis=0).astype(np.float32) if processed_windows else np.empty(empty_shape, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int64)
        groups = np.asarray(trial_groups, dtype=np.int64)

        if session_dir is not None:
            self._save_mi_windows(
                session_dir / window_filename,
                raw_windows=raw_X,
                processed_windows=X,
                labels=y,
                trial_ids=groups,
                window_sec=self._protocol.window_sec,
                stride_sec=self._protocol.stride_sec,
                quality_peak_abs_uv=np.asarray(quality_peak_abs_uv, dtype=np.float32),
                quality_clip_fraction=np.asarray(quality_clip_fraction, dtype=np.float32),
                quality_bad_channel_fraction=np.asarray(
                    quality_bad_channel_fraction,
                    dtype=np.float32,
                ),
                rejected_windows=rejected_windows,
                window_start_samples=np.asarray(window_start_samples, dtype=np.int64),
                window_stop_samples=np.asarray(window_stop_samples, dtype=np.int64),
                window_offsets_sec=np.asarray(window_offsets_sec, dtype=np.float32),
                quality_bad_channel_indices=np.asarray(
                    quality_bad_channel_indices,
                    dtype=np.str_,
                ),
                rejection_reason_counts=rejection_reason_counts,
            )
        if rejected_windows:
            self._console.print(
                f"[yellow]预处理质量控制剔除 {rejected_windows} 个伪迹窗；"
                f"保留 {X.shape[0]} 个有效4秒窗口。[/yellow]"
            )
        return raw_X, X, y, groups

    def _save_mi_windows(
        self,
        output_path: Path,
        *,
        raw_windows: np.ndarray,
        processed_windows: np.ndarray,
        labels: np.ndarray,
        trial_ids: np.ndarray,
        window_sec: float,
        stride_sec: float,
        quality_peak_abs_uv: np.ndarray,
        quality_clip_fraction: np.ndarray,
        quality_bad_channel_fraction: np.ndarray,
        rejected_windows: int,
        window_start_samples: np.ndarray,
        window_stop_samples: np.ndarray,
        window_offsets_sec: np.ndarray,
        quality_bad_channel_indices: np.ndarray,
        rejection_reason_counts: dict[str, int],
    ) -> None:
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                raw_windows=raw_windows,
                processed_windows=processed_windows,
                labels=labels,
                trial_ids=trial_ids,
                source_sfreq=np.asarray([self._source_sfreq], dtype=np.float32),
                sfreq=np.asarray([self._sfreq], dtype=np.float32),
                window_sec=np.asarray([window_sec], dtype=np.float32),
                step_sec=np.asarray([stride_sec], dtype=np.float32),
                quality_peak_abs_uv=quality_peak_abs_uv,
                quality_clip_fraction=quality_clip_fraction,
                quality_bad_channel_fraction=quality_bad_channel_fraction,
                quality_bad_channel_indices=quality_bad_channel_indices,
                window_start_samples=window_start_samples,
                window_stop_samples=window_stop_samples,
                window_offsets_sec=window_offsets_sec,
                quality_rejected_windows=np.asarray(
                    [rejected_windows],
                    dtype=np.int64,
                ),
                quality_rejection_reason_counts=np.asarray(
                    [
                        json.dumps(
                            rejection_reason_counts,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    ],
                    dtype=np.str_,
                ),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)

    def _get_continuous_eeg(self, *, session_dir: Path | None, recorder: SessionRecorder) -> np.ndarray:
        if session_dir is not None:
            return np.load(session_dir / "continuous_eeg.npy").astype(np.float32)
        return recorder.to_array()

    def _build_session_metadata(
        self,
        plan: SessionPlan,
        *,
        session_id: str,
        trials: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "protocol_name": "binary_hand_mi_collection_v1",
            "task_paradigm": "binary_hand_mi",
            "subject_mode": plan.subject_mode,
            "sfreq": self._sfreq,
            "source_sfreq": self._source_sfreq,
            "n_channels": self._acquirer.metadata.n_channels,
            "channel_names": list(
                getattr(self._acquirer.metadata, "channel_names", ())
            ),
            "channel_types": list(
                getattr(self._acquirer.metadata, "channel_types", ())
            ),
            "channel_selection": getattr(
                self._acquirer,
                "channel_diagnostics",
                {},
            ),
            "timing_diagnostics": getattr(
                self._acquirer,
                "timing_diagnostics",
                {},
            ),
            "window_sec": self._protocol.window_sec,
            "stride_sec": self._protocol.stride_sec,
            "motor_imagery_window_range_sec": [
                self._protocol.motor_imagery_start_offset_sec,
                self._protocol.motor_imagery_stop_offset_sec,
            ],
            "planned_collection_duration_sec": (
                plan.total_formal_trials * plan.trial_timing.total_sec
                + max(len(plan.blocks) - 1, 0) * plan.rest_between_blocks_sec
            ),
            "formal_trial_count": len(trials),
            "collection_mode": "fixed_session_plan",
            "validation_grouping": "trial_ids",
            "preprocessing": {
                **continuous_preprocessing_metadata(),
            },
            "continuous_span": "complete_collection_session",
            "trial_timing": {
                "fixation_sec": plan.trial_timing.fixation_sec,
                "cue_sec": plan.trial_timing.cue_sec,
                "control_sec": plan.trial_timing.control_sec,
            },
            "trial_stage_semantics": {
                "fixation": "green_cross_fixation_unlabeled",
                "cue": "left_or_right_movement_prompt_unlabeled",
                "control": "labeled_motor_imagery",
            },
            "label_map": LABEL_TO_ID,
            "trials": trials,
            "bad_trials": [],
            "low_quality_blocks": [],
            "provenance": self._build_provenance(),
        }

    def _write_session_summary(
        self,
        session_dir: Path | None,
        *,
        metrics: dict[str, float],
        windows_collected: int,
        session_metadata: dict[str, Any],
        training_performed: bool = True,
    ) -> None:
        if session_dir is None:
            return
        summary = dict(session_metadata)
        summary["training_performed"] = bool(training_performed)
        summary["model_path"] = str(self._model_path) if training_performed else None
        summary["windows_collected"] = windows_collected
        summary["metrics"] = metrics
        metadata_path = session_dir / "metadata.json"
        temporary = session_dir / ".metadata.json.tmp"
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, metadata_path)

    @staticmethod
    def _write_collection_summary(
        session_dir: Path | None,
        *,
        trials_collected: int,
        windows_collected: int,
        session_metadata: dict[str, Any],
    ) -> None:
        if session_dir is None:
            return
        summary = dict(session_metadata)
        summary["collection_mode"] = "acquisition_only"
        summary["trials_collected"] = trials_collected
        summary["windows_collected"] = windows_collected
        summary["preprocessing_performed"] = True
        summary["windowing_anchor"] = "motor_imagery_on_sample"
        summary["model_activity"] = "none"
        metadata_path = session_dir / "metadata.json"
        temporary = session_dir / ".metadata.json.tmp"
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, metadata_path)

    def _seal_session_bundle(
        self,
        session_dir: Path | None,
        *,
        include_model_files: bool = True,
    ) -> None:
        if session_dir is None:
            return
        metadata_path = session_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model_files: list[dict[str, Any]] = []
        candidate_model_paths = (
            self._model_path,
            Path(f"{self._model_path}.neuroonline.pt"),
        ) if include_model_files else ()
        for path in candidate_model_paths:
            if path.exists():
                model_files.append(
                    {
                        "path": str(path),
                        "sha256": self._sha256(path),
                        "size_bytes": path.stat().st_size,
                    }
                )
        checksums: list[dict[str, Any]] = []
        for path in sorted(session_dir.iterdir()):
            if (
                not path.is_file()
                or path == metadata_path
                or path.name in {
                    "checkpoint.json",
                    "events.jsonl",
                    "metadata.partial.json",
                }
            ):
                continue
            checksums.append(
                {
                    "path": path.name,
                    "sha256": self._sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
        eeg_path = session_dir / "continuous_eeg.npy"
        continuous_sample_count = 0
        if eeg_path.exists():
            eeg = np.load(eeg_path, mmap_mode="r")
            continuous_sample_count = int(eeg.shape[-1])
        packet_loss_count = int(
            float(
                (metadata.get("timing_diagnostics", {}) or {}).get(
                    "packet_loss_count",
                    0,
                )
            )
        )
        integrity_status = "complete"
        if packet_loss_count > 0:
            integrity_status = "source_packet_loss"
        elif continuous_sample_count <= 0:
            integrity_status = "missing_continuous_eeg"
        metadata["model_files"] = model_files
        metadata["integrity"] = {
            "status": integrity_status,
            "packet_loss_count": packet_loss_count,
            "continuous_sample_count": continuous_sample_count,
            "checksums": checksums,
        }
        temporary = session_dir / ".metadata.json.tmp"
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, metadata_path)

    def _build_provenance(self) -> dict[str, Any]:
        project_root = Path(__file__).resolve().parents[1]
        commit: str | None = None
        dirty: bool | None = None
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            dirty = bool(
                subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=project_root,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout.strip()
            )
        except (OSError, subprocess.SubprocessError):
            pass
        packages: dict[str, str | None] = {}
        package_names = ["numpy", "scipy", "scikit-learn"]
        if self._neuroonline_config is not None:
            package_names.append("torch")
        for package in package_names:
            try:
                packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                packages[package] = None
        encoded_config = json.dumps(
            self._experiment_config,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
        provenance = {
            "git": {"commit": commit, "dirty": dirty},
            "platform": platform.platform(),
            "python": sys.version,
            "packages": packages,
            "experiment_config": self._experiment_config,
            "experiment_config_sha256": hashlib.sha256(encoded_config).hexdigest(),
            "trial_sequence_random_seed": int(self._protocol.random_seed),
        }
        if self._neuroonline_config is not None:
            provenance["model_training_random_seed"] = int(
                self._neuroonline_config.random_seed
            )
        return provenance

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _print_instructions(self, plan: SessionPlan) -> None:
        self._console.print("[bold cyan]实验指导语[/bold cyan]")
        for line in RECOMMENDED_INSTRUCTIONS:
            self._console.print(f"- {line}")
        self._console.print(
            f"[bold cyan]Binary hand-MI trial[/bold cyan] fixation={plan.trial_timing.fixation_sec:.1f}s "
            f"movement_prompt={plan.trial_timing.cue_sec:.1f}s motor_imagery={plan.trial_timing.control_sec:.1f}s "
        )
        self._console.print(
            f"[bold cyan]MI 数据窗口[/bold cyan] window={self._protocol.window_sec:.1f}s stride={self._protocol.stride_sec:.1f}s "
            f"from motor_imagery [{self._protocol.motor_imagery_start_offset_sec:.1f}, {self._protocol.motor_imagery_stop_offset_sec:.1f}]s"
        )

    def _emit_event(self, recorder: SessionRecorder, event_name: str, **payload: Any) -> Any:
        event_time = time.monotonic()
        return recorder.add_event(
            event_name,
            timestamp_monotonic=event_time,
            event_code=PROTOCOL_EVENT_CODES[event_name],
            **payload,
        )

    @staticmethod
    def _persist_recorder(recorder: Any, **progress: Any) -> None:
        persist = getattr(recorder, "persist", None)
        if callable(persist):
            persist(**progress)

    def _sleep_with_recording(
        self,
        duration_sec: float,
        *,
        recorder: SessionRecorder,
        heartbeat: Callable[[], None] | None,
        stage_name: str = "",
        pause_control: CollectionPauseControl | None = None,
        interruptible: bool = False,
        incremental_save_interval_sec: float | None = None,
    ) -> None:
        total = max(float(duration_sec), 0.0)
        started_at = time.monotonic()
        deadline = started_at + total
        save_interval = (
            max(float(incremental_save_interval_sec), 0.1)
            if incremental_save_interval_sec is not None
            else None
        )
        next_incremental_save = (
            started_at + save_interval if save_interval is not None else None
        )
        self._update_stage_progress(stage_name=stage_name, elapsed_sec=0.0, duration_sec=total)
        while time.monotonic() < deadline:
            self._flush_recorder(recorder)
            now = time.monotonic()
            if (
                next_incremental_save is not None
                and save_interval is not None
                and now >= next_incremental_save
            ):
                self._persist_recorder(recorder)
                while next_incremental_save <= now:
                    next_incremental_save += save_interval
            if (
                interruptible
                and pause_control is not None
                and pause_control.pause_requested
            ):
                raise TrialDiscarded(stage_name)
            if heartbeat is not None:
                heartbeat()
            elapsed = min(time.monotonic() - started_at, total)
            self._update_stage_progress(stage_name=stage_name, elapsed_sec=elapsed, duration_sec=total)
            time.sleep(min(0.05, max(deadline - time.monotonic(), 0.0)))
        self._flush_recorder(recorder)
        self._update_stage_progress(stage_name=stage_name, elapsed_sec=total, duration_sec=total)
        if heartbeat is not None:
            heartbeat()

    def _wait_for_manual_resume(
        self,
        pause_control: CollectionPauseControl | None,
        *,
        recorder: SessionRecorder,
        heartbeat: Callable[[], None] | None,
        block_index: int,
        trial_index: int,
    ) -> None:
        if pause_control is None:
            return
        pause_control.begin_pause()
        self._emit_event(
            recorder,
            "manual_pause_start",
            phase="manual_pause",
            block_index=block_index,
            trial_index=trial_index,
        )
        self._persist_recorder(recorder)
        self._console.print("[bold yellow]休息（点击“继续采集”后恢复）[/bold yellow]")
        self._update_stage_progress(
            stage_name="手动休息",
            elapsed_sec=0.0,
            duration_sec=0.0,
        )
        next_incremental_save = time.monotonic() + REST_INCREMENTAL_SAVE_INTERVAL_SEC
        while pause_control.paused:
            self._flush_recorder(recorder)
            now = time.monotonic()
            if now >= next_incremental_save:
                self._persist_recorder(recorder)
                while next_incremental_save <= now:
                    next_incremental_save += REST_INCREMENTAL_SAVE_INTERVAL_SEC
            if heartbeat is not None:
                heartbeat()
            time.sleep(0.05)
        self._emit_event(
            recorder,
            "manual_pause_end",
            phase="manual_pause",
            block_index=block_index,
            trial_index=trial_index,
        )
        self._persist_recorder(recorder)

    def _update_stage_progress(self, *, stage_name: str, elapsed_sec: float, duration_sec: float) -> None:
        progress = getattr(self._console, "set_stage_progress", None)
        if callable(progress):
            progress(stage_name=stage_name, elapsed_sec=elapsed_sec, duration_sec=duration_sec)

    def _update_trial_progress(self, *, completed_trials: int, total_trials: int) -> None:
        progress = getattr(self._console, "set_trial_progress", None)
        if callable(progress):
            progress(
                completed_trials=completed_trials,
                total_trials=total_trials,
            )

    def _flush_recorder(self, recorder: SessionRecorder) -> None:
        try:
            samples = recorder.pull()
        except RuntimeError as exc:
            message = str(exc).lower()
            if "stream" in message and "not started" in message:
                raise RuntimeError(
                    "Collection interrupted: EEG stream stopped unexpectedly. "
                    "Please check BrainCo device power/network stability and rerun."
                ) from exc
            raise
        if samples.size == 0:
            return

    def _wait_for_first_samples(
        self,
        recorder: SessionRecorder,
        *,
        heartbeat: Callable[[], None] | None,
        timeout_sec: float = 10.0,
    ) -> None:
        """Keep the first fixation off-screen until the EEG stream is live."""

        deadline = time.monotonic() + max(float(timeout_sec), 0.1)
        self._update_stage_progress(
            stage_name="等待 EEG 数据流就绪",
            elapsed_sec=0.0,
            duration_sec=float(timeout_sec),
        )
        while time.monotonic() < deadline:
            samples = recorder.pull()
            if samples.size:
                return
            if heartbeat is not None:
                heartbeat()
            time.sleep(0.05)
        raise RuntimeError(
            f"EEG stream produced no samples within {float(timeout_sec):.1f} seconds."
        )

    def _save_metadata(
        self,
        *,
        metrics: dict[str, float],
        windows_collected: int,
        head_only: bool,
    ) -> None:
        metadata = {
            "model_path": str(self._model_path),
            "windows_collected": windows_collected,
            "training_window_seconds_collected": (
                windows_collected * self._protocol.window_sec
            ),
            "head_only": head_only,
            "sfreq": self._sfreq,
            "window_sec": self._protocol.window_sec,
            "step_sec": self._protocol.stride_sec,
            "metrics": metrics,
        }
        metadata_path = self._model_path.with_suffix(".metrics.yaml")
        with metadata_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(metadata, handle, sort_keys=False, allow_unicode=True)
