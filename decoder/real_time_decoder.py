"""Background realtime motor imagery decoding loop."""

from __future__ import annotations

from collections.abc import Callable
import copy
import hashlib
import importlib.metadata
import json
import logging
import platform
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import numpy as np
from rich.console import Console

from acquisition.base import AbstractAcquirer
from adaptation.neuroonline import (
    NeuroOnlineConfig,
    NeuroOnlineModelAdapter,
    NeuroOnlineStreamAdapter,
)
from models.factory import BaseModelAdapter, TorchModelAdapter
from utils.markers import LSLCommandOutlet, MarkerBackend
from utils.online_labels import CUED_PROTOCOL_VERSION, OnlineLabelSource
from utils.preprocessing import (
    ContinuousPreprocessingResult,
    PreprocessingResult,
    continuous_preprocessing_metadata,
    finalize_preprocessed_window,
    preprocess_eeg_continuous,
)
from utils.stream_writer import StreamWriter

LOGGER = logging.getLogger(__name__)

LABEL_NAMES = {0: "左手", 1: "右手"}
TEST_MODE_PROMPTS = {0: "想象左手", 1: "想象右手"}


def _integer_or(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


@dataclass(slots=True)
class PredictionResult:
    """One realtime decoding output."""

    label: str
    confidence: float
    uncertainty: float
    class_id: int | None


@dataclass(slots=True)
class _PendingCuedWindow:
    """One labeled window waiting for the future transition guard to close."""

    processed: np.ndarray
    probabilities: np.ndarray
    operational_prediction: int | None
    prediction_model_revision: int
    online_label: Any
    window_start: float
    window_end: float
    quality_accepted: bool
    training_role: str
    adaptation_eligible: bool
    record_payload: dict[str, Any] | None


class GameCommandOutlet(Protocol):
    """Command transport required by the continuous Unity driving protocol."""

    def push(self, command: str) -> None: ...

    def push_with_ack(self, command: str) -> dict[str, Any]: ...

    def close(self) -> None: ...


class RealTimeDecoder:
    """Continuously decode sliding EEG windows on a background thread."""

    def __init__(
        self,
        acquirer: AbstractAcquirer,
        model: BaseModelAdapter,
        console: Console,
        command_outlet: LSLCommandOutlet,
        game_command_outlet: GameCommandOutlet | None,
        *,
        sfreq: float,
        window_sec: float,
        step_sec: float,
        confidence_threshold: float,
        mc_dropout_passes: int,
        model_save_path: Path | None = None,
        online_label_source: OnlineLabelSource | None = None,
        status_callback: Callable[[dict[str, Any]], None] | None = None,
        thread_context: Any | None = None,
        stop_on_game_disconnect: bool = True,
        batch_update_config: dict[str, Any] | None = None,
        n_classes: int = 2,
        experiment_config: dict[str, Any] | None = None,
        model_name: str | None = None,
        model_source_path: Path | None = None,
    ) -> None:
        self._acquirer = acquirer
        self._model = model
        self._model_lock = threading.RLock()
        self._model_revision = 0
        self._console = console
        self._command_outlet = command_outlet
        self._game_command_outlet = game_command_outlet
        self._sfreq = sfreq
        self._window_sec = window_sec
        self._step_sec = step_sec
        self._confidence_threshold = confidence_threshold
        self._mc_dropout_passes = mc_dropout_passes
        self._n_classes = max(int(n_classes), 1)
        ar_game_config = (
            (((experiment_config or {}).get("output", {}) or {}).get("ar_game", {}) or {})
        )
        self._visual_onset_delay_sec = max(
            float(ar_game_config.get("visual_onset_delay_sec", 0.0)),
            0.0,
        )
        self._model_save_path = model_save_path
        self._online_label_source = online_label_source
        self._lane_transition_guard_sec = max(
            float(getattr(online_label_source, "lane_transition_guard_sec", 0.0)),
            0.0,
        )
        self._pending_cued_windows: list[_PendingCuedWindow] = []
        self._primary_decision_scenes: set[int] = set()
        self._primary_decision_window_bounds: dict[
            int,
            list[tuple[float, float]],
        ] = {}
        self._primary_decision_probabilities: dict[int, list[np.ndarray]] = {}
        source_metadata = getattr(online_label_source, "metadata", None)
        cue_metadata = source_metadata() if callable(source_metadata) else {}
        self._primary_windows_per_scene = max(
            int((cue_metadata or {}).get("primary_windows_per_scene", 1)),
            1,
        )
        self._primary_window_spacing_sec = max(
            float((cue_metadata or {}).get("primary_window_spacing_sec", 1.0)),
            self._step_sec,
        )
        self._control_released_at: dict[int, float] = {}
        self._last_processed_window_end_monotonic: float | None = None
        self._stale_source_windows_rejected = 0
        self._status_callback = status_callback
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_game_command: str | None = None
        self._last_game_transport_command: str | None = None
        self._last_game_transport_error: str | None = None
        self._last_game_transport_sent_at = 0.0
        self._last_game_movement_sent_at = 0.0
        self._game_command_keepalive_sec = max(0.2, min(0.5, step_sec * 1.1))
        self._game_session_started = False
        self._game_disconnect_message: str | None = None
        self._scene_sent_scene_index = -1
        self._scene_sent_label_id: int | None = None
        self._unity_scene_number_offset: int | None = None
        self._unity_scene_numbers: dict[int, int] = {}
        self._max_scenes: int | None = None
        self._scene_sync_error: str | None = None
        self._failed_scene_indices: set[int] = set()
        self._scene_started_at: dict[int, float] = {}
        self._scene_labels: dict[int, int] = {}
        self._scene_start_lanes: dict[int, int] = {}
        self._scene_safe_lanes: dict[int, int] = {}
        self._scene_end_recorded: set[int] = set()
        self._timestamp_fallback_warned = False
        self._stop_on_game_disconnect = bool(stop_on_game_disconnect)
        self._thread_context = thread_context
        self._experiment_config = copy.deepcopy(experiment_config or {})
        self._model_name = str(
            model_name or getattr(model, "model_name", type(model).__name__)
        )
        self._model_source_path = (
            None if model_source_path is None else Path(model_source_path)
        )
        self._run_id = uuid4().hex
        self._model_revision_records: list[dict[str, Any]] = []
        neuroonline_config = NeuroOnlineConfig.from_mapping(
            batch_update_config,
            window_duration_sec=window_sec,
        )
        self._neuroonline_adapter: NeuroOnlineStreamAdapter | None = None
        self._neuroonline_training_notice = False
        if neuroonline_config.enabled:
            if not isinstance(model, TorchModelAdapter):
                raise ValueError("NeuroOnline requires a PyTorch decoder model.")
            if model_save_path is None:
                raise ValueError("NeuroOnline adaptation requires model_save_path.")
            self._model = NeuroOnlineModelAdapter(
                model,
                config=neuroonline_config,
                state_path=model_save_path,
            )
            neuroonline_config = self._model.config
            self._neuroonline_adapter = NeuroOnlineStreamAdapter(
                config=neuroonline_config,
                update_callback=self._run_neuroonline_update,
                save_callback=self._save_current_model,
                completion_callback=self._on_neuroonline_update_complete,
                n_classes=n_classes,
            )

    def start(self) -> None:
        self._acquirer.start_stream()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._decode_loop, daemon=True)
        if self._thread_context is not None:
            try:
                from streamlit.runtime.scriptrunner import add_script_run_ctx
            except Exception:  # noqa: BLE001
                pass
            else:
                add_script_run_ctx(self._thread, self._thread_context)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        self._flush_pending_cued_windows(force=True)
        self._acquirer.stop_stream()
        if self._game_command_outlet is not None:
            try:
                self._game_command_outlet.push("STOP")
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("Failed to send final AR STOP: %s", exc)
            self._game_command_outlet.close()
        if self._neuroonline_adapter is not None:
            self._neuroonline_adapter.close()
        if self._online_label_source is not None:
            self._online_label_source.close()

    def run_forever(
        self,
        *,
        subject_id: str | None = None,
        save_dir: Path | None = None,
        record: bool = False,
        heartbeat: Callable[[], None] | None = None,
        max_scenes: int | None = None,
    ) -> None:
        if max_scenes is not None and int(max_scenes) < 1:
            raise ValueError("max_scenes must be at least 1 when provided")
        self._record = record
        self._subject_id = subject_id
        self._max_scenes = None if max_scenes is None else int(max_scenes)
        self._last_game_command = None
        self._last_game_transport_command = None
        self._last_game_transport_error = None
        self._last_game_transport_sent_at = 0.0
        self._last_game_movement_sent_at = 0.0
        self._game_session_started = False
        self._primary_decision_scenes.clear()
        self._primary_decision_window_bounds.clear()
        self._primary_decision_probabilities.clear()
        self._control_released_at.clear()
        self._last_processed_window_end_monotonic = None
        self._stale_source_windows_rejected = 0
        self._game_disconnect_message = None
        self._scene_sent_scene_index = -1
        self._scene_sent_label_id = None
        self._unity_scene_number_offset = None
        self._unity_scene_numbers.clear()
        self._scene_sync_error = None
        self._failed_scene_indices.clear()
        self._scene_started_at.clear()
        self._scene_labels.clear()
        self._scene_start_lanes.clear()
        self._scene_safe_lanes.clear()
        self._scene_end_recorded.clear()
        self._model_revision_records.clear()
        self._pending_cued_windows.clear()
        if record and subject_id:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            realtime_root = save_dir or Path("records_storage") / subject_id / "realtime"
            self._save_dir = realtime_root / timestamp
            self._writer = StreamWriter(self._save_dir)
            self._writer.start({
                "run_id": self._run_id,
                "subject_id": subject_id,
                "mode": "realtime",
                "sfreq": self._sfreq,
                "window_sec": self._window_sec,
                "step_sec": self._step_sec,
                "channels": self._acquirer.metadata.n_channels,
                "channel_names": list(
                    getattr(self._acquirer.metadata, "channel_names", ())
                ),
                "channel_types": list(
                    getattr(self._acquirer.metadata, "channel_types", ())
                ),
                "model_name": self._model_name,
                "model_revision": self._model_revision,
                "preprocessing": {
                    **continuous_preprocessing_metadata(),
                    "continuous_span": "retained_acquirer_history",
                },
                "online_adaptation": self._online_adaptation_status(),
                "online_label_source": self._online_label_source_metadata(),
                "provenance": self._build_run_provenance(),
            })
            self._writer.append_event(
                "session_start",
                run_id=self._run_id,
                subject_id=subject_id,
                model_name=self._model_name,
            )
            self._snapshot_model_revision(0, source="session_start")

        run_status = "running"
        run_error: str | None = None
        try:
            self._push_game_session_command("START")
            self.start()
            while not self._stop_event.is_set():
                self._sleep_with_heartbeat(min(0.1, max(self._step_sec, 0.1)), heartbeat)
                if heartbeat is not None:
                    heartbeat()
            if self._game_disconnect_message:
                raise RuntimeError(self._game_disconnect_message)
            run_status = "completed"
        except KeyboardInterrupt:
            run_status = "interrupted"
            self._console.print("\n[bold red]停止实时解码[/bold red]")
        except Exception as exc:
            run_status = "failed"
            run_error = str(exc)
            raise
        finally:
            self.stop()
            self._record_active_scene_end(outcome="incomplete", reason="session_stop")
            if heartbeat is not None:
                heartbeat()
            if hasattr(self, "_writer"):
                self._writer.append_event(
                    "session_stop",
                    status=run_status,
                    error=run_error,
                    model_revision=self._model_revision,
                )
                self._writer.stop()
                self._writer.finalize_manifest(
                    {
                        "status": run_status,
                        "error": run_error,
                        "model_revision": self._model_revision,
                        "model_revisions": list(self._model_revision_records),
                        "online_adaptation": self._online_adaptation_status(),
                        "online_label_source": self._online_label_source_metadata(),
                        "timing_diagnostics": getattr(
                            self._acquirer,
                            "timing_diagnostics",
                            {},
                        ),
                        "channel_selection": getattr(
                            self._acquirer,
                            "channel_diagnostics",
                            {},
                        ),
                    }
                )
                self._console.print(f"[bold green]实时数据已保存[/bold green] {self._save_dir}")

    def run_test_mode(
        self,
        *,
        subject_id: str,
        marker_backend: MarkerBackend,
        duration_sec: int,
        block_sec: float = 10.0,
        initial_rest_sec: float = 0.0,
        save_dir: Path | None = None,
        heartbeat: Callable[[], None] | None = None,
        stage_progress: Callable[[str, float, float], None] | None = None,
    ) -> dict[str, float | int | str]:
        """Run cue-based testing, save captured EEG/labels, and report accuracy."""

        self._console.print("[bold cyan]测试模式启动（有 cue）[/bold cyan]")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root_dir = save_dir or Path("records_storage") / subject_id / "test_mode" / timestamp
        writer = StreamWriter(root_dir)
        writer.start({
            "run_id": self._run_id,
            "subject_id": subject_id,
            "mode": "test_mode",
            "sfreq": self._sfreq,
            "window_sec": self._window_sec,
            "step_sec": self._step_sec,
            "channels": self._acquirer.metadata.n_channels,
            "channel_names": list(
                getattr(self._acquirer.metadata, "channel_names", ())
            ),
            "channel_types": list(
                getattr(self._acquirer.metadata, "channel_types", ())
            ),
            "preprocessing": {
                **continuous_preprocessing_metadata(),
                "continuous_span": "retained_acquirer_history",
            },
            "provenance": self._build_run_provenance(),
        })
        writer.append_event("session_start", run_id=self._run_id, mode="test_mode")
        
        def update_stage(stage_name: str, elapsed_sec: float, total_sec: float) -> None:
            if stage_progress is not None:
                stage_progress(stage_name, elapsed_sec, total_sec)

        self._acquirer.start_stream()
        if heartbeat is not None:
            heartbeat()
        if initial_rest_sec > 0:
            self._console.print(f"[bold yellow]Baseline 测试放松注视 ({initial_rest_sec:.0f}s)[/bold yellow]")
            update_stage("测试放松注视", 0.0, initial_rest_sec)
            self._sleep_with_stage_progress(
                initial_rest_sec,
                heartbeat=heartbeat,
                stage_name="测试放松注视",
                stage_progress=stage_progress,
            )
        started = time.monotonic()
        cue_index = 0
        labels = list(range(self._n_classes))
        collected_windows: list[np.ndarray] = []
        true_labels: list[int] = []
        pred_labels: list[int] = []
        confidences: list[float] = []
        quality_accepted: list[bool] = []
        run_status = "completed"
        run_error: str | None = None
        try:
            while time.monotonic() - started < duration_sec:
                label = labels[cue_index % len(labels)]
                cue_index += 1
                self._console.print(f"[bold yellow][cue][/bold yellow] {TEST_MODE_PROMPTS[label]}")
                if heartbeat is not None:
                    heartbeat()
                marker_backend.send(label)
                writer.append_event(
                    "test_cue_start",
                    cue_index=cue_index - 1,
                    label_id=label,
                    label_name=TEST_MODE_PROMPTS[label],
                )
                
                # IMPORTANT: Delay for window_sec before starting to evaluate this cue.
                # If window is 4s, the immediate chunk returned still mostly contains data PROR to the cue.
                # We need to give the subject time to react and the ring buffer time to fill with the new intent.
                update_stage(f"测试 {cue_index}: cue {TEST_MODE_PROMPTS[label]}", 0.0, self._window_sec)
                self._sleep_with_stage_progress(
                    self._window_sec,
                    heartbeat=heartbeat,
                    stage_name=f"测试 {cue_index}: cue {TEST_MODE_PROMPTS[label]}",
                    stage_progress=stage_progress,
                )
                
                # Now we predict on the new block length
                # Since we already waited window_sec, we subtract this from the block duration to keep blocks same length
                control_sec = max(0.1, block_sec - self._window_sec)
                control_started = time.monotonic()
                block_end = control_started + control_sec
                update_stage(f"测试 {cue_index}: 预测控制", 0.0, control_sec)
                
                while time.monotonic() < block_end and time.monotonic() - started < duration_sec:
                    loop_started = time.perf_counter()
                    try:
                        continuous, history_timestamps = self._acquire_preprocessed_history(
                            self._window_sec
                        )
                    except RuntimeError:
                        time.sleep(self._step_sec)
                        continue
                    window, timestamps, preprocessing = self._slice_preprocessed_history(
                        continuous,
                        history_timestamps,
                    )
                    window_start, window_end = self._resolve_window_time_bounds(timestamps)
                    processed = preprocessing.data
                    probability_batch, model_revision = self._predict_proba_with_revision(
                        processed[None, ...],
                        mc_dropout_passes=self._mc_dropout_passes,
                    )
                    probabilities = probability_batch[0]
                    raw_prediction = int(np.argmax(probabilities))
                    result = self._post_process(probabilities)
                    self._console.print(
                        f"[green][预测][/green] {result.label} "
                        f"(confidence: {result.confidence:.2f}, uncertainty: {result.uncertainty:.2f})"
                    )
                    self._command_outlet.push(result.label)
                    game_command = self._to_game_command(result)
                    self._push_game_command(game_command)
                    self._emit_status(result, game_command)
                    
                    pred_class = -1 if result.class_id is None else int(result.class_id)
                    timing_diagnostics = getattr(
                        self._acquirer,
                        "timing_diagnostics",
                        {},
                    ) or {}
                    writer.put(
                        window=window.astype(np.float32),
                        y_true=label,
                        y_pred=pred_class,
                        confidence=float(result.confidence),
                        raw_pred=raw_prediction,
                        model_revision=model_revision,
                        label_event_id=f"test-cue-{cue_index - 1:06d}",
                        probabilities=probabilities,
                        uncertainty=float(result.uncertainty),
                        window_start_monotonic=window_start,
                        window_end_monotonic=window_end,
                        scene_index=cue_index - 1,
                        scene_label=label,
                        mapped_command=game_command or "STOP",
                        transport_command=self._last_game_transport_command or "",
                        transport_success=(
                            self._last_game_transport_error is None
                            and self._last_game_transport_sent_at > 0.0
                        ),
                        transport_sent_at_monotonic=self._last_game_transport_sent_at,
                        transport_error=self._last_game_transport_error or "",
                        quality_accepted=preprocessing.quality.accepted,
                        quality_peak_abs_uv=preprocessing.quality.peak_abs_uv,
                        quality_clip_fraction=preprocessing.quality.clip_fraction,
                        quality_bad_channel_fraction=(
                            preprocessing.quality.bad_channel_fraction
                        ),
                        quality_reasons=preprocessing.quality.reasons,
                        quality_bad_channel_indices=(
                            preprocessing.quality.bad_channel_indices
                        ),
                        quality_nonfinite_fraction=(
                            preprocessing.quality.nonfinite_fraction
                        ),
                        timing_queueing_jitter_sec=float(
                            timing_diagnostics.get("queueing_jitter_sec", 0.0)
                        ),
                        timing_transport_delay_compensation_sec=float(
                            timing_diagnostics.get(
                                "transport_delay_compensation_sec",
                                0.0,
                            )
                        ),
                        timing_packet_arrival_monotonic=float(
                            timing_diagnostics.get(
                                "packet_arrival_monotonic",
                                float("nan"),
                            )
                        ),
                        timing_received_packets=float(
                            timing_diagnostics.get("received_packets", 0.0)
                        ),
                        timing_packet_loss_count=float(
                            timing_diagnostics.get("packet_loss_count", 0.0)
                        ),
                        timing_total_source_samples=float(
                            timing_diagnostics.get("total_source_samples", 0.0)
                        ),
                    )

                    true_labels.append(label)
                    pred_labels.append(pred_class)
                    confidences.append(float(result.confidence))
                    quality_accepted.append(preprocessing.quality.accepted)
                    if heartbeat is not None:
                        heartbeat()
                    update_stage(
                        f"测试 {cue_index}: 预测控制",
                        min(time.monotonic() - control_started, control_sec),
                        control_sec,
                    )
                    elapsed = time.perf_counter() - loop_started
                    self._sleep_with_heartbeat(max(0.0, self._step_sec - elapsed), heartbeat)
        except KeyboardInterrupt:
            run_status = "interrupted"
            self._console.print("\n[bold red]停止测试模式[/bold red]")
        except Exception as exc:
            run_status = "failed"
            run_error = str(exc)
            raise
        finally:
            self.stop()
            writer.append_event("session_stop", status=run_status, error=run_error)
            writer.stop()
            if run_status == "failed":
                writer.finalize_manifest({"status": run_status, "error": run_error})
            if heartbeat is not None:
                heartbeat()

        if not true_labels:
            writer.finalize_manifest(
                {"status": "no_windows", "error": "No EEG windows were collected."}
            )
            raise RuntimeError("Test mode did not collect any EEG windows.")

        y_true = np.asarray(true_labels, dtype=np.int64)
        y_pred = np.asarray(pred_labels, dtype=np.int64)
        pred_valid = y_pred >= 0
        quality_mask = np.asarray(quality_accepted, dtype=np.bool_)
        accuracy = float(np.mean(y_pred == y_true))
        valid_accuracy = float(np.mean(y_pred[pred_valid] == y_true[pred_valid])) if np.any(pred_valid) else 0.0
        quality_prediction_mask = quality_mask & pred_valid
        quality_accuracy = (
            float(
                np.mean(
                    y_pred[quality_prediction_mask]
                    == y_true[quality_prediction_mask]
                )
            )
            if np.any(quality_prediction_mask)
            else 0.0
        )
        
        writer.finalize_manifest({
            "status": run_status,
            "accuracy": accuracy,
            "valid_accuracy": valid_accuracy,
            "quality_accuracy": quality_accuracy,
            "quality_accepted_windows": int(np.sum(quality_mask)),
            "quality_rejected_windows": int(np.sum(~quality_mask)),
            "online_adaptation": self._online_adaptation_status(),
        })
        self._console.print(f"[bold green]测试数据已保存[/bold green] {root_dir}")

        return {
            "windows": len(true_labels),
            "accuracy": accuracy,
            "valid_accuracy": valid_accuracy,
            "quality_accuracy": quality_accuracy,
            "quality_accepted_windows": int(np.sum(quality_mask)),
        }

    def _decode_loop(self) -> None:
        while not self._stop_event.is_set():
            started_at = time.perf_counter()
            try:
                try:
                    continuous, history_timestamps = self._acquire_preprocessed_history(
                        self._window_sec
                    )
                except RuntimeError as exc:
                    if "Not enough data" in str(exc):
                        self._sleep_with_heartbeat(self._step_sec, None)
                        continue
                    raise
                window, timestamps, preprocessing = self._slice_preprocessed_history(
                    continuous,
                    history_timestamps,
                )
                window_start, window_end = self._resolve_window_time_bounds(timestamps)
                if (
                    self._last_processed_window_end_monotonic is not None
                    and window_end <= self._last_processed_window_end_monotonic
                ):
                    self._stale_source_windows_rejected += 1
                    writer = getattr(self, "_writer", None)
                    if writer is not None:
                        writer.append_event(
                            "source_window_rejected",
                            timestamp_monotonic=time.monotonic(),
                            reason="duplicate_or_non_increasing_window_end",
                            window_start_monotonic=window_start,
                            window_end_monotonic=window_end,
                            previous_window_end_monotonic=(
                                self._last_processed_window_end_monotonic
                            ),
                        )
                    self._sleep_with_heartbeat(self._step_sec, None)
                    continue
                self._last_processed_window_end_monotonic = window_end
                self._sync_game_scene()
                aligned = None
                alignment_target_end = self._primary_alignment_target_end()
                if (
                    alignment_target_end is not None
                    and window_end >= alignment_target_end
                ):
                    aligned = self._select_aligned_primary_window(
                        continuous.raw_data,
                        history_timestamps,
                    )
                if aligned is not None:
                    window, timestamps = aligned
                    start_index = int(
                        np.searchsorted(
                            history_timestamps,
                            float(timestamps[0]),
                            side="left",
                        )
                    )
                    window, timestamps, preprocessing = self._slice_preprocessed_history(
                        continuous,
                        history_timestamps,
                        start_index=start_index,
                    )
                    window_start, window_end = self._resolve_window_time_bounds(
                        timestamps
                    )
                processed = preprocessing.data
                probability_batch, model_revision = self._predict_proba_with_revision(
                    processed[None, ...],
                    mc_dropout_passes=self._mc_dropout_passes,
                )
                probabilities = probability_batch[0]
                raw_prediction = int(np.argmax(probabilities))
                result = self._post_process(probabilities)
                self._console.print(
                    f"[green][预测][/green] {result.label} "
                    f"(confidence: {result.confidence:.2f}, uncertainty: {result.uncertainty:.2f})"
                )
                self._command_outlet.push(result.label)

                online_label = self._get_online_label(
                    window_start=window_start,
                    window_end=window_end,
                )
                primary_decision_slot = self._claim_primary_decision_window(
                    online_label=online_label,
                    window_start=window_start,
                    window_end=window_end,
                )
                primary_decision = primary_decision_slot is not None
                scene_index = int(
                    (getattr(online_label, "payload", None) or {}).get(
                        "scene_index",
                        -1,
                    )
                )
                if primary_decision and preprocessing.quality.accepted:
                    self._primary_decision_probabilities.setdefault(
                        scene_index,
                        [],
                    ).append(np.asarray(probabilities, dtype=np.float64).copy())
                control_result = result
                if (
                    primary_decision
                    and scene_index in self._primary_decision_scenes
                ):
                    control_result = self._aggregate_primary_control_result(
                        scene_index
                    )
                control_gate_active = self._is_cued_control_gate_active()
                game_command = self._game_command_for_window(
                    control_result,
                    primary_decision=primary_decision,
                    control_gate_active=control_gate_active,
                )
                self._push_game_command(game_command)
                self._emit_status(
                    control_result,
                    None if control_gate_active else game_command,
                )
                if primary_decision:
                    lateral_control_released = (
                        scene_index in self._primary_decision_scenes
                    )
                    control_released_at = float("nan")
                    if lateral_control_released:
                        control_released_at = time.monotonic()
                        self._control_released_at[scene_index] = control_released_at
                    scene_started_at = getattr(self, "_scene_started_at", {}).get(
                        scene_index,
                    )
                    cue_metadata = self._online_label_source_metadata() or {}
                    boundary_guard = max(
                        float(cue_metadata.get("boundary_guard_sec", 0.5)),
                        0.0,
                    )
                    target_window_start = (
                        float(scene_started_at) + boundary_guard
                        + (int(primary_decision_slot) - 1)
                        * self._primary_window_spacing_sec
                        if scene_started_at is not None
                        else float("nan")
                    )
                    target_window_end = target_window_start + self._window_sec
                    writer = getattr(self, "_writer", None)
                    if writer is not None:
                        writer.append_event(
                            "primary_decision_window",
                            timestamp_monotonic=window_end,
                            scene_index=scene_index,
                            scene_number=scene_index + 1,
                            window_index=int(primary_decision_slot),
                            windows_required=self._primary_windows_per_scene,
                            window_start_monotonic=window_start,
                            window_end_monotonic=window_end,
                            target_window_start_monotonic=target_window_start,
                            target_window_end_monotonic=target_window_end,
                            window_start_alignment_error_sec=(
                                window_start - target_window_start
                            ),
                            window_end_alignment_error_sec=(
                                window_end - target_window_end
                            ),
                            control_released_at_monotonic=control_released_at,
                            decision_pipeline_latency_sec=(
                                control_released_at - window_end
                                if lateral_control_released
                                else float("nan")
                            ),
                            instruction_label_id=int(online_label.label_id),
                            raw_prediction=raw_prediction,
                            operational_prediction=(
                                -1 if result.class_id is None else int(result.class_id)
                            ),
                            confidence=float(result.confidence),
                            quality_accepted=bool(preprocessing.quality.accepted),
                            lateral_control_released=lateral_control_released,
                            ensemble_window_count=len(
                                self._primary_decision_probabilities.get(
                                    scene_index,
                                    [],
                                )
                            ),
                            ensemble_prediction=(
                                -1
                                if not lateral_control_released
                                or control_result.class_id is None
                                else int(control_result.class_id)
                            ),
                            ensemble_confidence=(
                                float(control_result.confidence)
                                if lateral_control_released
                                else float("nan")
                            ),
                        )
                record_payload = None
                if hasattr(self, "_record") and self._record and hasattr(self, "_writer"):
                    pred_class = -1 if result.class_id is None else int(result.class_id)
                    timing_diagnostics = getattr(
                        self._acquirer,
                        "timing_diagnostics",
                        {},
                    ) or {}
                    label_payload = (
                        getattr(online_label, "payload", None) or {}
                        if online_label is not None
                        else {}
                    )
                    scene_index = int(
                        label_payload.get(
                            "scene_index",
                            getattr(self, "_scene_sent_scene_index", -1),
                        )
                    )
                    record_payload = {
                        "window": window.astype(np.float32),
                        "y_pred": pred_class,
                        "confidence": float(result.confidence),
                        "raw_pred": raw_prediction,
                        "model_revision": model_revision,
                        "probabilities": probabilities,
                        "uncertainty": float(result.uncertainty),
                        "window_start_monotonic": window_start,
                        "window_end_monotonic": window_end,
                        "scene_index": scene_index,
                        "scene_start_lane": getattr(
                            self,
                            "_scene_start_lanes",
                            {},
                        ).get(scene_index, -9),
                        "scene_safe_lane": getattr(
                            self,
                            "_scene_safe_lanes",
                            {},
                        ).get(scene_index, -9),
                        "scene_current_lane": _integer_or(
                            label_payload.get(
                                "current_lane",
                                (
                                    self._online_label_source_status() or {}
                                ).get("current_lane", -9),
                            ),
                            -9,
                        ),
                        "instruction_label": getattr(
                            self,
                            "_scene_labels",
                            {},
                        ).get(scene_index, -1),
                        "vehicle_required_action": (
                            -1
                            if online_label is None
                            else int(online_label.label_id)
                        ),
                        "scene_failed": scene_index in self._failed_scene_indices,
                        "training_role": (
                            "primary_decision"
                            if primary_decision
                            else "continuous_context"
                            if online_label is not None
                            else "unlabeled"
                        ),
                        "adaptation_eligible": bool(primary_decision),
                        "control_gate_active": control_gate_active,
                        "mapped_command": game_command or "STOP",
                        "transport_command": self._last_game_transport_command or "",
                        "transport_success": (
                            self._last_game_transport_error is None
                            and self._last_game_transport_sent_at > 0.0
                        ),
                        "transport_sent_at_monotonic": self._last_game_transport_sent_at,
                        "transport_error": self._last_game_transport_error or "",
                        "quality_accepted": preprocessing.quality.accepted,
                        "quality_peak_abs_uv": preprocessing.quality.peak_abs_uv,
                        "quality_clip_fraction": preprocessing.quality.clip_fraction,
                        "quality_bad_channel_fraction": (
                            preprocessing.quality.bad_channel_fraction
                        ),
                        "quality_reasons": preprocessing.quality.reasons,
                        "quality_bad_channel_indices": (
                            preprocessing.quality.bad_channel_indices
                        ),
                        "quality_nonfinite_fraction": (
                            preprocessing.quality.nonfinite_fraction
                        ),
                        "timing_queueing_jitter_sec": float(
                            timing_diagnostics.get("queueing_jitter_sec", 0.0)
                        ),
                        "timing_transport_delay_compensation_sec": float(
                            timing_diagnostics.get(
                                "transport_delay_compensation_sec",
                                0.0,
                            )
                        ),
                        "timing_packet_arrival_monotonic": float(
                            timing_diagnostics.get(
                                "packet_arrival_monotonic",
                                float("nan"),
                            )
                        ),
                        "timing_received_packets": float(
                            timing_diagnostics.get("received_packets", 0.0)
                        ),
                        "timing_packet_loss_count": float(
                            timing_diagnostics.get("packet_loss_count", 0.0)
                        ),
                        "timing_total_source_samples": float(
                            timing_diagnostics.get("total_source_samples", 0.0)
                        ),
                    }

                if (
                    online_label is not None
                    and str(getattr(online_label, "source", "")) == "cued-protocol"
                    and self._lane_transition_guard_sec > 0.0
                    and not primary_decision
                ):
                    self._pending_cued_windows.append(
                        _PendingCuedWindow(
                            processed=processed.copy(),
                            probabilities=np.asarray(
                                probabilities,
                                dtype=np.float32,
                            ).copy(),
                            operational_prediction=result.class_id,
                            prediction_model_revision=model_revision,
                            online_label=online_label,
                            window_start=window_start,
                            window_end=window_end,
                            quality_accepted=bool(
                                preprocessing.quality.accepted
                            ),
                            training_role="continuous_context",
                            adaptation_eligible=False,
                            record_payload=record_payload,
                        )
                    )
                else:
                    self._finalize_realtime_window(
                        processed=processed,
                        probabilities=probabilities,
                        operational_prediction=result.class_id,
                        prediction_model_revision=model_revision,
                        online_label=online_label,
                        window_end=window_end,
                        quality_accepted=bool(preprocessing.quality.accepted),
                        training_role=(
                            "primary_decision"
                            if primary_decision
                            else "continuous_context"
                            if online_label is not None
                            else "unlabeled"
                        ),
                        adaptation_eligible=bool(primary_decision),
                        record_payload=record_payload,
                    )
                self._flush_pending_cued_windows()
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Realtime decoding failed")
                self._console.print(f"[red]解码失败：{exc}[/red]")

            elapsed = time.perf_counter() - started_at
            sleep_time = max(0.0, self._step_sec - elapsed)
            alignment_due = self._primary_alignment_target_end()
            if alignment_due is not None:
                until_due = alignment_due - time.monotonic()
                if until_due > 0.0:
                    sleep_time = min(sleep_time, until_due)
                else:
                    # Poll briefly until source-timestamped EEG through the exact
                    # target end has arrived instead of waiting another full step.
                    sleep_time = min(sleep_time, 0.05)
            self._sleep_with_heartbeat(sleep_time, None)

    def _acquire_preprocessed_history(
        self,
        min_history_sec: float,
    ) -> tuple[ContinuousPreprocessingResult, np.ndarray]:
        """Continuously transform retained source history before windowing."""

        continuous_getter = getattr(self._acquirer, "get_continuous_chunk", None)
        if callable(continuous_getter):
            source, source_timestamps = continuous_getter(min_history_sec)
        else:
            source, source_timestamps = self._acquirer.get_chunk(min_history_sec)
        source_array = np.asarray(source, dtype=np.float32)
        source_times = np.asarray(source_timestamps, dtype=np.float64).reshape(-1)
        if source_array.ndim != 2:
            raise RuntimeError(
                f"Unexpected continuous EEG shape: {source_array.shape}"
            )
        if source_times.size != source_array.shape[1]:
            raise RuntimeError(
                "Continuous EEG timestamp count does not match its sample count."
            )
        source_sfreq = float(
            getattr(
                self._acquirer,
                "continuous_sfreq",
                self._acquirer.metadata.sfreq,
            )
        )
        continuous = preprocess_eeg_continuous(
            source_array,
            source_sfreq=source_sfreq,
            target_sfreq=self._sfreq,
        )
        target_count = int(continuous.data.shape[1])
        required_target = int(round(float(min_history_sec) * self._sfreq))
        if target_count < required_target:
            raise RuntimeError(
                f"Not enough data after continuous preprocessing: "
                f"{target_count} < {required_target}"
            )

        if source_times.size and np.all(np.isfinite(source_times)):
            history_end = float(source_times[-1]) + (1.0 / source_sfreq)
            target_timestamps = history_end - (
                np.arange(target_count, 0, -1, dtype=np.float64) / self._sfreq
            )
        else:
            target_timestamps = np.arange(target_count, dtype=np.float64) / self._sfreq
        return continuous, target_timestamps

    def _slice_preprocessed_history(
        self,
        continuous: ContinuousPreprocessingResult,
        history_timestamps: np.ndarray,
        *,
        start_index: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, PreprocessingResult]:
        """Cut one fixed model window and apply deferred quality clipping."""

        sample_count = int(round(self._window_sec * self._sfreq))
        total_samples = int(continuous.data.shape[1])
        start = total_samples - sample_count if start_index is None else int(start_index)
        stop = start + sample_count
        timestamps = np.asarray(history_timestamps, dtype=np.float64).reshape(-1)
        if start < 0 or stop > total_samples or timestamps.size != total_samples:
            raise RuntimeError(
                f"Cannot cut {sample_count} samples from continuous history "
                f"with {total_samples} samples at start={start}."
            )

        source_start = int(
            round(start * continuous.source_sfreq / continuous.target_sfreq)
        )
        source_stop = int(
            round(stop * continuous.source_sfreq / continuous.target_sfreq)
        )
        source_start = max(source_start, 0)
        source_stop = min(
            max(source_stop, source_start + 1),
            continuous.source_nonfinite_mask.shape[1],
        )
        nonfinite_fraction = float(
            np.mean(
                continuous.source_nonfinite_mask[:, source_start:source_stop]
            )
        )
        result = finalize_preprocessed_window(
            continuous.data[:, start:stop],
            bad_channel_indices=continuous.bad_channel_indices,
            nonfinite_fraction=nonfinite_fraction,
        )
        return (
            continuous.raw_data[:, start:stop].copy(),
            timestamps[start:stop].copy(),
            result,
        )

    def _uses_cued_primary_alignment(self) -> bool:
        status = self._online_label_source_status()
        return bool(status and status.get("source") == "cued-protocol")

    def _primary_alignment_target_end(self) -> float | None:
        target = self._primary_alignment_target()
        return None if target is None else target[2]

    def _primary_alignment_target(
        self,
    ) -> tuple[int, float, float] | None:
        status = self._online_label_source_status()
        if not status or status.get("source") != "cued-protocol":
            return None
        scene_index = int(status.get("scene_index", -1))
        if (
            scene_index < 0
            or scene_index != getattr(self, "_scene_sent_scene_index", -1)
            or scene_index in getattr(self, "_primary_decision_scenes", set())
        ):
            return None
        scene_started_at = getattr(self, "_scene_started_at", {}).get(scene_index)
        if scene_started_at is None:
            return None
        bounds = getattr(self, "_primary_decision_window_bounds", {}).get(
            scene_index,
            [],
        )
        next_window_index = len(bounds)
        windows_required = max(
            int(getattr(self, "_primary_windows_per_scene", 1)),
            1,
        )
        if next_window_index >= windows_required:
            return None
        metadata = self._online_label_source_metadata() or {}
        boundary_guard = max(float(metadata.get("boundary_guard_sec", 0.5)), 0.0)
        spacing = max(
            float(getattr(self, "_primary_window_spacing_sec", self._step_sec)),
            self._step_sec,
        )
        target_start = (
            float(scene_started_at)
            + boundary_guard
            + next_window_index * spacing
        )
        return scene_index, target_start, target_start + self._window_sec

    def _select_aligned_primary_window(
        self,
        history_window: np.ndarray,
        history_timestamps: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Select the fixed current-Scene window from timestamped EEG history."""

        if not self._uses_cued_primary_alignment():
            return None
        domain = str(
            getattr(self._acquirer.metadata, "timestamp_domain", "relative")
        ).strip().lower()
        if domain != "monotonic":
            return None
        target = self._primary_alignment_target()
        if target is None:
            return None
        _, target_start, target_end = target
        timestamps = np.asarray(history_timestamps, dtype=np.float64).reshape(-1)
        data = np.asarray(history_window, dtype=np.float32)
        if (
            data.ndim != 2
            or timestamps.size != data.shape[-1]
            or timestamps.size < 2
            or not np.all(np.isfinite(timestamps))
            or not np.all(np.diff(timestamps) > 0.0)
        ):
            return None
        sample_period = 1.0 / float(self._sfreq)
        history_end = float(timestamps[-1]) + sample_period
        if history_end < target_end:
            return None
        sample_count = int(round(self._window_sec * self._sfreq))
        start_index = int(np.searchsorted(timestamps, target_start, side="left"))
        stop_index = start_index + sample_count
        if start_index < 0 or stop_index > timestamps.size:
            return None
        selected_timestamps = timestamps[start_index:stop_index]
        selected_start = float(selected_timestamps[0])
        selected_end = float(selected_timestamps[-1]) + sample_period
        tolerance = max(2.0 * sample_period, 0.01)
        if (
            selected_start < target_start - tolerance
            or abs(selected_end - target_end) > tolerance
        ):
            return None
        return (
            data[:, start_index:stop_index].copy(),
            selected_timestamps.copy(),
        )

    def _flush_pending_cued_windows(
        self,
        *,
        now: float | None = None,
        force: bool = False,
    ) -> None:
        """Finalize delayed labels after future lane-transition events are known."""

        pending = list(getattr(self, "_pending_cued_windows", []))
        if not pending:
            return
        timestamp = time.monotonic() if now is None else float(now)
        guard = max(float(getattr(self, "_lane_transition_guard_sec", 0.0)), 0.0)
        source = getattr(self, "_online_label_source", None)
        is_guarded = getattr(source, "is_window_transition_guarded", None)
        remaining: list[_PendingCuedWindow] = []
        for item in pending:
            matured = timestamp >= item.window_end + guard
            if not matured and not force:
                remaining.append(item)
                continue

            label_payload = getattr(item.online_label, "payload", None) or {}
            scene_index = int(label_payload.get("scene_index", -1))
            transition_guarded = bool(
                callable(is_guarded)
                and is_guarded(
                    scene_index=scene_index,
                    window_start=item.window_start,
                    window_end=item.window_end,
                )
            )
            shutdown_unconfirmed = bool(force and not matured)
            final_label = (
                None
                if transition_guarded or shutdown_unconfirmed
                else item.online_label
            )
            self._finalize_realtime_window(
                processed=item.processed,
                probabilities=item.probabilities,
                operational_prediction=item.operational_prediction,
                prediction_model_revision=item.prediction_model_revision,
                online_label=final_label,
                window_end=item.window_end,
                quality_accepted=item.quality_accepted,
                training_role=item.training_role,
                adaptation_eligible=item.adaptation_eligible,
                record_payload=item.record_payload,
            )
            writer = getattr(self, "_writer", None)
            if writer is not None and (transition_guarded or shutdown_unconfirmed):
                writer.append_event(
                    "training_label_rejected",
                    timestamp_monotonic=timestamp,
                    scene_index=scene_index,
                    window_start_monotonic=item.window_start,
                    window_end_monotonic=item.window_end,
                    original_label_id=int(item.online_label.label_id),
                    reason=(
                        "lane_transition_guard"
                        if transition_guarded
                        else "session_stopped_before_guard_confirmation"
                    ),
                    lane_transition_guard_sec=guard,
                )
        self._pending_cued_windows = remaining

    def _finalize_realtime_window(
        self,
        *,
        processed: np.ndarray,
        probabilities: np.ndarray,
        operational_prediction: int | None,
        prediction_model_revision: int,
        online_label: Any | None,
        window_end: float,
        quality_accepted: bool,
        training_role: str,
        adaptation_eligible: bool,
        record_payload: dict[str, Any] | None,
    ) -> None:
        """Commit one transition-safe window to adaptation and recording."""

        adaptation_committed = False
        label_in_task = bool(
            online_label is not None
            and 0 <= int(online_label.label_id) < self._n_classes
        )
        if label_in_task and quality_accepted and adaptation_eligible:
            adaptation_committed = self._handle_online_label(
                processed=processed,
                probabilities=probabilities,
                operational_prediction=operational_prediction,
                prediction_model_revision=prediction_model_revision,
                online_label=online_label,
                window_end=window_end,
            )
        if record_payload is None:
            return
        record_payload["training_role"] = str(training_role)
        record_payload["adaptation_eligible"] = bool(
            label_in_task and adaptation_eligible
        )
        record_payload["adaptation_committed"] = bool(adaptation_committed)
        self._writer.put(
            y_true=-1 if online_label is None else int(online_label.label_id),
            label_event_id=(
                "" if online_label is None else str(online_label.event_id)
            ),
            scene_label=-1 if online_label is None else int(online_label.label_id),
            **record_payload,
        )

    def _resolve_window_time_bounds(self, timestamps: np.ndarray) -> tuple[float, float]:
        """Resolve an EEG window on the same monotonic clock used by Unity."""

        domain = str(
            getattr(self._acquirer.metadata, "timestamp_domain", "relative")
        ).strip().lower()
        values = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if domain == "monotonic" and values.size:
            window_start = float(values[0])
            window_end = float(values[-1]) + (1.0 / float(self._sfreq))
            duration = window_end - window_start
            now = time.monotonic()
            if (
                np.all(np.isfinite(values))
                and window_end >= window_start
                and np.all(np.diff(values) >= 0.0)
                and abs(duration - self._window_sec) <= max(2.0 / self._sfreq, 0.02)
                and window_end <= now + max(2.0 / self._sfreq, 0.02)
            ):
                return window_start, window_end

        if domain == "monotonic" and not self._timestamp_fallback_warned:
            LOGGER.warning(
                "Acquirer supplied invalid monotonic timestamps; falling back to local retrieval time."
            )
            self._timestamp_fallback_warned = True
        window_end = time.monotonic()
        return window_end - self._window_sec, window_end

    def _handle_online_label(
        self,
        *,
        processed: np.ndarray,
        probabilities: np.ndarray,
        operational_prediction: int | None,
        prediction_model_revision: int,
        online_label: Any,
        window_end: float,
    ) -> bool:
        """Route one labeled window to the configured adaptation strategy."""

        label_id = int(online_label.label_id)
        event_id = str(getattr(online_label, "event_id", ""))
        if str(getattr(online_label, "source", "")) == "cued-protocol":
            event_id = (
                f"{event_id}-window-end-"
                f"{int(round(float(window_end) * 1_000_000.0))}"
            )
        if self._neuroonline_adapter is not None:
            accepted = self._neuroonline_adapter.add_window(
                processed,
                label_id,
                predicted_label=int(np.argmax(probabilities)),
                operational_predicted_label=operational_prediction,
                probabilities=probabilities,
                event_id=event_id,
                model_revision=prediction_model_revision,
                window_end_monotonic=window_end,
            )
            status = self._neuroonline_adapter.status()
            if status.get("training_in_background") and not self._neuroonline_training_notice:
                self._neuroonline_training_notice = True
                self._console.print(
                    "[bold cyan]NeuroOnline 已在后台训练候选模型，实时预测继续使用当前模型[/bold cyan]"
                )
            return bool(accepted)

        return True

    def _get_online_label(
        self,
        *,
        window_start: float,
        window_end: float,
    ) -> Any | None:
        if self._online_label_source is None:
            return None
        try:
            label = self._online_label_source.get_label(
                window_start=window_start,
                window_end=window_end,
            )
            if label is None or str(getattr(label, "source", "")) != "cued-protocol":
                return label
            payload = getattr(label, "payload", None) or {}
            label_scene_index = int(payload.get("scene_index", -1))
            if label_scene_index != getattr(self, "_scene_sent_scene_index", -1):
                return None
            return label
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to read online label: %s", exc)
            return None

    def _claim_primary_decision_window(
        self,
        *,
        online_label: Any | None,
        window_start: float,
        window_end: float,
    ) -> int | None:
        """Claim one of the fixed, causally clean training windows in a Scene."""

        if (
            online_label is None
            or str(getattr(online_label, "source", "")) != "cued-protocol"
        ):
            return None
        payload = getattr(online_label, "payload", None) or {}
        scene_index = int(payload.get("scene_index", -1))
        segment_index = int(payload.get("segment_index", -1))
        if (
            scene_index < 0
            or segment_index != 0
            or scene_index != getattr(self, "_scene_sent_scene_index", -1)
            or scene_index in getattr(self, "_primary_decision_scenes", set())
        ):
            return None
        if not hasattr(self, "_primary_decision_scenes"):
            self._primary_decision_scenes = set()
        if not hasattr(self, "_primary_decision_window_bounds"):
            self._primary_decision_window_bounds = {}
        if not hasattr(self, "_primary_windows_per_scene"):
            self._primary_windows_per_scene = 1
        if not hasattr(self, "_primary_window_spacing_sec"):
            self._primary_window_spacing_sec = 1.0
        bounds = self._primary_decision_window_bounds.setdefault(scene_index, [])
        if bounds:
            next_start = bounds[-1][0] + self._primary_window_spacing_sec
            tolerance = max(self._step_sec * 0.25, 1.0 / self._sfreq)
            if float(window_start) < next_start - tolerance:
                return None
        bounds.append((
            float(window_start),
            float(window_end),
        ))
        window_index = len(bounds)
        if window_index >= self._primary_windows_per_scene:
            self._primary_decision_scenes.add(scene_index)
        return window_index

    def _aggregate_primary_control_result(
        self,
        scene_index: int,
    ) -> PredictionResult:
        """Average quality-accepted primary-window probabilities for control."""

        probability_rows = getattr(
            self,
            "_primary_decision_probabilities",
            {},
        ).get(int(scene_index), [])
        if not probability_rows:
            return PredictionResult(
                label="不确定",
                confidence=0.0,
                uncertainty=1.0,
                class_id=None,
            )
        mean_probabilities = np.mean(
            np.stack(probability_rows, axis=0),
            axis=0,
        )
        return self._post_process(mean_probabilities)

    def _is_cued_control_gate_active(self) -> bool:
        """Hold lateral Unity commands until the current Scene has clean EEG."""

        status = self._online_label_source_status()
        if not status or status.get("source") != "cued-protocol":
            return False
        if status.get("phase") == "preparing":
            return True
        scene_index = int(status.get("scene_index", -1))
        return scene_index not in getattr(self, "_primary_decision_scenes", set())

    def _predict_proba(self, X: np.ndarray, *, mc_dropout_passes: int) -> np.ndarray:
        with self._model_lock:
            return self._model.predict_proba(X, mc_dropout_passes=mc_dropout_passes)

    def _predict_proba_with_revision(
        self,
        X: np.ndarray,
        *,
        mc_dropout_passes: int,
    ) -> tuple[np.ndarray, int]:
        with self._model_lock:
            probabilities = self._model.predict_proba(X, mc_dropout_passes=mc_dropout_passes)
            return probabilities, self._model_revision

    def _run_neuroonline_update(
        self,
        original: np.ndarray,
        time_masked: np.ndarray,
        frequency_masked: np.ndarray,
        labels: np.ndarray,
    ) -> dict[str, Any]:
        update_started_monotonic = time.monotonic()
        with self._model_lock:
            if not isinstance(self._model, NeuroOnlineModelAdapter):
                raise RuntimeError("NeuroOnline model adapter is not active")
            base_model_revision = self._model_revision
            candidate = copy.deepcopy(self._model)
        writer = getattr(self, "_writer", None)
        if writer is not None:
            writer.append_event(
                "model_update_start",
                timestamp_monotonic=update_started_monotonic,
                base_model_revision=base_model_revision,
                training_samples=int(labels.shape[0]),
                class_counts=np.bincount(
                    np.asarray(labels, dtype=np.int64),
                    minlength=self._n_classes,
                ).tolist(),
            )
        result = candidate.neuroonline_update(
            original,
            time_masked,
            frequency_masked,
            labels,
        )
        with self._model_lock:
            self._model = candidate
            self._model_revision += 1
            result["model_revision"] = self._model_revision
        result["base_model_revision"] = base_model_revision
        result["swap_timestamp_monotonic"] = time.monotonic()
        if writer is not None:
            writer.append_event(
                "model_swap",
                timestamp_monotonic=float(result["swap_timestamp_monotonic"]),
                base_model_revision=base_model_revision,
                model_revision=self._model_revision,
                training_samples=int(labels.shape[0]),
            )
        return result

    def _on_neuroonline_update_complete(self, result: dict[str, Any]) -> None:
        self._neuroonline_training_notice = False
        if result.get("error"):
            self._console.print(f"[bold red]NeuroOnline 后台更新失败[/bold red] {result['error']}")
        else:
            self._console.print(
                "[bold green]NeuroOnline 候选模型已原子切换[/bold green] "
                f"revision={int(result.get('model_revision', self._model_revision))} "
                f"loss={float(result.get('loss', 0.0)):.4f}"
            )
        writer = getattr(self, "_writer", None)
        if writer is not None:
            writer.append_event(
                "model_update_complete",
                model_revision=int(result.get("model_revision", self._model_revision)),
                **{
                    key: value
                    for key, value in result.items()
                    if key not in {"model_revision"}
                },
            )
        self._persist_online_adaptation_status()

    def _save_current_model(self) -> None:
        if self._model_save_path is None:
            return
        with self._model_lock:
            model_snapshot = copy.deepcopy(self._model)
            revision = getattr(self, "_model_revision", 0)
        self._model_save_path.parent.mkdir(parents=True, exist_ok=True)
        model_snapshot.save(self._model_save_path)
        self._snapshot_model_revision(
            revision,
            source="online_update",
            model_snapshot=model_snapshot,
        )

    def save_current_model(self) -> None:
        """Persist the decoder revision that is currently active under the model lock."""

        self._save_current_model()

    def _persist_online_adaptation_status(self) -> None:
        writer = getattr(self, "_writer", None)
        if writer is None:
            return
        writer.update_manifest(
            {
                "model_revision": self._model_revision,
                "online_adaptation": self._online_adaptation_status(),
                "online_label_source": self._online_label_source_metadata(),
            }
        )

    @staticmethod
    def _sleep_with_heartbeat(duration_sec: float, heartbeat: Callable[[], None] | None) -> None:
        remaining = max(float(duration_sec), 0.0)
        while remaining > 0:
            chunk = min(0.1, remaining)
            time.sleep(chunk)
            remaining -= chunk
            if heartbeat is not None:
                heartbeat()

    def _sleep_with_stage_progress(
        self,
        duration_sec: float,
        *,
        heartbeat: Callable[[], None] | None,
        stage_name: str,
        stage_progress: Callable[[str, float, float], None] | None,
    ) -> None:
        total = max(float(duration_sec), 0.0)
        started_at = time.monotonic()
        remaining = total
        while remaining > 0:
            chunk = min(0.1, remaining)
            time.sleep(chunk)
            remaining -= chunk
            elapsed = min(time.monotonic() - started_at, total)
            if stage_progress is not None:
                stage_progress(stage_name, elapsed, total)
            if heartbeat is not None:
                heartbeat()

    def _post_process(self, probabilities: np.ndarray) -> PredictionResult:
        best_index = int(np.argmax(probabilities))
        confidence = float(probabilities[best_index])
        uncertainty = float(1.0 - confidence)
        if confidence < self._confidence_threshold:
            return PredictionResult(
                label="不确定",
                confidence=confidence,
                uncertainty=uncertainty,
                class_id=None,
            )
        return PredictionResult(
            label=LABEL_NAMES.get(best_index, f"class-{best_index}"),
            confidence=confidence,
            uncertainty=uncertainty,
            class_id=best_index,
        )

    @staticmethod
    def _to_game_command(result: PredictionResult) -> str | None:
        if result.class_id == 0:
            return "LEFT"
        if result.class_id == 1:
            return "RIGHT"
        return None

    def _game_command_for_window(
        self,
        result: PredictionResult,
        *,
        primary_decision: bool,
        control_gate_active: bool,
    ) -> str | None:
        if control_gate_active:
            return None
        if self._uses_cued_primary_alignment() and not primary_decision:
            # Later windows are telemetry only; STOP does not change the lane
            # target selected by the single primary decision.
            return None
        return self._to_game_command(result)

    def _sync_game_scene(self) -> None:
        """Negotiate a reachable relative-action scene with authoritative Unity truth."""

        self._consume_game_scene_events()
        status = self._online_label_source_status()
        if not status or status.get("source") != "cued-protocol":
            return
        if status.get("protocol_mode") != "centered-single-decision":
            return
        if status.get("phase") == "preparing":
            return
        scene_index = int(status.get("scene_index", -1))
        label_value = status.get("label_id")
        label_id = -1 if label_value is None else int(label_value)
        # A Unity obstacle layout is immutable for the full Scene. LANE_SETTLED
        # may end the dynamic EEG truth after the requested lateral action, but
        # must never send another SCENE_* command or create another obstacle wall.
        if scene_index == getattr(self, "_scene_sent_scene_index", -1):
            return
        previous_scene = getattr(self, "_scene_sent_scene_index", -1)

        if label_id < 0:
            state_ack = self._push_game_scene_transport_command("SCENE_STATE")
            if state_ack is None:
                self._scene_sync_error = (
                    self._last_game_transport_error
                    or "Unity lane-state query failed"
                )
                return
            try:
                self._validate_unity_protocol_ack(
                    state_ack,
                    expected_ack="SCENE_STATE",
                    expected_scene_number=None,
                )
                unity_scene_number = int(state_ack["scene_number"])
                if unity_scene_number < 1:
                    raise ValueError(
                        f"Unity returned invalid next scene number {unity_scene_number}"
                    )
                if getattr(self, "_unity_scene_number_offset", None) is None:
                    # Unity deliberately keeps its counter while the game window stays
                    # open. Anchor this Python run to the first authoritative state ACK,
                    # then validate every following ACK/event against the fixed offset.
                    self._unity_scene_number_offset = (
                        unity_scene_number - (scene_index + 1)
                    )
                expected_unity_scene_number = self._unity_scene_number(scene_index)
                if unity_scene_number != expected_unity_scene_number:
                    raise ValueError(
                        f"expected Unity scene {expected_unity_scene_number}, "
                        f"received {unity_scene_number}"
                    )
                start_lane = int(state_ack.get("next_scene_start_lane", 0))
                if start_lane != 0:
                    raise ValueError(
                        "centered-scene protocol requires next_scene_start_lane=0"
                    )
                current_lane = int(state_ack["current_lane"])
                if current_lane not in {-1, 0, 1}:
                    raise ValueError(
                        f"Unity returned invalid current lane {current_lane}"
                    )

                if previous_scene >= 0:
                    failed_scene_indices = getattr(
                        self,
                        "_failed_scene_indices",
                        set(),
                    )
                    collision_recorded = previous_scene in failed_scene_indices
                    safe_lane = getattr(self, "_scene_safe_lanes", {}).get(
                        previous_scene
                    )
                    endpoint_reached = (
                        safe_lane is not None and current_lane == int(safe_lane)
                    )
                    self._record_scene_end(
                        previous_scene,
                        outcome=(
                            "success"
                            if endpoint_reached and not collision_recorded
                            else "failed"
                        ),
                        reason=(
                            "collision"
                            if collision_recorded
                            else "safe_lane_reached"
                            if endpoint_reached
                            else "endpoint_lane_mismatch"
                        ),
                        timestamp_monotonic=float(
                            state_ack.get(
                                "_received_at_monotonic",
                                time.monotonic(),
                            )
                        ),
                        endpoint_lane=current_lane,
                        endpoint_matches_safe_lane=endpoint_reached,
                    )
                    max_scenes = getattr(self, "_max_scenes", None)
                    if (
                        max_scenes is not None
                        and len(getattr(self, "_scene_end_recorded", set()))
                        >= max_scenes
                    ):
                        self._stop_event.set()
                        return
                prepare_scene = getattr(
                    self._online_label_source,
                    "prepare_scene",
                    None,
                )
                if not callable(prepare_scene):
                    raise RuntimeError(
                        "Relative-action label source does not support scene preparation."
                    )
                label_id = int(
                    prepare_scene(
                        scene_index=scene_index,
                        start_lane=start_lane,
                    )
                )
                status = self._online_label_source_status() or {}
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                self._abort_scene_protocol(f"invalid Unity lane-state ACK: {exc}")
                return

        command_by_label = {0: "SCENE_LEFT", 1: "SCENE_RIGHT"}
        command = command_by_label.get(label_id)
        if command is None:
            self._scene_sync_error = f"unsupported scene label id: {label_id}"
            return
        scene_ack = self._push_game_scene_transport_command(command)
        if scene_ack is not None:
            try:
                self._validate_unity_protocol_ack(
                    scene_ack,
                    expected_ack=command,
                    expected_scene_number=self._unity_scene_number(scene_index),
                )
                applied_label_name = str(scene_ack["applied_label"]).strip().lower()
                applied_label_id = {"left": 0, "right": 1}[
                    applied_label_name
                ]
                start_lane = int(scene_ack["start_lane"])
                safe_lane = int(scene_ack["safe_lane"])
                if applied_label_id != label_id:
                    raise ValueError(
                        f"requested label {label_id}, Unity applied {applied_label_id}"
                    )
            except (KeyError, TypeError, ValueError) as exc:
                self._abort_scene_protocol(f"invalid Unity scene ACK: {exc}")
                return
            confirm_scene = getattr(
                self._online_label_source,
                "confirm_scene_applied",
                None,
            )
            scene_ack_time = float(
                scene_ack.get(
                    "_received_at_monotonic",
                    self._last_game_transport_sent_at,
                )
            )
            scene_command_sent_at = float(
                scene_ack.get("_sent_at_monotonic", scene_ack_time)
            )
            scene_ack_round_trip_sec = max(
                float(
                    scene_ack.get(
                        "_ack_round_trip_sec",
                        scene_ack_time - scene_command_sent_at,
                    )
                ),
                0.0,
            )
            scene_visual_onset_time = scene_ack_time + max(
                float(getattr(self, "_visual_onset_delay_sec", 0.0)),
                0.0,
            )
            if callable(confirm_scene) and not confirm_scene(
                scene_index=scene_index,
                applied_label_id=applied_label_id,
                start_lane=start_lane,
                safe_lane=safe_lane,
                timestamp_monotonic=scene_visual_onset_time,
            ):
                self._abort_scene_protocol(
                    "Unity scene ACK did not match the prepared relative-action truth "
                    f"for scene {scene_index + 1}: label={applied_label_name}, "
                    f"start_lane={start_lane}, safe_lane={safe_lane}"
                )
                return
            self._scene_sent_scene_index = scene_index
            self._scene_sent_label_id = label_id
            if not hasattr(self, "_scene_started_at"):
                self._scene_started_at = {}
            if not hasattr(self, "_scene_labels"):
                self._scene_labels = {}
            if not hasattr(self, "_scene_start_lanes"):
                self._scene_start_lanes = {}
            if not hasattr(self, "_scene_safe_lanes"):
                self._scene_safe_lanes = {}
            if not hasattr(self, "_unity_scene_numbers"):
                self._unity_scene_numbers = {}
            self._scene_started_at[scene_index] = scene_visual_onset_time
            self._scene_labels[scene_index] = label_id
            self._scene_start_lanes[scene_index] = start_lane
            self._scene_safe_lanes[scene_index] = safe_lane
            self._unity_scene_numbers[scene_index] = int(scene_ack["scene_number"])
            writer = getattr(self, "_writer", None)
            if writer is not None:
                writer.append_event(
                    "scene_start",
                    timestamp_monotonic=scene_visual_onset_time,
                    scene_index=scene_index,
                    scene_number=scene_index + 1,
                    unity_scene_number=int(scene_ack["scene_number"]),
                    unity_scene_number_offset=self._unity_scene_number_offset,
                    label_id=label_id,
                    label_name=status.get("label_name"),
                    unity_command=command,
                    command_sent_at_monotonic=scene_command_sent_at,
                    ack_received_at_monotonic=scene_ack_time,
                    ack_round_trip_sec=scene_ack_round_trip_sec,
                    visual_onset_delay_sec=max(
                        float(getattr(self, "_visual_onset_delay_sec", 0.0)),
                        0.0,
                    ),
                    visual_onset_at_monotonic=scene_visual_onset_time,
                    ack_confirmed=True,
                    protocol_version=CUED_PROTOCOL_VERSION,
                    start_lane=start_lane,
                    safe_lane=safe_lane,
                    applied_label=applied_label_name,
                    planned_duration_sec=float(
                        self._online_label_source_metadata().get(
                            "scene_duration_sec",
                            0.0,
                        )
                        if self._online_label_source_metadata()
                        else 0.0
                    ),
                )
            self._scene_sync_error = None
            return
        self._scene_sync_error = self._last_game_transport_error or "scene command send failed"

    def _consume_game_scene_events(self) -> None:
        if self._game_command_outlet is None or self._online_label_source is None:
            return
        poll_events = getattr(self._game_command_outlet, "poll_events", None)
        mark_scene_failed = getattr(
            self._online_label_source,
            "mark_scene_failed",
            None,
        )
        if not callable(poll_events) or not callable(mark_scene_failed):
            return
        try:
            events = poll_events()
        except Exception as exc:  # noqa: BLE001
            self._last_game_transport_error = str(exc)
            LOGGER.warning("Failed to poll Unity scene events: %s", exc)
            if self._stop_on_game_disconnect:
                self._game_disconnect_message = f"Unity scene event connection lost: {exc}"
                self._stop_event.set()
            return

        for event in events:
            event_name = str(event.get("event", "")).strip().upper()
            if event_name == "LANE_SETTLED":
                self._handle_lane_settled_event(event)
                continue
            if event_name != "SCENE_FAILED":
                continue
            unity_scene_number = int(event.get("scene_number", 0))
            failed_scene_index = self._internal_scene_index_from_unity(
                unity_scene_number
            )
            if failed_scene_index is None:
                LOGGER.warning(
                    "Ignored Unity SCENE_FAILED before scene-number mapping was established: %s",
                    event,
                )
                continue
            if failed_scene_index != self._scene_sent_scene_index:
                LOGGER.warning(
                    "Ignored stale Unity SCENE_FAILED event unity_scene_number=%s; "
                    "current_unity_scene=%s",
                    unity_scene_number,
                    self._unity_scene_number(self._scene_sent_scene_index),
                )
                continue
            recorded = mark_scene_failed(
                timestamp_monotonic=float(
                    event.get("_received_at_monotonic", time.monotonic())
                ),
                expected_scene_index=failed_scene_index,
            )
            if recorded:
                if not hasattr(self, "_failed_scene_indices"):
                    self._failed_scene_indices = set()
                self._failed_scene_indices.add(failed_scene_index)
                writer = getattr(self, "_writer", None)
                if writer is not None:
                    writer.append_event(
                        "scene_failed",
                        timestamp_monotonic=float(
                            event.get("_received_at_monotonic", time.monotonic())
                        ),
                        scene_index=failed_scene_index,
                        scene_number=failed_scene_index + 1,
                        unity_scene_number=unity_scene_number,
                        label_id=self._scene_labels.get(
                            failed_scene_index,
                            self._scene_sent_label_id,
                        ),
                        unity_event=dict(event),
                    )
                self._console.print(
                    f"[bold yellow]Scene {failed_scene_index + 1} 避障失败；"
                    "保持当前安全车道和动态标签规则，到固定 Scene 边界再进入下一 Scene[/bold yellow]"
                )

    def _handle_lane_settled_event(self, event: dict[str, Any]) -> None:
        """Convert Unity's completed lane transition into a new truth segment."""

        if str(event.get("protocol_version", "")).strip() != CUED_PROTOCOL_VERSION:
            self._abort_scene_protocol(
                f"invalid LANE_SETTLED protocol version: {event!r}"
            )
            return
        unity_scene_number = int(event.get("scene_number", 0))
        scene_index = self._internal_scene_index_from_unity(unity_scene_number)
        if scene_index is None:
            LOGGER.warning(
                "Ignored Unity LANE_SETTLED before scene-number mapping was established: %s",
                event,
            )
            return
        if scene_index != self._scene_sent_scene_index:
            LOGGER.warning(
                "Ignored stale Unity LANE_SETTLED event unity_scene_number=%s; "
                "current_unity_scene=%s",
                unity_scene_number,
                self._unity_scene_number(self._scene_sent_scene_index),
            )
            return
        try:
            current_lane = int(event["current_lane"])
            safe_lane = int(event["safe_lane"])
        except (KeyError, TypeError, ValueError):
            self._abort_scene_protocol(f"invalid LANE_SETTLED payload: {event!r}")
            return
        expected_safe_lane = self._scene_safe_lanes.get(scene_index)
        if (
            current_lane not in {-1, 0, 1}
            or safe_lane not in {-1, 0, 1}
            or expected_safe_lane != safe_lane
        ):
            self._abort_scene_protocol(
                "Unity LANE_SETTLED event does not match active safe-lane truth: "
                f"{event!r}"
            )
            return
        update_current_lane = getattr(
            self._online_label_source,
            "update_current_lane",
            None,
        )
        transition_time = float(
            event.get("_received_at_monotonic", time.monotonic())
        )
        scene_started_at = getattr(self, "_scene_started_at", {}).get(scene_index)
        label_metadata = self._online_label_source_metadata() or {}
        scene_duration_sec = float(label_metadata.get("scene_duration_sec", 0.0))
        if (
            scene_started_at is not None
            and scene_duration_sec > 0.0
            and transition_time > scene_started_at + scene_duration_sec
        ):
            # Unity events are timestamped when Python receives them. An event
            # queued at the fixed boundary can therefore arrive a few
            # milliseconds after the Scene is already over. It must not alter
            # the next Scene's truth, but it is not a protocol failure either.
            writer = getattr(self, "_writer", None)
            if writer is not None:
                writer.append_event(
                    "lane_settled_ignored",
                    timestamp_monotonic=transition_time,
                    scene_index=scene_index,
                    scene_number=scene_index + 1,
                    unity_scene_number=unity_scene_number,
                    reason="received_after_fixed_scene_boundary",
                    scene_started_at_monotonic=scene_started_at,
                    scene_duration_sec=scene_duration_sec,
                    lateness_sec=(
                        transition_time - (scene_started_at + scene_duration_sec)
                    ),
                    unity_event=dict(event),
                )
            LOGGER.info(
                "Ignored late Unity LANE_SETTLED for completed Scene %s "
                "(lateness=%.3f ms)",
                scene_index + 1,
                (
                    transition_time
                    - (scene_started_at + scene_duration_sec)
                )
                * 1000.0,
            )
            return
        if not callable(update_current_lane) or not update_current_lane(
            scene_index=scene_index,
            current_lane=current_lane,
            safe_lane=safe_lane,
            timestamp_monotonic=transition_time,
        ):
            self._abort_scene_protocol(
                f"Rejected Unity LANE_SETTLED event: {event!r}"
            )
            return
        writer = getattr(self, "_writer", None)
        if writer is not None:
            writer.append_event(
                "lane_settled",
                timestamp_monotonic=transition_time,
                scene_index=scene_index,
                scene_number=scene_index + 1,
                unity_scene_number=unity_scene_number,
                current_lane=current_lane,
                safe_lane=safe_lane,
                dynamic_label_id=(
                    0 if current_lane > safe_lane
                    else 1 if current_lane < safe_lane
                    else 2
                ),
                unity_event=dict(event),
            )

    @staticmethod
    def _validate_unity_protocol_ack(
        response: dict[str, Any],
        *,
        expected_ack: str,
        expected_scene_number: int | None,
    ) -> None:
        if str(response.get("ack", "")).strip().upper() != expected_ack:
            raise ValueError(f"expected ACK {expected_ack}, received {response!r}")
        if (
            str(response.get("protocol_version", "")).strip()
            != CUED_PROTOCOL_VERSION
        ):
            raise ValueError(
                f"Unity runtime does not implement {CUED_PROTOCOL_VERSION}"
            )
        if "scene_number" not in response:
            raise ValueError(f"Unity ACK has no scene_number: {response!r}")
        scene_number = int(response["scene_number"])
        if (
            expected_scene_number is not None
            and scene_number != int(expected_scene_number)
        ):
            raise ValueError(
                f"expected scene {expected_scene_number}, "
                f"received {scene_number}"
            )

    def _unity_scene_number(self, scene_index: int) -> int:
        offset = getattr(self, "_unity_scene_number_offset", None)
        return int(scene_index) + 1 + (0 if offset is None else int(offset))

    def _internal_scene_index_from_unity(
        self,
        unity_scene_number: int,
    ) -> int | None:
        offset = getattr(self, "_unity_scene_number_offset", None)
        if offset is None:
            return None
        return int(unity_scene_number) - int(offset) - 1

    def _abort_scene_protocol(self, message: str) -> None:
        self._scene_sync_error = message
        self._last_game_transport_error = message
        LOGGER.error("Unity relative-action scene protocol failed: %s", message)
        if self._stop_on_game_disconnect:
            self._game_disconnect_message = message
            self._stop_event.set()

    def _push_game_scene_transport_command(
        self,
        command: str,
    ) -> dict[str, Any] | None:
        if self._game_command_outlet is None:
            return None
        push_with_ack = getattr(self._game_command_outlet, "push_with_ack", None)
        if not callable(push_with_ack):
            self._last_game_transport_error = "AR transport does not support Unity scene ACK"
            return None
        try:
            local_sent_at = time.monotonic()
            response = push_with_ack(command)
            local_received_at = time.monotonic()
            if not isinstance(response, dict):
                raise RuntimeError(
                    f"Unity returned no structured ACK for {command}"
                )
            response.setdefault("_sent_at_monotonic", local_sent_at)
            response.setdefault("_received_at_monotonic", local_received_at)
            response.setdefault(
                "_ack_round_trip_sec",
                max(
                    float(response["_received_at_monotonic"])
                    - float(response["_sent_at_monotonic"]),
                    0.0,
                ),
            )
            self._last_game_transport_command = command
            self._last_game_transport_error = None
            self._last_game_transport_sent_at = float(
                response["_sent_at_monotonic"]
            )
            return response
        except Exception as exc:  # noqa: BLE001
            self._last_game_transport_error = str(exc)
            LOGGER.warning("Failed to synchronize Unity scene '%s': %s", command, exc)
            if self._stop_on_game_disconnect:
                self._game_disconnect_message = f"Unity scene synchronization failed: {exc}"
                self._stop_event.set()
            return None

    def _push_game_command(self, command: str | None) -> None:
        if self._game_command_outlet is None:
            return

        if command is None:
            if self._last_game_command is None:
                self._push_game_keepalive()
                return
            self._push_game_transport_command("STOP", movement=True)
            self._last_game_command = None
            return

        self._push_game_session_command("START")
        now = time.monotonic()
        if (
            command == self._last_game_command
            and now - self._last_game_movement_sent_at < self._game_command_keepalive_sec
        ):
            return

        if self._push_game_transport_command(command, movement=True):
            self._last_game_command = command

    def _push_game_session_command(self, command: str) -> None:
        if self._game_command_outlet is None or self._game_session_started:
            return
        if self._push_game_transport_command(command):
            self._game_session_started = True

    def _push_game_keepalive(self) -> None:
        if not self._game_session_started:
            return
        now = time.monotonic()
        if now - self._last_game_movement_sent_at < self._game_command_keepalive_sec:
            return
        self._push_game_transport_command("STOP", movement=True)

    def _push_game_transport_command(self, command: str, *, movement: bool = False) -> bool:
        if self._game_command_outlet is None:
            return False
        try:
            self._game_command_outlet.push(command)
            self._last_game_transport_command = command
            self._last_game_transport_error = None
            self._last_game_transport_sent_at = time.monotonic()
            if movement:
                self._last_game_movement_sent_at = self._last_game_transport_sent_at
            return True
        except Exception as exc:  # noqa: BLE001
            self._last_game_transport_error = str(exc)
            LOGGER.warning("Failed to push AR game command '%s': %s", command, exc)
            if self._stop_on_game_disconnect:
                self._game_disconnect_message = f"Unity game connection lost: {exc}"
                self._stop_event.set()
            return False

    def _record_scene_end(
        self,
        scene_index: int,
        *,
        outcome: str,
        reason: str,
        timestamp_monotonic: float | None = None,
        endpoint_lane: int | None = None,
        endpoint_matches_safe_lane: bool | None = None,
    ) -> None:
        index = int(scene_index)
        scene_end_recorded = getattr(self, "_scene_end_recorded", set())
        if index < 0 or index in scene_end_recorded:
            return
        ended_at = (
            time.monotonic()
            if timestamp_monotonic is None
            else float(timestamp_monotonic)
        )
        started_at = getattr(self, "_scene_started_at", {}).get(index)
        failed_scene_indices = getattr(self, "_failed_scene_indices", set())
        writer = getattr(self, "_writer", None)
        if writer is not None:
            writer.append_event(
                "scene_end",
                timestamp_monotonic=ended_at,
                scene_index=index,
                scene_number=index + 1,
                unity_scene_number=getattr(self, "_unity_scene_numbers", {}).get(
                    index
                ),
                label_id=getattr(self, "_scene_labels", {}).get(index, -1),
                start_lane=getattr(self, "_scene_start_lanes", {}).get(index),
                safe_lane=getattr(self, "_scene_safe_lanes", {}).get(index),
                outcome=str(outcome),
                reason=str(reason),
                collision_recorded=index in failed_scene_indices,
                endpoint_lane=endpoint_lane,
                endpoint_matches_safe_lane=endpoint_matches_safe_lane,
                duration_sec=(
                    None if started_at is None else max(ended_at - started_at, 0.0)
                ),
            )
        if not hasattr(self, "_scene_end_recorded"):
            self._scene_end_recorded = set()
        self._scene_end_recorded.add(index)

    def _record_active_scene_end(self, *, outcome: str, reason: str) -> None:
        self._record_scene_end(
            getattr(self, "_scene_sent_scene_index", -1),
            outcome=outcome,
            reason=reason,
        )

    def _snapshot_model_revision(
        self,
        revision: int,
        *,
        source: str,
        model_snapshot: BaseModelAdapter | None = None,
    ) -> None:
        writer = getattr(self, "_writer", None)
        save_dir = getattr(self, "_save_dir", None)
        if writer is None or save_dir is None:
            return
        revision_value = int(revision)
        if any(
            int(record.get("model_revision", -1)) == revision_value
            for record in self._model_revision_records
        ):
            return
        revision_dir = Path(save_dir) / "model_revisions"
        revision_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = revision_dir / f"revision_{revision_value:04d}.pt"
        sidecar = Path(f"{checkpoint}.neuroonline.pt")
        if (
            revision_value == 0
            and self._model_source_path is not None
            and self._model_source_path.exists()
        ):
            shutil.copy2(self._model_source_path, checkpoint)
            source_sidecar = Path(f"{self._model_source_path}.neuroonline.pt")
            if source_sidecar.exists():
                shutil.copy2(source_sidecar, sidecar)
        elif model_snapshot is not None:
            model_snapshot.save(checkpoint)
        else:
            with self._model_lock:
                snapshot = copy.deepcopy(self._model)
            snapshot.save(checkpoint)
        record: dict[str, Any] = {
            "model_revision": revision_value,
            "source": str(source),
            "checkpoint": checkpoint.relative_to(save_dir).as_posix(),
            "checkpoint_sha256": self._sha256_file(checkpoint),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        if sidecar.exists():
            record["crm_sidecar"] = sidecar.relative_to(save_dir).as_posix()
            record["crm_sidecar_sha256"] = self._sha256_file(sidecar)
        self._model_revision_records.append(record)
        writer.append_event("model_checkpoint", **record)

    def _build_run_provenance(self) -> dict[str, Any]:
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

        package_versions: dict[str, str | None] = {}
        for package in ("numpy", "scipy", "torch", "scikit-learn"):
            try:
                package_versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                package_versions[package] = None
        config_json = json.dumps(
            self._experiment_config,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        model_files: list[dict[str, Any]] = []
        if self._model_source_path is not None and self._model_source_path.exists():
            model_files.append(
                {
                    "path": str(self._model_source_path),
                    "sha256": self._sha256_file(self._model_source_path),
                    "size_bytes": self._model_source_path.stat().st_size,
                }
            )
            source_sidecar = Path(f"{self._model_source_path}.neuroonline.pt")
            if source_sidecar.exists():
                model_files.append(
                    {
                        "path": str(source_sidecar),
                        "sha256": self._sha256_file(source_sidecar),
                        "size_bytes": source_sidecar.stat().st_size,
                    }
                )
        return {
            "run_id": self._run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git": {"commit": commit, "dirty": dirty},
            "platform": platform.platform(),
            "python": sys.version,
            "packages": package_versions,
            "experiment_config": self._experiment_config,
            "experiment_config_sha256": hashlib.sha256(config_json).hexdigest(),
            "random_seed": int(
                self._experiment_config.get("online_adaptation", {})
                .get("neuroonline", {})
                .get("random_seed", 42)
            ),
            "model_name": self._model_name,
            "initial_model_files": model_files,
        }

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _emit_status(self, result: PredictionResult, game_command: str | None) -> None:
        if self._status_callback is None:
            return

        payload = {
            "prediction": result.label,
            "confidence": result.confidence,
            "uncertainty": result.uncertainty,
            "class_id": result.class_id,
            "mapped_command": game_command or "STOP",
            "lateral_control_gate_active": self._is_cued_control_gate_active(),
            "last_transport_command": self._last_game_transport_command,
            "last_send_success": self._last_game_transport_error is None and self._last_game_transport_sent_at > 0.0,
            "last_send_error": self._last_game_transport_error,
            "model_revision": getattr(self, "_model_revision", 0),
            "timing_alignment": getattr(
                getattr(self, "_acquirer", None),
                "timing_diagnostics",
                {},
            ),
            "updated_at": time.time(),
        }
        adaptation_status = self._online_adaptation_status()
        if adaptation_status is not None:
            payload["online_adaptation"] = adaptation_status
        label_source_status = self._online_label_source_status()
        if label_source_status is not None:
            payload["online_label_source"] = label_source_status
        try:
            self._status_callback(payload)
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Failed to publish realtime decoder status: %s", exc)

    def _online_adaptation_status(self) -> dict[str, Any] | None:
        neuroonline_adapter = getattr(self, "_neuroonline_adapter", None)
        if neuroonline_adapter is not None:
            return neuroonline_adapter.status()
        return None

    def _online_label_source_status(self) -> dict[str, Any] | None:
        source = getattr(self, "_online_label_source", None)
        status = getattr(source, "status", None)
        if not callable(status):
            return None
        try:
            payload = status()
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Failed to read online label source status: %s", exc)
            return None
        if not isinstance(payload, dict):
            return None
        result = dict(payload)
        if result.get("source") == "cued-protocol":
            scene_index = int(result.get("scene_index", -1))
            label_value = result.get("label_id")
            result["scene_synced"] = (
                scene_index == getattr(self, "_scene_sent_scene_index", -1)
                and getattr(self, "_scene_sync_error", None) is None
            )
            result["primary_decision_collected"] = (
                scene_index
                in getattr(self, "_primary_decision_scenes", set())
            )
            result["primary_decision_windows_collected"] = len(
                getattr(self, "_primary_decision_window_bounds", {}).get(
                    scene_index,
                    [],
                )
            )
            result["primary_decision_windows_required"] = getattr(
                self,
                "_primary_windows_per_scene",
                1,
            )
            result["lateral_control_gate_active"] = (
                scene_index
                not in getattr(self, "_primary_decision_scenes", set())
            )
            result["control_released_at_monotonic"] = getattr(
                self,
                "_control_released_at",
                {},
            ).get(scene_index)
            result["scene_sync_error"] = getattr(self, "_scene_sync_error", None)
            result["unity_scene_number_offset"] = getattr(
                self,
                "_unity_scene_number_offset",
                None,
            )
            result["unity_scene_number"] = (
                self._unity_scene_number(scene_index)
                if getattr(self, "_unity_scene_number_offset", None) is not None
                else None
            )
        return result

    def _online_label_source_metadata(self) -> dict[str, Any] | None:
        source = getattr(self, "_online_label_source", None)
        metadata = getattr(source, "metadata", None)
        if callable(metadata):
            try:
                payload = metadata()
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("Failed to read online label source metadata: %s", exc)
            else:
                if isinstance(payload, dict):
                    return dict(payload)
        return self._online_label_source_status()
