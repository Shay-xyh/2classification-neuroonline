"""Background streamed writer for chunks of data."""
import hashlib
import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class RecordItem:
    window: np.ndarray
    y_true: int
    y_pred: int
    confidence: float
    raw_pred: int = -1
    model_revision: int = 0
    label_event_id: str = ""
    quality_accepted: bool = True
    quality_peak_abs_uv: float = 0.0
    quality_clip_fraction: float = 0.0
    quality_bad_channel_fraction: float = 0.0
    probabilities: np.ndarray | None = None
    uncertainty: float = 0.0
    window_start_monotonic: float = float("nan")
    window_end_monotonic: float = float("nan")
    window_start_unix: float = float("nan")
    window_end_unix: float = float("nan")
    scene_index: int = -1
    scene_label: int = -1
    scene_start_lane: int = -9
    scene_safe_lane: int = -9
    scene_current_lane: int = -9
    instruction_label: int = -1
    vehicle_required_action: int = -1
    scene_failed: bool = False
    training_role: str = "unlabeled"
    adaptation_eligible: bool = False
    adaptation_committed: bool = False
    control_gate_active: bool = False
    mapped_command: str = ""
    transport_command: str = ""
    transport_success: bool = False
    transport_sent_at_monotonic: float = float("nan")
    transport_error: str = ""
    quality_reasons: str = ""
    quality_bad_channel_indices: str = ""
    quality_nonfinite_fraction: float = 0.0
    timing_queueing_jitter_sec: float = 0.0
    timing_transport_delay_compensation_sec: float = 0.0
    timing_packet_arrival_monotonic: float = float("nan")
    timing_received_packets: float = 0.0
    timing_packet_loss_count: float = 0.0
    timing_total_source_samples: float = 0.0


class StreamWriter:
    """Writes chunks of recordings to avoid keeping all data in memory."""

    def __init__(self, output_dir: Path, chunk_size: int = 500, max_queue: int = 2000):
        self._output_dir = output_dir
        self._chunks_dir = output_dir / "chunks"
        self._events_path = output_dir / "events.jsonl"
        self._chunk_size = chunk_size
        self._queue: queue.Queue[RecordItem | None] = queue.Queue(maxsize=max_queue)
        self._dropped_records = 0
        self._total_windows = 0
        self._quality_rejected_windows = 0
        self._chunk_count = 0
        self._files: list[str] = []
        self._manifest_lock = threading.Lock()
        self._event_lock = threading.Lock()
        self._event_count = 0
        self._start_monotonic = time.monotonic()
        self._start_unix = time.time()

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self, metadata: dict) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._chunks_dir.mkdir(parents=True, exist_ok=True)
        self._start_monotonic = time.monotonic()
        self._start_unix = time.time()
        self._events_path.write_text("", encoding="utf-8")
        initial = dict(metadata)
        initial.setdefault("schema_version", "oi-mi-experiment-v2")
        initial.setdefault("run_id", self._output_dir.name)
        initial.setdefault("start_time", self._start_unix)
        initial.setdefault(
            "start_time_utc",
            datetime.fromtimestamp(self._start_unix, tz=timezone.utc).isoformat(),
        )
        initial.setdefault(
            "clock_mapping",
            {
                "start_monotonic": self._start_monotonic,
                "start_unix": self._start_unix,
                "window_clock": "source-mapped-monotonic",
            },
        )
        with self._manifest_lock:
            self._write_manifest_atomic(initial)

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            while True:
                try:
                    self._queue.put(None, timeout=0.1)
                    break
                except queue.Full:
                    continue
            self._thread.join()
            self._thread = None
            
    def put(
        self,
        window: np.ndarray,
        y_true: int,
        y_pred: int,
        confidence: float,
        *,
        raw_pred: int = -1,
        model_revision: int = 0,
        label_event_id: str = "",
        quality_accepted: bool = True,
        quality_peak_abs_uv: float = 0.0,
        quality_clip_fraction: float = 0.0,
        quality_bad_channel_fraction: float = 0.0,
        probabilities: np.ndarray | None = None,
        uncertainty: float = 0.0,
        window_start_monotonic: float = float("nan"),
        window_end_monotonic: float = float("nan"),
        scene_index: int = -1,
        scene_label: int = -1,
        scene_start_lane: int = -9,
        scene_safe_lane: int = -9,
        scene_current_lane: int = -9,
        instruction_label: int = -1,
        vehicle_required_action: int = -1,
        scene_failed: bool = False,
        training_role: str = "unlabeled",
        adaptation_eligible: bool = False,
        adaptation_committed: bool = False,
        control_gate_active: bool = False,
        mapped_command: str = "",
        transport_command: str = "",
        transport_success: bool = False,
        transport_sent_at_monotonic: float = float("nan"),
        transport_error: str = "",
        quality_reasons: tuple[str, ...] | list[str] | str = (),
        quality_bad_channel_indices: tuple[int, ...] | list[int] | str = (),
        quality_nonfinite_fraction: float = 0.0,
        timing_queueing_jitter_sec: float = 0.0,
        timing_transport_delay_compensation_sec: float = 0.0,
        timing_packet_arrival_monotonic: float = float("nan"),
        timing_received_packets: float = 0.0,
        timing_packet_loss_count: float = 0.0,
        timing_total_source_samples: float = 0.0,
    ) -> None:
        start_mono = float(window_start_monotonic)
        end_mono = float(window_end_monotonic)
        item = RecordItem(
            window=np.asarray(window, dtype=np.float32),
            y_true=int(y_true),
            y_pred=int(y_pred),
            confidence=float(confidence),
            raw_pred=int(raw_pred),
            model_revision=int(model_revision),
            label_event_id=str(label_event_id),
            quality_accepted=bool(quality_accepted),
            quality_peak_abs_uv=float(quality_peak_abs_uv),
            quality_clip_fraction=float(quality_clip_fraction),
            quality_bad_channel_fraction=float(quality_bad_channel_fraction),
            probabilities=(
                None
                if probabilities is None
                else np.asarray(probabilities, dtype=np.float32).reshape(-1)
            ),
            uncertainty=float(uncertainty),
            window_start_monotonic=start_mono,
            window_end_monotonic=end_mono,
            window_start_unix=self._monotonic_to_unix(start_mono),
            window_end_unix=self._monotonic_to_unix(end_mono),
            scene_index=int(scene_index),
            scene_label=int(scene_label),
            scene_start_lane=int(scene_start_lane),
            scene_safe_lane=int(scene_safe_lane),
            scene_current_lane=int(scene_current_lane),
            instruction_label=int(instruction_label),
            vehicle_required_action=int(vehicle_required_action),
            scene_failed=bool(scene_failed),
            training_role=str(training_role),
            adaptation_eligible=bool(adaptation_eligible),
            adaptation_committed=bool(adaptation_committed),
            control_gate_active=bool(control_gate_active),
            mapped_command=str(mapped_command),
            transport_command=str(transport_command),
            transport_success=bool(transport_success),
            transport_sent_at_monotonic=float(transport_sent_at_monotonic),
            transport_error=str(transport_error),
            quality_reasons=self._encode_sequence(quality_reasons),
            quality_bad_channel_indices=self._encode_sequence(
                quality_bad_channel_indices
            ),
            quality_nonfinite_fraction=float(quality_nonfinite_fraction),
            timing_queueing_jitter_sec=float(timing_queueing_jitter_sec),
            timing_transport_delay_compensation_sec=float(
                timing_transport_delay_compensation_sec
            ),
            timing_packet_arrival_monotonic=float(
                timing_packet_arrival_monotonic
            ),
            timing_received_packets=float(timing_received_packets),
            timing_packet_loss_count=float(timing_packet_loss_count),
            timing_total_source_samples=float(timing_total_source_samples),
        )
        try:
            self._queue.put_nowait(item)
            if not quality_accepted:
                self._quality_rejected_windows += 1
        except queue.Full:
            self._dropped_records += 1
            LOGGER.warning("StreamWriter queue full! Dropped record.")

    def append_event(
        self,
        event_type: str,
        *,
        timestamp_monotonic: float | None = None,
        timestamp_unix: float | None = None,
        **payload: Any,
    ) -> dict[str, Any]:
        """Append one recoverable JSONL event with both clock domains."""

        monotonic_value = (
            time.monotonic()
            if timestamp_monotonic is None
            else float(timestamp_monotonic)
        )
        unix_value = (
            self._monotonic_to_unix(monotonic_value)
            if timestamp_unix is None
            else float(timestamp_unix)
        )
        with self._event_lock:
            event = {
                "event_index": self._event_count,
                "event_type": str(event_type),
                "timestamp_monotonic": monotonic_value,
                "timestamp_unix": unix_value,
                "timestamp_utc": datetime.fromtimestamp(
                    unix_value,
                    tz=timezone.utc,
                ).isoformat(),
                "payload": self._json_safe(payload),
            }
            with self._events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._event_count += 1
        return event

    def update_manifest(self, extra: dict) -> None:
        manifest_path = self._output_dir / "manifest.json"
        with self._manifest_lock:
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except FileNotFoundError:
                metadata = {}

            metadata.update({
                "chunk_size": self._chunk_size,
                "chunk_count": self._chunk_count,
                "total_windows": self._total_windows,
                "quality_rejected_windows": self._quality_rejected_windows,
                "quality_accepted_windows": (
                    self._total_windows - self._quality_rejected_windows
                ),
                "dropped_records": self._dropped_records,
                "event_count": self._event_count,
                "events_file": self._events_path.name,
                "files": list(self._files),
                "end_time": time.time(),
                "end_time_utc": datetime.now(timezone.utc).isoformat(),
            })
            metadata.update(extra)
            self._write_manifest_atomic(metadata)

    def finalize_manifest(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """Recompute metrics from disk, add checksums, and atomically seal manifest."""

        scientific_metrics = self._compute_scientific_metrics()
        timing_integrity = self._timing_integrity()
        integrity_status = "complete"
        if self._dropped_records > 0:
            integrity_status = "records_dropped"
        elif timing_integrity["packet_loss_count"] > 0:
            integrity_status = "source_packet_loss"
        elif timing_integrity["invalid_window_timestamps"] > 0:
            integrity_status = "invalid_window_timestamps"
        payload: dict[str, Any] = {
            "scientific_metrics": scientific_metrics,
            "integrity": {
                "status": integrity_status,
                "dropped_records": self._dropped_records,
                "timing": timing_integrity,
                "checksums": self._checksums(),
            },
        }
        if extra:
            payload.update(extra)
        self.update_manifest(payload)
        return payload

    def _writer_loop(self) -> None:
        buffer = []
        
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.1)
                
                if item is None:
                    continue
                    
                buffer.append(item)
                
                if len(buffer) >= self._chunk_size:
                    self._flush_buffer(buffer)
                    buffer = []
                    
            except queue.Empty:
                continue
                
        if buffer:
            self._flush_buffer(buffer)

    def _flush_buffer(self, buffer: list[RecordItem]) -> None:
        if not buffer:
            return
            
        windows = np.stack([item.window for item in buffer])
        y_trues = np.array([item.y_true for item in buffer])
        y_preds = np.array([item.y_pred for item in buffer])
        confidences = np.array([item.confidence for item in buffer])
        raw_predictions = np.array([item.raw_pred for item in buffer], dtype=np.int64)
        model_revisions = np.array([item.model_revision for item in buffer], dtype=np.int64)
        label_event_ids = np.array([item.label_event_id for item in buffer], dtype=np.str_)
        quality_accepted = np.array(
            [item.quality_accepted for item in buffer],
            dtype=np.bool_,
        )
        quality_peak_abs_uv = np.array(
            [item.quality_peak_abs_uv for item in buffer],
            dtype=np.float32,
        )
        quality_clip_fraction = np.array(
            [item.quality_clip_fraction for item in buffer],
            dtype=np.float32,
        )
        quality_bad_channel_fraction = np.array(
            [item.quality_bad_channel_fraction for item in buffer],
            dtype=np.float32,
        )
        probability_width = max(
            (int(item.probabilities.size) for item in buffer if item.probabilities is not None),
            default=0,
        )
        probabilities = np.full(
            (len(buffer), probability_width),
            np.nan,
            dtype=np.float32,
        )
        for index, item in enumerate(buffer):
            if item.probabilities is not None:
                probabilities[index, : item.probabilities.size] = item.probabilities
        uncertainties = np.asarray(
            [item.uncertainty for item in buffer],
            dtype=np.float32,
        )
        window_start_monotonic = np.asarray(
            [item.window_start_monotonic for item in buffer],
            dtype=np.float64,
        )
        window_end_monotonic = np.asarray(
            [item.window_end_monotonic for item in buffer],
            dtype=np.float64,
        )
        window_start_unix = np.asarray(
            [item.window_start_unix for item in buffer],
            dtype=np.float64,
        )
        window_end_unix = np.asarray(
            [item.window_end_unix for item in buffer],
            dtype=np.float64,
        )
        scene_indices = np.asarray([item.scene_index for item in buffer], dtype=np.int64)
        scene_labels = np.asarray([item.scene_label for item in buffer], dtype=np.int64)
        scene_start_lanes = np.asarray(
            [item.scene_start_lane for item in buffer],
            dtype=np.int8,
        )
        scene_safe_lanes = np.asarray(
            [item.scene_safe_lane for item in buffer],
            dtype=np.int8,
        )
        scene_current_lanes = np.asarray(
            [item.scene_current_lane for item in buffer],
            dtype=np.int8,
        )
        instruction_labels = np.asarray(
            [item.instruction_label for item in buffer],
            dtype=np.int64,
        )
        vehicle_required_actions = np.asarray(
            [item.vehicle_required_action for item in buffer],
            dtype=np.int64,
        )
        scene_failed = np.asarray([item.scene_failed for item in buffer], dtype=np.bool_)
        training_roles = np.asarray(
            [item.training_role for item in buffer],
            dtype=np.str_,
        )
        adaptation_eligible = np.asarray(
            [item.adaptation_eligible for item in buffer],
            dtype=np.bool_,
        )
        adaptation_committed = np.asarray(
            [item.adaptation_committed for item in buffer],
            dtype=np.bool_,
        )
        control_gate_active = np.asarray(
            [item.control_gate_active for item in buffer],
            dtype=np.bool_,
        )
        mapped_commands = np.asarray([item.mapped_command for item in buffer], dtype=np.str_)
        transport_commands = np.asarray(
            [item.transport_command for item in buffer],
            dtype=np.str_,
        )
        transport_success = np.asarray(
            [item.transport_success for item in buffer],
            dtype=np.bool_,
        )
        transport_sent_at_monotonic = np.asarray(
            [item.transport_sent_at_monotonic for item in buffer],
            dtype=np.float64,
        )
        transport_errors = np.asarray(
            [item.transport_error for item in buffer],
            dtype=np.str_,
        )
        quality_reasons = np.asarray([item.quality_reasons for item in buffer], dtype=np.str_)
        quality_bad_channel_indices = np.asarray(
            [item.quality_bad_channel_indices for item in buffer],
            dtype=np.str_,
        )
        quality_nonfinite_fraction = np.asarray(
            [item.quality_nonfinite_fraction for item in buffer],
            dtype=np.float32,
        )
        timing_queueing_jitter_sec = np.asarray(
            [item.timing_queueing_jitter_sec for item in buffer],
            dtype=np.float64,
        )
        timing_transport_delay_compensation_sec = np.asarray(
            [item.timing_transport_delay_compensation_sec for item in buffer],
            dtype=np.float64,
        )
        timing_packet_arrival_monotonic = np.asarray(
            [item.timing_packet_arrival_monotonic for item in buffer],
            dtype=np.float64,
        )
        timing_received_packets = np.asarray(
            [item.timing_received_packets for item in buffer],
            dtype=np.float64,
        )
        timing_packet_loss_count = np.asarray(
            [item.timing_packet_loss_count for item in buffer],
            dtype=np.float64,
        )
        timing_total_source_samples = np.asarray(
            [item.timing_total_source_samples for item in buffer],
            dtype=np.float64,
        )
        
        chunk_name = f"chunk_{self._chunk_count:06d}.npz"
        chunk_path = self._chunks_dir / chunk_name
        
        temporary_path = chunk_path.with_suffix(".npz.tmp")
        with temporary_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                eeg_windows=windows,
                labels_true=y_trues,
                labels_pred=y_preds,
                confidences=confidences,
                probabilities=probabilities,
                uncertainties=uncertainties,
                predictions_raw=raw_predictions,
                model_revisions=model_revisions,
                label_event_ids=label_event_ids,
                window_start_monotonic=window_start_monotonic,
                window_end_monotonic=window_end_monotonic,
                window_start_unix=window_start_unix,
                window_end_unix=window_end_unix,
                scene_indices=scene_indices,
                scene_labels=scene_labels,
                scene_start_lanes=scene_start_lanes,
                scene_safe_lanes=scene_safe_lanes,
                scene_current_lanes=scene_current_lanes,
                instruction_labels=instruction_labels,
                vehicle_required_actions=vehicle_required_actions,
                scene_failed_at_prediction=scene_failed,
                training_roles=training_roles,
                adaptation_eligible=adaptation_eligible,
                adaptation_committed=adaptation_committed,
                control_gate_active=control_gate_active,
                mapped_commands=mapped_commands,
                transport_commands=transport_commands,
                transport_success=transport_success,
                transport_sent_at_monotonic=transport_sent_at_monotonic,
                transport_errors=transport_errors,
                quality_accepted=quality_accepted,
                quality_peak_abs_uv=quality_peak_abs_uv,
                quality_clip_fraction=quality_clip_fraction,
                quality_bad_channel_fraction=quality_bad_channel_fraction,
                quality_nonfinite_fraction=quality_nonfinite_fraction,
                quality_reasons=quality_reasons,
                quality_bad_channel_indices=quality_bad_channel_indices,
                timing_queueing_jitter_sec=timing_queueing_jitter_sec,
                timing_transport_delay_compensation_sec=(
                    timing_transport_delay_compensation_sec
                ),
                timing_packet_arrival_monotonic=(
                    timing_packet_arrival_monotonic
                ),
                timing_received_packets=timing_received_packets,
                timing_packet_loss_count=timing_packet_loss_count,
                timing_total_source_samples=timing_total_source_samples,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, chunk_path)
        
        self._files.append(chunk_name)
        self._chunk_count += 1
        self._total_windows += len(buffer)

    def _compute_scientific_metrics(self) -> dict[str, Any]:
        arrays: dict[str, list[np.ndarray]] = {}
        for chunk_name in self._files:
            path = self._chunks_dir / chunk_name
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as chunk:
                for key in (
                    "labels_true",
                    "labels_pred",
                    "predictions_raw",
                    "probabilities",
                    "quality_accepted",
                    "scene_indices",
                    "training_roles",
                    "adaptation_eligible",
                    "adaptation_committed",
                ):
                    if key in chunk:
                        arrays.setdefault(key, []).append(np.asarray(chunk[key]))
                    elif key == "training_roles" and "labels_true" in chunk:
                        arrays.setdefault(key, []).append(
                            np.full(len(chunk["labels_true"]), "", dtype=np.str_)
                        )
        if not arrays.get("labels_true"):
            return {"evaluated_windows": 0}

        y_true = np.concatenate(arrays["labels_true"]).astype(np.int64)
        y_operational = np.concatenate(arrays["labels_pred"]).astype(np.int64)
        y_raw = np.concatenate(arrays["predictions_raw"]).astype(np.int64)
        quality = np.concatenate(arrays["quality_accepted"]).astype(np.bool_)
        scene_indices = np.concatenate(arrays.get("scene_indices", [])).astype(np.int64)
        probabilities = np.concatenate(arrays.get("probabilities", []), axis=0)
        training_roles = np.concatenate(arrays["training_roles"]).astype(np.str_)
        adaptation_eligible = np.concatenate(
            arrays.get(
                "adaptation_eligible",
                [np.zeros(y_true.shape, dtype=np.bool_)],
            )
        ).astype(np.bool_)
        adaptation_committed = np.concatenate(
            arrays.get(
                "adaptation_committed",
                [np.zeros(y_true.shape, dtype=np.bool_)],
            )
        ).astype(np.bool_)
        class_count = max(
            probabilities.shape[1] if probabilities.ndim == 2 else 0,
            int(np.max(y_true[y_true >= 0]) + 1) if np.any(y_true >= 0) else 0,
        )
        valid = quality & (y_true >= 0)
        raw_valid = valid & (y_raw >= 0)
        operational_valid = valid & (y_operational >= 0)
        raw_metrics = self._classification_metrics(
            y_true[raw_valid],
            y_raw[raw_valid],
            class_count,
        )
        operational_metrics = self._classification_metrics(
            y_true[valid],
            y_operational[valid],
            class_count,
            abstain_value=-1,
        )
        operational_metrics["coverage"] = (
            float(np.sum(operational_valid) / np.sum(valid)) if np.any(valid) else 0.0
        )
        operational_metrics["selective_accuracy"] = (
            float(np.mean(y_operational[operational_valid] == y_true[operational_valid]))
            if np.any(operational_valid)
            else 0.0
        )
        operational_metrics["abstained_windows"] = int(np.sum(valid & ~operational_valid))

        primary_valid = valid & adaptation_eligible
        primary_raw_valid = primary_valid & (y_raw >= 0)
        primary_operational_valid = primary_valid & (y_operational >= 0)
        primary_raw_metrics = self._classification_metrics(
            y_true[primary_raw_valid],
            y_raw[primary_raw_valid],
            class_count,
        )
        primary_operational_metrics = self._classification_metrics(
            y_true[primary_valid],
            y_operational[primary_valid],
            class_count,
            abstain_value=-1,
        )
        primary_operational_metrics["coverage"] = (
            float(np.sum(primary_operational_valid) / np.sum(primary_valid))
            if np.any(primary_valid)
            else 0.0
        )
        primary_operational_metrics["selective_accuracy"] = (
            float(
                np.mean(
                    y_operational[primary_operational_valid]
                    == y_true[primary_operational_valid]
                )
            )
            if np.any(primary_operational_valid)
            else 0.0
        )
        primary_operational_metrics["abstained_windows"] = int(
            np.sum(primary_valid & ~primary_operational_valid)
        )

        continuous_valid = valid & (training_roles == "continuous_context")
        continuous_raw_valid = continuous_valid & (y_raw >= 0)
        continuous_operational_valid = continuous_valid & (y_operational >= 0)
        continuous_raw_metrics = self._classification_metrics(
            y_true[continuous_raw_valid],
            y_raw[continuous_raw_valid],
            class_count,
        )
        continuous_operational_metrics = self._classification_metrics(
            y_true[continuous_valid],
            y_operational[continuous_valid],
            class_count,
            abstain_value=-1,
        )
        continuous_operational_metrics["coverage"] = (
            float(np.sum(continuous_operational_valid) / np.sum(continuous_valid))
            if np.any(continuous_valid)
            else 0.0
        )
        continuous_operational_metrics["selective_accuracy"] = (
            float(
                np.mean(
                    y_operational[continuous_operational_valid]
                    == y_true[continuous_operational_valid]
                )
            )
            if np.any(continuous_operational_valid)
            else 0.0
        )
        continuous_operational_metrics["abstained_windows"] = int(
            np.sum(continuous_valid & ~continuous_operational_valid)
        )

        scene_truth: list[int] = []
        scene_prediction: list[int] = []
        for scene_index in np.unique(
            scene_indices[primary_valid & (scene_indices >= 0)]
        ):
            mask = primary_valid & (scene_indices == scene_index)
            if not np.any(mask):
                continue
            truth_values = y_true[mask]
            truth = int(np.bincount(truth_values, minlength=class_count).argmax())
            scene_probs = probabilities[mask]
            if scene_probs.ndim != 2 or scene_probs.shape[1] != class_count:
                continue
            finite_rows = np.all(np.isfinite(scene_probs), axis=1)
            if not np.any(finite_rows):
                continue
            scene_truth.append(truth)
            scene_prediction.append(int(np.mean(scene_probs[finite_rows], axis=0).argmax()))
        scene_metrics = self._classification_metrics(
            np.asarray(scene_truth, dtype=np.int64),
            np.asarray(scene_prediction, dtype=np.int64),
            class_count,
        )

        scene_payloads = self._scene_outcome_payloads()
        completed_payloads = [
            payload
            for payload in scene_payloads.values()
            if payload.get("outcome") in {"success", "failed"}
        ]
        completed = [str(payload["outcome"]) for payload in completed_payloads]
        success_count = sum(value == "success" for value in completed)
        task_by_label: dict[str, dict[str, Any]] = {}
        for label_id in range(class_count):
            rows = [
                payload
                for payload in completed_payloads
                if payload.get("label_id") == label_id
            ]
            successes = sum(payload.get("outcome") == "success" for payload in rows)
            task_by_label[str(label_id)] = {
                "completed_scenes": len(rows),
                "successful_scenes": successes,
                "failed_scenes": len(rows) - successes,
                "success_rate": float(successes / len(rows)) if rows else 0.0,
            }
        failure_reasons: dict[str, int] = {}
        for payload in completed_payloads:
            if payload.get("outcome") != "failed":
                continue
            reason = str(payload.get("reason", "unknown"))
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        task_metrics = {
            "completed_scenes": len(completed),
            "successful_scenes": success_count,
            "failed_scenes": sum(value == "failed" for value in completed),
            "success_rate": float(success_count / len(completed)) if completed else 0.0,
            "success_rate_wilson_95": self._wilson_interval(success_count, len(completed)),
            "endpoint_verified_scenes": sum(
                payload.get("endpoint_matches_safe_lane") is not None
                for payload in completed_payloads
            ),
            "by_label_id": task_by_label,
            "failure_reasons": failure_reasons,
        }
        return {
            "definitions": {
                "evaluation_filter": "quality_accepted & labels_true>=0",
                "primary_decision_filter": (
                    "quality_accepted & labels_true>=0 & adaptation_eligible"
                ),
                "continuous_dynamic_filter": (
                    "quality_accepted & labels_true>=0 & "
                    "training_roles=='continuous_context'"
                ),
                "raw_window_prediction": "argmax(probabilities) before online update",
                "operational_prediction": "confidence-thresholded command; abstention=-1",
                "balanced_accuracy": "macro recall over the fixed class set; missing-class recall=0",
                "scene_prediction": (
                    "mean probability across quality-accepted causally clean "
                    "primary windows in the same Scene"
                ),
                "car_task_success": (
                    "no collision event and Unity endpoint_lane==safe_lane; "
                    "report separately from decoding accuracy because confidence "
                    "abstention maps to STOP"
                ),
            },
            "evaluated_windows": int(np.sum(valid)),
            "primary_decision_windows": int(np.sum(primary_valid)),
            "primary_decision_scenes": int(
                np.unique(scene_indices[primary_valid & (scene_indices >= 0)]).size
            ),
            "primary_decision_quality_rejected": int(
                np.sum(adaptation_eligible & ~quality)
            ),
            "continuous_dynamic_windows": int(np.sum(continuous_valid)),
            "adaptation_committed_windows": int(np.sum(adaptation_committed)),
            "excluded_unlabeled_windows": int(np.sum(y_true < 0)),
            "excluded_quality_windows": int(np.sum((y_true >= 0) & ~quality)),
            "raw_window": raw_metrics,
            "operational_window": operational_metrics,
            "primary_decision": {
                "raw": primary_raw_metrics,
                "operational": primary_operational_metrics,
            },
            "continuous_dynamic": {
                "raw": continuous_raw_metrics,
                "operational": continuous_operational_metrics,
            },
            "scene_classification": scene_metrics,
            "car_task": task_metrics,
        }

    @staticmethod
    def _classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_count: int,
        *,
        abstain_value: int | None = None,
    ) -> dict[str, Any]:
        classes = max(int(class_count), 1)
        confusion = np.zeros((classes, classes), dtype=np.int64)
        for truth, prediction in zip(y_true, y_pred, strict=False):
            if 0 <= int(truth) < classes and 0 <= int(prediction) < classes:
                confusion[int(truth), int(prediction)] += 1
        support = np.bincount(
            np.asarray(y_true, dtype=np.int64)[np.asarray(y_true) >= 0],
            minlength=classes,
        )[:classes]
        recalls = np.divide(
            np.diag(confusion),
            support,
            out=np.zeros(classes, dtype=np.float64),
            where=support > 0,
        )
        total = int(len(y_true))
        correct = int(np.sum(np.asarray(y_true) == np.asarray(y_pred)))
        result = {
            "samples": total,
            "correct": correct,
            "accuracy": float(correct / total) if total else 0.0,
            "balanced_accuracy": float(np.mean(recalls)),
            "all_classes_observed": bool(np.all(support > 0)),
            "support": support.tolist(),
            "per_class_recall": recalls.tolist(),
            "confusion_matrix": confusion.tolist(),
            "accuracy_wilson_95": StreamWriter._wilson_interval(correct, total),
        }
        if abstain_value is not None:
            result["abstain_value"] = int(abstain_value)
        return result

    def _scene_outcomes(self) -> dict[int, str]:
        return {
            scene_index: str(payload.get("outcome", "incomplete"))
            for scene_index, payload in self._scene_outcome_payloads().items()
        }

    def _scene_outcome_payloads(self) -> dict[int, dict[str, Any]]:
        outcomes: dict[int, dict[str, Any]] = {}
        if not self._events_path.exists():
            return outcomes
        for line in self._events_path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event_type") != "scene_end":
                continue
            payload = event.get("payload", {}) or {}
            try:
                scene_index = int(payload["scene_index"])
            except (KeyError, TypeError, ValueError):
                continue
            outcomes[scene_index] = dict(payload)
        return outcomes

    def _timing_integrity(self) -> dict[str, Any]:
        packet_loss_count = 0
        invalid_timestamps = 0
        maximum_jitter = 0.0
        evaluated = 0
        for chunk_name in self._files:
            path = self._chunks_dir / chunk_name
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as chunk:
                if (
                    "window_start_monotonic" in chunk
                    and "window_end_monotonic" in chunk
                ):
                    starts = np.asarray(
                        chunk["window_start_monotonic"],
                        dtype=np.float64,
                    )
                    ends = np.asarray(
                        chunk["window_end_monotonic"],
                        dtype=np.float64,
                    )
                    evaluated += int(starts.size)
                    invalid_timestamps += int(
                        np.sum(
                            ~np.isfinite(starts)
                            | ~np.isfinite(ends)
                            | (ends <= starts)
                        )
                    )
                if "timing_packet_loss_count" in chunk:
                    losses = np.asarray(
                        chunk["timing_packet_loss_count"],
                        dtype=np.float64,
                    )
                    if losses.size and np.any(np.isfinite(losses)):
                        packet_loss_count = max(
                            packet_loss_count,
                            int(np.nanmax(losses)),
                        )
                if "timing_queueing_jitter_sec" in chunk:
                    jitter = np.asarray(
                        chunk["timing_queueing_jitter_sec"],
                        dtype=np.float64,
                    )
                    if jitter.size and np.any(np.isfinite(jitter)):
                        maximum_jitter = max(
                            maximum_jitter,
                            float(np.nanmax(jitter)),
                        )
        return {
            "evaluated_windows": evaluated,
            "invalid_window_timestamps": invalid_timestamps,
            "packet_loss_count": packet_loss_count,
            "maximum_queueing_jitter_sec": maximum_jitter,
        }

    def _checksums(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        paths = [self._events_path]
        paths.extend(self._chunks_dir / name for name in self._files)
        revisions_dir = self._output_dir / "model_revisions"
        if revisions_dir.exists():
            paths.extend(path for path in revisions_dir.rglob("*") if path.is_file())
        for path in paths:
            if not path.exists():
                continue
            records.append(
                {
                    "path": path.relative_to(self._output_dir).as_posix(),
                    "sha256": self._sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
        return records

    def _write_manifest_atomic(self, metadata: dict[str, Any]) -> None:
        manifest_path = self._output_dir / "manifest.json"
        temporary_path = self._output_dir / ".manifest.json.tmp"
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(
                self._json_safe(metadata),
                handle,
                ensure_ascii=False,
                indent=2,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, manifest_path)

    def _monotonic_to_unix(self, value: float) -> float:
        if not np.isfinite(value):
            return float("nan")
        return self._start_unix + (float(value) - self._start_monotonic)

    @staticmethod
    def _encode_sequence(value: tuple[Any, ...] | list[Any] | str) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(list(value), ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): StreamWriter._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [StreamWriter._json_safe(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    @staticmethod
    def _wilson_interval(successes: int, total: int) -> list[float]:
        if total <= 0:
            return [0.0, 0.0]
        z = 1.959963984540054
        n = float(total)
        p = float(successes) / n
        denominator = 1.0 + (z * z / n)
        centre = (p + (z * z / (2.0 * n))) / denominator
        margin = (
            z
            * np.sqrt((p * (1.0 - p) / n) + (z * z / (4.0 * n * n)))
            / denominator
        )
        return [float(max(0.0, centre - margin)), float(min(1.0, centre + margin))]
