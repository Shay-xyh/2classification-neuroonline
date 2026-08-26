"""Replay recorded realtime EEG through causal NeuroOnline updates.

The recorder stores raw EEG windows even though realtime inference consumes
preprocessed windows. This tool reconstructs that preprocessing, predicts each
committed window before revealing its label, and synchronously replays the
latest labeled history at the configured update boundaries.
"""

from __future__ import annotations

import argparse
from collections import deque
import copy
from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Callable

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adaptation.neuroonline import (  # noqa: E402
    NEUROONLINE_TRAINING_MECHANICS_VERSION,
    NeuroOnlineConfig,
    NeuroOnlineModelAdapter,
    _frequency_mask,
    _time_mask,
)
from models.factory import ModelFactory, TorchModelAdapter  # noqa: E402
from utils.preprocessing import preprocess_eeg_window  # noqa: E402


@dataclass(slots=True)
class RealtimeData:
    raw_windows: np.ndarray
    labels: np.ndarray
    scene_indices: np.ndarray
    event_ids: np.ndarray
    window_end_monotonic: np.ndarray
    recorded_quality_accepted: np.ndarray
    recorded_quality_peak_abs_uv: np.ndarray
    recorded_quality_clip_fraction: np.ndarray
    source_chunks: np.ndarray
    source_rows: np.ndarray
    chunk_artifacts: list[dict[str, Any]]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def snapshot_files(roots: list[Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for root in roots:
        if not root.exists():
            continue
        paths = [root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file())
        for path in paths:
            info = artifact(path)
            result[str(path.resolve())] = {
                "size_bytes": info["size_bytes"],
                "sha256": info["sha256"],
            }
    return result


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def classification_metrics(
    truth: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_classes: int,
) -> dict[str, Any]:
    truth = np.asarray(truth, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if truth.size == 0:
        return {"samples": 0}
    predictions = probabilities.argmax(axis=1)
    labels = np.arange(n_classes, dtype=np.int64)
    matrix = confusion_matrix(truth, predictions, labels=labels)
    totals = matrix.sum(axis=1)
    recalls = np.divide(
        np.diag(matrix),
        totals,
        out=np.zeros(n_classes, dtype=np.float64),
        where=totals > 0,
    )
    if np.unique(np.concatenate((truth, predictions))).size < 2:
        kappa = -1.0
    else:
        kappa = float(cohen_kappa_score(truth, predictions))
        if not np.isfinite(kappa):
            kappa = -1.0
    observed = totals > 0
    return {
        "samples": int(truth.size),
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_accuracy": float(np.mean(recalls[observed])),
        "kappa": kappa,
        "macro_f1": float(
            f1_score(truth, predictions, average="macro", zero_division=0)
        ),
        "worst_observed_class_recall": float(np.min(recalls[observed])),
        "all_classes_observed": bool(np.all(observed)),
        "all_classes_predicted": bool(
            set(np.unique(predictions).tolist()) == set(labels.tolist())
        ),
        "per_class_recall": {
            str(label): float(recalls[label]) for label in labels
        },
        "predicted_class_counts": {
            str(label): int(np.sum(predictions == label)) for label in labels
        },
        "true_class_counts": {
            str(label): int(np.sum(truth == label)) for label in labels
        },
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def aggregate_scenes(
    labels: np.ndarray,
    probabilities: np.ndarray,
    scene_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scene_order = list(dict.fromkeys(np.asarray(scene_indices, dtype=np.int64).tolist()))
    scene_labels: list[int] = []
    scene_probabilities: list[np.ndarray] = []
    for scene in scene_order:
        mask = scene_indices == scene
        unique_labels = np.unique(labels[mask])
        if unique_labels.size != 1:
            raise ValueError(f"Scene {scene} contains multiple labels: {unique_labels.tolist()}.")
        scene_labels.append(int(unique_labels[0]))
        scene_probabilities.append(np.mean(probabilities[mask], axis=0))
    return (
        np.asarray(scene_labels, dtype=np.int64),
        np.stack(scene_probabilities).astype(np.float32),
        np.asarray(scene_order, dtype=np.int64),
    )


def metric_bundle(
    labels: np.ndarray,
    probabilities: np.ndarray,
    scene_indices: np.ndarray,
    *,
    n_classes: int,
) -> dict[str, Any]:
    scene_labels, scene_probabilities, _ = aggregate_scenes(
        labels,
        probabilities,
        scene_indices,
    )
    return {
        "window": classification_metrics(
            labels,
            probabilities,
            n_classes=n_classes,
        ),
        "scene": classification_metrics(
            scene_labels,
            scene_probabilities,
            n_classes=n_classes,
        ),
    }


def load_committed_data(recording_dir: Path) -> RealtimeData:
    chunk_paths = sorted((recording_dir / "chunks").glob("chunk_*.npz"))
    if not chunk_paths:
        raise FileNotFoundError(f"No realtime chunks found under {recording_dir}.")
    required = {
        "eeg_windows",
        "labels_true",
        "scene_indices",
        "label_event_ids",
        "window_end_monotonic",
        "training_roles",
        "adaptation_committed",
        "quality_accepted",
        "quality_peak_abs_uv",
        "quality_clip_fraction",
    }
    collected: dict[str, list[np.ndarray]] = {
        "raw_windows": [],
        "labels": [],
        "scene_indices": [],
        "event_ids": [],
        "window_end_monotonic": [],
        "recorded_quality_accepted": [],
        "recorded_quality_peak_abs_uv": [],
        "recorded_quality_clip_fraction": [],
        "source_chunks": [],
        "source_rows": [],
    }
    chunk_artifacts: list[dict[str, Any]] = []
    for chunk_path in chunk_paths:
        with np.load(chunk_path, allow_pickle=False) as payload:
            missing = sorted(required - set(payload.files))
            if missing:
                raise ValueError(f"{chunk_path.name} is missing: {', '.join(missing)}")
            committed = payload["adaptation_committed"].astype(bool)
            rows = np.flatnonzero(committed)
            roles = payload["training_roles"][committed].astype(str)
            if np.any(roles != "primary_decision"):
                raise ValueError(f"{chunk_path.name} has committed non-primary windows.")
            collected["raw_windows"].append(payload["eeg_windows"][committed].astype(np.float32))
            collected["labels"].append(payload["labels_true"][committed].astype(np.int64))
            collected["scene_indices"].append(payload["scene_indices"][committed].astype(np.int64))
            collected["event_ids"].append(payload["label_event_ids"][committed].astype(str))
            collected["window_end_monotonic"].append(
                payload["window_end_monotonic"][committed].astype(np.float64)
            )
            collected["recorded_quality_accepted"].append(
                payload["quality_accepted"][committed].astype(bool)
            )
            collected["recorded_quality_peak_abs_uv"].append(
                payload["quality_peak_abs_uv"][committed].astype(np.float32)
            )
            collected["recorded_quality_clip_fraction"].append(
                payload["quality_clip_fraction"][committed].astype(np.float32)
            )
            collected["source_chunks"].append(
                np.full(rows.size, chunk_path.name, dtype=f"<U{len(chunk_path.name)}")
            )
            collected["source_rows"].append(rows.astype(np.int64))
        chunk_info = artifact(chunk_path)
        chunk_info["committed_windows"] = int(rows.size)
        chunk_artifacts.append(chunk_info)

    values = {
        name: np.concatenate(parts, axis=0)
        for name, parts in collected.items()
    }
    timestamps = values["window_end_monotonic"]
    if timestamps.size == 0:
        raise ValueError("The recording contains no committed adaptation windows.")
    if not np.all(np.diff(timestamps) > 0.0):
        raise ValueError("Committed windows are not in strict chronological order.")
    if values["raw_windows"].ndim != 3:
        raise ValueError(f"Expected raw windows [N,C,T], got {values['raw_windows'].shape}.")
    return RealtimeData(chunk_artifacts=chunk_artifacts, **values)


def preprocess_windows(
    data: RealtimeData,
    *,
    sfreq: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    processed: list[np.ndarray] = []
    accepted: list[bool] = []
    peaks: list[float] = []
    clip_fractions: list[float] = []
    reason_counts: dict[str, int] = {}
    for window in data.raw_windows:
        result = preprocess_eeg_window(window, sfreq=sfreq)
        processed.append(result.data)
        accepted.append(result.quality.accepted)
        peaks.append(result.quality.peak_abs_uv)
        clip_fractions.append(result.quality.clip_fraction)
        for reason in result.quality.reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    windows = np.stack(processed).astype(np.float32)
    accepted_array = np.asarray(accepted, dtype=bool)
    peaks_array = np.asarray(peaks, dtype=np.float64)
    clips_array = np.asarray(clip_fractions, dtype=np.float64)
    quality = {
        "accepted_windows": int(accepted_array.sum()),
        "rejected_windows": int((~accepted_array).sum()),
        "reason_counts": reason_counts,
        "matches_recorded_accepted": bool(
            np.array_equal(accepted_array, data.recorded_quality_accepted)
        ),
        "max_recorded_peak_abs_error_uv": float(
            np.max(np.abs(peaks_array - data.recorded_quality_peak_abs_uv))
        ),
        "max_recorded_clip_fraction_error": float(
            np.max(np.abs(clips_array - data.recorded_quality_clip_fraction))
        ),
    }
    return windows, quality


def distribution_stats(windows: np.ndarray) -> dict[str, Any]:
    array = np.asarray(windows)
    return {
        "shape": list(array.shape),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "absolute_maximum": float(np.max(np.abs(array))),
    }


def load_checkpoint_config(
    checkpoint: Path,
    *,
    compatible_legacy_versions: tuple[int, ...] = (),
) -> tuple[NeuroOnlineConfig, dict[str, Any]]:
    sidecar = Path(f"{checkpoint}.neuroonline.pt")
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload = (
        checkpoint_payload.get("neuroonline")
        if isinstance(checkpoint_payload, dict)
        else None
    )
    if not isinstance(payload, dict):
        if not sidecar.exists():
            raise FileNotFoundError(
                "NeuroOnline state was not embedded in the checkpoint and no "
                f"sidecar exists: {sidecar}"
            )
        payload = torch.load(sidecar, map_location="cpu", weights_only=True)
    mechanics_version = int(payload.get("training_mechanics_version", 1))
    allowed_versions = {
        NEUROONLINE_TRAINING_MECHANICS_VERSION,
        *(int(version) for version in compatible_legacy_versions),
    }
    if mechanics_version not in allowed_versions:
        raise ValueError(
            f"Checkpoint mechanics v{mechanics_version} does not match current "
            f"v{NEUROONLINE_TRAINING_MECHANICS_VERSION}."
        )
    saved = payload.get("config", {}) or {}
    allowed = {field.name for field in fields(NeuroOnlineConfig)}
    config_values = {name: value for name, value in saved.items() if name in allowed}
    config = NeuroOnlineConfig(**config_values)
    if not config.enabled:
        config = NeuroOnlineConfig(**{**asdict(config), "enabled": True})
    return config, payload


def _predict_probabilities(
    adapter: Any,
    windows: np.ndarray,
    *,
    mc_dropout_passes: int,
) -> np.ndarray:
    if mc_dropout_passes == 1:
        return adapter.predict_proba(windows)
    return adapter.predict_proba(
        windows,
        mc_dropout_passes=mc_dropout_passes,
    )


def causal_replay(
    adapter: Any,
    windows: np.ndarray,
    labels: np.ndarray,
    scene_indices: np.ndarray,
    *,
    config: NeuroOnlineConfig,
    n_classes: int,
    mc_dropout_passes: int = 1,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    atomic_scene_groups: bool = False,
    max_update_seen: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    original: deque[np.ndarray] = deque(maxlen=config.recent_samples)
    time_views: deque[np.ndarray] = deque(maxlen=config.recent_samples)
    frequency_views: deque[np.ndarray] = deque(maxlen=config.recent_samples)
    replay_labels: deque[int] = deque(maxlen=config.recent_samples)
    mask_generator = torch.Generator().manual_seed(config.random_seed)
    probabilities: list[np.ndarray] = []
    model_revisions: list[int] = []
    update_history: list[dict[str, Any]] = []
    revision = 0

    index = 0
    next_update_at = config.history_threshold
    while index < len(windows):
        group_end = index + 1
        if atomic_scene_groups:
            while (
                group_end < len(windows)
                and scene_indices[group_end] == scene_indices[index]
            ):
                group_end += 1
        group_probabilities = _predict_probabilities(
            adapter,
            windows[index:group_end],
            mc_dropout_passes=mc_dropout_passes,
        )
        probabilities.extend(
            np.asarray(probability, dtype=np.float32)
            for probability in group_probabilities
        )
        model_revisions.extend([revision] * (group_end - index))

        if max_update_seen is not None and index >= max_update_seen:
            index = group_end
            continue

        for row in range(index, group_end):
            window = windows[row]
            tensor = torch.as_tensor(window, dtype=torch.float32).unsqueeze(0)
            time_view = _time_mask(tensor, config.mask_ratio, mask_generator)[0].numpy()
            frequency_view = _frequency_mask(tensor, config.mask_ratio, mask_generator)[0].numpy()
            original.append(window.copy())
            time_views.append(time_view)
            frequency_views.append(frequency_view)
            replay_labels.append(int(labels[row]))

        seen = group_end
        should_update = (
            seen >= next_update_at
            if atomic_scene_groups
            else seen >= config.history_threshold
            and (seen - config.history_threshold) % config.update_stride == 0
        )
        should_update = should_update and (
            max_update_seen is None or seen <= max_update_seen
        )
        if not should_update:
            index = group_end
            continue
        update_x = np.stack(original).astype(np.float32)
        update_time = np.stack(time_views).astype(np.float32)
        update_frequency = np.stack(frequency_views).astype(np.float32)
        update_y = np.asarray(replay_labels, dtype=np.int64)
        started = time.perf_counter()
        update_metrics = adapter.neuroonline_update(
            update_x,
            update_time,
            update_frequency,
            update_y,
            learning_rate=config.learning_rate,
            epochs=config.epochs,
            batch_size=config.update_batch_size,
        )
        duration = time.perf_counter() - started
        revision += 1
        partial_probabilities = np.stack(probabilities[:seen]).astype(np.float32)
        history_entry = {
            "update": revision,
            "trigger_seen_labeled_windows": seen,
            "snapshot_first_window_id": seen - len(update_y) + 1,
            "snapshot_last_window_id": seen,
            "snapshot_samples": int(len(update_y)),
            "snapshot_class_counts": np.bincount(
                update_y,
                minlength=n_classes,
            ).astype(int).tolist(),
            "duration_sec": float(duration),
            "cumulative_prequential": metric_bundle(
                labels[:seen],
                partial_probabilities,
                scene_indices[:seen],
                n_classes=n_classes,
            ),
            **{key: float(value) for key, value in update_metrics.items()},
        }
        update_history.append(history_entry)
        if progress_callback is not None:
            progress_callback(history_entry)
        if atomic_scene_groups:
            while next_update_at <= seen:
                next_update_at += config.update_stride
        index = group_end

    return (
        np.stack(probabilities).astype(np.float32),
        np.asarray(model_revisions, dtype=np.int64),
        update_history,
    )


def causal_guarded_replay(
    adapter: Any,
    windows: np.ndarray,
    labels: np.ndarray,
    scene_indices: np.ndarray,
    *,
    config: NeuroOnlineConfig,
    n_classes: int,
    mc_dropout_passes: int = 1,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Any, np.ndarray, np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    """Train shadow candidates and replace only after causal accuracy gains."""

    original: deque[np.ndarray] = deque(maxlen=config.recent_samples)
    time_views: deque[np.ndarray] = deque(maxlen=config.recent_samples)
    frequency_views: deque[np.ndarray] = deque(maxlen=config.recent_samples)
    replay_labels: deque[int] = deque(maxlen=config.recent_samples)
    mask_generator = torch.Generator().manual_seed(config.random_seed)
    probabilities: list[np.ndarray] = []
    model_revisions: list[int] = []
    update_history: list[dict[str, Any]] = []
    replacement_history: list[dict[str, Any]] = []
    active_adapter = adapter
    active_revision = 0
    active_correct = 0
    pending: dict[str, Any] | None = None

    def decide_pending(seen: int) -> None:
        nonlocal active_adapter, active_revision, pending
        if pending is None or pending["evaluated_windows"] == 0:
            return
        candidate_hypothetical_correct = (
            pending["correct_before_evaluation"] + pending["candidate_correct"]
        )
        accepted = candidate_hypothetical_correct > active_correct
        if accepted:
            active_adapter = pending["adapter"]
            active_revision += 1
        entry = {
            "candidate_update": pending["candidate_update"],
            "trained_after_window_id": pending["trained_after_window_id"],
            "evaluation_first_window_id": pending["evaluation_first_window_id"],
            "evaluation_last_window_id": seen,
            "evaluated_windows": pending["evaluated_windows"],
            "active_correct_on_evaluation": pending["active_correct"],
            "candidate_correct_on_evaluation": pending["candidate_correct"],
            "active_cumulative_correct": active_correct,
            "candidate_hypothetical_cumulative_correct": candidate_hypothetical_correct,
            "active_cumulative_accuracy": float(active_correct / seen),
            "candidate_hypothetical_cumulative_accuracy": float(
                candidate_hypothetical_correct / seen
            ),
            "cumulative_accuracy_delta": float(
                (candidate_hypothetical_correct - active_correct) / seen
            ),
            "accepted": accepted,
            "active_revision_after_decision": active_revision,
        }
        replacement_history.append(entry)
        update_history[pending["candidate_update"] - 1]["replacement"] = entry
        if progress_callback is not None:
            progress_callback({"kind": "replacement", **entry})
        pending = None

    for index, (window, label) in enumerate(zip(windows, labels, strict=True)):
        probability = _predict_probabilities(
            active_adapter,
            window[None, ...],
            mc_dropout_passes=mc_dropout_passes,
        )[0]
        probabilities.append(np.asarray(probability, dtype=np.float32))
        model_revisions.append(active_revision)
        active_prediction = int(np.argmax(probability))
        active_is_correct = int(active_prediction == int(label))
        active_correct += active_is_correct
        if pending is not None:
            candidate_probability = _predict_probabilities(
                pending["adapter"],
                window[None, ...],
                mc_dropout_passes=mc_dropout_passes,
            )[0]
            pending["evaluated_windows"] += 1
            pending["active_correct"] += active_is_correct
            pending["candidate_correct"] += int(
                int(np.argmax(candidate_probability)) == int(label)
            )

        tensor = torch.as_tensor(window, dtype=torch.float32).unsqueeze(0)
        time_view = _time_mask(tensor, config.mask_ratio, mask_generator)[0].numpy()
        frequency_view = _frequency_mask(tensor, config.mask_ratio, mask_generator)[0].numpy()
        original.append(window.copy())
        time_views.append(time_view)
        frequency_views.append(frequency_view)
        replay_labels.append(int(label))

        seen = index + 1
        should_train = (
            seen >= config.history_threshold
            and (seen - config.history_threshold) % config.update_stride == 0
        )
        if not should_train:
            continue

        decide_pending(seen)
        candidate = copy.deepcopy(active_adapter)
        update_x = np.stack(original).astype(np.float32)
        update_time = np.stack(time_views).astype(np.float32)
        update_frequency = np.stack(frequency_views).astype(np.float32)
        update_y = np.asarray(replay_labels, dtype=np.int64)
        started = time.perf_counter()
        update_metrics = candidate.neuroonline_update(
            update_x,
            update_time,
            update_frequency,
            update_y,
            learning_rate=config.learning_rate,
            epochs=config.epochs,
            batch_size=config.update_batch_size,
        )
        duration = time.perf_counter() - started
        candidate_update = len(update_history) + 1
        partial_probabilities = np.stack(probabilities).astype(np.float32)
        history_entry = {
            "candidate_update": candidate_update,
            "trigger_seen_labeled_windows": seen,
            "snapshot_first_window_id": seen - len(update_y) + 1,
            "snapshot_last_window_id": seen,
            "snapshot_samples": int(len(update_y)),
            "snapshot_class_counts": np.bincount(
                update_y,
                minlength=n_classes,
            ).astype(int).tolist(),
            "duration_sec": float(duration),
            "cumulative_prequential": metric_bundle(
                labels[:seen],
                partial_probabilities,
                scene_indices[:seen],
                n_classes=n_classes,
            ),
            "replacement": None,
            **{key: float(value) for key, value in update_metrics.items()},
        }
        update_history.append(history_entry)
        pending = {
            "adapter": candidate,
            "candidate_update": candidate_update,
            "trained_after_window_id": seen,
            "evaluation_first_window_id": seen + 1,
            "evaluated_windows": 0,
            "correct_before_evaluation": active_correct,
            "active_correct": 0,
            "candidate_correct": 0,
        }
        if progress_callback is not None:
            progress_callback({"kind": "training", **history_entry})

    decide_pending(len(windows))
    return (
        active_adapter,
        np.stack(probabilities).astype(np.float32),
        np.asarray(model_revisions, dtype=np.int64),
        update_history,
        replacement_history,
    )


def predict_in_batches(
    adapter: Any,
    windows: np.ndarray,
    *,
    batch_size: int = 64,
    mc_dropout_passes: int = 1,
) -> np.ndarray:
    batches = [
        _predict_probabilities(
            adapter,
            windows[start : start + batch_size],
            mc_dropout_passes=mc_dropout_passes,
        )
        for start in range(0, len(windows), batch_size)
    ]
    return np.concatenate(batches).astype(np.float32)


def phase_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    scene_indices: np.ndarray,
    model_revisions: np.ndarray,
    *,
    n_classes: int,
) -> list[dict[str, Any]]:
    phases: list[dict[str, Any]] = []
    for revision in np.unique(model_revisions):
        indices = np.flatnonzero(model_revisions == revision)
        phases.append(
            {
                "model_revision": int(revision),
                "first_window_id": int(indices[0] + 1),
                "last_window_id": int(indices[-1] + 1),
                **metric_bundle(
                    labels[indices],
                    probabilities[indices],
                    scene_indices[indices],
                    n_classes=n_classes,
                ),
            }
        )
    return phases


def build_adapter(
    checkpoint: Path,
    *,
    model_name: str,
    n_chans: int,
    n_times: int,
    n_classes: int,
    sfreq: float,
    config: NeuroOnlineConfig,
) -> NeuroOnlineModelAdapter:
    base = ModelFactory.get(
        model_name,
        n_chans=n_chans,
        n_times=n_times,
        n_classes=n_classes,
        sfreq=sfreq,
    )
    if not isinstance(base, TorchModelAdapter):
        raise TypeError("NeuroOnline replay requires a PyTorch model.")
    base.load(checkpoint)
    return NeuroOnlineModelAdapter(base, config=config, state_path=checkpoint)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.mc_dropout_passes < 1:
        raise ValueError("MC dropout passes must be at least 1.")
    checkpoint = args.checkpoint.resolve()
    recording_dir = args.recording.resolve()
    output_dir = args.output.resolve()
    sidecar = Path(f"{checkpoint}.neuroonline.pt")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    input_model_before = artifact(checkpoint)
    input_sidecar_before = artifact(sidecar)
    protected_before = snapshot_files(args.protected_model_dir)
    config, sidecar_payload = load_checkpoint_config(checkpoint)
    time_overrides = {
        "first_update_seconds": args.first_update_seconds,
        "update_stride_seconds": args.update_stride_seconds,
        "recent_history_seconds": args.recent_history_seconds,
        "update_batch_seconds": (
            config.update_batch_seconds
            if args.update_batch_seconds is None
            else args.update_batch_seconds
        ),
    }
    config = replace(
        config,
        **time_overrides,
        learning_rate=(
            config.learning_rate if args.learning_rate is None else args.learning_rate
        ),
        epochs=config.epochs if args.epochs is None else args.epochs,
        mask_ratio=(
            config.mask_ratio if args.mask_ratio is None else args.mask_ratio
        ),
        consistency_weight=(
            config.consistency_weight
            if args.consistency_weight is None
            else args.consistency_weight
        ),
        random_seed=(
            config.random_seed if args.random_seed is None else args.random_seed
        ),
    )
    selected_seconds = {
        name: float(config.time_budget[name]["requested_seconds"])
        for name in ("first_update", "update_stride", "recent_history")
    }
    expected_seconds = {
        "first_update": 32.0,
        "update_stride": 32.0,
        "recent_history": 640.0,
    }
    if any(
        not np.isclose(selected_seconds[name], seconds)
        for name, seconds in expected_seconds.items()
    ):
        raise ValueError(
            "This current-protocol replay requires first/stride/history "
            f"budgets of 32/32/640 window-seconds; got {selected_seconds}."
        )

    data = load_committed_data(recording_dir)
    labels_present = set(np.unique(data.labels).tolist())
    if labels_present != set(range(args.n_classes)):
        raise ValueError(f"Committed labels do not cover all classes: {sorted(labels_present)}")
    processed, quality = preprocess_windows(data, sfreq=args.sfreq)
    if processed.shape != data.raw_windows.shape:
        raise ValueError(
            f"Preprocessed shape {processed.shape} differs from raw shape {data.raw_windows.shape}."
        )
    calibration_stats = None
    if args.calibration_dataset is not None:
        with np.load(args.calibration_dataset, allow_pickle=False) as payload:
            calibration_stats = distribution_stats(payload[args.calibration_feature_key])

    seed_everything(config.random_seed)
    adapter = build_adapter(
        checkpoint,
        model_name=args.model_name,
        n_chans=processed.shape[1],
        n_times=processed.shape[2],
        n_classes=args.n_classes,
        sfreq=args.sfreq,
        config=config,
    )

    def show_progress(entry: dict[str, Any]) -> None:
        if entry.get("kind") == "replacement":
            print(
                f"GATE candidate={entry['candidate_update']} "
                f"evaluated={entry['evaluated_windows']} "
                f"active_acc={entry['active_cumulative_accuracy']:.4f} "
                f"candidate_acc={entry['candidate_hypothetical_cumulative_accuracy']:.4f} "
                f"accepted={entry['accepted']}",
                flush=True,
            )
            return
        window_metrics = entry["cumulative_prequential"]["window"]
        print(
            f"UPDATE {entry.get('update', entry.get('candidate_update'))} "
            f"after={entry['trigger_seen_labeled_windows']} "
            f"loss={entry['loss']:.6f} acc={window_metrics['accuracy']:.4f} "
            f"bacc={window_metrics['balanced_accuracy']:.4f} "
            f"duration={entry['duration_sec']:.2f}s",
            flush=True,
        )

    started = time.perf_counter()
    replacement_history: list[dict[str, Any]] = []
    if args.replacement_gate == "cumulative_accuracy":
        (
            adapter,
            prequential_probabilities,
            revisions,
            update_history,
            replacement_history,
        ) = causal_guarded_replay(
            adapter,
            processed,
            data.labels,
            data.scene_indices,
            config=config,
            n_classes=args.n_classes,
            mc_dropout_passes=args.mc_dropout_passes,
            progress_callback=show_progress,
        )
    else:
        prequential_probabilities, revisions, update_history = causal_replay(
            adapter,
            processed,
            data.labels,
            data.scene_indices,
            config=config,
            n_classes=args.n_classes,
            mc_dropout_passes=args.mc_dropout_passes,
            progress_callback=show_progress,
        )
    replay_duration = time.perf_counter() - started
    expected_updates = len(
        [
            seen
            for seen in range(1, len(processed) + 1)
            if seen >= config.history_threshold
            and (seen - config.history_threshold) % config.update_stride == 0
        ]
    )
    if len(update_history) != expected_updates:
        raise RuntimeError(
            f"Expected {expected_updates} updates, observed {len(update_history)}."
        )

    posthoc_probabilities = predict_in_batches(
        adapter,
        processed,
        mc_dropout_passes=args.mc_dropout_passes,
    )
    output_model = output_dir / args.output_model_name
    adapter.save(output_model)
    output_sidecar = Path(f"{output_model}.neuroonline.pt")

    input_model_after = artifact(checkpoint)
    input_sidecar_after = artifact(sidecar)
    protected_after = snapshot_files(args.protected_model_dir)
    input_unchanged = (
        input_model_before["sha256"] == input_model_after["sha256"]
        and input_sidecar_before["sha256"] == input_sidecar_after["sha256"]
    )
    protected_unchanged = protected_before == protected_after
    if not input_unchanged or not protected_unchanged:
        raise RuntimeError("A protected input model changed during isolated replay.")

    np.savez_compressed(
        output_dir / "prequential_predictions.npz",
        labels=data.labels,
        scene_indices=data.scene_indices,
        event_ids=data.event_ids,
        window_end_monotonic=data.window_end_monotonic,
        source_chunks=data.source_chunks,
        source_rows=data.source_rows,
        model_revisions=revisions,
        prequential_probabilities=prequential_probabilities,
        posthoc_probabilities=posthoc_probabilities,
    )
    prequential = metric_bundle(
        data.labels,
        prequential_probabilities,
        data.scene_indices,
        n_classes=args.n_classes,
    )
    posthoc = metric_bundle(
        data.labels,
        posthoc_probabilities,
        data.scene_indices,
        n_classes=args.n_classes,
    )
    final_window = posthoc["window"]
    final_scene = posthoc["scene"]
    collapse = {
        "posthoc_window_class_collapse": bool(
            not final_window["all_classes_predicted"]
            or final_window["worst_observed_class_recall"] <= 0.0
        ),
        "posthoc_scene_class_collapse": bool(
            not final_scene["all_classes_predicted"]
            or final_scene["worst_observed_class_recall"] <= 0.0
        ),
    }
    summary = {
        "schema_version": 1,
        "simulation": "causal_predict_then_update_neuroonline_realtime_replay",
        "mc_dropout_passes": int(args.mc_dropout_passes),
        "replacement_policy": {
            "gate": args.replacement_gate,
            "criterion": (
                "strictly_higher_hypothetical_cumulative_prequential_accuracy_"
                "on_future_shadow_windows"
                if args.replacement_gate == "cumulative_accuracy"
                else "replace_immediately_after_training"
            ),
            "ties_accepted": False,
            "candidates_accepted": int(
                sum(bool(entry["accepted"]) for entry in replacement_history)
            ),
            "candidates_rejected": int(
                sum(not bool(entry["accepted"]) for entry in replacement_history)
            ),
        },
        "training_mechanics_version": int(
            sidecar_payload.get("training_mechanics_version", 1)
        ),
        "source_recording": str(recording_dir),
        "source_chunks": data.chunk_artifacts,
        "source_checkpoint": {
            "model": input_model_before,
            "neuroonline_sidecar": input_sidecar_before,
        },
        "output_checkpoint": {
            "model": artifact(output_model),
            "neuroonline_sidecar": artifact(output_sidecar),
        },
        "input_checkpoint_unchanged": input_unchanged,
        "protected_model_files": {
            "roots": [str(path.resolve()) for path in args.protected_model_dir],
            "files_checked": len(protected_before),
            "unchanged": protected_unchanged,
        },
        "live_model_overwritten": False,
        "model_name": args.model_name,
        "sfreq": args.sfreq,
        "n_classes": args.n_classes,
        "config": asdict(config),
        "objective": (
            "CE(original)+CE(time_masked)+CE(freq_masked)+"
            "lambda*mean(MSE(time,original),MSE(freq,original))"
        ),
        "stream": {
            "committed_windows": int(len(data.labels)),
            "scenes": int(len(np.unique(data.scene_indices))),
            "strictly_chronological": True,
            "class_counts": {
                str(label): int(np.sum(data.labels == label))
                for label in range(args.n_classes)
            },
            "expected_update_triggers": [
                int(entry["trigger_seen_labeled_windows"])
                for entry in update_history
            ],
            "updates_completed": len(update_history),
            "untrained_tail_windows": int(
                len(processed) - update_history[-1]["trigger_seen_labeled_windows"]
                if update_history
                else len(processed)
            ),
        },
        "preprocessing": {
            "raw": distribution_stats(data.raw_windows),
            "reconstructed_processed": distribution_stats(processed),
            "calibration_processed_reference": calibration_stats,
            "quality": quality,
        },
        "prequential": prequential,
        "prequential_phases": phase_metrics(
            data.labels,
            prequential_probabilities,
            data.scene_indices,
            revisions,
            n_classes=args.n_classes,
        ),
        "update_history": update_history,
        "replacement_history": replacement_history,
        "posthoc_final_model_on_all_stream_windows": posthoc,
        "class_collapse": collapse,
        "duration_sec": float(replay_duration),
    }
    save_json(output_dir / "simulation_summary.json", summary)
    print(f"SUMMARY {output_dir / 'simulation_summary.json'}", flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="cbramod")
    parser.add_argument("--sfreq", type=float, default=200.0)
    parser.add_argument("--n-classes", type=int, default=2)
    parser.add_argument("--first-update-seconds", type=float, default=32.0)
    parser.add_argument("--update-stride-seconds", type=float, default=32.0)
    parser.add_argument("--recent-history-seconds", type=float, default=640.0)
    parser.add_argument("--update-batch-seconds", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--mask-ratio", type=float)
    parser.add_argument("--consistency-weight", type=float)
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--mc-dropout-passes", type=int, default=1)
    parser.add_argument("--calibration-dataset", type=Path)
    parser.add_argument("--calibration-feature-key", default="processed_windows")
    parser.add_argument(
        "--replacement-gate",
        choices=("none", "cumulative_accuracy"),
        default="none",
        help=(
            "Train shadow candidates and replace only when future-window "
            "cumulative accuracy strictly increases."
        ),
    )
    parser.add_argument(
        "--protected-model-dir",
        type=Path,
        action="append",
        default=[],
        help="Directory or model file whose hashes must remain unchanged.",
    )
    parser.add_argument(
        "--output-model-name",
        default="cbramod_seed2026_after_online_replay.pt",
    )
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
