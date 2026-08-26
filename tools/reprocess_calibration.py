"""Rebuild calibration windows from raw-rate continuous EEG and trial metadata."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from utils.preprocessing import (
    continuous_preprocessing_metadata,
    finalize_preprocessed_window,
    preprocess_eeg_continuous,
)


def _window_specs(
    *,
    window_sec: float,
    stride_sec: float,
    control_start_sec: float,
    control_stop_sec: float,
) -> list[float]:
    last_start = control_stop_sec - window_sec
    if last_start < control_start_sec:
        return []
    count = int(np.floor((last_start - control_start_sec) / stride_sec + 1e-9)) + 1
    return [control_start_sec + index * stride_sec for index in range(count)]


def _restore_peak_only(reasons: tuple[str, ...], *, enabled: bool) -> bool:
    return bool(enabled and set(reasons) == {"extreme_amplitude"})


def build_windows(
    continuous_eeg: np.ndarray,
    trials: list[dict[str, Any]],
    *,
    source_sfreq: float,
    target_sfreq: float,
    window_sec: float,
    stride_sec: float,
    control_start_sec: float,
    control_stop_sec: float,
    channel_indices: np.ndarray | None = None,
    allow_peak_only: bool = False,
) -> dict[str, np.ndarray]:
    """Create target-rate raw and processed windows without changing source files."""

    if continuous_eeg.ndim != 2:
        raise ValueError(f"Expected continuous EEG shaped (channels, time), got {continuous_eeg.shape}.")

    source_window_samples = int(round(window_sec * source_sfreq))
    target_window_samples = int(round(window_sec * target_sfreq))
    offsets_sec = _window_specs(
        window_sec=window_sec,
        stride_sec=stride_sec,
        control_start_sec=control_start_sec,
        control_stop_sec=control_stop_sec,
    )
    if not offsets_sec:
        raise ValueError("The configured control interval cannot contain a complete window.")

    raw_windows: list[np.ndarray] = []
    processed_windows: list[np.ndarray] = []
    labels: list[int] = []
    trial_ids: list[int] = []
    block_indices: list[int] = []
    trial_indices: list[int] = []
    windows_in_trial: list[int] = []
    starts_source: list[int] = []
    starts_target: list[int] = []
    clip_fractions: list[float] = []
    peak_abs_uv: list[float] = []
    bad_channel_fractions: list[float] = []
    bad_channel_indices: list[str] = []
    rejected_block_indices: list[int] = []
    rejected_trial_indices: list[int] = []
    rejected_window_indices: list[int] = []
    rejected_reasons: list[str] = []
    rejected_peak_abs_uv: list[float] = []
    rejected_clip_fraction: list[float] = []
    rejected_peak_channel_indices: list[int] = []
    peak_only_restored: list[bool] = []
    rejection_reason_counts: dict[str, int] = {}
    rejected_windows = 0
    selected_channels = (
        np.arange(continuous_eeg.shape[0], dtype=np.int64)
        if channel_indices is None
        else np.asarray(channel_indices, dtype=np.int64)
    )
    if selected_channels.ndim != 1 or selected_channels.size == 0:
        raise ValueError("channel_indices must select at least one channel.")
    if selected_channels.min() < 0 or selected_channels.max() >= continuous_eeg.shape[0]:
        raise ValueError("channel_indices contains a channel outside the continuous EEG array.")
    selected_continuous = np.asarray(
        continuous_eeg[selected_channels],
        dtype=np.float32,
    )
    continuous = preprocess_eeg_continuous(
        selected_continuous,
        source_sfreq=source_sfreq,
        target_sfreq=target_sfreq,
    )

    for trial_id, trial in enumerate(trials):
        control_on = int(trial["motor_imagery_on_sample"])
        for window_in_trial, offset_sec in enumerate(offsets_sec):
            start_source = control_on + int(round(offset_sec * source_sfreq))
            stop_source = start_source + source_window_samples
            if start_source < 0 or stop_source > continuous_eeg.shape[1]:
                continue

            start_target = int(round(start_source * target_sfreq / source_sfreq))
            stop_target = start_target + target_window_samples
            target_window = continuous.raw_data[:, start_target:stop_target]
            if target_window.shape[-1] != target_window_samples:
                raise RuntimeError(
                    f"Continuous preprocessing produced {target_window.shape[-1]} points; "
                    f"expected {target_window_samples}."
                )

            filtered_window = continuous.data[:, start_target:stop_target]
            nonfinite_fraction = float(
                np.mean(
                    continuous.source_nonfinite_mask[
                        :,
                        start_source:stop_source,
                    ]
                )
            )
            result = finalize_preprocessed_window(
                filtered_window,
                bad_channel_indices=continuous.bad_channel_indices,
                nonfinite_fraction=nonfinite_fraction,
            )
            restored_peak_only = _restore_peak_only(
                result.quality.reasons,
                enabled=allow_peak_only,
            )
            if not result.quality.accepted and not restored_peak_only:
                rejected_windows += 1
                rejected_block_indices.append(int(trial.get("block_index", -1)))
                rejected_trial_indices.append(int(trial.get("trial_index", trial_id)))
                rejected_window_indices.append(window_in_trial)
                rejected_reasons.append(
                    json.dumps(list(result.quality.reasons), separators=(",", ":"))
                )
                rejected_peak_abs_uv.append(float(result.quality.peak_abs_uv))
                rejected_clip_fraction.append(float(result.quality.clip_fraction))
                channel_peaks = np.max(np.abs(filtered_window), axis=1)
                rejected_peak_channel_indices.append(int(np.argmax(channel_peaks)))
                for reason in result.quality.reasons:
                    rejection_reason_counts[reason] = (
                        rejection_reason_counts.get(reason, 0) + 1
                    )
                continue

            raw_windows.append(target_window)
            clip_fractions.append(result.quality.clip_fraction)
            peak_abs_uv.append(result.quality.peak_abs_uv)
            bad_channel_fractions.append(result.quality.bad_channel_fraction)
            bad_channel_indices.append(
                json.dumps(
                    list(result.quality.bad_channel_indices),
                    separators=(",", ":"),
                )
            )
            peak_only_restored.append(restored_peak_only)
            processed_windows.append(result.data)
            labels.append(int(trial["label_id"]))
            trial_ids.append(int(trial.get("source_trial_id", trial_id)))
            block_indices.append(int(trial.get("block_index", -1)))
            trial_indices.append(int(trial.get("trial_index", trial_id)))
            windows_in_trial.append(window_in_trial)
            starts_source.append(start_source)
            starts_target.append(start_target)

    if not raw_windows:
        raise RuntimeError(
            "No windows could be reconstructed from the supplied EEG and trial metadata: "
            f"rejected={rejected_windows}, reasons={rejection_reason_counts}."
        )

    return {
        "raw_windows": np.stack(raw_windows).astype(np.float32),
        "processed_windows": np.stack(processed_windows).astype(np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
        "trial_ids": np.asarray(trial_ids, dtype=np.int64),
        "block_indices": np.asarray(block_indices, dtype=np.int64),
        "trial_indices": np.asarray(trial_indices, dtype=np.int64),
        "window_indices": np.asarray(windows_in_trial, dtype=np.int64),
        "window_start_source": np.asarray(starts_source, dtype=np.int64),
        "window_start_target": np.asarray(starts_target, dtype=np.int64),
        "quality_clip_fraction": np.asarray(clip_fractions, dtype=np.float32),
        "quality_peak_abs_uv": np.asarray(peak_abs_uv, dtype=np.float32),
        "quality_bad_channel_fraction": np.asarray(
            bad_channel_fractions,
            dtype=np.float32,
        ),
        "quality_bad_channel_indices": np.asarray(
            bad_channel_indices,
            dtype=np.str_,
        ),
        "quality_peak_only_restored": np.asarray(
            peak_only_restored, dtype=np.bool_
        ),
        "quality_rejected_windows": np.asarray([rejected_windows], dtype=np.int64),
        "quality_rejection_reason_counts": np.asarray(
            [
                json.dumps(
                    rejection_reason_counts,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ],
            dtype=np.str_,
        ),
        "rejected_block_indices": np.asarray(rejected_block_indices, dtype=np.int64),
        "rejected_trial_indices": np.asarray(rejected_trial_indices, dtype=np.int64),
        "rejected_window_indices": np.asarray(rejected_window_indices, dtype=np.int64),
        "rejected_reasons": np.asarray(rejected_reasons, dtype=np.str_),
        "rejected_peak_abs_uv": np.asarray(rejected_peak_abs_uv, dtype=np.float32),
        "rejected_clip_fraction": np.asarray(
            rejected_clip_fraction, dtype=np.float32
        ),
        "rejected_peak_channel_indices": np.asarray(
            rejected_peak_channel_indices, dtype=np.int64
        ),
        "continuous_bad_channel_indices": np.asarray(
            list(continuous.bad_channel_indices), dtype=np.int64
        ),
        "selected_channels": selected_channels,
        "source_sfreq": np.asarray([source_sfreq], dtype=np.float32),
        "sfreq": np.asarray([target_sfreq], dtype=np.float32),
        "window_sec": np.asarray([window_sec], dtype=np.float32),
        "step_sec": np.asarray([stride_sec], dtype=np.float32),
    }


def _save_dataset(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _select_trials(
    trials: list[dict[str, Any]],
    *,
    excluded_blocks: tuple[int, ...],
    exclusion_notes: dict[int, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalized_blocks = tuple(sorted(set(int(value) for value in excluded_blocks)))
    available_blocks = {int(trial.get("block_index", -1)) for trial in trials}
    missing_blocks = set(normalized_blocks) - available_blocks
    if missing_blocks:
        raise ValueError(
            "Excluded calibration blocks are absent from metadata: "
            f"{sorted(missing_blocks)}."
        )
    unused_notes = set(exclusion_notes) - set(normalized_blocks)
    if unused_notes:
        raise ValueError(
            "Exclusion notes were supplied for blocks that are not excluded: "
            f"{sorted(unused_notes)}."
        )

    selected = [
        trial
        for trial in trials
        if int(trial.get("block_index", -1)) not in normalized_blocks
    ]
    if not selected:
        raise ValueError("Manual block exclusions removed every calibration trial.")

    excluded_entries: list[dict[str, Any]] = []
    for block_index in normalized_blocks:
        block_trials = [
            trial
            for trial in trials
            if int(trial.get("block_index", -1)) == block_index
        ]
        excluded_entries.append(
            {
                "block_index": block_index,
                "reason": exclusion_notes.get(
                    block_index,
                    "operator_reported_interference",
                ),
                "trial_count": len(block_trials),
                "class_counts": dict(
                    sorted(Counter(str(trial["label"]) for trial in block_trials).items())
                ),
            }
        )

    return selected, {
        "policy": "whole_block_exclusion_before_windowing",
        "excluded_blocks": excluded_entries,
        "source_trial_count": len(trials),
        "excluded_trial_count": len(trials) - len(selected),
        "included_trial_count": len(selected),
        "included_class_counts": dict(
            sorted(Counter(str(trial["label"]) for trial in selected).items())
        ),
    }


def promote_corrected_datasets(paths: list[Path]) -> list[Path]:
    """Atomically promote verified corrected datasets while retaining originals."""

    promoted: list[Path] = []
    for corrected_path in paths:
        if not corrected_path.name.endswith("_corrected.npz"):
            raise ValueError(f"Not a corrected calibration dataset: {corrected_path}")
        with np.load(corrected_path) as payload:
            if "processed_windows" not in payload or "labels" not in payload:
                raise ValueError(f"Corrected dataset is missing required arrays: {corrected_path}")
            window_count = int(payload["processed_windows"].shape[0])
            label_count = int(payload["labels"].shape[0])
        if window_count <= 0 or label_count != window_count:
            raise ValueError(
                f"Refusing to promote unusable corrected dataset {corrected_path}: "
                f"windows={window_count}, labels={label_count}."
            )

        canonical_name = corrected_path.name.replace("_corrected.npz", ".npz")
        canonical_path = corrected_path.with_name(canonical_name)
        backup_path = canonical_path.with_name(
            f"{canonical_path.stem}.pre_reprocess{canonical_path.suffix}"
        )
        if canonical_path.exists() and not backup_path.exists():
            shutil.copy2(canonical_path, backup_path)

        temporary = canonical_path.with_suffix(canonical_path.suffix + ".promote.tmp")
        shutil.copy2(corrected_path, temporary)
        os.replace(temporary, canonical_path)
        promoted.append(canonical_path)
    return promoted


def reprocess_session(
    session_dir: Path,
    output_dir: Path,
    *,
    source_sfreq: float,
    target_sfreq: float,
    eeg_channel_count: int | None = None,
    excluded_blocks: tuple[int, ...] = (),
    exclusion_notes: dict[int, str] | None = None,
    allow_peak_only: bool = False,
    stride_sec: float | None = None,
) -> list[Path]:
    metadata_path = session_dir / "metadata.json"
    continuous_path = session_dir / "continuous_eeg.npy"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    continuous = np.load(continuous_path, mmap_mode="r")
    if eeg_channel_count is not None:
        if eeg_channel_count <= 0 or eeg_channel_count > continuous.shape[0]:
            raise ValueError(
                f"eeg_channel_count must be between 1 and {continuous.shape[0]}, "
                f"got {eeg_channel_count}."
            )
        channel_indices = np.arange(eeg_channel_count, dtype=np.int64)
    else:
        channel_indices = None

    control_start_sec, control_stop_sec = (
        float(value) for value in metadata["motor_imagery_window_range_sec"]
    )
    effective_stride_sec = (
        float(metadata.get("stride_sec", 0.5))
        if stride_sec is None
        else float(stride_sec)
    )
    if effective_stride_sec <= 0.0:
        raise ValueError("stride_sec must be positive.")
    source_trials = [
        {**trial, "source_trial_id": trial_id}
        for trial_id, trial in enumerate(metadata["trials"])
    ]
    trials, manual_exclusions = _select_trials(
        source_trials,
        excluded_blocks=excluded_blocks,
        exclusion_notes=exclusion_notes or {},
    )

    datasets = [
        (
            "training_windows_main_corrected.npz",
            float(metadata["window_sec"]),
            effective_stride_sec,
        ),
    ]

    written: list[Path] = []
    for filename, window_sec, dataset_stride in datasets:
        payload = build_windows(
            continuous,
            trials,
            source_sfreq=source_sfreq,
            target_sfreq=target_sfreq,
            window_sec=window_sec,
            stride_sec=dataset_stride,
            control_start_sec=control_start_sec,
            control_stop_sec=control_stop_sec,
            channel_indices=channel_indices,
            allow_peak_only=allow_peak_only,
        )
        payload["excluded_block_indices"] = np.asarray(
            sorted(set(int(value) for value in excluded_blocks)),
            dtype=np.int64,
        )
        payload["source_trial_count"] = np.asarray(
            [manual_exclusions["source_trial_count"]],
            dtype=np.int64,
        )
        payload["included_trial_count"] = np.asarray(
            [manual_exclusions["included_trial_count"]],
            dtype=np.int64,
        )
        payload["allow_peak_only"] = np.asarray([allow_peak_only], dtype=np.bool_)
        output_path = output_dir / filename
        _save_dataset(output_path, payload)
        written.append(output_path)

    corrected_metadata = dict(metadata)
    corrected_metadata["original_formal_trial_count"] = int(
        metadata.get("formal_trial_count", len(source_trials))
    )
    corrected_metadata["formal_trial_count"] = len(trials)
    corrected_metadata["trials"] = trials
    corrected_metadata["stride_sec"] = effective_stride_sec
    corrected_metadata["auxiliary_windows_exported"] = False
    corrected_metadata["manual_exclusions"] = manual_exclusions
    restored_count = 0
    for path in written:
        with np.load(path, allow_pickle=False) as payload:
            restored_count += int(np.sum(payload["quality_peak_only_restored"]))
    corrected_metadata["quality_policy_override"] = {
        "allow_peak_only": bool(allow_peak_only),
        "restored_window_count": restored_count,
        "remaining_policy": (
            "reject excessive_clipping; restore windows whose only reason was "
            "extreme_amplitude"
            if allow_peak_only
            else "default preprocessing quality policy"
        ),
    }
    corrected_metadata["preprocessing"] = {
        **continuous_preprocessing_metadata(),
        "source_sfreq": source_sfreq,
        "target_sfreq": target_sfreq,
        "resampling": "scipy.signal.resample_poly",
        "continuous_span": "complete_calibration_session",
        "selected_channel_indices": (
            list(range(eeg_channel_count)) if eeg_channel_count is not None else "all"
        ),
        "grouping_fields": ["trial_ids", "block_indices", "trial_indices", "window_indices"],
        "quality_fields": [
            "quality_clip_fraction",
            "quality_peak_abs_uv",
            "quality_bad_channel_fraction",
            "quality_rejected_windows",
        ],
    }
    (output_dir / "metadata_corrected.json").write_text(
        json.dumps(corrected_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source-sfreq", type=float, required=True)
    parser.add_argument("--target-sfreq", type=float, default=200.0)
    parser.add_argument(
        "--stride-sec",
        type=float,
        help="Override the calibration-only window stride recorded by the source session.",
    )
    parser.add_argument(
        "--eeg-channel-count",
        type=int,
        help="Keep only the leading EEG channels; excludes trailing ECG/EOG auxiliaries.",
    )
    parser.add_argument(
        "--promote-main",
        action="store_true",
        help=(
            "After validation, atomically promote *_corrected.npz for training and "
            "retain each previous dataset as *.pre_reprocess.npz."
        ),
    )
    parser.add_argument(
        "--exclude-block",
        action="append",
        default=[],
        type=int,
        metavar="INDEX",
        help="Exclude one original block index before windowing; may be repeated.",
    )
    parser.add_argument(
        "--exclusion-note",
        action="append",
        default=[],
        metavar="INDEX=TEXT",
        help="Record the reason for an excluded block; may be repeated.",
    )
    parser.add_argument(
        "--allow-peak-only",
        action="store_true",
        help=(
            "Restore windows rejected only for a peak above 300 uV; windows with "
            "excessive clipping remain rejected."
        ),
    )
    return parser.parse_args()


def _parse_exclusion_notes(values: list[str]) -> dict[int, str]:
    notes: dict[int, str] = {}
    for value in values:
        block_text, separator, reason = value.partition("=")
        if not separator or not reason.strip():
            raise ValueError(
                f"Invalid exclusion note {value!r}; expected INDEX=TEXT."
            )
        block_index = int(block_text)
        if block_index in notes:
            raise ValueError(f"Duplicate exclusion note for block {block_index}.")
        notes[block_index] = reason.strip()
    return notes


def main() -> None:
    args = _parse_args()
    exclusion_notes = _parse_exclusion_notes(args.exclusion_note)
    written = reprocess_session(
        args.session_dir.resolve(),
        args.output_dir.resolve(),
        source_sfreq=args.source_sfreq,
        target_sfreq=args.target_sfreq,
        eeg_channel_count=args.eeg_channel_count,
        excluded_blocks=tuple(args.exclude_block),
        exclusion_notes=exclusion_notes,
        allow_peak_only=args.allow_peak_only,
        stride_sec=args.stride_sec,
    )
    for path in written:
        with np.load(path) as payload:
            print(
                f"corrected={path} windows={int(payload['processed_windows'].shape[0])} "
                f"rejected={int(payload['quality_rejected_windows'][0])}"
            )
    if args.promote_main:
        for path in promote_corrected_datasets(written):
            print(f"promoted={path}")


if __name__ == "__main__":
    main()
