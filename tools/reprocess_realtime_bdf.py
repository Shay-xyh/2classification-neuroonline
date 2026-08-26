"""Rebuild target-rate labeled windows from a realtime BDF and its manifest."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pyedflib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.reprocess_calibration import build_windows
from utils.preprocessing import (
    continuous_preprocessing_metadata,
    finalize_preprocessed_window,
    preprocess_eeg_continuous,
)


LABEL_NAME_TO_ID = {"left": 0, "right": 1, "idle": 2}


def _max_recorded_trial(session_dir: Path) -> int:
    maximum = -1
    for chunk_path in sorted((session_dir / "chunks").glob("*.npz")):
        with np.load(chunk_path, allow_pickle=False) as payload:
            for event_id in payload["label_event_ids"]:
                text = str(event_id)
                if text.startswith("cue-"):
                    maximum = max(maximum, int(text.rsplit("-", 1)[-1]))
    if maximum < 0:
        raise RuntimeError("No cue trial identifiers were found in the realtime chunks.")
    return maximum


def _load_labeled_online_windows(
    session_dir: Path,
    *,
    match_channel: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    templates: list[np.ndarray] = []
    labels: list[int] = []
    trial_ids: list[int] = []
    stream_indices: list[int] = []
    stream_index = 0
    for chunk_path in sorted((session_dir / "chunks").glob("*.npz")):
        with np.load(chunk_path, allow_pickle=False) as payload:
            windows = payload["eeg_windows"]
            true_labels = payload["labels_true"].astype(np.int64)
            event_ids = payload["label_event_ids"]
            if match_channel >= windows.shape[1]:
                raise ValueError(
                    f"Match channel {match_channel} is outside {windows.shape[1]} channels."
                )
            for local_index, event_id in enumerate(event_ids):
                event_text = str(event_id)
                label = int(true_labels[local_index])
                if not event_text.startswith("cue-") or label < 0:
                    continue
                templates.append(
                    np.asarray(windows[local_index, match_channel], dtype=np.float32)
                )
                labels.append(label)
                trial_ids.append(int(event_text.rsplit("-", 1)[-1]))
                stream_indices.append(stream_index + local_index)
            stream_index += len(event_ids)
    if not templates:
        raise RuntimeError("No labeled cued-protocol windows were found in realtime chunks.")
    return (
        np.stack(templates),
        np.asarray(labels, dtype=np.int64),
        np.asarray(trial_ids, dtype=np.int64),
        np.asarray(stream_indices, dtype=np.int64),
    )


def _match_templates_to_bdf(
    bdf_signal: np.ndarray,
    templates: np.ndarray,
    *,
    tolerance_uv: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Locate raw online windows in BDF using several exact-amplitude fingerprints."""

    template_samples = templates.shape[1]
    searchable = np.asarray(bdf_signal[:-template_samples], dtype=np.float32)
    offsets = np.unique(
        np.asarray(
            [0, template_samples // 7, template_samples // 3, 2 * template_samples // 3, template_samples - 1],
            dtype=np.int64,
        )
    )
    matches = np.full(len(templates), -1, dtype=np.int64)
    mean_absolute_errors = np.full(len(templates), np.nan, dtype=np.float32)
    previous_match = -1
    for index, template in enumerate(templates):
        if float(np.std(template)) < 1e-6:
            continue
        candidates = np.flatnonzero(
            np.abs(searchable - float(template[0])) <= tolerance_uv
        )
        for offset in offsets[1:]:
            if candidates.size == 0:
                break
            candidates = candidates[
                np.abs(bdf_signal[candidates + offset] - float(template[offset]))
                <= tolerance_uv
            ]
        chronological = candidates[candidates >= previous_match]
        if chronological.size:
            candidates = chronological
        if candidates.size == 0:
            continue
        errors = np.asarray(
            [
                np.mean(
                    np.abs(
                        bdf_signal[start : start + template_samples]
                        - template
                    )
                )
                for start in candidates
            ],
            dtype=np.float64,
        )
        best = int(np.argmin(errors))
        if float(errors[best]) > tolerance_uv:
            continue
        match = int(candidates[best])
        matches[index] = match
        mean_absolute_errors[index] = float(errors[best])
        previous_match = match
    return matches, mean_absolute_errors


def reprocess_realtime_from_waveform_matches(
    bdf_path: Path,
    session_dir: Path,
    output_dir: Path,
    *,
    target_sfreq: float,
    eeg_channel_count: int,
    match_channel: int = 0,
    match_tolerance_uv: float = 0.08,
) -> Path:
    """Recover correctly timed 2-second windows from labeled online raw windows."""

    manifest = json.loads((session_dir / "manifest.json").read_text(encoding="utf-8"))
    templates, labels, trial_ids, stream_indices = _load_labeled_online_windows(
        session_dir,
        match_channel=match_channel,
    )
    source = pyedflib.EdfReader(str(bdf_path))
    try:
        source_sfreqs = [
            float(source.getSampleFrequency(index)) for index in range(eeg_channel_count)
        ]
        if not np.allclose(source_sfreqs, source_sfreqs[0]):
            raise ValueError(f"Selected BDF channels have inconsistent rates: {source_sfreqs}")
        source_sfreq = source_sfreqs[0]
        match_signal = source.readSignal(match_channel).astype(np.float32)
        matched_starts, match_errors = _match_templates_to_bdf(
            match_signal,
            templates,
            tolerance_uv=match_tolerance_uv,
        )
        keep = matched_starts >= 0
        if not np.any(keep):
            raise RuntimeError("None of the labeled online windows could be matched to BDF.")
        templates = templates[keep]
        labels = labels[keep]
        trial_ids = trial_ids[keep]
        stream_indices = stream_indices[keep]
        matched_starts = matched_starts[keep]
        match_errors = match_errors[keep]

        window_sec = float(manifest["window_sec"])
        stored_samples = int(templates.shape[1])
        source_window_samples = int(round(window_sec * source_sfreq))
        target_window_samples = int(round(window_sec * target_sfreq))
        window_ends_source = matched_starts + stored_samples
        window_starts_source = window_ends_source - source_window_samples
        valid = window_starts_source >= 0
        if not np.all(valid):
            templates = templates[valid]
            labels = labels[valid]
            trial_ids = trial_ids[valid]
            stream_indices = stream_indices[valid]
            matched_starts = matched_starts[valid]
            match_errors = match_errors[valid]
            window_ends_source = window_ends_source[valid]
            window_starts_source = window_starts_source[valid]

        segment_start = int(window_starts_source.min())
        segment_stop = int(window_ends_source.max())
        source_segment = np.empty(
            (eeg_channel_count, segment_stop - segment_start),
            dtype=np.float32,
        )
        for channel in range(eeg_channel_count):
            source_segment[channel] = source.readSignal(
                channel,
                start=segment_start,
                n=segment_stop - segment_start,
            ).astype(np.float32)
        continuous = preprocess_eeg_continuous(
            source_segment,
            source_sfreq=source_sfreq,
            target_sfreq=target_sfreq,
        )
        channel_labels = [
            source.getLabel(index).strip() for index in range(eeg_channel_count)
        ]
    finally:
        source.close()

    sequence = list(manifest["online_label_source"]["sequence"])
    expected_labels = np.asarray(
        [LABEL_NAME_TO_ID[str(sequence[trial % len(sequence)])] for trial in trial_ids],
        dtype=np.int64,
    )
    if not np.array_equal(labels, expected_labels):
        mismatch_count = int(np.sum(labels != expected_labels))
        raise RuntimeError(
            f"{mismatch_count} matched windows disagree with the manifest label sequence."
        )

    processed_windows: list[np.ndarray] = []
    raw_windows: list[np.ndarray] = []
    clip_fractions: list[float] = []
    peak_abs_uv: list[float] = []
    for start_source, stop_source in zip(
        window_starts_source,
        window_ends_source,
        strict=True,
    ):
        local_start_source = int(start_source) - segment_start
        local_stop_source = int(stop_source) - segment_start
        start_target = int(
            round(local_start_source * target_sfreq / source_sfreq)
        )
        stop_target = start_target + target_window_samples
        raw_windows.append(continuous.raw_data[:, start_target:stop_target])
        result = finalize_preprocessed_window(
            continuous.data[:, start_target:stop_target],
            bad_channel_indices=continuous.bad_channel_indices,
            nonfinite_fraction=float(
                np.mean(
                    continuous.source_nonfinite_mask[
                        :,
                        local_start_source:local_stop_source,
                    ]
                )
            ),
        )
        clip_fractions.append(result.quality.clip_fraction)
        peak_abs_uv.append(result.quality.peak_abs_uv)
        processed_windows.append(result.data)

    per_trial_count: dict[int, int] = {}
    window_indices: list[int] = []
    for trial_id in trial_ids:
        trial = int(trial_id)
        window_indices.append(per_trial_count.get(trial, 0))
        per_trial_count[trial] = per_trial_count.get(trial, 0) + 1

    payload = {
        "raw_windows": np.stack(raw_windows).astype(np.float32),
        "processed_windows": np.stack(processed_windows).astype(np.float32),
        "labels": labels,
        "trial_ids": trial_ids,
        "block_indices": np.full(len(labels), -1, dtype=np.int64),
        "trial_indices": trial_ids.copy(),
        "window_indices": np.asarray(window_indices, dtype=np.int64),
        "window_start_source": window_starts_source.astype(np.int64),
        "window_end_source": window_ends_source.astype(np.int64),
        "matched_online_window_start_source": matched_starts.astype(np.int64),
        "online_stream_indices": stream_indices,
        "waveform_match_mae_uv": match_errors.astype(np.float32),
        "quality_clip_fraction": np.asarray(clip_fractions, dtype=np.float32),
        "quality_peak_abs_uv": np.asarray(peak_abs_uv, dtype=np.float32),
        "selected_channels": np.arange(eeg_channel_count, dtype=np.int64),
        "source_sfreq": np.asarray([source_sfreq], dtype=np.float32),
        "sfreq": np.asarray([target_sfreq], dtype=np.float32),
        "window_sec": np.asarray([window_sec], dtype=np.float32),
        "step_sec": np.asarray([float(manifest["step_sec"])], dtype=np.float32),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "training_windows_realtime_waveform_aligned.npz"
    np.savez_compressed(output_path, **payload)

    corrected_metadata = {
        "subject_id": manifest["subject_id"],
        "session_id": session_dir.name,
        "source_bdf": str(bdf_path.resolve()),
        "source_sfreq": source_sfreq,
        "target_sfreq": target_sfreq,
        "selected_channel_indices": list(range(eeg_channel_count)),
        "selected_channel_labels": channel_labels,
        "alignment_basis": (
            "exact waveform matching of labeled online raw windows to BDF; "
            "the corrected 2-second window ends with the matched 0.5-second raw window"
        ),
        "labeled_online_windows": int(len(keep)),
        "matched_windows": int(len(labels)),
        "unmatched_windows": int(np.sum(~keep)),
        "matched_trials": int(len(np.unique(trial_ids))),
        "match_channel": match_channel,
        "match_tolerance_uv": match_tolerance_uv,
        "match_mae_uv": {
            "mean": float(np.mean(match_errors)),
            "max": float(np.max(match_errors)),
        },
        "window_sec": window_sec,
        "step_sec": float(manifest["step_sec"]),
        "preprocessing": {
            **continuous_preprocessing_metadata(),
            "source_sfreq": source_sfreq,
            "target_sfreq": target_sfreq,
            "resampling": "scipy.signal.resample_poly",
            "continuous_span": "matched_realtime_bdf_segment",
        },
    }
    (output_dir / "metadata_waveform_aligned.json").write_text(
        json.dumps(corrected_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def reprocess_realtime(
    bdf_path: Path,
    session_dir: Path,
    output_dir: Path,
    *,
    target_sfreq: float,
    eeg_channel_count: int,
    scene_sec: float,
    timezone_name: str,
) -> Path:
    manifest = json.loads((session_dir / "manifest.json").read_text(encoding="utf-8"))
    source = pyedflib.EdfReader(str(bdf_path))
    try:
        source_sfreqs = [float(source.getSampleFrequency(index)) for index in range(eeg_channel_count)]
        if not np.allclose(source_sfreqs, source_sfreqs[0]):
            raise ValueError(f"Selected BDF channels have inconsistent rates: {source_sfreqs}")
        source_sfreq = source_sfreqs[0]
        bdf_start_local = source.getStartdatetime().replace(tzinfo=None)
        session_start_local = datetime.fromtimestamp(
            float(manifest["start_time"]),
            ZoneInfo(timezone_name),
        ).replace(tzinfo=None)
        base_sample = int(round((session_start_local - bdf_start_local).total_seconds() * source_sfreq))
        if base_sample < 0:
            raise ValueError("Realtime session starts before the BDF recording.")

        max_trial = _max_recorded_trial(session_dir)
        trial_count = max_trial + 1
        control_start_sec, control_stop_sec = (
            float(value) for value in manifest["online_label_source"]["valid_control_range_sec"]
        )
        required_end = base_sample + int(
            round(((trial_count - 1) * scene_sec + control_stop_sec) * source_sfreq)
        )
        if required_end > source.getNSamples()[0]:
            raise ValueError(
                f"Requested trial interval ends at BDF sample {required_end}, "
                f"but the file contains {source.getNSamples()[0]} samples."
            )

        segment_samples = required_end - base_sample
        continuous = np.empty((eeg_channel_count, segment_samples), dtype=np.float32)
        for channel in range(eeg_channel_count):
            continuous[channel] = source.readSignal(
                channel,
                start=base_sample,
                n=segment_samples,
            ).astype(np.float32)
        channel_labels = [source.getLabel(index).strip() for index in range(eeg_channel_count)]
    finally:
        source.close()

    sequence = list(manifest["online_label_source"]["sequence"])
    trials = []
    for trial_index in range(trial_count):
        label_name = str(sequence[trial_index % len(sequence)])
        trials.append(
            {
                "label_id": LABEL_NAME_TO_ID[label_name],
                "label": label_name,
                "block_index": -1,
                "trial_index": trial_index,
                "motor_imagery_on_sample": int(round(trial_index * scene_sec * source_sfreq)),
            }
        )

    payload = build_windows(
        continuous,
        trials,
        source_sfreq=source_sfreq,
        target_sfreq=target_sfreq,
        window_sec=float(manifest["window_sec"]),
        stride_sec=float(manifest["step_sec"]),
        control_start_sec=control_start_sec,
        control_stop_sec=control_stop_sec,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "training_windows_realtime_corrected.npz"
    np.savez_compressed(output_path, **payload)

    corrected_metadata = {
        "subject_id": manifest["subject_id"],
        "session_id": session_dir.name,
        "source_bdf": str(bdf_path.resolve()),
        "source_sfreq": source_sfreq,
        "target_sfreq": target_sfreq,
        "selected_channel_indices": list(range(eeg_channel_count)),
        "selected_channel_labels": channel_labels,
        "bdf_start_local": bdf_start_local.isoformat(),
        "session_start_local": session_start_local.isoformat(),
        "session_start_bdf_sample": base_sample,
        "timezone": timezone_name,
        "alignment_basis": "manifest epoch and BDF local header clock",
        "scene_sec": scene_sec,
        "trial_count": trial_count,
        "motor_imagery_window_range_sec": [control_start_sec, control_stop_sec],
        "window_sec": float(manifest["window_sec"]),
        "step_sec": float(manifest["step_sec"]),
        "label_sequence_repeats": True,
        "preprocessing": {
            **continuous_preprocessing_metadata(),
            "source_sfreq": source_sfreq,
            "target_sfreq": target_sfreq,
            "resampling": "scipy.signal.resample_poly",
            "continuous_span": "complete_reconstructed_bdf_segment",
        },
    }
    (output_dir / "metadata_corrected.json").write_text(
        json.dumps(corrected_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bdf_path", type=Path)
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--target-sfreq", type=float, default=200.0)
    parser.add_argument("--eeg-channel-count", type=int, default=59)
    parser.add_argument("--scene-sec", type=float, default=10.0)
    parser.add_argument("--timezone", default="Asia/Hong_Kong")
    parser.add_argument(
        "--alignment",
        choices=["waveform", "clock"],
        default="waveform",
        help="Waveform alignment recovers actual labeled online windows; clock is legacy.",
    )
    parser.add_argument("--match-channel", type=int, default=0)
    parser.add_argument("--match-tolerance-uv", type=float, default=0.08)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.alignment == "waveform":
        output = reprocess_realtime_from_waveform_matches(
            args.bdf_path.resolve(),
            args.session_dir.resolve(),
            args.output_dir.resolve(),
            target_sfreq=args.target_sfreq,
            eeg_channel_count=args.eeg_channel_count,
            match_channel=args.match_channel,
            match_tolerance_uv=args.match_tolerance_uv,
        )
    else:
        output = reprocess_realtime(
            args.bdf_path.resolve(),
            args.session_dir.resolve(),
            args.output_dir.resolve(),
            target_sfreq=args.target_sfreq,
            eeg_channel_count=args.eeg_channel_count,
            scene_sec=args.scene_sec,
            timezone_name=args.timezone,
        )
    print(output)


if __name__ == "__main__":
    main()
