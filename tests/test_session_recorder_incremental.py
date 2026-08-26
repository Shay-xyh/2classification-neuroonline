from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from adaptation.session_recorder import SessionRecorder


class IncrementalFakeAcquirer:
    def __init__(self, chunks: list[np.ndarray]) -> None:
        self.metadata = SimpleNamespace(
            n_channels=2,
            timestamp_domain="relative",
        )
        self._chunks = list(chunks)

    def get_new_samples(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._chunks:
            raise RuntimeError("stream is not started")
        chunk = self._chunks.pop(0)
        return chunk, np.arange(chunk.shape[1], dtype=np.float64)


def _load_raw_chunks(session_dir: Path) -> np.ndarray:
    chunks = [
        np.load(path, allow_pickle=False)
        for path in sorted((session_dir / "raw_chunks").glob("chunk_*.npy"))
    ]
    return np.concatenate(chunks, axis=1)


def _assert_no_absolute_time_fields(payload: object) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            lowered = str(key).lower()
            assert "unix" not in lowered
            assert "utc" not in lowered
            _assert_no_absolute_time_fields(value)
    elif isinstance(payload, list):
        for value in payload:
            _assert_no_absolute_time_fields(value)


def test_incremental_checkpoint_only_advances_with_durable_data(tmp_path: Path) -> None:
    first = np.arange(8, dtype=np.float32).reshape(2, 4)
    second = np.arange(8, 16, dtype=np.float32).reshape(2, 4)
    session_dir = tmp_path / "session_incremental"
    recorder = SessionRecorder(
        IncrementalFakeAcquirer([first, second]),  # type: ignore[arg-type]
        sfreq=250.0,
        n_channels=2,
        output_dir=session_dir,
        session_id=session_dir.name,
        total_trials=2,
    )

    recorder.pull()
    recorder.add_event("motor_imagery_left_on", label="left")
    recorder.persist(
        completed_trials=1,
        total_trials=2,
        last_completed_block=1,
        last_completed_trial_in_block=1,
        wait=True,
    )

    checkpoint = json.loads((session_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["state"] == "collecting"
    assert checkpoint["completed_trials"] == 1
    assert checkpoint["sample_count"] == 4
    assert checkpoint["event_count"] == 1
    _assert_no_absolute_time_fields(checkpoint)
    np.testing.assert_array_equal(_load_raw_chunks(session_dir), first)

    recorder.pull()
    recorder.add_event("motor_imagery_right_on", label="right")
    recorder.abort(error="simulated disconnect")

    checkpoint = json.loads((session_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["state"] == "failed"
    assert checkpoint["completed_trials"] == 1
    assert checkpoint["sample_count"] == 8
    assert checkpoint["event_count"] == 2
    assert checkpoint["error"] == "simulated disconnect"
    np.testing.assert_array_equal(
        _load_raw_chunks(session_dir),
        np.concatenate([first, second], axis=1),
    )
    journal = [
        json.loads(line)
        for line in (session_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["name"] for event in journal] == [
        "motor_imagery_left_on",
        "motor_imagery_right_on",
    ]

    recovery = SessionRecorder.recover_partial(session_dir)
    assert recovery["completed_trials"] == 1
    _assert_no_absolute_time_fields(recovery)
    _assert_no_absolute_time_fields(
        json.loads((session_dir / "metadata.partial.json").read_text(encoding="utf-8"))
    )
    np.testing.assert_array_equal(
        np.load(session_dir / "continuous_eeg.partial.npy", allow_pickle=False),
        np.concatenate([first, second], axis=1),
    )
    recovered_events = json.loads(
        (session_dir / "events.partial.json").read_text(encoding="utf-8")
    )
    assert len(recovered_events) == 2


def test_normal_export_preserves_final_format_and_cleans_staging(tmp_path: Path) -> None:
    eeg = np.arange(12, dtype=np.float32).reshape(2, 6)
    session_dir = tmp_path / "session_complete"
    recorder = SessionRecorder(
        IncrementalFakeAcquirer([eeg]),  # type: ignore[arg-type]
        sfreq=250.0,
        n_channels=2,
        output_dir=session_dir,
        session_id=session_dir.name,
        total_trials=1,
    )
    recorder.pull()
    recorder.add_event("motor_imagery_left_on", label="left")
    recorder.persist(
        completed_trials=1,
        last_completed_block=1,
        last_completed_trial_in_block=1,
    )

    recorder.export(
        session_dir,
        metadata={"session_id": session_dir.name, "formal_trial_count": 1},
    )
    recorder.mark_processing_complete()
    SessionRecorder.prepare_final_bundle(session_dir)
    SessionRecorder.finalize_session(session_dir)

    np.testing.assert_array_equal(
        np.load(session_dir / "continuous_eeg.npy", allow_pickle=False),
        eeg,
    )
    events = json.loads((session_dir / "events.json").read_text(encoding="utf-8"))
    assert [event["name"] for event in events] == ["motor_imagery_left_on"]
    checkpoint = json.loads((session_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["state"] == "complete"
    assert checkpoint["completed_trials"] == 1
    _assert_no_absolute_time_fields(checkpoint)
    assert not (session_dir / "raw_chunks").exists()
    assert not (session_dir / "events.jsonl").exists()
    assert not (session_dir / "metadata.partial.json").exists()


