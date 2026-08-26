"""Continuous EEG and sample-aligned event recording for protocol sessions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import queue
import shutil
import threading
import time
from typing import Any

import numpy as np

from acquisition.base import AbstractAcquirer


INCREMENTAL_SCHEMA_VERSION = 1


@dataclass(slots=True)
class SessionEvent:
    name: str
    sample_index: int
    payload: dict[str, Any]


@dataclass(slots=True)
class _WriteRequest:
    chunk_index: int | None
    eeg: np.ndarray | None
    events: tuple[dict[str, Any], ...]
    checkpoint: dict[str, Any]
    done: threading.Event


class SessionRecorder:
    """Collect continuous EEG and aligned events during one protocol session.

    When ``output_dir`` is supplied, raw EEG, events, and an atomic progress
    checkpoint are durably committed throughout collection. Writes happen on a
    dedicated thread so filesystem latency cannot delay stimulus transitions.
    """

    def __init__(
        self,
        acquirer: AbstractAcquirer,
        *,
        sfreq: float,
        n_channels: int,
        output_dir: Path | None = None,
        session_id: str | None = None,
        total_trials: int | None = None,
    ) -> None:
        self._acquirer = acquirer
        self._sfreq = float(sfreq)
        self._n_channels = int(n_channels)
        self._chunks: list[np.ndarray] = []
        self._events: list[SessionEvent] = []
        self._sample_count = 0
        self._latest_sample_end_monotonic: float | None = None

        self._output_dir = Path(output_dir) if output_dir is not None else None
        self._session_id = session_id
        self._total_trials = int(total_trials) if total_trials is not None else None
        self._raw_chunks_dir: Path | None = None
        self._events_journal_path: Path | None = None
        self._checkpoint_path: Path | None = None
        self._write_queue: queue.Queue[_WriteRequest | None] | None = None
        self._writer_thread: threading.Thread | None = None
        self._writer_error: BaseException | None = None
        self._writer_error_lock = threading.Lock()
        self._writer_closed = False
        self._next_chunk_index = 0
        self._enqueued_event_count = 0
        self._checkpoint_state: dict[str, Any] = {
            "schema_version": INCREMENTAL_SCHEMA_VERSION,
            "session_id": session_id,
            "state": "initializing",
            "completed_trials": 0,
            "total_trials": self._total_trials,
            "sample_count": 0,
            "event_count": 0,
            "chunk_count": 0,
        }
        if self._output_dir is not None:
            self._initialize_incremental_output()

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def events(self) -> list[SessionEvent]:
        return list(self._events)

    @property
    def checkpoint_path(self) -> Path | None:
        return self._checkpoint_path

    def pull(self) -> np.ndarray:
        self._raise_writer_error()
        samples, timestamps = self._acquirer.get_new_samples()
        if samples.size == 0:
            return np.empty((self._n_channels, 0), dtype=np.float32)
        if samples.ndim != 2 or samples.shape[0] < self._n_channels:
            raise RuntimeError(f"Unexpected incremental EEG shape: {samples.shape}")
        eeg = np.asarray(samples[: self._n_channels], dtype=np.float32)
        self._chunks.append(eeg)
        self._sample_count += int(eeg.shape[1])
        values = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if (
            str(getattr(self._acquirer.metadata, "timestamp_domain", "relative")).lower()
            == "monotonic"
            and values.size == eeg.shape[1]
            and values.size
            and np.all(np.isfinite(values))
        ):
            self._latest_sample_end_monotonic = float(values[-1]) + (1.0 / self._sfreq)
        return eeg

    def add_event(
        self,
        name: str,
        *,
        timestamp_monotonic: float | None = None,
        **payload: Any,
    ) -> SessionEvent:
        self._raise_writer_error()
        event_time = (
            time.monotonic()
            if timestamp_monotonic is None
            else float(timestamp_monotonic)
        )
        sample_index = self._sample_count
        alignment_method = "latest-received-sample"
        if self._latest_sample_end_monotonic is not None:
            sample_index += int(
                round((event_time - self._latest_sample_end_monotonic) * self._sfreq)
            )
            sample_index = max(sample_index, 0)
            alignment_method = "source-clock-projection"
        event_payload = dict(payload)
        event_payload["alignment_method"] = alignment_method
        event = SessionEvent(
            name=name,
            sample_index=sample_index,
            payload=event_payload,
        )
        self._events.append(event)
        return event

    def persist(
        self,
        *,
        completed_trials: int | None = None,
        total_trials: int | None = None,
        last_completed_block: int | None = None,
        last_completed_trial_in_block: int | None = None,
        state: str | None = None,
        error: str | None = None,
        wait: bool = False,
    ) -> None:
        """Queue all data accumulated since the previous durable commit."""

        if self._output_dir is None:
            return
        if self._writer_closed:
            raise RuntimeError("Incremental session writer is already closed.")
        self._raise_writer_error()
        assert self._write_queue is not None

        eeg: np.ndarray | None = None
        chunk_index: int | None = None
        if self._chunks:
            eeg = np.concatenate(self._chunks, axis=1).astype(np.float32, copy=False)
            self._chunks.clear()
            chunk_index = self._next_chunk_index
            self._next_chunk_index += 1

        new_events = tuple(
            asdict(event) for event in self._events[self._enqueued_event_count :]
        )
        checkpoint = dict(self._checkpoint_state)
        if completed_trials is not None:
            checkpoint["completed_trials"] = int(completed_trials)
        if total_trials is not None:
            checkpoint["total_trials"] = int(total_trials)
        if last_completed_block is not None:
            checkpoint["last_completed_block"] = int(last_completed_block)
        if last_completed_trial_in_block is not None:
            checkpoint["last_completed_trial_in_block"] = int(
                last_completed_trial_in_block
            )
        if state is not None:
            checkpoint["state"] = str(state)
        if error is not None:
            checkpoint["error"] = str(error)
        elif state != "failed":
            checkpoint.pop("error", None)
        checkpoint.update(
            {
                "sample_count": int(self._sample_count),
                "event_count": len(self._events),
                "chunk_count": self._next_chunk_index,
            }
        )
        request = _WriteRequest(
            chunk_index=chunk_index,
            eeg=eeg,
            events=new_events,
            checkpoint=checkpoint,
            done=threading.Event(),
        )
        self._write_queue.put(request)
        self._enqueued_event_count = len(self._events)
        self._checkpoint_state = checkpoint
        if wait:
            request.done.wait()
            self._raise_writer_error()

    def abort(self, *, error: str) -> None:
        """Durably retain a failed partial session without masking its cause."""

        if self._output_dir is None or self._writer_closed:
            return
        try:
            self.persist(state="failed", error=error, wait=True)
        finally:
            self._close_writer()

    def export(self, output_dir: Path, *, metadata: dict[str, Any]) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.pull()
        except RuntimeError as exc:
            # Calibration stops the stream before export; keep buffered chunks in that case.
            if not self._is_stream_not_started_error(exc):
                raise

        if self._output_dir is not None:
            self.persist(
                completed_trials=int(metadata.get("formal_trial_count", 0)),
                total_trials=self._total_trials,
                state="raw_exporting",
                wait=True,
            )
            self._close_writer()
        eeg = self._assembled_array()
        self._save_npy_atomic(output_dir / "continuous_eeg.npy", eeg)
        events_path = output_dir / "events.json"
        events_temporary = output_dir / ".events.json.tmp"
        with events_temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                [asdict(event) for event in self._events],
                handle,
                ensure_ascii=False,
                indent=2,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(events_temporary, events_path)
        metadata_path = output_dir / "metadata.json"
        self._write_json_atomic(metadata_path, metadata)
        self._update_checkpoint_direct(state="raw_exported")
        return output_dir

    def mark_processing_complete(self) -> None:
        if self._output_dir is None:
            return
        self._update_checkpoint_direct(state="processing_complete")

    def mark_processing_failed(self, *, error: str) -> None:
        if self._output_dir is None:
            return
        self._update_checkpoint_direct(
            state="processing_failed",
            error=str(error),
        )

    @classmethod
    def prepare_final_bundle(cls, output_dir: Path | None) -> None:
        """Remove staging copies after final raw/events/windows files exist."""

        if output_dir is None:
            return
        output_dir = Path(output_dir)
        cls._update_checkpoint_at_path(output_dir, state="finalizing")
        cleanup_errors: list[str] = []
        try:
            (output_dir / "events.jsonl").unlink(missing_ok=True)
        except OSError as exc:
            cleanup_errors.append(f"events.jsonl: {exc}")
        raw_chunks_dir = output_dir / "raw_chunks"
        if raw_chunks_dir.is_dir():
            try:
                shutil.rmtree(raw_chunks_dir)
            except OSError as exc:
                cleanup_errors.append(f"raw_chunks: {exc}")
        try:
            (output_dir / "metadata.partial.json").unlink(missing_ok=True)
        except OSError as exc:
            cleanup_errors.append(f"metadata.partial.json: {exc}")
        if cleanup_errors:
            cls._update_checkpoint_at_path(
                output_dir,
                staging_cleanup_errors=cleanup_errors,
            )

    @classmethod
    def finalize_session(cls, output_dir: Path | None) -> None:
        if output_dir is None:
            return
        cls._update_checkpoint_at_path(
            Path(output_dir),
            state="complete",
        )

    @classmethod
    def recover_partial(cls, output_dir: Path) -> dict[str, Any]:
        """Materialize only the data covered by the last durable checkpoint."""

        output_dir = Path(output_dir)
        checkpoint_path = output_dir / "checkpoint.json"
        partial_metadata_path = output_dir / "metadata.partial.json"
        events_journal_path = output_dir / "events.jsonl"
        raw_chunks_dir = output_dir / "raw_chunks"
        if not checkpoint_path.is_file() or not partial_metadata_path.is_file():
            raise FileNotFoundError(
                "Partial session requires checkpoint.json and metadata.partial.json."
            )
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        partial_metadata = json.loads(
            partial_metadata_path.read_text(encoding="utf-8")
        )
        chunk_count = int(checkpoint.get("chunk_count", 0))
        sample_count = int(checkpoint.get("sample_count", 0))
        n_channels = int(partial_metadata["n_channels"])
        chunks: list[np.ndarray] = []
        for index in range(chunk_count):
            path = raw_chunks_dir / f"chunk_{index:06d}.npy"
            if not path.is_file():
                raise RuntimeError(f"Checkpointed EEG chunk is missing: {path.name}")
            chunk = np.load(path, allow_pickle=False)
            if chunk.ndim != 2 or chunk.shape[0] != n_channels:
                raise RuntimeError(
                    f"Invalid checkpointed EEG shape in {path.name}: {chunk.shape}"
                )
            chunks.append(np.asarray(chunk, dtype=np.float32))
        eeg = (
            np.concatenate(chunks, axis=1).astype(np.float32, copy=False)
            if chunks
            else np.empty((n_channels, 0), dtype=np.float32)
        )
        if eeg.shape[1] != sample_count:
            raise RuntimeError(
                "Checkpoint/sample mismatch: "
                f"checkpoint={sample_count}, chunks={eeg.shape[1]}"
            )

        event_count = int(checkpoint.get("event_count", 0))
        events: list[dict[str, Any]] = []
        if event_count:
            if not events_journal_path.is_file():
                raise RuntimeError("Checkpointed event journal is missing.")
            with events_journal_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if len(events) >= event_count:
                        break
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(
                            "Event journal is truncated before the checkpointed event count."
                        ) from exc
        if len(events) != event_count:
            raise RuntimeError(
                "Checkpoint/event mismatch: "
                f"checkpoint={event_count}, journal={len(events)}"
            )

        eeg_path = output_dir / "continuous_eeg.partial.npy"
        events_path = output_dir / "events.partial.json"
        recovery_path = output_dir / "recovery.json"
        cls._save_npy_atomic(eeg_path, eeg)
        cls._write_json_list_atomic(events_path, events)
        recovery = {
            "status": "incomplete_recovered",
            "source_checkpoint": checkpoint,
            "continuous_eeg_path": eeg_path.name,
            "events_path": events_path.name,
            "sample_count": sample_count,
            "event_count": event_count,
            "completed_trials": int(checkpoint.get("completed_trials", 0)),
        }
        cls._write_json_atomic(recovery_path, recovery)
        return recovery

    def to_array(self) -> np.ndarray:
        try:
            self.pull()
        except RuntimeError as exc:
            if not self._is_stream_not_started_error(exc):
                raise
        if self._output_dir is not None and not self._writer_closed:
            self.persist(wait=True)
        return self._assembled_array()

    def _initialize_incremental_output(self) -> None:
        assert self._output_dir is not None
        self._output_dir.mkdir(parents=True, exist_ok=False)
        self._raw_chunks_dir = self._output_dir / "raw_chunks"
        self._raw_chunks_dir.mkdir()
        self._events_journal_path = self._output_dir / "events.jsonl"
        with self._events_journal_path.open("xb") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        self._checkpoint_path = self._output_dir / "checkpoint.json"
        partial_metadata = {
            "schema_version": INCREMENTAL_SCHEMA_VERSION,
            "session_id": self._session_id,
            "source_sfreq": self._sfreq,
            "n_channels": self._n_channels,
            "total_trials": self._total_trials,
        }
        self._write_json_atomic(
            self._output_dir / "metadata.partial.json", partial_metadata
        )
        self._checkpoint_state["state"] = "collecting"
        self._write_json_atomic(self._checkpoint_path, self._checkpoint_state)
        self._write_queue = queue.Queue(maxsize=8)
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name=f"session-writer-{self._session_id or 'anonymous'}",
            daemon=False,
        )
        self._writer_thread.start()

    def _writer_loop(self) -> None:
        assert self._write_queue is not None
        while True:
            request = self._write_queue.get()
            try:
                if request is None:
                    return
                self._write_request(request)
            except BaseException as exc:  # noqa: BLE001
                with self._writer_error_lock:
                    if self._writer_error is None:
                        self._writer_error = exc
            finally:
                if request is not None:
                    request.done.set()
                self._write_queue.task_done()

    def _write_request(self, request: _WriteRequest) -> None:
        self._raise_writer_error()
        if request.eeg is not None:
            assert request.chunk_index is not None
            assert self._raw_chunks_dir is not None
            self._save_npy_atomic(
                self._raw_chunks_dir / f"chunk_{request.chunk_index:06d}.npy",
                request.eeg,
            )
        if request.events:
            assert self._events_journal_path is not None
            with self._events_journal_path.open("a", encoding="utf-8", newline="\n") as handle:
                for event in request.events:
                    handle.write(
                        json.dumps(event, ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    )
                handle.flush()
                os.fsync(handle.fileno())
        assert self._checkpoint_path is not None
        self._write_json_atomic(self._checkpoint_path, request.checkpoint)

    def _assembled_array(self) -> np.ndarray:
        chunks: list[np.ndarray] = []
        if self._raw_chunks_dir is not None and self._raw_chunks_dir.is_dir():
            for path in sorted(self._raw_chunks_dir.glob("chunk_*.npy")):
                chunks.append(np.load(path, allow_pickle=False))
        chunks.extend(self._chunks)
        if not chunks:
            return np.empty((self._n_channels, 0), dtype=np.float32)
        return np.concatenate(chunks, axis=1).astype(np.float32, copy=False)

    def _close_writer(self) -> None:
        if self._writer_closed or self._write_queue is None:
            return
        self._write_queue.put(None)
        self._write_queue.join()
        if self._writer_thread is not None:
            self._writer_thread.join()
        self._writer_closed = True
        self._raise_writer_error()

    def _raise_writer_error(self) -> None:
        with self._writer_error_lock:
            error = self._writer_error
        if error is not None:
            raise RuntimeError(f"Incremental session write failed: {error}") from error

    def _update_checkpoint_direct(self, **updates: Any) -> None:
        if self._checkpoint_path is None:
            return
        checkpoint = dict(self._checkpoint_state)
        checkpoint.update(updates)
        self._write_json_atomic(self._checkpoint_path, checkpoint)
        self._checkpoint_state = checkpoint

    @classmethod
    def _update_checkpoint_at_path(cls, output_dir: Path, **updates: Any) -> None:
        checkpoint_path = output_dir / "checkpoint.json"
        if not checkpoint_path.is_file():
            return
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        checkpoint.update(updates)
        cls._write_json_atomic(checkpoint_path, checkpoint)

    @staticmethod
    def _is_stream_not_started_error(exc: RuntimeError) -> bool:
        message = str(exc).lower()
        return "not started" in message and "stream" in message

    @staticmethod
    def _save_npy_atomic(path: Path, array: np.ndarray) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.save(handle, array)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    @staticmethod
    def _write_json_list_atomic(path: Path, payload: list[dict[str, Any]]) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
