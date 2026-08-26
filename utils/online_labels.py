"""Realtime intent-label sources for online decoder updates."""

from __future__ import annotations

import json
import logging
import random
import secrets
import threading
import time
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

LOGGER = logging.getLogger(__name__)

LABEL_NAME_TO_ID = {"left": 0, "right": 1}
LABEL_ID_TO_NAME = {value: key for key, value in LABEL_NAME_TO_ID.items()}
CUED_PROTOCOL_VERSION = "continuous-scene-v5-centered-single-decision"


@dataclass(frozen=True, slots=True)
class OnlineLabel:
    """One realtime label event aligned by local monotonic time."""

    label_id: int
    label_name: str
    timestamp_monotonic: float
    expires_at_monotonic: float
    event_id: str = ""
    source: str = "manual"
    payload: dict[str, Any] | None = None

    def is_active_for(self, *, window_start: float, window_end: float) -> bool:
        """Return whether this label overlaps a decoding window."""

        return self.timestamp_monotonic <= window_end and self.expires_at_monotonic >= window_start


class OnlineLabelSource:
    """Base class for optional realtime labels."""

    def get_label(self, *, window_start: float, window_end: float) -> OnlineLabel | None:
        del window_start, window_end
        return None

    def close(self) -> None:
        return


class ManualOnlineLabelSource(OnlineLabelSource):
    """Thread-safe label source updated by an operator or another process."""

    def __init__(self, *, default_ttl_sec: float = 2.0) -> None:
        self._default_ttl_sec = max(float(default_ttl_sec), 0.05)
        self._lock = threading.Lock()
        self._latest: OnlineLabel | None = None
        self._event_counter = 0

    def set_label(
        self,
        label: str | int,
        *,
        ttl_sec: float | None = None,
        source: str = "manual",
        timestamp_monotonic: float | None = None,
        payload: dict[str, Any] | None = None,
    ) -> OnlineLabel:
        label_id, label_name = coerce_label(label)
        ts = time.monotonic() if timestamp_monotonic is None else float(timestamp_monotonic)
        ttl = self._default_ttl_sec if ttl_sec is None else max(float(ttl_sec), 0.05)
        with self._lock:
            self._event_counter += 1
            event = OnlineLabel(
                label_id=label_id,
                label_name=label_name,
                timestamp_monotonic=ts,
                expires_at_monotonic=ts + ttl,
                event_id=f"manual-{self._event_counter}",
                source=source,
                payload=dict(payload) if payload else None,
            )
            self._latest = event
        return event

    def clear(self) -> None:
        with self._lock:
            self._latest = None

    def get_label(self, *, window_start: float, window_end: float) -> OnlineLabel | None:
        with self._lock:
            event = self._latest
        if event is None:
            return None
        if not event.is_active_for(window_start=window_start, window_end=window_end):
            return None
        return event


class SimulatedOnlineLabelSource(OnlineLabelSource):
    """Drive a label-aware dummy acquirer through balanced synthetic trials."""

    def __init__(
        self,
        acquirer: Any,
        *,
        trial_sec: float = 6.0,
        settle_sec: float = 2.0,
        seed: int = 17,
        clock: Any = time.monotonic,
    ) -> None:
        if not hasattr(acquirer, "set_intent"):
            raise TypeError("Simulated labels require an acquirer with set_intent(label).")
        self._acquirer = acquirer
        self._trial_sec = max(float(trial_sec), 1.0)
        self._settle_sec = min(max(float(settle_sec), 0.0), self._trial_sec * 0.8)
        self._clock = clock
        labels = list(LABEL_ID_TO_NAME)
        random.Random(seed).shuffle(labels)
        self._sequence = tuple(labels)
        self._started_at = float(self._clock())
        self._active_cycle = -1
        self._set_cycle(0)

    def get_label(self, *, window_start: float, window_end: float) -> OnlineLabel | None:
        del window_start
        now = float(self._clock())
        elapsed = max(now - self._started_at, 0.0)
        cycle = int(elapsed // self._trial_sec)
        self._set_cycle(cycle)
        cycle_started = self._started_at + (cycle * self._trial_sec)
        if now - cycle_started < self._settle_sec:
            return None
        label_id = self._sequence[cycle % len(self._sequence)]
        return OnlineLabel(
            label_id=label_id,
            label_name=LABEL_ID_TO_NAME[label_id],
            timestamp_monotonic=cycle_started + self._settle_sec,
            expires_at_monotonic=cycle_started + self._trial_sec,
            event_id=f"sim-{cycle:06d}",
            source="label-aware-dummy",
            payload={"cycle": cycle},
        )

    def _set_cycle(self, cycle: int) -> None:
        if cycle == self._active_cycle:
            return
        label_id = self._sequence[cycle % len(self._sequence)]
        self._acquirer.set_intent(label_id)
        self._active_cycle = cycle


class CuedOnlineLabelSource(OnlineLabelSource):
    """Generate balanced relative-action truth for the centered car task.

    Unity resets the car to the center lane before every scene, so LEFT and
    RIGHT are always feasible and the label sequence is independent of the
    previous model output. A scene is not labelable until Unity confirms the
    centered start lane, safe lane, and applied relative action.
    """

    def __init__(
        self,
        sequence: list[str | int],
        *,
        scene_duration_sec: float,
        start_delay_sec: float = 5.0,
        boundary_guard_sec: float = 0.5,
        lane_transition_guard_sec: float = 0.0,
        primary_windows_per_scene: int = 1,
        primary_window_spacing_sec: float = 1.0,
        pool_rng: random.Random | None = None,
        sequence_seed: int | None = None,
        clock: Any = time.monotonic,
    ) -> None:
        if not sequence:
            raise ValueError("Car scene protocol requires at least one scene.")
        self._sequence = tuple(coerce_label(label)[0] for label in sequence)
        self._scene_duration_sec = max(float(scene_duration_sec), 0.1)
        self._boundary_guard_sec = min(
            max(float(boundary_guard_sec), 0.0),
            self._scene_duration_sec / 2.0,
        )
        self._lane_transition_guard_sec = max(
            float(lane_transition_guard_sec),
            0.0,
        )
        self._primary_windows_per_scene = max(
            int(primary_windows_per_scene),
            1,
        )
        self._primary_window_spacing_sec = max(
            float(primary_window_spacing_sec),
            0.0,
        )
        self._clock = clock
        self._started_at = float(clock()) + max(float(start_delay_sec), 0.0)
        self._lock = threading.RLock()
        self._scene_index = 0
        self._scene_started_at = self._started_at
        self._confirmed_scene_index = -1
        self._prepared_scene_index = -1
        self._prepared_label_id: int | None = None
        self._prepared_start_lane: int | None = None
        self._confirmed_safe_lane: int | None = None
        self._current_lane: int | None = None
        self._label_segments: list[tuple[float, int | None, int]] = []
        self._lane_transition_events: list[tuple[int, float]] = []
        self._label_transition_count = 0
        self._pending_sequence = list(self._sequence)
        self._pool_rng = pool_rng
        self._sequence_seed = None if sequence_seed is None else int(sequence_seed)
        self._pool_index = 0
        self._pool_class_counts = {
            label_id: self._sequence.count(label_id)
            for label_id in LABEL_ID_TO_NAME
        }
        self._applied_label_counts = {label_id: 0 for label_id in LABEL_ID_TO_NAME}
        self._last_transition_reason = "start"
        self._failed_scenes = 0
        self._active_scene_failed = False
        self._last_failed_scene_index = -1

    def get_label(self, *, window_start: float, window_end: float) -> OnlineLabel | None:
        state = self.status(now=window_end)
        if state["phase"] != "control" or not bool(state["scene_confirmed"]):
            return None
        valid_from = (
            float(state["valid_from_monotonic"]) + self._boundary_guard_sec
        )
        valid_until = (
            float(state["valid_until_monotonic"]) - self._boundary_guard_sec
        )
        if float(window_start) < valid_from or float(window_end) > valid_until:
            return None
        with self._lock:
            matching_index = -1
            for index, (started_at, _label_id, _lane) in enumerate(
                self._label_segments
            ):
                if started_at <= float(window_start):
                    matching_index = index
                else:
                    break
            if matching_index < 0:
                return None
            segment_start, label_id, current_lane = self._label_segments[
                matching_index
            ]
            if label_id is None:
                return None
            segment_end = valid_until
            if matching_index + 1 < len(self._label_segments):
                segment_end = min(
                    segment_end,
                    self._label_segments[matching_index + 1][0],
                )
            if float(window_end) > segment_end:
                return None
            scene_index = int(state["scene_index"])
            return OnlineLabel(
                label_id=label_id,
                label_name=LABEL_ID_TO_NAME[label_id],
                timestamp_monotonic=max(valid_from, segment_start),
                expires_at_monotonic=segment_end,
                event_id=f"scene-{scene_index:06d}-segment-{matching_index:03d}",
                source="cued-protocol",
                payload={
                    "scene_index": scene_index,
                    "segment_index": matching_index,
                    "current_lane": current_lane,
                    "safe_lane": self._confirmed_safe_lane,
                },
            )

    def prepare_scene(self, *, scene_index: int, start_lane: int) -> int:
        """Choose the next balanced action for Unity's centered scene start."""

        reported_lane = int(start_lane)
        if reported_lane not in {-1, 0, 1}:
            raise ValueError(f"Unity reported invalid lane state: {reported_lane}")
        lane = 0
        with self._lock:
            if int(scene_index) != self._scene_index:
                raise ValueError(
                    f"Cannot prepare scene {scene_index}; active scene is {self._scene_index}."
                )
            if self._prepared_scene_index == self._scene_index:
                if self._prepared_start_lane != lane:
                    raise ValueError(
                        "Unity start lane changed after the scene was prepared."
                    )
                assert self._prepared_label_id is not None
                return self._prepared_label_id

            if not self._pending_sequence:
                self._pool_index += 1
                if self._pool_rng is None:
                    self._pending_sequence = list(self._sequence)
                else:
                    from adaptation.mi_protocol import generate_block_sequence

                    self._pending_sequence = generate_block_sequence(
                        {
                            LABEL_ID_TO_NAME[label_id]: count
                            for label_id, count in self._pool_class_counts.items()
                        },
                        rng=self._pool_rng,
                    )
                    self._pending_sequence = [
                        LABEL_NAME_TO_ID[label]
                        for label in self._pending_sequence
                    ]
            chosen = self._pending_sequence.pop(0)

            self._prepared_scene_index = self._scene_index
            self._prepared_label_id = chosen
            self._prepared_start_lane = lane
            self._confirmed_safe_lane = None
            return chosen

    def status(self, *, now: float | None = None) -> dict[str, Any]:
        timestamp = float(self._clock()) if now is None else float(now)
        with self._lock:
            if timestamp < self._started_at:
                return self._status_payload(
                    phase="preparing",
                    scene_index=0,
                    phase_remaining_sec=self._started_at - timestamp,
                )
            self._advance_timeouts_locked(timestamp)
            scene_started = self._scene_started_at
            return self._status_payload(
                phase="control",
                scene_index=self._scene_index,
                phase_remaining_sec=max(
                    self._scene_duration_sec - max(timestamp - scene_started, 0.0),
                    0.0,
                ),
                valid_from_monotonic=scene_started,
                valid_until_monotonic=scene_started + self._scene_duration_sec,
            )

    def mark_scene_failed(
        self,
        *,
        timestamp_monotonic: float | None = None,
        expected_scene_index: int | None = None,
    ) -> bool:
        """Record a collision without changing the fixed-duration scene clock."""

        timestamp = (
            float(self._clock())
            if timestamp_monotonic is None
            else float(timestamp_monotonic)
        )
        with self._lock:
            if timestamp < self._started_at:
                return False
            if (
                expected_scene_index is not None
                and int(expected_scene_index) != self._scene_index
            ):
                return False
            if self._active_scene_failed:
                return False
            self._active_scene_failed = True
            self._last_failed_scene_index = self._scene_index
            self._failed_scenes += 1
            return True

    def confirm_scene_applied(
        self,
        *,
        scene_index: int,
        applied_label_id: int,
        start_lane: int,
        safe_lane: int,
        timestamp_monotonic: float | None = None,
    ) -> bool:
        """Validate Unity's authoritative relative-action ACK and anchor time."""

        timestamp = (
            float(self._clock())
            if timestamp_monotonic is None
            else float(timestamp_monotonic)
        )
        with self._lock:
            if int(scene_index) != self._scene_index:
                return False
            if self._prepared_scene_index != self._scene_index:
                return False
            if self._prepared_label_id != int(applied_label_id):
                return False
            if self._prepared_start_lane != int(start_lane):
                return False
            action_delta = {0: -1, 1: 1}
            if int(applied_label_id) not in action_delta:
                return False
            expected_safe_lane = int(start_lane) + action_delta[int(applied_label_id)]
            if (
                int(start_lane) not in {-1, 0, 1}
                or int(safe_lane) not in {-1, 0, 1}
                or expected_safe_lane != int(safe_lane)
            ):
                return False
            # An unconfirmed Scene has no authoritative start time yet.  Unity
            # may need longer than one Scene duration to launch or render the
            # first layout, so the local placeholder must never invalidate an
            # otherwise exact ACK.  The fixed five-second clock begins here.
            self._scene_started_at = max(self._scene_started_at, timestamp)
            self._confirmed_scene_index = self._scene_index
            self._confirmed_safe_lane = int(safe_lane)
            self._current_lane = int(start_lane)
            self._label_segments = [
                (timestamp, int(applied_label_id), int(start_lane))
            ]
            self._applied_label_counts[int(applied_label_id)] += 1
            return True

    def update_current_lane(
        self,
        *,
        scene_index: int,
        current_lane: int,
        safe_lane: int,
        timestamp_monotonic: float | None = None,
    ) -> bool:
        """Start a new truth segment after Unity confirms a completed lane change."""

        timestamp = (
            float(self._clock())
            if timestamp_monotonic is None
            else float(timestamp_monotonic)
        )
        lane = int(current_lane)
        safe = int(safe_lane)
        if lane not in {-1, 0, 1} or safe not in {-1, 0, 1}:
            return False
        with self._lock:
            if (
                int(scene_index) != self._scene_index
                or self._confirmed_scene_index != self._scene_index
                or self._confirmed_safe_lane != safe
            ):
                return False
            if timestamp < self._scene_started_at:
                return False
            if timestamp > self._scene_started_at + self._scene_duration_sec:
                return False
            desired_label = _relative_action_label(
                current_lane=lane,
                safe_lane=safe,
            )
            if self._label_segments:
                last_started_at, last_label, last_lane = self._label_segments[-1]
                if last_label == desired_label and last_lane == lane:
                    return True
                timestamp = max(timestamp, last_started_at)
            self._label_segments.append((timestamp, desired_label, lane))
            self._lane_transition_events.append((self._scene_index, timestamp))
            self._current_lane = lane
            self._label_transition_count += 1
            self._last_transition_reason = "lane_settled"
            return True

    def metadata(self) -> dict[str, Any]:
        return {
            "source": "cued-protocol",
            "protocol_mode": "centered-single-decision",
            "protocol_version": CUED_PROTOCOL_VERSION,
            "label_semantics": "centered-relative-action-to-fixed-safe-lane",
            "scene_start_lane": 0,
            "balance_pool_scenes": len(self._sequence),
            "sequence": [LABEL_ID_TO_NAME[label] for label in self._sequence],
            "sequence_seed": self._sequence_seed,
            "pool_index": self._pool_index,
            "pool_class_counts": {
                LABEL_ID_TO_NAME[label_id]: count
                for label_id, count in self._pool_class_counts.items()
            },
            "applied_label_counts": {
                LABEL_ID_TO_NAME[label]: count
                for label, count in self._applied_label_counts.items()
            },
            "prepared_start_lane": self._prepared_start_lane,
            "confirmed_safe_lane": self._confirmed_safe_lane,
            "current_lane": self._current_lane,
            "label_transition_count": self._label_transition_count,
            "label_segments": [
                {
                    "started_at_monotonic": started_at,
                    "label_id": label_id,
                    "label_name": (
                        None if label_id is None else LABEL_ID_TO_NAME[label_id]
                    ),
                    "current_lane": current_lane,
                }
                for started_at, label_id, current_lane in self._label_segments
            ],
            "scene_duration_sec": self._scene_duration_sec,
            "boundary_guard_sec": self._boundary_guard_sec,
            "lane_transition_guard_sec": self._lane_transition_guard_sec,
            "primary_windows_per_scene": self._primary_windows_per_scene,
            "primary_window_spacing_sec": self._primary_window_spacing_sec,
            "lane_transition_events": [
                {
                    "scene_index": scene_index,
                    "timestamp_monotonic": timestamp,
                }
                for scene_index, timestamp in self._lane_transition_events
            ],
            "confirmed_scene_index": self._confirmed_scene_index,
            "failed_scenes": self._failed_scenes,
            "active_scene_failed": self._active_scene_failed,
            "last_failed_scene_index": self._last_failed_scene_index,
            "last_transition_reason": self._last_transition_reason,
        }

    def _advance_timeouts_locked(self, timestamp: float) -> None:
        # A local clock must never manufacture later scenes while the current
        # layout is still waiting for Unity's authoritative ACK.  Advancing an
        # unconfirmed scene used to make the GUI count upward after a transport
        # timeout even though Unity had not applied any new obstacle wall.
        if self._confirmed_scene_index != self._scene_index:
            return

        elapsed = timestamp - self._scene_started_at
        if elapsed >= self._scene_duration_sec:
            # Advance exactly one scene.  The new scene remains pending at this
            # index until SCENE_STATE and SCENE_* both succeed; a delayed GUI
            # refresh therefore cannot skip scene numbers or labels.
            self._scene_started_at = timestamp
            self._scene_index += 1
            self._confirmed_scene_index = -1
            self._prepared_scene_index = -1
            self._prepared_label_id = None
            self._prepared_start_lane = None
            self._confirmed_safe_lane = None
            self._current_lane = None
            self._label_segments = []
            self._active_scene_failed = False
            self._last_transition_reason = "timeout"

    def _status_payload(
        self,
        *,
        phase: str,
        scene_index: int,
        phase_remaining_sec: float,
        valid_from_monotonic: float | None = None,
        valid_until_monotonic: float | None = None,
    ) -> dict[str, Any]:
        label_id = (
            self._label_segments[-1][1]
            if (
                scene_index == self._confirmed_scene_index
                and self._label_segments
            )
            else self._prepared_label_id
            if scene_index == self._prepared_scene_index
            else None
        )
        return {
            "source": "cued-protocol",
            "protocol_mode": "centered-single-decision",
            "protocol_version": CUED_PROTOCOL_VERSION,
            "phase": phase,
            "scene_index": scene_index,
            "scene_number": 0 if phase == "preparing" else scene_index + 1,
            "label_id": label_id,
            "label_name": (
                None if label_id is None else LABEL_ID_TO_NAME[label_id]
            ),
            "start_lane": self._prepared_start_lane,
            "safe_lane": self._confirmed_safe_lane,
            "current_lane": self._current_lane,
            "phase_remaining_sec": float(phase_remaining_sec),
            "scene_confirmed": scene_index == self._confirmed_scene_index,
            "scene_failed": self._active_scene_failed,
            "valid_from_monotonic": valid_from_monotonic,
            "valid_until_monotonic": valid_until_monotonic,
        }

    @property
    def lane_transition_guard_sec(self) -> float:
        """Future-confirmation delay required for transition-safe labels."""

        return self._lane_transition_guard_sec

    def is_window_transition_guarded(
        self,
        *,
        scene_index: int,
        window_start: float,
        window_end: float,
    ) -> bool:
        """Return whether a window touches a lane transition's guard interval."""

        start = float(window_start)
        end = float(window_end)
        guard = self._lane_transition_guard_sec
        if guard <= 0.0 or end < start:
            return False
        with self._lock:
            return any(
                event_scene_index == int(scene_index)
                and start < transition_time + guard
                and end > transition_time - guard
                for event_scene_index, transition_time in self._lane_transition_events
            )


def _relative_action_label(*, current_lane: int, safe_lane: int) -> int | None:
    if current_lane > safe_lane:
        return LABEL_NAME_TO_ID["left"]
    if current_lane < safe_lane:
        return LABEL_NAME_TO_ID["right"]
    return None


def build_cued_online_label_source(
    config: dict[str, Any],
    *,
    clock: Any = time.monotonic,
) -> CuedOnlineLabelSource:
    """Build the car experiment's balanced cue source from the project config."""

    from adaptation.mi_protocol import generate_block_sequence

    adaptation = config.get("online_adaptation", {}) or {}
    cue_config = adaptation.get("cued_labels", {}) or {}
    from utils.timebase import seconds_to_windows

    window_duration_sec = float(config.get("window_sec", 4.0))
    if "balance_pool_window_seconds_per_class" in cue_config:
        balance_pool_per_class = seconds_to_windows(
            cue_config["balance_pool_window_seconds_per_class"],
            window_duration_sec,
        )
    else:
        # Compatibility only: old configs expressed this budget as a raw count.
        balance_pool_per_class = max(
            int(cue_config.get("balance_pool_per_class", cue_config.get("trials_per_class", 32))),
            1,
        )
    configured_seed = cue_config.get("random_seed")
    sequence_seed = (
        int(configured_seed)
        if configured_seed is not None
        else secrets.randbits(32)
    )
    pool_rng = random.Random(sequence_seed)
    sequence = generate_block_sequence(
        {label: balance_pool_per_class for label in LABEL_NAME_TO_ID},
        rng=pool_rng,
    )
    timing = config.get("protocol", {}).get("trial_timing", {}) or {}
    return CuedOnlineLabelSource(
        sequence,
        start_delay_sec=float(cue_config.get("start_delay_sec", 5.0)),
        scene_duration_sec=float(
            cue_config.get("scene_duration_sec", timing.get("control_sec", 5.0))
        ),
        boundary_guard_sec=float(cue_config.get("boundary_guard_sec", 0.5)),
        lane_transition_guard_sec=float(
            cue_config.get("lane_transition_guard_sec", 0.5)
        ),
        primary_windows_per_scene=int(
            cue_config.get("primary_windows_per_scene", 1)
        ),
        primary_window_spacing_sec=float(
            cue_config.get("primary_window_spacing_sec", 1.0)
        ),
        pool_rng=pool_rng,
        sequence_seed=sequence_seed,
        clock=clock,
    )


class ManualLabelHttpServer:
    """Small HTTP server that lets external tools post realtime labels."""

    def __init__(
        self,
        label_source: ManualOnlineLabelSource,
        *,
        host: str = "127.0.0.1",
        port: int = 8776,
    ) -> None:
        self.label_source = label_source
        self.host = host
        self.port = int(port)
        self._httpd = ThreadingHTTPServer((host, self.port), _make_handler(self))
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="oi-mi-label-http-server",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()
        LOGGER.info("Manual label server listening on http://%s:%s", self.host, self.port)

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=1.0)


def coerce_label(label: str | int) -> tuple[int, str]:
    if isinstance(label, int):
        if label not in LABEL_ID_TO_NAME:
            raise ValueError(f"Unsupported label id: {label}")
        return label, LABEL_ID_TO_NAME[label]

    normalized = str(label).strip().lower()
    aliases = {
        "0": "left",
        "1": "right",
        "左": "left",
        "右": "right",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in LABEL_NAME_TO_ID:
        raise ValueError(f"Unsupported label: {label}")
    return LABEL_NAME_TO_ID[normalized], normalized


def _make_handler(runtime: ManualLabelHttpServer) -> type[BaseHTTPRequestHandler]:
    class LabelHandler(BaseHTTPRequestHandler):
        def do_OPTIONS(self) -> None:  # noqa: N802
            self._write_json(HTTPStatus.NO_CONTENT, {})

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/api/label":
                self._write_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})
                return
            now = time.monotonic()
            label = runtime.label_source.get_label(window_start=now, window_end=now)
            self._write_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "label": None if label is None else asdict(label),
                },
            )

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/label":
                self._write_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})
                return
            try:
                payload = self._read_json()
                event = runtime.label_source.set_label(
                    payload.get("label", ""),
                    ttl_sec=payload.get("ttl_sec"),
                    source=str(payload.get("source", "manual")),
                    payload={key: value for key, value in payload.items() if key not in {"label", "ttl_sec", "source"}},
                )
            except Exception as exc:  # noqa: BLE001
                self._write_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
                return
            self._write_json(HTTPStatus.OK, {"ok": True, "label": asdict(event)})

        def do_DELETE(self) -> None:  # noqa: N802
            if self.path != "/api/label":
                self._write_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})
                return
            runtime.label_source.clear()
            self._write_json(HTTPStatus.OK, {"ok": True})

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            LOGGER.debug("label-server " + format, *args)

        def _read_json(self) -> dict[str, Any]:
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise ValueError("Invalid Content-Length") from exc
            raw = self.rfile.read(max(content_length, 0))
            try:
                payload = json.loads(raw.decode("utf-8")) if raw else {}
            except json.JSONDecodeError as exc:
                raise ValueError("Request body must be JSON") from exc
            if not isinstance(payload, dict):
                raise ValueError("Request body must be a JSON object")
            return payload

        def _write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET,POST,DELETE,OPTIONS")
            self.end_headers()
            if self.command != "OPTIONS":
                self.wfile.write(raw)

    return LabelHandler
