from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from decoder.real_time_decoder import RealTimeDecoder, _PendingCuedWindow
from utils.online_labels import (
    CUED_PROTOCOL_VERSION,
    CuedOnlineLabelSource,
    build_cued_online_label_source,
)


class _Clock:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class _GameOutlet:
    def __init__(self, *, current_lane: int = 0) -> None:
        self.commands: list[str] = []
        self.events: list[dict[str, object]] = []
        self.current_lane = current_lane
        self.next_scene_number = 1

    def push(self, command: str) -> None:
        self.commands.append(command)

    def push_with_ack(self, command: str) -> dict[str, object]:
        self.push(command)
        if command == "SCENE_STATE":
            return {
                "ack": command,
                "protocol_version": CUED_PROTOCOL_VERSION,
                "scene_number": self.next_scene_number,
                "current_lane": self.current_lane,
                "next_scene_start_lane": 0,
            }
        label_by_command = {
            "SCENE_LEFT": ("left", -1),
            "SCENE_RIGHT": ("right", 1),
        }
        label, delta = label_by_command[command]
        start_lane = 0
        safe_lane = start_lane + delta
        if safe_lane not in {-1, 0, 1}:
            raise RuntimeError("unreachable relative action")
        response = {
            "ack": command,
            "protocol_version": CUED_PROTOCOL_VERSION,
            "scene_number": self.next_scene_number,
            "start_lane": start_lane,
            "safe_lane": safe_lane,
            "applied_label": label,
        }
        self.next_scene_number += 1
        return response

    def poll_events(self) -> list[dict[str, object]]:
        events = list(self.events)
        self.events.clear()
        return events


def _bare_decoder(
    source: CuedOnlineLabelSource,
    outlet: _GameOutlet,
) -> RealTimeDecoder:
    decoder = RealTimeDecoder.__new__(RealTimeDecoder)
    decoder._online_label_source = source
    decoder._game_command_outlet = outlet
    decoder._last_game_transport_command = None
    decoder._last_game_transport_error = None
    decoder._last_game_transport_sent_at = 0.0
    decoder._last_game_movement_sent_at = 0.0
    decoder._stop_on_game_disconnect = True
    decoder._scene_sent_scene_index = -1
    decoder._scene_sent_label_id = None
    decoder._unity_scene_number_offset = None
    decoder._unity_scene_numbers = {}
    decoder._max_scenes = None
    decoder._scene_sync_error = None
    decoder._game_disconnect_message = None
    decoder._step_sec = 0.5
    decoder._sfreq = 200.0
    decoder._window_sec = 2.0
    decoder._confidence_threshold = 0.5
    decoder._primary_windows_per_scene = 2
    decoder._primary_window_spacing_sec = 1.0
    decoder._primary_decision_scenes = set()
    decoder._primary_decision_window_bounds = {}
    decoder._primary_decision_probabilities = {}
    decoder._console = type(
        "_Console",
        (),
        {"print": lambda self, *args, **kwargs: None},
    )()
    decoder._stop_event = type(
        "_StopEvent",
        (),
        {
            "__init__": lambda self: setattr(self, "was_set", False),
            "set": lambda self: setattr(self, "was_set", True),
        },
    )()
    return decoder


class CuedOnlineLabelTests(unittest.TestCase):
    def test_two_spaced_primary_windows_release_cued_lateral_control(self) -> None:
        clock = _Clock(100.0)
        source = CuedOnlineLabelSource(
            ["left", "right"],
            scene_duration_sec=5.0,
            start_delay_sec=0.0,
            boundary_guard_sec=0.5,
            primary_windows_per_scene=2,
            clock=clock,
        )
        outlet = _GameOutlet(current_lane=0)
        decoder = _bare_decoder(source, outlet)
        self.assertEqual(source.prepare_scene(scene_index=0, start_lane=0), 0)
        self.assertTrue(
            source.confirm_scene_applied(
                scene_index=0,
                applied_label_id=0,
                start_lane=0,
                safe_lane=-1,
                timestamp_monotonic=100.0,
            )
        )
        decoder._scene_sent_scene_index = 0
        decoder._scene_sent_label_id = 0
        self.assertTrue(decoder._is_cued_control_gate_active())

        label = source.get_label(window_start=100.5, window_end=102.5)
        self.assertIsNotNone(label)
        self.assertEqual(
            decoder._claim_primary_decision_window(
                online_label=label,
                window_start=100.5,
                window_end=102.5,
            ),
            1,
        )
        self.assertTrue(decoder._is_cued_control_gate_active())
        self.assertIsNone(
            decoder._claim_primary_decision_window(
                online_label=label,
                window_start=101.0,
                window_end=103.0,
            )
        )
        self.assertTrue(decoder._is_cued_control_gate_active())
        self.assertEqual(
            decoder._claim_primary_decision_window(
                online_label=label,
                window_start=101.5,
                window_end=103.5,
            ),
            2,
        )
        self.assertFalse(decoder._is_cued_control_gate_active())
        self.assertEqual(
            decoder._primary_decision_window_bounds[0],
            [(100.5, 102.5), (101.5, 103.5)],
        )
        self.assertIsNone(
            decoder._claim_primary_decision_window(
                online_label=label,
                window_start=102.0,
                window_end=104.0,
            )
        )

    def test_default_single_primary_window_releases_control_at_first_window(self) -> None:
        clock = _Clock(100.0)
        source = CuedOnlineLabelSource(
            ["left"],
            scene_duration_sec=5.0,
            start_delay_sec=0.0,
            boundary_guard_sec=0.5,
            clock=clock,
        )
        outlet = _GameOutlet(current_lane=0)
        decoder = _bare_decoder(source, outlet)
        decoder._primary_windows_per_scene = source.metadata()[
            "primary_windows_per_scene"
        ]
        self.assertEqual(decoder._primary_windows_per_scene, 1)
        self.assertEqual(source.prepare_scene(scene_index=0, start_lane=0), 0)
        self.assertTrue(
            source.confirm_scene_applied(
                scene_index=0,
                applied_label_id=0,
                start_lane=0,
                safe_lane=-1,
                timestamp_monotonic=100.0,
            )
        )
        decoder._scene_sent_scene_index = 0
        label = source.get_label(window_start=100.5, window_end=102.5)

        self.assertEqual(
            decoder._claim_primary_decision_window(
                online_label=label,
                window_start=100.5,
                window_end=102.5,
            ),
            1,
        )
        self.assertFalse(decoder._is_cued_control_gate_active())
        self.assertIsNone(
            decoder._claim_primary_decision_window(
                online_label=label,
                window_start=101.5,
                window_end=103.5,
            )
        )

    def test_only_primary_window_can_issue_lateral_command(self) -> None:
        clock = _Clock(100.0)
        source = CuedOnlineLabelSource(
            ["left"],
            scene_duration_sec=5.0,
            start_delay_sec=0.0,
            boundary_guard_sec=0.5,
            clock=clock,
        )
        decoder = _bare_decoder(source, _GameOutlet())
        result = SimpleNamespace(class_id=0)

        self.assertIsNone(
            decoder._game_command_for_window(
                result,
                primary_decision=False,
                control_gate_active=True,
            )
        )
        self.assertEqual(
            decoder._game_command_for_window(
                result,
                primary_decision=True,
                control_gate_active=False,
            ),
            "LEFT",
        )
        self.assertIsNone(
            decoder._game_command_for_window(
                result,
                primary_decision=False,
                control_gate_active=False,
            )
        )

    def test_primary_window_is_cut_at_exact_ack_relative_source_times(self) -> None:
        clock = _Clock(100.0)
        source = CuedOnlineLabelSource(
            ["left"],
            scene_duration_sec=5.0,
            start_delay_sec=0.0,
            boundary_guard_sec=0.5,
            clock=clock,
        )
        decoder = _bare_decoder(source, _GameOutlet())
        decoder._acquirer = SimpleNamespace(
            metadata=SimpleNamespace(timestamp_domain="monotonic")
        )
        decoder._primary_windows_per_scene = 1
        decoder._scene_sent_scene_index = 0
        decoder._scene_started_at = {0: 100.0}
        self.assertEqual(source.prepare_scene(scene_index=0, start_lane=0), 0)
        self.assertTrue(
            source.confirm_scene_applied(
                scene_index=0,
                applied_label_id=0,
                start_lane=0,
                safe_lane=-1,
                timestamp_monotonic=100.0,
            )
        )
        timestamps = 99.5 + np.arange(700, dtype=np.float64) / 200.0
        history = np.tile(np.arange(700, dtype=np.float32), (2, 1))

        aligned = decoder._select_aligned_primary_window(history, timestamps)

        self.assertIsNotNone(aligned)
        assert aligned is not None
        window, window_timestamps = aligned
        self.assertEqual(window.shape, (2, 400))
        self.assertAlmostEqual(float(window_timestamps[0]), 100.5)
        self.assertAlmostEqual(float(window_timestamps[-1]) + 1.0 / 200.0, 102.5)
        np.testing.assert_array_equal(window[0], np.arange(200, 600, dtype=np.float32))

    def test_primary_alignment_waits_until_target_eeg_has_arrived(self) -> None:
        clock = _Clock(100.0)
        source = CuedOnlineLabelSource(
            ["left"],
            scene_duration_sec=5.0,
            start_delay_sec=0.0,
            boundary_guard_sec=0.5,
            clock=clock,
        )
        decoder = _bare_decoder(source, _GameOutlet())
        decoder._acquirer = SimpleNamespace(
            metadata=SimpleNamespace(timestamp_domain="monotonic")
        )
        decoder._primary_windows_per_scene = 1
        decoder._scene_sent_scene_index = 0
        decoder._scene_started_at = {0: 100.0}
        self.assertEqual(source.prepare_scene(scene_index=0, start_lane=0), 0)
        self.assertTrue(
            source.confirm_scene_applied(
                scene_index=0,
                applied_label_id=0,
                start_lane=0,
                safe_lane=-1,
                timestamp_monotonic=100.0,
            )
        )
        timestamps = 99.5 + np.arange(560, dtype=np.float64) / 200.0
        history = np.zeros((2, 560), dtype=np.float32)

        self.assertIsNone(
            decoder._select_aligned_primary_window(history, timestamps)
        )

    def test_explicit_multi_window_alignment_advances_to_next_slot(self) -> None:
        clock = _Clock(100.0)
        source = CuedOnlineLabelSource(
            ["left"],
            scene_duration_sec=5.0,
            start_delay_sec=0.0,
            boundary_guard_sec=0.5,
            primary_windows_per_scene=2,
            primary_window_spacing_sec=1.0,
            clock=clock,
        )
        decoder = _bare_decoder(source, _GameOutlet())
        decoder._acquirer = SimpleNamespace(
            metadata=SimpleNamespace(timestamp_domain="monotonic")
        )
        decoder._scene_sent_scene_index = 0
        decoder._scene_started_at = {0: 100.0}
        decoder._primary_decision_window_bounds = {0: [(100.5, 102.5)]}
        self.assertEqual(source.prepare_scene(scene_index=0, start_lane=0), 0)
        self.assertTrue(
            source.confirm_scene_applied(
                scene_index=0,
                applied_label_id=0,
                start_lane=0,
                safe_lane=-1,
                timestamp_monotonic=100.0,
            )
        )
        timestamps = 100.5 + np.arange(600, dtype=np.float64) / 200.0
        history = np.tile(np.arange(600, dtype=np.float32), (2, 1))

        aligned = decoder._select_aligned_primary_window(history, timestamps)

        self.assertIsNotNone(aligned)
        assert aligned is not None
        _, window_timestamps = aligned
        self.assertAlmostEqual(float(window_timestamps[0]), 101.5)
        self.assertAlmostEqual(float(window_timestamps[-1]) + 1.0 / 200.0, 103.5)

    def test_scene_ack_preserves_send_and_receive_timing(self) -> None:
        class TimedOutlet(_GameOutlet):
            def push_with_ack(self, command: str) -> dict[str, object]:
                response = super().push_with_ack(command)
                response["_sent_at_monotonic"] = 99.95
                response["_received_at_monotonic"] = 100.0
                response["_ack_round_trip_sec"] = 0.05
                return response

        decoder = _bare_decoder(
            CuedOnlineLabelSource(
                ["left"],
                scene_duration_sec=5.0,
                start_delay_sec=0.0,
            ),
            TimedOutlet(),
        )

        response = decoder._push_game_scene_transport_command("SCENE_LEFT")

        self.assertIsNotNone(response)
        self.assertAlmostEqual(decoder._last_game_transport_sent_at, 99.95)
        assert response is not None
        self.assertAlmostEqual(float(response["_received_at_monotonic"]), 100.0)
        self.assertAlmostEqual(float(response["_ack_round_trip_sec"]), 0.05)

    def test_primary_control_averages_only_quality_accepted_windows(self) -> None:
        decoder = _bare_decoder(
            CuedOnlineLabelSource(
                ["left"],
                scene_duration_sec=5.0,
                start_delay_sec=0.0,
            ),
            _GameOutlet(),
        )
        decoder._primary_decision_probabilities = {
            0: [
                np.asarray([0.8, 0.1, 0.1], dtype=np.float32),
                np.asarray([0.2, 0.7, 0.1], dtype=np.float32),
            ]
        }

        result = decoder._aggregate_primary_control_result(0)

        self.assertEqual(result.class_id, 0)
        self.assertAlmostEqual(result.confidence, 0.5)
        rejected = decoder._aggregate_primary_control_result(1)
        self.assertIsNone(rejected.class_id)
        self.assertEqual(rejected.confidence, 0.0)

    def test_cued_primary_windows_have_unique_adaptation_event_ids(self) -> None:
        class Adapter:
            def __init__(self) -> None:
                self.event_ids: list[str] = []

            def add_window(self, *_args, event_id: str, **_kwargs) -> bool:
                self.event_ids.append(event_id)
                return True

            @staticmethod
            def status() -> dict[str, bool]:
                return {"training_in_background": False}

        decoder = _bare_decoder(
            CuedOnlineLabelSource(
                ["left"],
                scene_duration_sec=5.0,
                start_delay_sec=0.0,
            ),
            _GameOutlet(),
        )
        adapter = Adapter()
        decoder._neuroonline_adapter = adapter
        decoder._batch_adapter = None
        decoder._neuroonline_training_notice = False
        label = SimpleNamespace(
            label_id=0,
            event_id="scene-000000-segment-000",
            source="cued-protocol",
        )
        payload = {
            "processed": np.zeros((2, 400), dtype=np.float32),
            "probabilities": np.asarray([0.7, 0.2, 0.1], dtype=np.float32),
            "operational_prediction": 0,
            "prediction_model_revision": 0,
            "online_label": label,
        }

        first = decoder._handle_online_label(**payload, window_end=102.5)
        second = decoder._handle_online_label(**payload, window_end=103.5)

        self.assertTrue(first)
        self.assertTrue(second)
        self.assertEqual(len(set(adapter.event_ids)), 2)
        self.assertTrue(adapter.event_ids[0].endswith("102500000"))
        self.assertTrue(adapter.event_ids[1].endswith("103500000"))

    def test_only_windows_inside_one_scene_receive_labels(self) -> None:
        clock = _Clock(100.0)
        source = CuedOnlineLabelSource(
            ["left", "right"],
            scene_duration_sec=5.0,
            start_delay_sec=0.0,
            boundary_guard_sec=0.0,
            clock=clock,
        )
        self.assertEqual(source.prepare_scene(scene_index=0, start_lane=0), 0)
        self.assertTrue(source.confirm_scene_applied(
            scene_index=0,
            applied_label_id=0,
            start_lane=0,
            safe_lane=-1,
            timestamp_monotonic=100.0,
        ))

        clock.value = 102.0
        label = source.get_label(window_start=100.0, window_end=102.0)
        self.assertIsNotNone(label)
        assert label is not None
        self.assertEqual(label.label_name, "left")
        self.assertEqual(label.event_id, "scene-000000-segment-000")

        clock.value = 105.5
        self.assertIsNone(source.get_label(window_start=103.5, window_end=105.5))

    def test_delayed_first_unity_ack_anchors_scene_instead_of_expiring(self) -> None:
        clock = _Clock(100.0)
        source = CuedOnlineLabelSource(
            ["right"],
            scene_duration_sec=5.0,
            start_delay_sec=0.0,
            boundary_guard_sec=0.5,
            clock=clock,
        )
        self.assertEqual(source.prepare_scene(scene_index=0, start_lane=0), 1)

        clock.value = 112.0
        self.assertTrue(
            source.confirm_scene_applied(
                scene_index=0,
                applied_label_id=1,
                start_lane=0,
                safe_lane=1,
                timestamp_monotonic=112.0,
            )
        )
        status = source.status(now=112.0)
        self.assertEqual(status["scene_index"], 0)
        self.assertEqual(status["valid_from_monotonic"], 112.0)
        self.assertEqual(status["valid_until_monotonic"], 117.0)

    def test_unconfirmed_scene_never_advances_from_local_time_alone(self) -> None:
        clock = _Clock(0.0)
        source = CuedOnlineLabelSource(
            ["left", "right"],
            scene_duration_sec=5.0,
            start_delay_sec=0.0,
            boundary_guard_sec=0.0,
            clock=clock,
        )

        clock.value = 30.0
        status = source.status()
        self.assertEqual(status["phase"], "control")
        self.assertEqual(status["protocol_mode"], "centered-single-decision")
        self.assertEqual(status["scene_index"], 0)
        self.assertEqual(status["scene_number"], 1)
        self.assertIsNone(status["label_name"])
        self.assertEqual(source.metadata()["balance_pool_scenes"], 2)
        self.assertNotIn("total_trials", status)

    def test_confirmed_scene_advances_only_one_step_after_long_pause(self) -> None:
        clock = _Clock(0.0)
        source = CuedOnlineLabelSource(
            ["left", "right"],
            scene_duration_sec=5.0,
            start_delay_sec=0.0,
            boundary_guard_sec=0.0,
            clock=clock,
        )
        self.assertEqual(source.prepare_scene(scene_index=0, start_lane=0), 0)
        self.assertTrue(source.confirm_scene_applied(
            scene_index=0,
            applied_label_id=0,
            start_lane=0,
            safe_lane=-1,
            timestamp_monotonic=0.0,
        ))

        clock.value = 30.0
        status = source.status()
        self.assertEqual(status["scene_index"], 1)
        self.assertFalse(status["scene_confirmed"])

        clock.value = 60.0
        self.assertEqual(source.status()["scene_index"], 1)

    def test_start_delay_has_no_label_or_hidden_control_phase(self) -> None:
        clock = _Clock(10.0)
        source = CuedOnlineLabelSource(
            ["right"],
            scene_duration_sec=5.0,
            start_delay_sec=1.0,
            boundary_guard_sec=0.0,
            clock=clock,
        )

        self.assertEqual(source.status()["phase"], "preparing")
        self.assertIsNone(source.get_label(window_start=9.0, window_end=10.0))
        clock.value = 11.0
        self.assertEqual(source.status()["phase"], "control")

    def test_unity_ack_anchors_scene_start_and_boundary_guard(self) -> None:
        clock = _Clock(100.0)
        source = CuedOnlineLabelSource(
            ["left"],
            scene_duration_sec=5.0,
            start_delay_sec=0.0,
            boundary_guard_sec=0.1,
            clock=clock,
        )

        self.assertIsNone(source.get_label(window_start=100.0, window_end=102.0))
        clock.value = 100.25
        self.assertEqual(source.prepare_scene(scene_index=0, start_lane=0), 0)
        self.assertTrue(
            source.confirm_scene_applied(
                scene_index=0,
                applied_label_id=0,
                start_lane=0,
                safe_lane=-1,
                timestamp_monotonic=100.25,
            )
        )
        self.assertAlmostEqual(source.status()["valid_from_monotonic"], 100.25)
        self.assertIsNone(source.get_label(window_start=100.25, window_end=102.25))
        self.assertIsNotNone(source.get_label(window_start=100.35, window_end=102.35))

    def test_online_training_interval_matches_calibration_half_second_offsets(self) -> None:
        clock = _Clock(100.0)
        source = CuedOnlineLabelSource(
            ["left"],
            scene_duration_sec=5.0,
            start_delay_sec=0.0,
            boundary_guard_sec=0.5,
            clock=clock,
        )
        self.assertEqual(source.prepare_scene(scene_index=0, start_lane=0), 0)
        self.assertTrue(source.confirm_scene_applied(
            scene_index=0,
            applied_label_id=0,
            start_lane=0,
            safe_lane=-1,
            timestamp_monotonic=100.0,
        ))

        clock.value = 104.5
        self.assertIsNone(source.get_label(window_start=100.49, window_end=102.49))
        self.assertIsNotNone(source.get_label(window_start=100.5, window_end=102.5))
        self.assertIsNotNone(source.get_label(window_start=102.5, window_end=104.5))
        self.assertIsNone(source.get_label(window_start=102.51, window_end=104.51))

    def test_lane_settled_splits_truth_and_rejects_crossing_windows(self) -> None:
        clock = _Clock(100.0)
        source = CuedOnlineLabelSource(
            ["left"],
            scene_duration_sec=5.0,
            start_delay_sec=0.0,
            boundary_guard_sec=0.0,
            clock=clock,
        )
        self.assertEqual(source.prepare_scene(scene_index=0, start_lane=0), 0)
        self.assertTrue(source.confirm_scene_applied(
            scene_index=0,
            applied_label_id=0,
            start_lane=0,
            safe_lane=-1,
            timestamp_monotonic=100.0,
        ))
        self.assertTrue(source.update_current_lane(
            scene_index=0,
            current_lane=-1,
            safe_lane=-1,
            timestamp_monotonic=102.0,
        ))

        left = source.get_label(window_start=100.0, window_end=102.0)
        crossing = source.get_label(window_start=101.0, window_end=103.0)
        after_arrival = source.get_label(window_start=102.0, window_end=104.0)
        self.assertIsNotNone(left)
        self.assertEqual(left.label_name, "left")
        self.assertIsNone(crossing)
        self.assertIsNone(after_arrival)
        self.assertEqual(source.metadata()["label_transition_count"], 1)

    def test_lane_transition_guard_rejects_windows_touching_both_sides(self) -> None:
        clock = _Clock(100.0)
        source = CuedOnlineLabelSource(
            ["left"],
            scene_duration_sec=5.0,
            start_delay_sec=0.0,
            boundary_guard_sec=0.0,
            lane_transition_guard_sec=0.5,
            clock=clock,
        )
        self.assertEqual(source.prepare_scene(scene_index=0, start_lane=0), 0)
        self.assertTrue(source.confirm_scene_applied(
            scene_index=0,
            applied_label_id=0,
            start_lane=0,
            safe_lane=-1,
            timestamp_monotonic=100.0,
        ))
        self.assertTrue(source.update_current_lane(
            scene_index=0,
            current_lane=-1,
            safe_lane=-1,
            timestamp_monotonic=102.0,
        ))

        self.assertFalse(source.is_window_transition_guarded(
            scene_index=0,
            window_start=100.0,
            window_end=101.5,
        ))
        self.assertTrue(source.is_window_transition_guarded(
            scene_index=0,
            window_start=101.25,
            window_end=101.75,
        ))
        self.assertTrue(source.is_window_transition_guarded(
            scene_index=0,
            window_start=102.0,
            window_end=103.0,
        ))
        self.assertFalse(source.is_window_transition_guarded(
            scene_index=0,
            window_start=102.5,
            window_end=103.5,
        ))
        self.assertEqual(source.metadata()["lane_transition_guard_sec"], 0.5)

    def test_decoder_delays_then_rejects_pre_transition_training_label(self) -> None:
        clock = _Clock(100.0)
        source = CuedOnlineLabelSource(
            ["left"],
            scene_duration_sec=5.0,
            start_delay_sec=0.0,
            boundary_guard_sec=0.0,
            lane_transition_guard_sec=0.5,
            clock=clock,
        )
        self.assertEqual(source.prepare_scene(scene_index=0, start_lane=0), 0)
        self.assertTrue(source.confirm_scene_applied(
            scene_index=0,
            applied_label_id=0,
            start_lane=0,
            safe_lane=-1,
            timestamp_monotonic=100.0,
        ))
        label = source.get_label(window_start=100.0, window_end=101.2)
        self.assertIsNotNone(label)
        assert label is not None
        self.assertTrue(source.update_current_lane(
            scene_index=0,
            current_lane=-1,
            safe_lane=-1,
            timestamp_monotonic=101.5,
        ))

        class Writer:
            def __init__(self) -> None:
                self.records: list[dict] = []
                self.events: list[tuple[str, dict]] = []

            def put(self, **payload) -> None:
                self.records.append(payload)

            def append_event(self, event_type: str, **payload) -> None:
                self.events.append((event_type, payload))

        decoder = RealTimeDecoder.__new__(RealTimeDecoder)
        decoder._online_label_source = source
        decoder._lane_transition_guard_sec = 0.5
        decoder._writer = Writer()
        decoder._pending_cued_windows = [
            _PendingCuedWindow(
                processed=np.zeros((2, 400), dtype=np.float32),
                probabilities=np.asarray([0.8, 0.1, 0.1], dtype=np.float32),
                operational_prediction=0,
                prediction_model_revision=0,
                online_label=label,
                window_start=100.0,
                window_end=101.2,
                quality_accepted=True,
                training_role="continuous_context",
                adaptation_eligible=False,
                record_payload={"window": np.zeros((2, 400), dtype=np.float32)},
            )
        ]
        decoder._handle_online_label = lambda **_kwargs: self.fail(
            "guarded label reached online adaptation"
        )

        decoder._flush_pending_cued_windows(now=101.6)
        self.assertEqual(len(decoder._pending_cued_windows), 1)
        decoder._flush_pending_cued_windows(now=101.7)
        self.assertEqual(decoder._pending_cued_windows, [])
        self.assertEqual(decoder._writer.records[0]["y_true"], -1)
        self.assertEqual(
            decoder._writer.events[0][1]["reason"],
            "lane_transition_guard",
        )

    def test_decoder_accepts_dynamic_lane_truth_from_unity_event(self) -> None:
        clock = _Clock(100.0)
        source = CuedOnlineLabelSource(
            ["left"],
            scene_duration_sec=5.0,
            start_delay_sec=0.0,
            boundary_guard_sec=0.0,
            clock=clock,
        )
        outlet = _GameOutlet(current_lane=0)
        decoder = _bare_decoder(source, outlet)

        from unittest import mock

        with mock.patch("decoder.real_time_decoder.time.monotonic", return_value=100.0):
            decoder._sync_game_scene()
        outlet.events.append(
            {
                "event": "LANE_SETTLED",
                "protocol_version": CUED_PROTOCOL_VERSION,
                "scene_number": 1,
                "current_lane": -1,
                "safe_lane": -1,
            }
        )
        clock.value = 102.0
        with mock.patch("decoder.real_time_decoder.time.monotonic", return_value=102.0):
            decoder._consume_game_scene_events()

        self.assertIsNone(
            decoder._get_online_label(window_start=101.0, window_end=103.0)
        )
        after_arrival = decoder._get_online_label(window_start=102.0, window_end=104.0)
        self.assertIsNone(after_arrival)

    def test_collision_marks_failure_without_ending_fixed_scene(self) -> None:
        clock = _Clock(100.0)
        source = CuedOnlineLabelSource(
            ["left", "right"],
            scene_duration_sec=10.0,
            start_delay_sec=0.0,
            boundary_guard_sec=0.0,
            clock=clock,
        )
        self.assertEqual(source.prepare_scene(scene_index=0, start_lane=0), 0)
        self.assertTrue(source.confirm_scene_applied(
            scene_index=0,
            applied_label_id=0,
            start_lane=0,
            safe_lane=-1,
            timestamp_monotonic=100.0,
        ))

        clock.value = 104.0
        self.assertTrue(
            source.mark_scene_failed(
                timestamp_monotonic=104.0,
                expected_scene_index=0,
            )
        )
        status = source.status()
        self.assertEqual(status["scene_index"], 0)
        self.assertEqual(status["label_name"], "left")
        self.assertTrue(status["scene_failed"])
        self.assertEqual(source.metadata()["failed_scenes"], 1)
        self.assertFalse(
            source.mark_scene_failed(
                timestamp_monotonic=104.1,
                expected_scene_index=0,
            )
        )

        clock.value = 110.0
        status = source.status()
        self.assertEqual(status["scene_index"], 1)
        self.assertIsNone(status["label_name"])
        self.assertFalse(status["scene_failed"])

    def test_buffered_collision_is_recorded_then_scene_times_out_normally(self) -> None:
        clock = _Clock(0.0)
        source = CuedOnlineLabelSource(
            ["left", "right"],
            scene_duration_sec=7.0,
            start_delay_sec=0.0,
            boundary_guard_sec=0.0,
            clock=clock,
        )
        self.assertEqual(source.prepare_scene(scene_index=0, start_lane=0), 0)
        self.assertTrue(source.confirm_scene_applied(
            scene_index=0,
            applied_label_id=0,
            start_lane=0,
            safe_lane=-1,
            timestamp_monotonic=0.0,
        ))

        clock.value = 7.1
        self.assertTrue(
            source.mark_scene_failed(
                timestamp_monotonic=7.1,
                expected_scene_index=0,
            )
        )
        status = source.status()
        self.assertEqual(status["scene_index"], 1)
        self.assertIsNone(status["label_name"])
        self.assertEqual(source.metadata()["failed_scenes"], 1)

    def test_scene_truth_is_required_before_label_is_accepted(self) -> None:
        clock = _Clock(100.0)
        source = CuedOnlineLabelSource(
            ["right"],
            scene_duration_sec=5.0,
            start_delay_sec=0.0,
            boundary_guard_sec=0.0,
            clock=clock,
        )
        outlet = _GameOutlet()
        decoder = RealTimeDecoder.__new__(RealTimeDecoder)
        decoder._online_label_source = source
        decoder._game_command_outlet = outlet
        decoder._last_game_transport_command = None
        decoder._last_game_transport_error = None
        decoder._last_game_transport_sent_at = 0.0
        decoder._last_game_movement_sent_at = 0.0
        decoder._stop_on_game_disconnect = False
        decoder._scene_sent_scene_index = -1
        decoder._scene_sent_label_id = None
        decoder._scene_sync_error = None
        decoder._console = type("_Console", (), {"print": lambda self, *args, **kwargs: None})()
        decoder._stop_event = type("_StopEvent", (), {"set": lambda self: None})()

        self.assertIsNone(decoder._get_online_label(window_start=100.0, window_end=102.0))
        from unittest import mock

        with mock.patch("decoder.real_time_decoder.time.monotonic", return_value=100.0):
            decoder._sync_game_scene()
        self.assertEqual(outlet.commands, ["SCENE_STATE", "SCENE_RIGHT"])
        label = decoder._get_online_label(window_start=100.0, window_end=102.0)
        self.assertIsNotNone(label)
        assert label is not None
        self.assertEqual(label.label_name, "right")

    def test_visual_onset_delay_anchors_scene_and_primary_window(self) -> None:
        clock = _Clock(100.0)
        source = CuedOnlineLabelSource(
            ["right"],
            scene_duration_sec=5.0,
            start_delay_sec=0.0,
            boundary_guard_sec=0.5,
            clock=clock,
        )
        decoder = _bare_decoder(source, _GameOutlet())
        decoder._visual_onset_delay_sec = 0.02

        from unittest import mock

        with mock.patch("decoder.real_time_decoder.time.monotonic", return_value=100.0):
            decoder._sync_game_scene()

        self.assertAlmostEqual(decoder._scene_started_at[0], 100.02)
        self.assertIsNone(
            decoder._get_online_label(window_start=100.5, window_end=102.5)
        )
        self.assertIsNotNone(
            decoder._get_online_label(window_start=100.52, window_end=102.52)
        )

    def test_reconnect_anchors_persistent_unity_scene_counter(self) -> None:
        clock = _Clock(100.0)
        source = CuedOnlineLabelSource(
            ["left", "right"],
            scene_duration_sec=5.0,
            start_delay_sec=0.0,
            boundary_guard_sec=0.0,
            clock=clock,
        )
        outlet = _GameOutlet(current_lane=0)
        outlet.next_scene_number = 7
        decoder = _bare_decoder(source, outlet)

        from unittest import mock

        with mock.patch("decoder.real_time_decoder.time.monotonic", return_value=100.0):
            decoder._sync_game_scene()

        self.assertEqual(decoder._unity_scene_number_offset, 6)
        self.assertEqual(decoder._unity_scene_numbers[0], 7)
        self.assertEqual(decoder._scene_sent_scene_index, 0)
        self.assertEqual(outlet.commands, ["SCENE_STATE", "SCENE_LEFT"])

        clock.value = 105.0
        outlet.current_lane = -1
        with mock.patch("decoder.real_time_decoder.time.monotonic", return_value=105.0):
            decoder._sync_game_scene()

        self.assertEqual(decoder._unity_scene_numbers[1], 8)
        self.assertEqual(decoder._scene_sent_scene_index, 1)
        self.assertEqual(
            outlet.commands,
            ["SCENE_STATE", "SCENE_LEFT", "SCENE_STATE", "SCENE_RIGHT"],
        )

    def test_lane_settled_changes_truth_without_rebuilding_same_scene(self) -> None:
        clock = _Clock(100.0)
        source = CuedOnlineLabelSource(
            ["right"],
            scene_duration_sec=5.0,
            start_delay_sec=0.0,
            boundary_guard_sec=0.0,
            clock=clock,
        )
        outlet = _GameOutlet(current_lane=0)
        decoder = _bare_decoder(source, outlet)

        from unittest import mock

        with mock.patch("decoder.real_time_decoder.time.monotonic", return_value=100.0):
            decoder._sync_game_scene()
        self.assertEqual(outlet.commands, ["SCENE_STATE", "SCENE_RIGHT"])

        outlet.events.append(
            {
                "event": "LANE_SETTLED",
                "protocol_version": CUED_PROTOCOL_VERSION,
                "scene_number": 1,
                "current_lane": 1,
                "safe_lane": 1,
                "_received_at_monotonic": 102.0,
            }
        )
        clock.value = 102.0
        with mock.patch("decoder.real_time_decoder.time.monotonic", return_value=102.0):
            decoder._sync_game_scene()

        self.assertIsNone(source.status()["label_name"])
        self.assertEqual(outlet.commands, ["SCENE_STATE", "SCENE_RIGHT"])
        self.assertEqual(decoder._scene_sent_scene_index, 0)

    def test_late_lane_settled_is_ignored_without_stopping_next_scene(self) -> None:
        clock = _Clock(100.0)
        source = CuedOnlineLabelSource(
            ["right"],
            scene_duration_sec=5.0,
            start_delay_sec=0.0,
            boundary_guard_sec=0.0,
            clock=clock,
        )
        outlet = _GameOutlet(current_lane=0)
        decoder = _bare_decoder(source, outlet)

        class _Writer:
            def __init__(self) -> None:
                self.events: list[tuple[str, dict[str, object]]] = []

            def append_event(self, event_type: str, **payload: object) -> None:
                self.events.append((event_type, payload))

        decoder._writer = _Writer()

        from unittest import mock

        with mock.patch("decoder.real_time_decoder.time.monotonic", return_value=100.0):
            decoder._sync_game_scene()
        outlet.events.append(
            {
                "event": "LANE_SETTLED",
                "protocol_version": CUED_PROTOCOL_VERSION,
                "scene_number": 1,
                "current_lane": 1,
                "safe_lane": 1,
                "_received_at_monotonic": 105.015,
            }
        )
        clock.value = 105.015
        with mock.patch(
            "decoder.real_time_decoder.time.monotonic",
            return_value=105.015,
        ):
            decoder._consume_game_scene_events()

        self.assertFalse(decoder._stop_event.was_set)
        self.assertEqual(source.status(now=104.999)["label_name"], "right")
        ignored = [
            payload
            for event_type, payload in decoder._writer.events
            if event_type == "lane_settled_ignored"
        ]
        self.assertEqual(len(ignored), 1)
        self.assertEqual(
            ignored[0]["reason"],
            "received_after_fixed_scene_boundary",
        )

    def test_scene_limit_stops_before_next_obstacle_layout(self) -> None:
        clock = _Clock(100.0)
        source = CuedOnlineLabelSource(
            ["left", "right"],
            scene_duration_sec=5.0,
            start_delay_sec=0.0,
            boundary_guard_sec=0.0,
            clock=clock,
        )
        outlet = _GameOutlet(current_lane=0)
        decoder = _bare_decoder(source, outlet)
        decoder._max_scenes = 1

        from unittest import mock

        with mock.patch("decoder.real_time_decoder.time.monotonic", return_value=100.0):
            decoder._sync_game_scene()
        clock.value = 105.0
        outlet.current_lane = -1
        with mock.patch("decoder.real_time_decoder.time.monotonic", return_value=105.0):
            decoder._sync_game_scene()

        self.assertTrue(decoder._stop_event.was_set)
        self.assertEqual(
            outlet.commands,
            ["SCENE_STATE", "SCENE_LEFT", "SCENE_STATE"],
        )
        self.assertEqual(decoder._scene_end_recorded, {0})

    def test_scene_boundary_rejects_unreached_safe_lane(self) -> None:
        class _Writer:
            def __init__(self) -> None:
                self.events: list[tuple[str, dict[str, object]]] = []

            def append_event(self, event_type: str, **payload: object) -> None:
                self.events.append((event_type, payload))

        clock = _Clock(100.0)
        source = CuedOnlineLabelSource(
            ["left"],
            scene_duration_sec=5.0,
            start_delay_sec=0.0,
            boundary_guard_sec=0.0,
            clock=clock,
        )
        outlet = _GameOutlet(current_lane=0)
        decoder = _bare_decoder(source, outlet)
        decoder._max_scenes = 1
        decoder._writer = _Writer()

        from unittest import mock

        with mock.patch("decoder.real_time_decoder.time.monotonic", return_value=100.0):
            decoder._sync_game_scene()
        clock.value = 105.0
        outlet.current_lane = 0
        with mock.patch("decoder.real_time_decoder.time.monotonic", return_value=105.0):
            decoder._sync_game_scene()

        scene_end = [
            payload
            for event_type, payload in decoder._writer.events
            if event_type == "scene_end"
        ]
        self.assertEqual(len(scene_end), 1)
        self.assertEqual(scene_end[0]["outcome"], "failed")
        self.assertEqual(scene_end[0]["reason"], "endpoint_lane_mismatch")
        self.assertFalse(scene_end[0]["endpoint_matches_safe_lane"])

    def test_cued_source_reshuffles_each_balanced_pool(self) -> None:
        config = {
            "online_adaptation": {
                "cued_labels": {
                    "enabled": True,
                    "balance_pool_per_class": 2,
                    "random_seed": 17,
                    "scene_duration_sec": 1.0,
                    "start_delay_sec": 0.0,
                    "boundary_guard_sec": 0.0,
                }
            },
            "protocol": {"trial_timing": {"control_sec": 1.0}},
        }
        clock = _Clock(0.0)
        source = build_cued_online_label_source(config, clock=clock)
        first_pool = list(source.metadata()["sequence"])
        for scene_index in range(len(first_pool)):
            label = source.prepare_scene(scene_index=scene_index, start_lane=0)
            safe_lane = {0: -1, 1: 1}[label]
            self.assertTrue(
                source.confirm_scene_applied(
                    scene_index=scene_index,
                    applied_label_id=label,
                    start_lane=0,
                    safe_lane=safe_lane,
                    timestamp_monotonic=clock.value,
                )
            )
            clock.value += 1.0
            source.status()
        second_pool = []
        for offset in range(len(first_pool)):
            scene_index = len(first_pool) + offset
            label = source.prepare_scene(scene_index=scene_index, start_lane=0)
            second_pool.append(label)
            safe_lane = {0: -1, 1: 1}[label]
            self.assertTrue(
                source.confirm_scene_applied(
                    scene_index=scene_index,
                    applied_label_id=label,
                    start_lane=0,
                    safe_lane=safe_lane,
                    timestamp_monotonic=clock.value,
                )
            )
            clock.value += 1.0
            source.status()
        self.assertEqual(
            {label: second_pool.count(label) for label in set(second_pool)},
            {0: 2, 1: 2},
        )
        self.assertNotEqual(first_pool, second_pool)

    def test_unity_failure_waits_for_fixed_boundary_before_next_scene(self) -> None:
        clock = _Clock(100.0)
        source = CuedOnlineLabelSource(
            ["left", "right"],
            scene_duration_sec=10.0,
            start_delay_sec=0.0,
            boundary_guard_sec=0.0,
            clock=clock,
        )
        outlet = _GameOutlet()
        decoder = RealTimeDecoder.__new__(RealTimeDecoder)
        decoder._online_label_source = source
        decoder._game_command_outlet = outlet
        decoder._last_game_transport_command = None
        decoder._last_game_transport_error = None
        decoder._last_game_transport_sent_at = 0.0
        decoder._last_game_movement_sent_at = 0.0
        decoder._stop_on_game_disconnect = False
        decoder._scene_sent_scene_index = -1
        decoder._scene_sent_label_id = None
        decoder._scene_sync_error = None
        decoder._console = type("_Console", (), {"print": lambda self, *args, **kwargs: None})()
        decoder._stop_event = type("_StopEvent", (), {"set": lambda self: None})()

        from unittest import mock

        with mock.patch("decoder.real_time_decoder.time.monotonic", return_value=100.0):
            decoder._sync_game_scene()
        self.assertEqual(outlet.commands, ["SCENE_STATE", "SCENE_LEFT"])

        clock.value = 103.0
        outlet.events.append({"event": "SCENE_FAILED", "scene_number": 1})
        with mock.patch("decoder.real_time_decoder.time.monotonic", return_value=103.0):
            decoder._sync_game_scene()

        self.assertEqual(outlet.commands, ["SCENE_STATE", "SCENE_LEFT"])
        self.assertEqual(source.status()["scene_index"], 0)
        self.assertTrue(source.status()["scene_failed"])
        self.assertEqual(source.metadata()["failed_scenes"], 1)

        clock.value = 110.0
        outlet.current_lane = -1
        with mock.patch("decoder.real_time_decoder.time.monotonic", return_value=110.0):
            decoder._sync_game_scene()

        self.assertEqual(
            outlet.commands,
            ["SCENE_STATE", "SCENE_LEFT", "SCENE_STATE", "SCENE_RIGHT"],
        )
        self.assertEqual(source.status()["scene_index"], 1)
        self.assertFalse(source.status()["scene_failed"])

    def test_repeated_direction_resets_to_center_after_failure(self) -> None:
        clock = _Clock(0.0)
        source = CuedOnlineLabelSource(
            ["right", "right", "left"],
            scene_duration_sec=5.0,
            start_delay_sec=0.0,
            boundary_guard_sec=0.0,
            clock=clock,
        )
        outlet = _GameOutlet(current_lane=-1)
        decoder = RealTimeDecoder.__new__(RealTimeDecoder)
        decoder._online_label_source = source
        decoder._game_command_outlet = outlet
        decoder._last_game_transport_command = None
        decoder._last_game_transport_error = None
        decoder._last_game_transport_sent_at = 0.0
        decoder._last_game_movement_sent_at = 0.0
        decoder._stop_on_game_disconnect = False
        decoder._scene_sent_scene_index = -1
        decoder._scene_sent_label_id = None
        decoder._scene_sync_error = None
        decoder._console = type("_Console", (), {"print": lambda self, *args, **kwargs: None})()
        decoder._stop_event = type("_StopEvent", (), {"set": lambda self: None})()

        from unittest import mock

        with mock.patch("decoder.real_time_decoder.time.monotonic", return_value=0.0):
            decoder._sync_game_scene()
        self.assertEqual(decoder._scene_labels[0], 1)
        self.assertEqual(decoder._scene_start_lanes[0], 0)
        self.assertEqual(decoder._scene_safe_lanes[0], 1)

        clock.value = 5.0
        outlet.current_lane = -1  # The car failed and never left its lane.
        with mock.patch("decoder.real_time_decoder.time.monotonic", return_value=5.0):
            decoder._sync_game_scene()
        self.assertEqual(decoder._scene_start_lanes[1], 0)
        self.assertEqual(decoder._scene_labels[1], 1)
        self.assertEqual(decoder._scene_safe_lanes[1], 1)

    def test_64_scene_pool_stays_balanced_and_centered(self) -> None:
        clock = _Clock(0.0)
        source = CuedOnlineLabelSource(
            ["left"] * 32 + ["right"] * 32,
            scene_duration_sec=5.0,
            start_delay_sec=0.0,
            boundary_guard_sec=0.0,
            clock=clock,
        )
        counts = {0: 0, 1: 0}
        reported_lane = -1
        for scene_index in range(64):
            source.status()
            label = source.prepare_scene(
                scene_index=scene_index,
                start_lane=reported_lane,
            )
            safe_lane = {0: -1, 1: 1}[label]
            self.assertTrue(
                source.confirm_scene_applied(
                    scene_index=scene_index,
                    applied_label_id=label,
                    start_lane=0,
                    safe_lane=safe_lane,
                    timestamp_monotonic=clock.value,
                )
            )
            counts[label] += 1
            reported_lane = safe_lane
            clock.value += 5.0

        self.assertEqual(counts, {0: 32, 1: 32})

    def test_wrong_unity_safe_lane_aborts_without_creating_training_label(self) -> None:
        class _WrongSafeLaneOutlet(_GameOutlet):
            def push_with_ack(self, command: str) -> dict[str, object]:
                response = super().push_with_ack(command)
                if command != "SCENE_STATE":
                    response["safe_lane"] = 1
                return response

        clock = _Clock(0.0)
        source = CuedOnlineLabelSource(
            ["left"],
            scene_duration_sec=5.0,
            start_delay_sec=0.0,
            boundary_guard_sec=0.0,
            clock=clock,
        )
        decoder = _bare_decoder(source, _WrongSafeLaneOutlet(current_lane=0))

        from unittest import mock

        with mock.patch("decoder.real_time_decoder.time.monotonic", return_value=0.0):
            decoder._sync_game_scene()

        self.assertTrue(decoder._stop_event.was_set)
        self.assertIn("safe", decoder._game_disconnect_message.lower())
        self.assertIsNone(
            decoder._get_online_label(window_start=0.0, window_end=2.0)
        )

    def test_old_or_wrong_unity_protocol_aborts_before_scene_command(self) -> None:
        class _OldProtocolOutlet(_GameOutlet):
            def push_with_ack(self, command: str) -> dict[str, object]:
                response = super().push_with_ack(command)
                response["protocol_version"] = "continuous-scene-v2"
                return response

        clock = _Clock(0.0)
        source = CuedOnlineLabelSource(
            ["right"],
            scene_duration_sec=5.0,
            start_delay_sec=0.0,
            boundary_guard_sec=0.0,
            clock=clock,
        )
        outlet = _OldProtocolOutlet(current_lane=0)
        decoder = _bare_decoder(source, outlet)

        from unittest import mock

        with mock.patch("decoder.real_time_decoder.time.monotonic", return_value=0.0):
            decoder._sync_game_scene()

        self.assertEqual(outlet.commands, ["SCENE_STATE"])
        self.assertTrue(decoder._stop_event.was_set)
        self.assertEqual(decoder._scene_sent_scene_index, -1)
        self.assertIsNone(
            decoder._get_online_label(window_start=0.0, window_end=2.0)
        )


if __name__ == "__main__":
    unittest.main()
