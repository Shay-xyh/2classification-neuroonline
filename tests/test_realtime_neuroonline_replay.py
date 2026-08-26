from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from adaptation.neuroonline import NeuroOnlineConfig
from tools.simulate_neuroonline_realtime import (
    causal_guarded_replay,
    causal_replay,
    load_committed_data,
)


class _RecordingAdapter:
    def __init__(self) -> None:
        self.revision = 0
        self.updates: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    def predict_proba(self, windows: np.ndarray) -> np.ndarray:
        result = np.zeros((len(windows), 3), dtype=np.float32)
        result[:, self.revision % 3] = 1.0
        return result

    def neuroonline_update(
        self,
        original: np.ndarray,
        time_view: np.ndarray,
        frequency_view: np.ndarray,
        labels: np.ndarray,
        **_kwargs: object,
    ) -> dict[str, float]:
        self.updates.append(
            (original.copy(), time_view.copy(), frequency_view.copy(), labels.copy())
        )
        self.revision += 1
        return {"updated": float(len(labels)), "loss": float(self.revision)}


class _GateAdapter:
    def __init__(self) -> None:
        self.revision = 0

    def predict_proba(self, windows: np.ndarray) -> np.ndarray:
        result = np.zeros((len(windows), 3), dtype=np.float32)
        result[:, self.revision % 3] = 1.0
        return result

    def neuroonline_update(self, *args: object, **kwargs: object) -> dict[str, float]:
        del args, kwargs
        self.revision += 1
        return {"updated": 4.0, "loss": float(self.revision)}


class RealtimeNeuroOnlineReplayTests(unittest.TestCase):
    def test_causal_replay_predicts_before_updates_and_reuses_fixed_masks(self) -> None:
        config = NeuroOnlineConfig(
            enabled=True,
            random_seed=19,
            history_threshold=4,
            update_stride=2,
            recent_samples=4,
            update_batch_size=2,
            epochs=1,
            mask_ratio=0.3,
        )
        windows = np.arange(8 * 2 * 10, dtype=np.float32).reshape(8, 2, 10)
        labels = np.arange(8, dtype=np.int64) % 3
        scenes = np.arange(8, dtype=np.int64)
        adapter = _RecordingAdapter()

        probabilities, revisions, history = causal_replay(
            adapter,
            windows,
            labels,
            scenes,
            config=config,
            n_classes=3,
        )

        self.assertEqual([item["trigger_seen_labeled_windows"] for item in history], [4, 6, 8])
        np.testing.assert_array_equal(revisions, [0, 0, 0, 0, 1, 1, 2, 2])
        np.testing.assert_array_equal(probabilities.argmax(axis=1), revisions % 3)
        np.testing.assert_array_equal(adapter.updates[0][0], windows[:4])
        np.testing.assert_array_equal(adapter.updates[1][0], windows[2:6])
        np.testing.assert_array_equal(adapter.updates[0][1][2:], adapter.updates[1][1][:2])
        np.testing.assert_array_equal(adapter.updates[0][2][2:], adapter.updates[1][2][:2])

    def test_causal_replay_predicts_both_trial_windows_before_atomic_update(self) -> None:
        config = NeuroOnlineConfig(
            enabled=True,
            random_seed=19,
            history_threshold=3,
            update_stride=3,
            recent_samples=6,
            update_batch_size=2,
            epochs=1,
            mask_ratio=0.3,
        )
        windows = np.arange(6 * 2 * 10, dtype=np.float32).reshape(6, 2, 10)
        labels = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
        scenes = np.asarray([10, 10, 11, 11, 12, 12], dtype=np.int64)
        adapter = _RecordingAdapter()

        probabilities, revisions, history = causal_replay(
            adapter,
            windows,
            labels,
            scenes,
            config=config,
            n_classes=3,
            atomic_scene_groups=True,
        )

        self.assertEqual([item["trigger_seen_labeled_windows"] for item in history], [4, 6])
        np.testing.assert_array_equal(revisions, [0, 0, 0, 0, 1, 1])
        np.testing.assert_array_equal(probabilities.argmax(axis=1), revisions % 3)
        np.testing.assert_array_equal(adapter.updates[0][0], windows[:4])

    def test_causal_replay_keeps_temporal_holdout_labels_sealed(self) -> None:
        config = NeuroOnlineConfig(
            enabled=True,
            random_seed=19,
            history_threshold=2,
            update_stride=2,
            recent_samples=4,
            update_batch_size=2,
            epochs=1,
            mask_ratio=0.3,
        )
        windows = np.ones((8, 2, 10), dtype=np.float32)
        labels = np.repeat(np.arange(4, dtype=np.int64) % 3, 2)
        scenes = np.repeat(np.arange(4, dtype=np.int64), 2)
        adapter = _RecordingAdapter()

        _, revisions, history = causal_replay(
            adapter,
            windows,
            labels,
            scenes,
            config=config,
            n_classes=3,
            atomic_scene_groups=True,
            max_update_seen=4,
        )

        self.assertEqual([item["trigger_seen_labeled_windows"] for item in history], [2, 4])
        np.testing.assert_array_equal(revisions, [0, 0, 1, 1, 2, 2, 2, 2])
        self.assertEqual(len(adapter.updates), 2)

    def test_guarded_replay_accepts_only_strict_future_cumulative_gain(self) -> None:
        config = NeuroOnlineConfig(
            enabled=True,
            random_seed=3,
            history_threshold=4,
            update_stride=2,
            recent_samples=4,
            update_batch_size=2,
            epochs=1,
            mask_ratio=0.2,
        )
        windows = np.ones((8, 2, 10), dtype=np.float32)
        labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
        scenes = np.arange(8, dtype=np.int64)

        final_adapter, probabilities, revisions, updates, replacements = (
            causal_guarded_replay(
                _GateAdapter(),
                windows,
                labels,
                scenes,
                config=config,
                n_classes=3,
            )
        )

        self.assertTrue(replacements[0]["accepted"])
        self.assertFalse(replacements[1]["accepted"])
        self.assertEqual(replacements[0]["evaluated_windows"], 2)
        self.assertGreater(
            replacements[0]["candidate_hypothetical_cumulative_accuracy"],
            replacements[0]["active_cumulative_accuracy"],
        )
        self.assertEqual(final_adapter.revision, 1)
        np.testing.assert_array_equal(revisions, [0, 0, 0, 0, 0, 0, 1, 1])
        np.testing.assert_array_equal(probabilities.argmax(axis=1), revisions)
        self.assertEqual(len(updates), 3)

    def test_loader_filters_only_committed_rows_in_chunk_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recording = Path(tmp)
            chunks = recording / "chunks"
            chunks.mkdir()
            for chunk_index in range(2):
                count = 3
                base = chunk_index * count
                np.savez_compressed(
                    chunks / f"chunk_{chunk_index:06d}.npz",
                    eeg_windows=np.full((count, 2, 8), base, dtype=np.float32),
                    labels_true=np.asarray([0, 1, 2], dtype=np.int64),
                    scene_indices=np.arange(base, base + count, dtype=np.int64),
                    label_event_ids=np.asarray([f"event-{base + i}" for i in range(count)]),
                    window_end_monotonic=np.arange(base + 1, base + count + 1, dtype=np.float64),
                    training_roles=np.asarray(["primary_decision"] * count),
                    adaptation_committed=np.asarray([True, False, True]),
                    quality_accepted=np.ones(count, dtype=bool),
                    quality_peak_abs_uv=np.ones(count, dtype=np.float32),
                    quality_clip_fraction=np.zeros(count, dtype=np.float32),
                )

            data = load_committed_data(recording)

            np.testing.assert_array_equal(data.source_rows, [0, 2, 0, 2])
            np.testing.assert_array_equal(data.labels, [0, 2, 0, 2])
            np.testing.assert_array_equal(data.window_end_monotonic, [1.0, 3.0, 4.0, 6.0])


if __name__ == "__main__":
    unittest.main()
