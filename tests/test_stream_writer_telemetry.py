from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.stream_writer import StreamWriter
from tools.verify_experiment_bundle import verify_bundle


class StreamWriterTelemetryTests(unittest.TestCase):
    def test_persists_raw_predictions_revisions_and_adaptation_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            writer = StreamWriter(root, chunk_size=2)
            writer.start({"mode": "realtime"})
            writer.put(
                np.zeros((2, 8), dtype=np.float32),
                y_true=1,
                y_pred=-1,
                confidence=0.4,
                raw_pred=1,
                model_revision=2,
                label_event_id="cue-000001",
                training_role="primary_decision",
                adaptation_eligible=True,
                adaptation_committed=False,
                control_gate_active=False,
                scene_current_lane=0,
                instruction_label=1,
                vehicle_required_action=1,
                quality_accepted=False,
                quality_peak_abs_uv=420.0,
                quality_clip_fraction=0.02,
                quality_bad_channel_fraction=0.25,
            )
            writer.stop()
            writer.update_manifest({"online_adaptation": {"update_count": 2}})

            with np.load(root / "chunks" / "chunk_000000.npz") as payload:
                self.assertEqual(payload["predictions_raw"].tolist(), [1])
                self.assertEqual(payload["model_revisions"].tolist(), [2])
                self.assertEqual(payload["label_event_ids"].tolist(), ["cue-000001"])
                self.assertEqual(payload["training_roles"].tolist(), ["primary_decision"])
                self.assertEqual(payload["adaptation_eligible"].tolist(), [True])
                self.assertEqual(payload["adaptation_committed"].tolist(), [False])
                self.assertEqual(payload["scene_current_lanes"].tolist(), [0])
                self.assertEqual(payload["instruction_labels"].tolist(), [1])
                self.assertEqual(payload["vehicle_required_actions"].tolist(), [1])
                self.assertEqual(payload["quality_accepted"].tolist(), [False])
                self.assertEqual(payload["quality_peak_abs_uv"].tolist(), [420.0])
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["online_adaptation"]["update_count"], 2)
            self.assertEqual(manifest["quality_rejected_windows"], 1)
            self.assertEqual(manifest["quality_accepted_windows"], 0)

    def test_final_manifest_contains_recomputable_scientific_metrics_and_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            writer = StreamWriter(root, chunk_size=3)
            writer.start({"mode": "realtime", "n_classes": 3})
            for index, (truth, raw, operational) in enumerate(
                [(0, 0, 0), (1, 1, -1), (2, 1, 1)]
            ):
                writer.append_event(
                    "scene_start",
                    timestamp_monotonic=10.0 + (index * 5.0),
                    scene_index=index,
                    label_id=truth,
                )
                probabilities = np.full(3, 0.05, dtype=np.float32)
                probabilities[raw] = 0.9
                writer.put(
                    np.full((2, 8), index, dtype=np.float32),
                    y_true=truth,
                    y_pred=operational,
                    raw_pred=raw,
                    confidence=float(probabilities[raw]),
                    probabilities=probabilities,
                    uncertainty=0.1,
                    model_revision=index,
                    label_event_id=f"scene-{index:06d}",
                    window_start_monotonic=10.5 + (index * 5.0),
                    window_end_monotonic=12.5 + (index * 5.0),
                    scene_index=index,
                    scene_label=truth,
                    scene_start_lane=(-1, 0, 1)[index],
                    scene_safe_lane=(0, 0, 0)[index],
                    instruction_label=truth,
                    vehicle_required_action=truth,
                    training_role="primary_decision",
                    adaptation_eligible=True,
                    adaptation_committed=True,
                    mapped_command="STOP" if operational < 0 else str(operational),
                    quality_reasons=(),
                    quality_bad_channel_indices=(),
                )
                writer.append_event(
                    "scene_end",
                    timestamp_monotonic=15.0 + (index * 5.0),
                    scene_index=index,
                    label_id=truth,
                    outcome="failed" if index == 2 else "success",
                    reason="endpoint_lane_mismatch" if index == 2 else "safe_lane_reached",
                    endpoint_lane=(0, 0, 1)[index],
                    endpoint_matches_safe_lane=index != 2,
                )
            writer.stop()
            writer.finalize_manifest()

            with np.load(root / "chunks" / "chunk_000000.npz") as payload:
                self.assertEqual(payload["probabilities"].shape, (3, 3))
                self.assertEqual(payload["scene_indices"].tolist(), [0, 1, 2])
                self.assertEqual(payload["scene_start_lanes"].tolist(), [-1, 0, 1])
                self.assertEqual(payload["scene_safe_lanes"].tolist(), [0, 0, 0])
                self.assertTrue(np.all(np.isfinite(payload["window_start_monotonic"])))
                self.assertEqual(payload["mapped_commands"].tolist(), ["0", "STOP", "1"])

            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            metrics = manifest["scientific_metrics"]
            self.assertAlmostEqual(metrics["raw_window"]["accuracy"], 2.0 / 3.0)
            self.assertAlmostEqual(
                metrics["raw_window"]["balanced_accuracy"],
                2.0 / 3.0,
            )
            self.assertAlmostEqual(
                metrics["operational_window"]["coverage"],
                2.0 / 3.0,
            )
            self.assertEqual(metrics["primary_decision_windows"], 3)
            self.assertEqual(metrics["adaptation_committed_windows"], 3)
            self.assertAlmostEqual(
                metrics["primary_decision"]["raw"]["balanced_accuracy"],
                2.0 / 3.0,
            )
            self.assertEqual(metrics["car_task"]["completed_scenes"], 3)
            self.assertAlmostEqual(metrics["car_task"]["success_rate"], 2.0 / 3.0)
            self.assertEqual(metrics["car_task"]["endpoint_verified_scenes"], 3)
            self.assertEqual(
                metrics["car_task"]["by_label_id"]["2"]["failed_scenes"],
                1,
            )
            self.assertEqual(
                metrics["car_task"]["failure_reasons"],
                {"endpoint_lane_mismatch": 1},
            )
            self.assertEqual(manifest["integrity"]["status"], "complete")
            checksum_paths = {
                item["path"] for item in manifest["integrity"]["checksums"]
            }
            self.assertIn("events.jsonl", checksum_paths)
            self.assertIn("chunks/chunk_000000.npz", checksum_paths)
            self.assertTrue(verify_bundle(root)["ok"])

    def test_continuous_metrics_exclude_primary_decision_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            writer = StreamWriter(root, chunk_size=2)
            writer.start({"mode": "realtime", "n_classes": 3})
            writer.put(
                np.zeros((2, 8), dtype=np.float32),
                y_true=0,
                y_pred=0,
                raw_pred=0,
                confidence=0.9,
                probabilities=np.asarray([0.9, 0.05, 0.05], dtype=np.float32),
                training_role="primary_decision",
                adaptation_eligible=True,
            )
            writer.put(
                np.ones((2, 8), dtype=np.float32),
                y_true=1,
                y_pred=2,
                raw_pred=2,
                confidence=0.9,
                probabilities=np.asarray([0.05, 0.05, 0.9], dtype=np.float32),
                training_role="continuous_context",
                adaptation_eligible=False,
            )
            writer.stop()
            writer.finalize_manifest()

            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            metrics = manifest["scientific_metrics"]
            self.assertEqual(metrics["raw_window"]["samples"], 2)
            self.assertEqual(metrics["primary_decision"]["raw"]["samples"], 1)
            self.assertEqual(metrics["primary_decision"]["raw"]["accuracy"], 1.0)
            self.assertEqual(metrics["continuous_dynamic_windows"], 1)
            self.assertEqual(metrics["continuous_dynamic"]["raw"]["samples"], 1)
            self.assertEqual(metrics["continuous_dynamic"]["raw"]["accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
