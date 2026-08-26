"""Focused tests for the NeuroOnline integration."""

from __future__ import annotations

import tempfile
import unittest
from unittest import mock
from pathlib import Path
import threading
import time

import numpy as np
import torch
from torch import nn
from click.testing import CliRunner

from adaptation.neuroonline import (
    NEUROONLINE_TRAINING_MECHANICS_VERSION,
    NeuroOnlineConfig,
    NeuroOnlineModelAdapter,
    NeuroOnlineStreamAdapter,
    _frequency_mask,
    _time_mask,
)
from models.factory import TorchModelAdapter
from cli import cli


class _TinyDecoder(nn.Module):
    def __init__(self, n_classes: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(4, n_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs).squeeze(-1))


class _BatchNormDropoutDecoder(nn.Module):
    def __init__(self, n_classes: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(4),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(4, n_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs).squeeze(-1))


class _TinyCBraDecoder(nn.Module):
    def __init__(self, n_classes: int = 3) -> None:
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.projection = nn.Linear(2, 4)
        self.backbone.encoder = nn.Module()
        self.backbone.encoder.layers = nn.ModuleList(
            [nn.Linear(4, 4) for _ in range(4)]
        )
        self.classifier = nn.Linear(4, n_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone.projection(inputs.mean(dim=-1))
        for layer in self.backbone.encoder.layers:
            hidden = torch.relu(layer(hidden))
        return self.classifier(hidden)


def _separable_tiny_data(samples_per_class: int = 10) -> tuple[np.ndarray, np.ndarray]:
    labels = np.repeat(np.arange(3, dtype=np.int64), samples_per_class)
    inputs = np.zeros((labels.size, 2, 16), dtype=np.float32)
    inputs[labels == 0, 0, :] = 2.0
    inputs[labels == 1, 1, :] = 2.0
    inputs[labels == 2, :, :] = -2.0
    inputs += np.random.default_rng(7).normal(0.0, 0.02, inputs.shape).astype(np.float32)
    return inputs, labels


class NeuroOnlineTests(unittest.TestCase):
    def test_defaults_start_updating_after_64_windows(self) -> None:
        config = NeuroOnlineConfig.from_mapping(
            {"enabled": True, "strategy": "neuroonline"}
        )
        self.assertEqual(config.history_threshold, 8)
        self.assertEqual(config.update_stride, 8)
        self.assertEqual(config.recent_samples, 160)

    def test_second_based_config_resolves_for_four_second_windows(self) -> None:
        config = NeuroOnlineConfig.from_mapping(
            {
                "enabled": True,
                "strategy": "neuroonline",
                "neuroonline": {
                    "window_duration_sec": 4.0,
                    "update_batch_seconds": 64.0,
                    "first_update_seconds": 32.0,
                    "update_stride_seconds": 32.0,
                    "recent_history_seconds": 640.0,
                    "offline_batch_seconds": 32.0,
                },
            }
        )
        self.assertEqual(config.update_batch_size, 16)
        self.assertEqual(config.history_threshold, 8)
        self.assertEqual(config.update_stride, 8)
        self.assertEqual(config.recent_samples, 160)
        self.assertEqual(config.offline_batch_size, 8)
        self.assertEqual(
            config.time_budget["recent_history"]["actual_window_seconds"],
            640.0,
        )

    def test_offline_augmentation_parameters_are_separate_from_online(self) -> None:
        config = NeuroOnlineConfig.from_mapping(
            {
                "enabled": True,
                "strategy": "neuroonline",
                "neuroonline": {
                    "mask_ratio": 0.7,
                    "consistency_weight": 0.1,
                    "offline_mask_ratio": 0.1,
                    "offline_consistency_weight": 1.5,
                },
            }
        )
        self.assertEqual(config.mask_ratio, 0.7)
        self.assertEqual(config.consistency_weight, 0.1)
        self.assertEqual(config.offline_mask_ratio, 0.1)
        self.assertEqual(config.offline_consistency_weight, 1.5)

    def setUp(self) -> None:
        torch.manual_seed(7)
        np.random.seed(7)

    def test_config_uses_neuroonline_as_the_only_enabled_strategy(self) -> None:
        selected = NeuroOnlineConfig.from_mapping(
            {
                "enabled": True,
                "strategy": "neuroonline",
                "neuroonline": {"history_threshold": 8, "update_stride": 2},
            }
        )
        default_strategy = NeuroOnlineConfig.from_mapping({"enabled": True})
        self.assertTrue(selected.enabled)
        self.assertEqual(selected.history_threshold, 8)
        self.assertEqual(selected.update_stride, 2)
        self.assertTrue(default_strategy.enabled)

    def test_identity_modulation_preserves_predictions_before_online_update(self) -> None:
        base = TorchModelAdapter("tiny", _TinyDecoder())
        inputs = np.random.randn(5, 2, 16).astype(np.float32)
        expected = base.predict_proba(inputs)
        wrapped = NeuroOnlineModelAdapter(base, config=NeuroOnlineConfig(enabled=True))
        actual = wrapped.predict_proba(inputs)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)

    def test_mc_dropout_does_not_update_batchnorm_statistics(self) -> None:
        inputs = np.random.randn(5, 2, 16).astype(np.float32)
        base = TorchModelAdapter("batchnorm-dropout", _BatchNormDropoutDecoder())
        batchnorm = base.model.features[1]
        before_mean = batchnorm.running_mean.detach().clone()
        before_batches = batchnorm.num_batches_tracked.detach().clone()

        base.predict_proba(inputs, mc_dropout_passes=4)

        torch.testing.assert_close(batchnorm.running_mean, before_mean)
        torch.testing.assert_close(batchnorm.num_batches_tracked, before_batches)

        wrapped_base = TorchModelAdapter(
            "batchnorm-dropout",
            _BatchNormDropoutDecoder(),
        )
        wrapped = NeuroOnlineModelAdapter(
            wrapped_base,
            config=NeuroOnlineConfig(enabled=True, prompt_count=4),
        )
        wrapped_batchnorm = wrapped_base.model.features[1]
        wrapped_before_mean = wrapped_batchnorm.running_mean.detach().clone()
        wrapped_before_batches = wrapped_batchnorm.num_batches_tracked.detach().clone()

        wrapped.predict_proba(inputs, mc_dropout_passes=4)

        torch.testing.assert_close(wrapped_batchnorm.running_mean, wrapped_before_mean)
        torch.testing.assert_close(
            wrapped_batchnorm.num_batches_tracked,
            wrapped_before_batches,
        )

    def test_mask_ratio_uses_independent_elementwise_time_and_frequency_masks(self) -> None:
        inputs = torch.ones(3, 2, 20)
        expected_time_generator = torch.Generator().manual_seed(11)
        expected_time_mask = torch.rand(
            inputs.shape,
            generator=expected_time_generator,
        ) < 0.25
        time_masked = _time_mask(
            inputs,
            0.25,
            torch.Generator().manual_seed(11),
        )
        self.assertTrue(torch.equal(time_masked == 0, expected_time_mask))
        self.assertFalse(torch.equal(time_masked[0, 0] == 0, time_masked[0, 1] == 0))

        signal = torch.randn(3, 2, 20)
        original_spectrum = torch.fft.rfft(signal, dim=-1)
        expected_frequency_generator = torch.Generator().manual_seed(11)
        expected_frequency_mask = torch.rand(
            original_spectrum.shape,
            generator=expected_frequency_generator,
        ) < 0.25
        expected_frequency_view = torch.fft.irfft(
            original_spectrum.masked_fill(expected_frequency_mask, 0.0 + 0.0j),
            n=signal.shape[-1],
            dim=-1,
        )
        frequency_masked = _frequency_mask(
            signal,
            0.25,
            torch.Generator().manual_seed(11),
        )
        torch.testing.assert_close(frequency_masked, expected_frequency_view)
        self.assertFalse(
            torch.equal(expected_frequency_mask[0, 0], expected_frequency_mask[0, 1])
        )

    def test_objective_applies_ce_to_all_three_views_and_preserves_lambda(self) -> None:
        class CountingCriterion(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.calls = 0

            def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
                del labels
                self.calls += 1
                return logits.sum() * 0.0 + float(self.calls)

        config = NeuroOnlineConfig(
            enabled=True,
            consistency_weight=0.7,
            prompt_count=4,
        )
        wrapped = NeuroOnlineModelAdapter(
            TorchModelAdapter("tiny", _TinyDecoder()),
            config=config,
        )
        original = torch.randn(6, 2, 16)
        wrapped._prepare_training(original[:1])
        criterion = CountingCriterion()
        loss, classification, consistency = wrapped._training_objective(
            original,
            original * 0.5,
            original * 0.25,
            torch.arange(6) % 3,
            criterion,
        )

        self.assertEqual(criterion.calls, 3)
        torch.testing.assert_close(
            classification,
            torch.tensor(6.0, device=classification.device),
        )
        torch.testing.assert_close(
            loss,
            classification + config.consistency_weight * consistency,
        )

    def test_update_policies_freeze_expected_cbramod_parameters(self) -> None:
        inputs = torch.randn(2, 2, 16)
        base = TorchModelAdapter("tiny-cbramod", _TinyCBraDecoder())
        wrapped = NeuroOnlineModelAdapter(
            base,
            config=NeuroOnlineConfig(enabled=True, prompt_count=4),
        )

        modulator = wrapped._prepare_training(inputs[:1], update_policy="head")
        self.assertTrue(all(p.requires_grad for p in base.model.classifier.parameters()))
        self.assertTrue(all(p.requires_grad for p in modulator.parameters()))
        self.assertFalse(any(p.requires_grad for p in base.model.backbone.parameters()))

        wrapped._prepare_training(inputs[:1], update_policy="last2")
        layers = base.model.backbone.encoder.layers
        self.assertFalse(any(p.requires_grad for p in layers[0].parameters()))
        self.assertFalse(any(p.requires_grad for p in layers[1].parameters()))
        self.assertTrue(all(p.requires_grad for p in layers[2].parameters()))
        self.assertTrue(all(p.requires_grad for p in layers[3].parameters()))

    def test_differential_optimizer_groups_preserve_learning_rates(self) -> None:
        inputs = torch.randn(2, 2, 16)
        base = TorchModelAdapter("tiny-cbramod", _TinyCBraDecoder())
        wrapped = NeuroOnlineModelAdapter(
            base,
            config=NeuroOnlineConfig(enabled=True, prompt_count=4),
        )
        modulator = wrapped._prepare_training(inputs[:1], update_policy="last2")

        groups = wrapped._optimizer_groups(
            modulator,
            head_learning_rate=1e-4,
            backbone_learning_rate=1e-6,
        )

        self.assertEqual([group["group_name"] for group in groups], ["backbone", "head_crm"])
        self.assertEqual([group["lr"] for group in groups], [1e-6, 1e-4])

    def test_full_neuroonline_objective_updates_backbone_and_persists_crm(self) -> None:
        config = NeuroOnlineConfig(
            enabled=True,
            learning_rate=1e-3,
            update_batch_size=4,
            epochs=1,
            history_threshold=4,
            update_stride=2,
            recent_samples=4,
            prompt_count=4,
        )
        base = TorchModelAdapter("tiny", _TinyDecoder())
        wrapped = NeuroOnlineModelAdapter(base, config=config)
        original = np.random.randn(8, 2, 16).astype(np.float32)
        time_view = original.copy()
        time_view[..., ::2] = 0.0
        frequency_view = original * 0.5
        labels = np.arange(8, dtype=np.int64) % 3
        before = base.model.features[0].weight.detach().clone()
        metrics = wrapped.neuroonline_update(original, time_view, frequency_view, labels)
        after = base.model.features[0].weight.detach().clone()

        self.assertEqual(metrics["updated"], 8.0)
        self.assertGreater(metrics["loss"], 0.0)
        self.assertIn("gate_alpha", metrics)
        self.assertIn("gate_beta", metrics)
        self.assertFalse(torch.equal(before, after))
        self.assertIsNotNone(wrapped._modulator)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "tiny.pt"
            expected = wrapped.predict_proba(original[:3])
            wrapped.save(path)
            self.assertTrue(path.exists())
            self.assertTrue(Path(f"{path}.neuroonline.pt").exists())
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["checkpoint_format"], "neuroonline_bundle_v1")
            self.assertIn("neuroonline", checkpoint)

            Path(f"{path}.neuroonline.pt").unlink()
            restored_base = TorchModelAdapter("tiny", _TinyDecoder())
            restored_base.load(path)
            restored = NeuroOnlineModelAdapter(
                restored_base,
                config=config,
                state_path=path,
            )
            pending_copy = Path(tmp_dir) / "pending-copy.pt"
            restored.save(pending_copy)
            pending_payload = torch.load(
                pending_copy,
                map_location="cpu",
                weights_only=True,
            )
            self.assertIn("neuroonline", pending_payload)
            np.testing.assert_allclose(
                restored.predict_proba(original[:3]),
                expected,
                rtol=1e-6,
                atol=1e-7,
            )
            restored.load(path)
            np.testing.assert_allclose(
                restored.predict_proba(original[:3]),
                expected,
                rtol=1e-6,
                atol=1e-7,
            )

    def test_offline_fit_initializes_and_trains_complete_neuroonline_checkpoint(self) -> None:
        config = NeuroOnlineConfig(
            enabled=True,
            prompt_count=4,
            offline_epochs=20,
            offline_batch_size=6,
            offline_learning_rate=1e-2,
        )
        wrapped = NeuroOnlineModelAdapter(TorchModelAdapter("tiny", _TinyDecoder()), config=config)
        inputs, labels = _separable_tiny_data()
        epoch_states: dict[
            int,
            tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
        ] = {}

        def capture_epoch(
            epoch: int,
            total_epochs: int,
            metrics: dict[str, float],
        ) -> None:
            del total_epochs, metrics
            assert wrapped._modulator is not None
            epoch_states[epoch] = (
                {
                    name: value.detach().cpu().clone()
                    for name, value in wrapped.base.model.state_dict().items()
                },
                {
                    name: value.detach().cpu().clone()
                    for name, value in wrapped._modulator.state_dict().items()
                },
            )

        metrics = wrapped.fit(
            inputs,
            labels,
            epochs=1,
            batch_size=2,
            learning_rate=1e-5,
            progress_callback=capture_epoch,
        )
        self.assertIsNotNone(wrapped._modulator)
        self.assertIn("val_kappa", metrics)
        self.assertIn("val_balanced_accuracy", metrics)
        self.assertGreaterEqual(metrics["val_acc"], 0.0)
        self.assertEqual(metrics["epochs_completed"], 20.0)
        self.assertGreaterEqual(metrics["best_epoch"], 1.0)
        self.assertEqual(len(epoch_states), 20)
        best_model_state, best_modulator_state = epoch_states[int(metrics["best_epoch"])]
        for name, value in wrapped.base.model.state_dict().items():
            torch.testing.assert_close(value.cpu(), best_model_state[name])
        assert wrapped._modulator is not None
        for name, value in wrapped._modulator.state_dict().items():
            torch.testing.assert_close(value.cpu(), best_modulator_state[name])

    def test_fit_with_split_rejects_trial_leakage(self) -> None:
        inputs, labels = _separable_tiny_data()
        groups = np.repeat(np.arange(labels.size // 2, dtype=np.int64), 2)
        wrapped = NeuroOnlineModelAdapter(
            TorchModelAdapter("tiny", _TinyDecoder()),
            config=NeuroOnlineConfig(enabled=True, prompt_count=4),
        )

        with self.assertRaisesRegex(ValueError, "same trial"):
            wrapped.fit_with_split(
                inputs,
                labels,
                train_indices=np.arange(0, labels.size, 2),
                validation_indices=np.arange(1, labels.size, 2),
                groups=groups,
            )

    def test_full_fit_uses_every_window_without_validation_selection(self) -> None:
        inputs, labels = _separable_tiny_data()
        wrapped = NeuroOnlineModelAdapter(
            TorchModelAdapter("tiny", _TinyDecoder()),
            config=NeuroOnlineConfig(
                enabled=True,
                prompt_count=4,
                offline_epochs=2,
                offline_batch_size=6,
                offline_learning_rate=1e-2,
            ),
        )

        metrics = wrapped.fit_full(inputs, labels)

        self.assertEqual(metrics["epochs_completed"], 2.0)
        self.assertEqual(metrics["scheduler_horizon_epochs"], 2.0)
        self.assertIn("train_balanced_accuracy", metrics)
        self.assertIn("train_worst_class_recall", metrics)

    def test_full_fit_can_stop_before_scheduler_horizon(self) -> None:
        inputs, labels = _separable_tiny_data()
        wrapped = NeuroOnlineModelAdapter(
            TorchModelAdapter("tiny", _TinyDecoder()),
            config=NeuroOnlineConfig(
                enabled=True,
                prompt_count=4,
                offline_epochs=3,
                offline_batch_size=6,
                offline_learning_rate=1e-2,
            ),
        )

        metrics = wrapped.fit_full(inputs, labels, epochs=1)

        self.assertEqual(metrics["epochs_completed"], 1.0)
        self.assertEqual(metrics["scheduler_horizon_epochs"], 3.0)

    def test_checkpoint_restores_model_settings_without_overriding_online_seed(self) -> None:
        selected = NeuroOnlineConfig(
            enabled=True,
            prompt_count=4,
            mask_ratio=0.5,
            consistency_weight=0.3,
            random_seed=2026,
        )
        wrapped = NeuroOnlineModelAdapter(
            TorchModelAdapter("tiny", _TinyDecoder()),
            config=selected,
        )
        inputs = np.random.randn(2, 2, 16).astype(np.float32)
        wrapped.predict_proba(inputs)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "tiny.pt"
            wrapped.save(model_path)
            restored = NeuroOnlineModelAdapter(
                TorchModelAdapter("tiny", _TinyDecoder()),
                config=NeuroOnlineConfig(
                    enabled=True,
                    prompt_count=4,
                    mask_ratio=0.3,
                    consistency_weight=0.1,
                    random_seed=42,
                ),
                state_path=model_path,
            )
            self.assertEqual(restored.config.mask_ratio, 0.3)
            self.assertEqual(restored.config.consistency_weight, 0.1)
            self.assertEqual(restored.config.random_seed, 42)

    def test_offline_seed_is_independent_from_online_seed(self) -> None:
        config = NeuroOnlineConfig.from_mapping(
            {"neuroonline": {"random_seed": 2026, "offline_random_seed": 42}}
        )

        self.assertEqual(config.random_seed, 2026)
        self.assertEqual(config.effective_offline_random_seed, 42)

    def test_train_from_records_restores_neuroonline_main_and_crm_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            records = root / "records" / "S001" / "calibration" / "20260722_120000"
            records.mkdir(parents=True)
            inputs, labels = _separable_tiny_data(samples_per_class=10)
            binary = labels < 2
            inputs = inputs[binary]
            labels = labels[binary]
            inputs = np.repeat(inputs, 50, axis=2)
            np.savez_compressed(
                records / "training_windows_main.npz",
                raw_windows=inputs,
                processed_windows=inputs,
                labels=labels,
            )
            config_path = root / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "subject_id: S001",
                        "model_name: cbramod",
                        "device_type: neuracle",
                        "hardware_dummy_mode: false",
                        "sfreq: 200",
                        "n_classes: 2",
                        "window_sec: 4.0",
                        "step_sec: 0.5",
                        "calibration_epochs: 20",
                        "batch_size: 4",
                        "learning_rate: 0.001",
                        "storage:",
                        f"  models_dir: {str(root / 'models')!r}",
                        f"  records_dir: {str(root / 'records')!r}",
                        "online_adaptation:",
                        "  enabled: true",
                        "  strategy: neuroonline",
                        "  neuroonline:",
                        "    prompt_count: 4",
                        "    mask_ratio: 0.1",
                        "    offline_epochs: 20",
                        "    offline_batch_size: 4",
                        "    offline_learning_rate: 0.01",
                    ]
                ),
                encoding="utf-8",
            )

            with mock.patch(
                "models.factory.ModelFactory.get",
                return_value=TorchModelAdapter("cbramod", _TinyDecoder(n_classes=2)),
            ):
                result = CliRunner().invoke(
                    cli,
                    [
                        "--config",
                        str(config_path),
                        "train-from-records",
                        "--subject",
                        "S001",
                        "--model",
                        "cbramod",
                        "--device",
                        "neuracle",
                        "--session",
                        "20260722_120000",
                    ],
                )

            self.assertEqual(result.exit_code, 0, f"{result.output}\n{result.exception!r}")
            model_path = root / "models" / "S001" / "neuracle" / "cbramod.pt"
            self.assertTrue(model_path.exists())
            self.assertTrue(Path(f"{model_path}.neuroonline.pt").exists())
            self.assertTrue(model_path.with_suffix(".metrics.yaml").exists())
            self.assertIn("CRM 已保存", result.output)
            self.assertIn("训练指标已保存", result.output)

    def test_stream_updates_at_threshold_then_stride(self) -> None:
        calls: list[int] = []

        def update(
            original: np.ndarray,
            time_view: np.ndarray,
            frequency_view: np.ndarray,
            labels: np.ndarray,
        ) -> dict[str, float]:
            self.assertEqual(original.shape, time_view.shape)
            self.assertEqual(original.shape, frequency_view.shape)
            self.assertEqual(original.shape[0], labels.shape[0])
            calls.append(original.shape[0])
            return {"updated": float(original.shape[0]), "loss": 1.0}

        adapter = NeuroOnlineStreamAdapter(
            config=NeuroOnlineConfig(
                enabled=True,
                history_threshold=4,
                update_stride=2,
                recent_samples=4,
                prompt_count=4,
            ),
            update_callback=update,
        )
        for index in range(7):
            label = index % 2
            adapter.add_window(
                np.full((2, 16), index, dtype=np.float32),
                label,
                predicted_label=label,
            )

        self.assertTrue(adapter.wait_for_idle(timeout_sec=2.0))
        self.assertEqual(calls, [4, 4])
        status = adapter.status()
        self.assertEqual(status["seen_labeled_windows"], 7)
        self.assertEqual(status["update_count"], 2)
        self.assertEqual(status["samples_until_update"], 1)
        self.assertEqual(status["prequential"]["evaluated_windows"], 7)
        self.assertEqual(status["prequential"]["balanced_accuracy"], 1.0)
        self.assertEqual(status["prequential"]["confusion_matrix"], [[4, 0], [0, 3]])
        self.assertEqual(len(status["update_history"]), 2)
        first_update = status["update_history"][0]
        self.assertEqual(first_update["trigger_seen_labeled_windows"], 4)
        self.assertEqual(first_update["snapshot_first_window_id"], 1)
        self.assertEqual(first_update["snapshot_last_window_id"], 4)
        self.assertEqual(first_update["snapshot_samples"], 4)
        self.assertAlmostEqual(status["progress"], 0.5)

    def test_stream_update_runs_in_background(self) -> None:
        update_started = threading.Event()
        allow_update = threading.Event()

        def update(*_arrays: np.ndarray) -> dict[str, float]:
            update_started.set()
            allow_update.wait(timeout=2.0)
            return {"updated": 4.0, "loss": 1.0}

        adapter = NeuroOnlineStreamAdapter(
            config=NeuroOnlineConfig(
                enabled=True,
                history_threshold=4,
                update_stride=2,
                recent_samples=4,
            ),
            update_callback=update,
        )
        started_at = time.perf_counter()
        for index in range(4):
            adapter.add_window(np.zeros((2, 16), dtype=np.float32), index % 2)

        self.assertTrue(update_started.wait(timeout=1.0))
        self.assertLess(time.perf_counter() - started_at, 0.5)
        self.assertTrue(adapter.status()["training_in_background"])
        allow_update.set()
        self.assertTrue(adapter.wait_for_idle(timeout_sec=2.0))
        self.assertEqual(adapter.status()["update_count"], 1)

    def test_stream_rejects_duplicate_event_and_non_increasing_timestamp(self) -> None:
        adapter = NeuroOnlineStreamAdapter(
            config=NeuroOnlineConfig(
                enabled=True,
                history_threshold=64,
                update_stride=64,
                recent_samples=64,
            ),
            update_callback=lambda *_arrays: {"updated": 1.0},
        )
        window = np.zeros((2, 16), dtype=np.float32)
        self.assertTrue(
            adapter.add_window(
                window,
                0,
                event_id="scene-000001-primary",
                window_end_monotonic=10.0,
            )
        )
        self.assertFalse(
            adapter.add_window(
                window,
                0,
                event_id="scene-000001-primary",
                window_end_monotonic=10.5,
            )
        )
        self.assertFalse(
            adapter.add_window(
                window,
                1,
                event_id="scene-000002-primary",
                window_end_monotonic=9.5,
            )
        )
        status = adapter.status()
        self.assertEqual(status["seen_labeled_windows"], 1)
        self.assertEqual(status["duplicate_windows_rejected"], 1)
        self.assertEqual(status["stale_windows_rejected"], 1)


if __name__ == "__main__":
    unittest.main()
