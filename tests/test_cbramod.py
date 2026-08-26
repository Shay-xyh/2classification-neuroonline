"""CBraMod integration tests."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch

from adaptation.neuroonline import NeuroOnlineConfig, NeuroOnlineModelAdapter
from models.cbramod import CBraModClassifier
from models.factory import DEFAULT_CBRAMOD_WEIGHTS, ModelFactory, TorchModelAdapter


@unittest.skipUnless(DEFAULT_CBRAMOD_WEIGHTS.is_file(), "CBraMod weights not downloaded")
class CBraModIntegrationTests(unittest.TestCase):
    def test_official_checkpoint_and_classifier_forward(self) -> None:
        adapter = ModelFactory.get(
            "cbramod",
            n_chans=4,
            n_times=400,
            sfreq=200,
            n_classes=3,
        )
        probabilities = adapter.predict_proba(np.zeros((2, 4, 400), dtype=np.float32))
        self.assertEqual(probabilities.shape, (2, 3))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)

    def test_neuroonline_receives_cbramod_patch_tokens(self) -> None:
        base = ModelFactory.get(
            "cbramod",
            n_chans=4,
            n_times=400,
            sfreq=200,
            n_classes=3,
        )
        self.assertIsInstance(base, TorchModelAdapter)
        adapter = NeuroOnlineModelAdapter(
            base,
            config=NeuroOnlineConfig(enabled=True, prompt_count=2),
        )
        probabilities = adapter.predict_proba(np.zeros((1, 4, 400), dtype=np.float32))
        self.assertEqual(probabilities.shape, (1, 3))
        self.assertEqual(adapter._feature_shape, (200, 4, 2))

    def test_rejects_incompatible_sampling_rate_and_window(self) -> None:
        with self.assertRaisesRegex(ValueError, "200 Hz"):
            CBraModClassifier(
                n_chans=4,
                n_times=400,
                sfreq=250,
                n_classes=3,
                pretrained_path=DEFAULT_CBRAMOD_WEIGHTS,
            )
        with self.assertRaisesRegex(ValueError, "whole number"):
            CBraModClassifier(
                n_chans=4,
                n_times=300,
                sfreq=200,
                n_classes=3,
                pretrained_path=DEFAULT_CBRAMOD_WEIGHTS,
            )
