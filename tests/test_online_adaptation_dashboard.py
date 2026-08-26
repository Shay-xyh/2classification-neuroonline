from __future__ import annotations

import unittest

from utils import online_adaptation_dashboard as dashboard


class _FakeColumn:
    def __init__(self, owner: "_FakeStreamlit") -> None:
        self.owner = owner

    def __enter__(self) -> "_FakeColumn":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def metric(self, *args: object, **kwargs: object) -> None:
        self.owner.metrics += 1
        self.owner.metric_calls.append((args, kwargs))


class _FakeStreamlit:
    def __init__(self) -> None:
        self.metrics = 0
        self.metric_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.dataframes = 0
        self.line_charts = 0
        self.successes = 0

    def columns(self, count: int) -> list[_FakeColumn]:
        return [_FakeColumn(self) for _ in range(count)]

    def dataframe(self, *_args: object, **_kwargs: object) -> None:
        self.dataframes += 1

    def line_chart(self, *_args: object, **_kwargs: object) -> None:
        self.line_charts += 1

    def success(self, *_args: object, **_kwargs: object) -> None:
        self.successes += 1

    def __getattr__(self, _name: str):
        return lambda *_args, **_kwargs: None


class OnlineAdaptationDashboardTests(unittest.TestCase):
    def test_continuous_cue_dashboard_uses_cumulative_scene(self) -> None:
        fake = _FakeStreamlit()

        dashboard.render_online_cue_panel(
            {
                "source": "cued-protocol",
                "phase": "control",
                "label_name": "left",
                "phase_remaining_sec": 2.5,
                "scene_number": 97,
                "continuous": True,
            },
            ui=fake,
        )

        self.assertIn((("累计 Scene", 97), {}), fake.metric_calls)
        self.assertFalse(any("轮" in str(args) for args, _ in fake.metric_calls))

    def test_neuroonline_dashboard_renders_diagnostics(self) -> None:
        adaptation = {
            "enabled": True,
            "strategy": "neuroonline",
            "state": "collecting",
            "update_count": 1,
            "buffered_windows": 3,
            "seen_labeled_windows": 3,
            "samples_until_update": 1,
            "next_update_step": 4,
            "progress": 0.75,
            "class_counts": {"0": 1, "1": 1, "2": 1},
            "prequential": {
                "balanced_accuracy": 2 / 3,
                "per_class_accuracy": {"0": 1.0, "1": 0.0, "2": 1.0},
                "confusion_matrix": [[1, 0, 0], [1, 0, 0], [0, 0, 1]],
            },
            "last_result": {
                "loss": 1.0,
                "classification_loss": 0.8,
                "consistency_loss": 0.2,
                "duration_sec": 0.1,
            },
            "update_history": [
                {
                    "update": 1,
                    "loss": 1.0,
                    "classification_loss": 0.8,
                    "consistency_loss": 0.2,
                    "gate_alpha": 0.01,
                    "gate_beta": -0.01,
                    "prequential_accuracy": 2 / 3,
                    "prequential_balanced_accuracy": 2 / 3,
                }
            ],
        }
        fake = _FakeStreamlit()

        dashboard.render_online_adaptation_panel(adaptation, ui=fake)

        self.assertEqual(fake.metrics, 7)
        self.assertEqual(fake.dataframes, 2)
        self.assertEqual(fake.line_charts, 3)
        self.assertEqual(fake.successes, 1)

    def test_disabled_dashboard_is_empty(self) -> None:
        fake = _FakeStreamlit()

        dashboard.render_online_adaptation_panel({"enabled": False}, ui=fake)

        self.assertEqual(fake.metrics, 0)
        self.assertEqual(fake.dataframes, 0)
        self.assertEqual(fake.line_charts, 0)


if __name__ == "__main__":
    unittest.main()
