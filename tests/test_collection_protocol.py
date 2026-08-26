from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from adaptation.calibrator import Calibrator, CollectionPauseControl, TrialDiscarded
from adaptation.mi_protocol import ProtocolConfig, SessionPlan, TrialTiming


def test_single_trial_uses_expected_events_and_timing() -> None:
    protocol = ProtocolConfig.from_config(
        {"protocol": {}}
    )
    calibrator = Calibrator.__new__(Calibrator)
    calibrator._protocol = protocol
    messages: list[str] = []
    calibrator._console = SimpleNamespace(print=lambda message: messages.append(str(message)))
    events: list[tuple[str, dict[str, object]]] = []
    sample_indices = iter((0, 0, 250, 1_250))

    def emit_event(recorder, name: str, **payload):
        events.append((name, payload))
        return SimpleNamespace(sample_index=next(sample_indices))

    stages: list[tuple[str, float]] = []
    calibrator._emit_event = emit_event
    calibrator._sleep_with_recording = lambda duration, **kwargs: stages.append(
        (kwargs["stage_name"], duration)
    )

    result = calibrator._run_trial(
        label="right",
        recorder=object(),
        heartbeat=None,
        trial_index=0,
        block_index=0,
    )

    assert [name for name, _payload in events] == [
        "fixation_on",
        "cue_right_on",
        "motor_imagery_right_on",
        "motor_imagery_off",
    ]
    assert "label" not in events[0][1]
    assert events[1][1]["target_hand"] == "right"
    assert "label" not in events[1][1]
    assert events[2][1]["label"] == "right"
    assert events[2][1]["label_id"] == 1
    assert stages == [
        ("Block 1 / Trial 1: fixation", 2.0),
        ("Block 1 / Trial 1: movement prompt right", 2.0),
        ("Block 1 / Trial 1: motor imagery right", 4.0),
    ]
    assert result is not None
    assert result["motor_imagery_on_sample"] == 250
    assert result["motor_imagery_off_sample"] == 1_250


def test_removed_protocol_modes_cannot_change_the_fixed_session_plan() -> None:
    protocol = ProtocolConfig.from_config(
        {
            "protocol": {
                "continuous_collection": True,
                "practice_labels": ["left"],
                "practice_repetitions": 99,
                "baseline_segments": [{"name": "rest", "duration_sec": 600}],
                "minimum_calibration_window_seconds": 9999,
            }
        }
    )

    assert not hasattr(protocol, "continuous_collection")
    assert not hasattr(protocol, "practice_labels")
    assert not hasattr(protocol, "baseline_segments")
    assert protocol.collection_blocks == 9
    assert protocol.collection_trials_per_class_per_block == 50


def test_pause_request_interrupts_the_current_trial_stage() -> None:
    calibrator = Calibrator.__new__(Calibrator)
    calibrator._flush_recorder = lambda recorder: None
    calibrator._update_stage_progress = lambda **kwargs: None
    control = CollectionPauseControl()
    control.request_pause()

    with pytest.raises(TrialDiscarded):
        calibrator._sleep_with_recording(
            1.0,
            recorder=object(),
            heartbeat=None,
            stage_name="trial stage",
            pause_control=control,
            interruptible=True,
        )


def test_automatic_break_is_explicitly_bracketed_by_events() -> None:
    calibrator = Calibrator.__new__(Calibrator)
    calibrator._console = SimpleNamespace(print=lambda message: None)
    events: list[str] = []
    calibrator._emit_event = lambda recorder, name, **payload: events.append(name)
    break_states: list[bool] = []
    control = CollectionPauseControl()
    calibrator._sleep_with_recording = lambda duration, **kwargs: break_states.append(
        control.automatic_break
    )
    calibrator._run_formal_blocks(
        SessionPlan(
            subject_mode="fixed_session",
            blocks=[[], []],
            rest_between_blocks_sec=180.0,
            trial_timing=TrialTiming(),
        ),
        recorder=object(),
        heartbeat=None,
        trials=[],
        pause_control=control,
    )

    assert events == [
        "block_start",
        "block_end",
        "automatic_break_start",
        "automatic_break_end",
        "block_start",
        "block_end",
    ]
    assert break_states == [True]
    assert not control.automatic_break


def test_discarded_trial_is_recollected_with_the_same_label() -> None:
    calibrator = Calibrator.__new__(Calibrator)
    calibrator._console = SimpleNamespace(print=lambda message: None)
    events: list[str] = []
    calibrator._emit_event = lambda recorder, name, **payload: events.append(name)
    attempts: list[tuple[str, int]] = []

    def run_trial(**kwargs):
        attempts.append((kwargs["label"], kwargs["attempt_index"]))
        if kwargs["attempt_index"] == 0:
            raise TrialDiscarded("fixation")
        return {"label": kwargs["label"], "attempt_index": kwargs["attempt_index"]}

    calibrator._run_trial = run_trial
    calibrator._wait_for_manual_resume = (
        lambda control, **kwargs: control.resume() if control is not None else None
    )
    trials: list[dict] = []
    control = CollectionPauseControl()
    plan = SessionPlan(
        subject_mode="fixed_session",
        blocks=[["left"]],
        rest_between_blocks_sec=180.0,
        trial_timing=TrialTiming(2.0, 2.0, 4.0),
    )

    calibrator._run_formal_blocks(
        plan,
        recorder=object(),
        heartbeat=None,
        trials=trials,
        pause_control=control,
    )

    assert attempts == [("left", 0), ("left", 1)]
    assert trials == [{"label": "left", "attempt_index": 1}]
    assert events == ["block_start", "trial_discarded", "block_end"]


def test_collection_only_saves_session_without_training_model(tmp_path: Path) -> None:
    protocol = ProtocolConfig.from_config({"protocol": {}})
    messages: list[str] = []
    calibrator = Calibrator(
        acquirer=SimpleNamespace(
            metadata=SimpleNamespace(n_channels=3),
            source_sfreq=250.0,
        ),
        model=None,
        console=SimpleNamespace(print=lambda message: messages.append(message)),
        sfreq=200.0,
        window_sec=2.0,
        step_sec=0.5,
        session_records_dir=tmp_path,
        protocol_config=protocol,
        online_adaptation_config={"enabled": True, "strategy": "neuroonline"},
        experiment_config={
            "subject_id": "S001",
            "sfreq": 200,
            "protocol": {"collection_blocks": 9},
            "model_name": "must-not-enter-collection-metadata",
            "calibration_epochs": 50,
            "online_adaptation": {"enabled": True},
        },
    )
    assert calibrator._model is None
    assert calibrator._neuroonline_config is None
    assert calibrator._experiment_config == {
        "subject_id": "S001",
        "sfreq": 200,
        "protocol": {"collection_blocks": 9},
    }
    session_metadata: dict = {"formal_trial_count": 12}
    processed = np.empty((12, 3, 400), dtype=np.float32)
    calibrator._collect_session_data = lambda **kwargs: (
        tmp_path,
        processed.copy(),
        processed,
        np.zeros(12, dtype=np.int64),
        np.arange(12, dtype=np.int64),
        session_metadata,
    )
    summaries: list[dict] = []
    calibrator._write_collection_summary = lambda session_dir, **kwargs: summaries.append(
        {"session_dir": session_dir, **kwargs}
    )
    sealed: list[tuple[Path, bool]] = []
    calibrator._seal_session_bundle = (
        lambda session_dir, *, include_model_files=True: sealed.append(
            (session_dir, include_model_files)
        )
    )

    result = calibrator.collect()

    assert result.continuous_eeg_path == tmp_path / "continuous_eeg.npy"
    assert result.events_path == tmp_path / "events.json"
    assert result.windows_path == tmp_path / "mi_windows.npz"
    assert result.trials_collected == 12
    assert result.windows_collected == 12
    assert "training" not in session_metadata
    assert summaries[0]["trials_collected"] == 12
    assert summaries[0]["windows_collected"] == 12
    assert sealed == [(tmp_path, False)]
    assert any("采集完成，数据已保存" in message for message in messages)
