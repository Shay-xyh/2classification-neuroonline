from __future__ import annotations

import pytest

from tools.reprocess_calibration import (
    _parse_exclusion_notes,
    _restore_peak_only,
    _select_trials,
)


def _trial(block_index: int, trial_index: int, label: str) -> dict[str, object]:
    return {
        "block_index": block_index,
        "trial_index": trial_index,
        "label": label,
        "label_id": {"left": 0, "right": 1, "idle": 2}[label],
    }


def test_select_trials_records_balanced_whole_block_exclusion() -> None:
    trials = [
        _trial(block, trial, label)
        for block in (3, 9, 10)
        for trial, label in enumerate(("left", "right", "idle"))
    ]

    selected, report = _select_trials(
        trials,
        excluded_blocks=(9, 3, 9),
        exclusion_notes={3: "cognitive interference", 9: "frontal artifact"},
    )

    assert [trial["block_index"] for trial in selected] == [10, 10, 10]
    assert report["source_trial_count"] == 9
    assert report["excluded_trial_count"] == 6
    assert report["included_trial_count"] == 3
    assert report["included_class_counts"] == {"idle": 1, "left": 1, "right": 1}
    assert [entry["block_index"] for entry in report["excluded_blocks"]] == [3, 9]


def test_select_trials_rejects_unknown_block() -> None:
    with pytest.raises(ValueError, match="absent from metadata"):
        _select_trials(
            [_trial(0, 0, "left")],
            excluded_blocks=(3,),
            exclusion_notes={},
        )


def test_parse_exclusion_notes_requires_index_and_text() -> None:
    assert _parse_exclusion_notes(["3=cognitive interference"]) == {
        3: "cognitive interference"
    }
    with pytest.raises(ValueError, match="INDEX=TEXT"):
        _parse_exclusion_notes(["3"])


def test_restore_peak_only_never_restores_clipping_failure() -> None:
    assert _restore_peak_only(("extreme_amplitude",), enabled=True)
    assert not _restore_peak_only(("extreme_amplitude",), enabled=False)
    assert not _restore_peak_only(("excessive_clipping",), enabled=True)
    assert not _restore_peak_only(
        ("extreme_amplitude", "excessive_clipping"), enabled=True
    )
