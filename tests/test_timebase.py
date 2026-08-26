from __future__ import annotations

import pytest

from utils.timebase import resolved_time_budget, seconds_to_windows, windows_to_seconds


def test_seconds_to_windows_never_shortens_requested_budget() -> None:
    assert seconds_to_windows(32.0, 4.0) == 8
    assert seconds_to_windows(32.0, 3.0) == 11
    assert windows_to_seconds(11, 3.0) == 33.0


def test_resolved_budget_reports_requested_and_actual_seconds() -> None:
    assert resolved_time_budget(32.0, 3.0) == {
        "requested_seconds": 32.0,
        "window_duration_seconds": 3.0,
        "windows": 11,
        "actual_window_seconds": 33.0,
    }


@pytest.mark.parametrize("seconds,duration", [(0.0, 4.0), (32.0, 0.0)])
def test_invalid_time_budget_is_rejected(seconds: float, duration: float) -> None:
    with pytest.raises(ValueError):
        seconds_to_windows(seconds, duration)
