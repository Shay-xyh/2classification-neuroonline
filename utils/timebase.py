"""Shared time-to-window conversions for training and online adaptation.

Experiment quantities are configured in seconds.  Window counts are derived
only after the model window duration is known.  Conversion rounds upward so a
requested time budget is never silently shortened.
"""

from __future__ import annotations

import math


def seconds_to_windows(seconds: float, window_duration_sec: float) -> int:
    """Return the minimum whole-window count covering ``seconds``."""

    seconds = float(seconds)
    window_duration_sec = float(window_duration_sec)
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise ValueError("seconds must be finite and positive")
    if not math.isfinite(window_duration_sec) or window_duration_sec <= 0.0:
        raise ValueError("window_duration_sec must be finite and positive")
    return max(1, int(math.ceil(seconds / window_duration_sec - 1e-12)))


def windows_to_seconds(windows: int, window_duration_sec: float) -> float:
    """Return the represented window-seconds for a whole-window count."""

    windows = int(windows)
    window_duration_sec = float(window_duration_sec)
    if windows < 0:
        raise ValueError("windows must be non-negative")
    if not math.isfinite(window_duration_sec) or window_duration_sec <= 0.0:
        raise ValueError("window_duration_sec must be finite and positive")
    return float(windows * window_duration_sec)


def resolved_time_budget(seconds: float, window_duration_sec: float) -> dict[str, float | int]:
    """Describe a requested duration and its realizable whole-window budget."""

    windows = seconds_to_windows(seconds, window_duration_sec)
    return {
        "requested_seconds": float(seconds),
        "window_duration_seconds": float(window_duration_sec),
        "windows": windows,
        "actual_window_seconds": windows_to_seconds(windows, window_duration_sec),
    }
