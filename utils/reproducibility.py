"""Deterministic experiment seeding shared by CLI and GUI entry points."""

from __future__ import annotations

import random

import numpy as np


def seed_experiment(seed: int) -> dict[str, object]:
    """Seed model initialization/training and request deterministic torch kernels."""

    value = int(seed)
    random.seed(value)
    np.random.seed(value)
    result: dict[str, object] = {
        "seed": value,
        "python_random": True,
        "numpy": True,
        "torch": False,
        "deterministic_algorithms": False,
    }
    try:
        import torch
    except ImportError:
        return result
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    result["torch"] = True
    result["deterministic_algorithms"] = True
    return result
