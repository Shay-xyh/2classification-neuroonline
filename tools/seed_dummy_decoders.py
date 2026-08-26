"""Train and export decoder weights for hardware-free dummy EEG testing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.factory import ModelFactory
from utils.preprocessing import filter_and_transform

DEFAULT_PROFILE = {
    "n_chans": 59,
    "sfreq": 200.0,
    "window_sec": 4.0,
    "n_classes": 2,
    "n_samples": 240,
    "epochs": 12,
    "batch_size": 32,
    "learning_rate": 0.001,
    "seed": 17,
}


def _asset_name(model_name: str, n_chans: int, n_times: int) -> str:
    return f"{model_name}_{n_chans}x{n_times}.pt"


def generate_synthetic_dataset(
    *,
    n_samples: int,
    n_chans: int,
    n_times: int,
    sfreq: float,
    n_classes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_classes, size=n_samples, dtype=np.int64)
    processed_windows: list[np.ndarray] = []
    time_axis = np.arange(n_times, dtype=np.float64) / sfreq

    for label in labels:
        alpha_hz = 8.0 + 2.0 * float(label)
        beta_hz = 18.0 + 1.5 * float(label)
        phase = rng.random(n_chans, dtype=np.float64) * (2.0 * np.pi)
        alpha = np.sin(2.0 * np.pi * alpha_hz * time_axis[None, :] + phase[:, None])
        beta = np.sin(2.0 * np.pi * beta_hz * time_axis[None, :] + (phase[:, None] * 0.5))
        noise = rng.normal(0.0, 5.0, size=(n_chans, n_times))
        raw = (6.0 * alpha + 3.0 * beta + noise).astype(np.float32)
        processed_windows.append(filter_and_transform(raw, sfreq=sfreq))

    return np.stack(processed_windows, axis=0).astype(np.float32), labels


def seed_profile(
    *,
    output_dir: Path,
    model_names: list[str],
    profile: dict[str, float | int],
) -> list[dict[str, object]]:
    n_chans = int(profile["n_chans"])
    sfreq = float(profile["sfreq"])
    window_sec = float(profile["window_sec"])
    n_classes = int(profile["n_classes"])
    n_times = int(round(sfreq * window_sec))
    X, y = generate_synthetic_dataset(
        n_samples=int(profile["n_samples"]),
        n_chans=n_chans,
        n_times=n_times,
        sfreq=sfreq,
        n_classes=n_classes,
        seed=int(profile["seed"]),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[dict[str, object]] = []
    for model_name in model_names:
        adapter = ModelFactory.get(
            model_name,
            n_chans=n_chans,
            sfreq=sfreq,
            n_classes=n_classes,
            n_times=n_times,
        )
        metrics = adapter.fit(
            X,
            y,
            epochs=int(profile["epochs"]),
            batch_size=int(profile["batch_size"]),
            learning_rate=float(profile["learning_rate"]),
            head_only=False,
        )
        asset_path = output_dir / _asset_name(model_name, n_chans, n_times)
        adapter.save(asset_path)
        metrics_path = asset_path.with_suffix(asset_path.suffix + ".metrics.yaml")
        with metrics_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                {
                    "model_name": model_name,
                    "profile": profile,
                    "metrics": metrics,
                    "asset_path": str(asset_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                },
                handle,
                sort_keys=False,
                allow_unicode=True,
            )
        saved.append(
            {
                "model_name": model_name,
                "asset_path": str(asset_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "metrics": metrics,
            }
        )
    return saved


def write_manifest(
    *,
    output_dir: Path,
    profiles: list[dict[str, object]],
) -> Path:
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump({"profiles": profiles}, handle, ensure_ascii=False, indent=2)
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "assets" / "dummy_decoders",
        help="Directory for bundled dummy decoder assets.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["cbramod"],
        help="Model registry names to export.",
    )
    parser.add_argument("--n-chans", type=int, default=DEFAULT_PROFILE["n_chans"])
    parser.add_argument("--sfreq", type=float, default=DEFAULT_PROFILE["sfreq"])
    parser.add_argument("--window-sec", type=float, default=DEFAULT_PROFILE["window_sec"])
    parser.add_argument("--n-classes", type=int, default=DEFAULT_PROFILE["n_classes"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = dict(DEFAULT_PROFILE)
    profile.update(
        {
            "n_chans": args.n_chans,
            "sfreq": args.sfreq,
            "window_sec": args.window_sec,
            "n_classes": args.n_classes,
        }
    )
    n_times = int(round(float(profile["sfreq"]) * float(profile["window_sec"])))
    saved = seed_profile(
        output_dir=args.output_dir,
        model_names=list(args.models),
        profile=profile,
    )
    manifest_path = write_manifest(
        output_dir=args.output_dir,
        profiles=[
            {
                "n_chans": int(profile["n_chans"]),
                "sfreq": float(profile["sfreq"]),
                "window_sec": float(profile["window_sec"]),
                "n_times": n_times,
                "n_classes": int(profile["n_classes"]),
                "models": saved,
            }
        ],
    )
    print(f"Saved dummy decoder assets to {args.output_dir}")
    print(f"Manifest: {manifest_path}")
    for item in saved:
        print(f"- {item['model_name']}: {item['asset_path']} metrics={item['metrics']}")


if __name__ == "__main__":
    main()
