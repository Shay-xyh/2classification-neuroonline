"""Verify oi-mi calibration or realtime experiment bundles before archival."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


REALTIME_REQUIRED_ARRAYS = {
    "eeg_windows",
    "labels_true",
    "labels_pred",
    "predictions_raw",
    "probabilities",
    "model_revisions",
    "window_start_monotonic",
    "window_end_monotonic",
    "scene_indices",
    "quality_accepted",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_bundle(root: Path) -> dict[str, Any]:
    metadata_path = root / "manifest.json"
    if not metadata_path.exists():
        metadata_path = root / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError("Bundle has neither manifest.json nor metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    integrity = metadata.get("integrity", {}) or {}
    errors: list[str] = []
    verified_files = 0
    for record in integrity.get("checksums", []) or []:
        path = root / str(record.get("path", ""))
        if not path.exists():
            errors.append(f"missing file: {path.relative_to(root)}")
            continue
        actual = sha256(path)
        if actual != str(record.get("sha256", "")):
            errors.append(f"checksum mismatch: {path.relative_to(root)}")
        else:
            verified_files += 1

    chunk_files = sorted((root / "chunks").glob("chunk_*.npz"))
    for chunk_path in chunk_files:
        with np.load(chunk_path, allow_pickle=False) as chunk:
            missing = REALTIME_REQUIRED_ARRAYS.difference(chunk.files)
            if missing:
                errors.append(
                    f"{chunk_path.name} missing arrays: {sorted(missing)}"
                )
                continue
            row_count = int(chunk["labels_true"].shape[0])
            for key in REALTIME_REQUIRED_ARRAYS:
                if int(chunk[key].shape[0]) != row_count:
                    errors.append(
                        f"{chunk_path.name} row mismatch: {key}"
                    )
            starts = np.asarray(chunk["window_start_monotonic"], dtype=np.float64)
            ends = np.asarray(chunk["window_end_monotonic"], dtype=np.float64)
            if np.any(~np.isfinite(starts) | ~np.isfinite(ends) | (ends <= starts)):
                errors.append(f"{chunk_path.name} has invalid window timestamps")

    if int(metadata.get("dropped_records", 0) or 0) > 0:
        errors.append("stream writer dropped records")
    if str(integrity.get("status", "complete")) != "complete":
        errors.append(f"integrity status is {integrity.get('status')}")
    return {
        "ok": not errors,
        "bundle": str(root.resolve()),
        "verified_files": verified_files,
        "chunk_files": len(chunk_files),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    result = verify_bundle(args.bundle)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
