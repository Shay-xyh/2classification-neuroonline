"""Download and verify the official CBraMod foundation checkpoint."""

from __future__ import annotations

import hashlib
from pathlib import Path
import urllib.request


URL = "https://huggingface.co/weighting666/CBraMod/resolve/main/pretrained_weights.pth"
SHA256 = "0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178"
TARGET = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "pretrained"
    / "cbramod_pretrained_weights.pth"
)


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    temporary = TARGET.with_suffix(".download")
    urllib.request.urlretrieve(URL, temporary)
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    if digest != SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"CBraMod checkpoint SHA-256 mismatch: {digest}")
    temporary.replace(TARGET)
    print(f"CBraMod pretrained weights ready: {TARGET}")


if __name__ == "__main__":
    main()
