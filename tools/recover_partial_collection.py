"""Recover the conservative, checkpointed prefix of an interrupted session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adaptation.session_recorder import SessionRecorder


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover checkpointed EEG/events from an interrupted collection."
    )
    parser.add_argument("session_dir", type=Path)
    args = parser.parse_args()
    recovery = SessionRecorder.recover_partial(args.session_dir)
    print(json.dumps(recovery, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


