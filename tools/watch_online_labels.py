"""Watch labels posted to the optional manual-label test endpoint."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.online_labels import ManualLabelHttpServer, ManualOnlineLabelSource


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start a small /api/label server and print incoming labels.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address. Use 0.0.0.0 for LAN tests.")
    parser.add_argument("--port", type=int, default=8776, help="HTTP label server port.")
    parser.add_argument("--ttl-sec", type=float, default=2.0, help="Default label lifetime.")
    parser.add_argument("--poll-sec", type=float, default=0.1, help="Display refresh interval.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = ManualOnlineLabelSource(default_ttl_sec=args.ttl_sec)
    server = ManualLabelHttpServer(source, host=args.host, port=args.port)
    server.start()
    print(f"Listening for labels at http://{args.host}:{args.port}/api/label")
    print(
        "POST label=left/right from a manual test client. "
        "The formal car protocol does not use this endpoint. Press Ctrl+C to stop."
    )

    last_seen: tuple[str, float] | None = None
    try:
        while True:
            now = time.monotonic()
            label = source.get_label(window_start=now, window_end=now)
            if label is not None:
                fingerprint = (label.label_name, label.timestamp_monotonic)
                if fingerprint != last_seen:
                    payload = label.payload or {}
                    print(
                        f"{time.strftime('%H:%M:%S')} label={label.label_name} "
                        f"id={label.label_id} source={label.source} payload={payload}",
                        flush=True,
                    )
                    last_seen = fingerprint
            time.sleep(max(args.poll_sec, 0.02))
    except KeyboardInterrupt:
        print("\nStopping label watcher.")
    finally:
        server.close()


if __name__ == "__main__":
    main()
