"""Run a Unity-only car scene flow test without EEG or simulated signals."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.markers import ArTcpCommandSender  # noqa: E402
from utils.unity_runtime import REQUIRED_RUNTIME_PROTOCOL  # noqa: E402


SCENES = {
    "SCENE_LEFT": ("LEFT", -1),
    "SCENE_RIGHT": ("RIGHT", 1),
}


def _balanced_random_commands(count: int, rng: random.Random) -> list[str]:
    commands = list(SCENES)
    sequence = [commands[index % len(commands)] for index in range(count)]
    rng.shuffle(sequence)
    return sequence


def _validate_ack(ack: dict[str, object], command: str) -> tuple[int, int]:
    if str(ack.get("ack", "")).upper() != command:
        raise RuntimeError(f"unexpected ACK for {command}: {ack}")
    if str(ack.get("protocol_version", "")) != REQUIRED_RUNTIME_PROTOCOL:
        raise RuntimeError(f"unexpected Unity protocol: {ack}")
    start_lane = int(ack["start_lane"])
    safe_lane = int(ack["safe_lane"])
    expected_safe_lane = SCENES[command][1]
    if start_lane != 0 or safe_lane != expected_safe_lane:
        raise RuntimeError(
            f"invalid scene layout for {command}: start={start_lane}, safe={safe_lane}"
        )
    return int(ack["scene_number"]), safe_lane


def run_test(
    *,
    host: str,
    port: int,
    scene_count: int,
    scene_duration_sec: float,
    release_offset_sec: float,
    seed: int,
) -> dict[str, object]:
    rng = random.Random(seed)
    commands = _balanced_random_commands(scene_count, rng)
    sender = ArTcpCommandSender(host, port, timeout_sec=3.0)
    completed: list[dict[str, object]] = []
    try:
        ready = sender.push_with_ack("SCENE_STATE")
        if str(ready.get("protocol_version", "")) != REQUIRED_RUNTIME_PROTOCOL:
            raise RuntimeError(f"Unity is not running the required car protocol: {ready}")

        for index, command in enumerate(commands, start=1):
            movement_command, _ = SCENES[command]
            ack = sender.push_with_ack(command)
            unity_scene_number, safe_lane = _validate_ack(ack, command)
            started_at = time.monotonic()
            last_heartbeat = float("-inf")
            failure_event: dict[str, object] | None = None

            while time.monotonic() - started_at < scene_duration_sec:
                now = time.monotonic()
                elapsed = now - started_at
                gated_command = (
                    "STOP" if elapsed < release_offset_sec else movement_command
                )
                if now - last_heartbeat >= 0.2:
                    sender.push(gated_command)
                    last_heartbeat = now
                for event in sender.poll_events():
                    if (
                        str(event.get("event", "")).upper() == "SCENE_FAILED"
                        and int(event.get("scene_number", -1)) == unity_scene_number
                    ):
                        failure_event = event
                        break
                if failure_event is not None:
                    break
                time.sleep(0.02)

            if failure_event is not None:
                raise RuntimeError(
                    f"scene {index}/{scene_count} collided: {json.dumps(failure_event)}"
                )

            state = sender.push_with_ack("SCENE_STATE")
            current_lane = int(state["current_lane"])
            if current_lane != safe_lane:
                raise RuntimeError(
                    f"scene {index}/{scene_count} ended in lane {current_lane}, "
                    f"expected safe lane {safe_lane}"
                )
            result = {
                "test_scene": index,
                "unity_scene": unity_scene_number,
                "command": command,
                "safe_lane": safe_lane,
                "current_lane": current_lane,
                "release_offset_sec": release_offset_sec,
                "outcome": "success",
            }
            completed.append(result)
            print(json.dumps(result, ensure_ascii=True), flush=True)

        return {
            "seed": seed,
            "requested_scenes": scene_count,
            "completed_scenes": len(completed),
            "successful_scenes": len(completed),
            "failed_scenes": 0,
            "all_successful": len(completed) == scene_count,
            "signal_source": None,
            "release_offset_sec": release_offset_sec,
        }
    finally:
        sender.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--scenes", type=int, default=10)
    parser.add_argument("--scene-duration-sec", type=float, default=5.0)
    parser.add_argument("--release-offset-sec", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=int(time.time_ns() & 0xFFFFFFFF))
    args = parser.parse_args()
    if args.scenes <= 0 or args.scene_duration_sec <= 0:
        parser.error("--scenes and --scene-duration-sec must be positive")
    if not 0 <= args.release_offset_sec < args.scene_duration_sec:
        parser.error("--release-offset-sec must be within the Scene duration")

    summary = run_test(
        host=args.host,
        port=args.port,
        scene_count=args.scenes,
        scene_duration_sec=args.scene_duration_sec,
        release_offset_sec=args.release_offset_sec,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
