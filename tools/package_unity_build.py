"""Validate and package the oi-mi Unity runtime as a GitHub Release asset."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.unity_runtime import (
    DEFAULT_BUILD_DIR,
    RUNTIME_MANIFEST_FILENAME,
    resolve_project_path,
    validate_unity_runtime,
    write_unity_runtime_manifest,
)

DEFAULT_OUTPUT = PROJECT_ROOT / "ARPrototype3D-windows-x64.zip"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=DEFAULT_BUILD_DIR,
        help="Unity Windows build directory inside oi-mi.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination zip path.",
    )
    parser.add_argument(
        "--build-id",
        default=None,
        help="Release build identifier. Defaults to the current UTC timestamp.",
    )
    args = parser.parse_args(argv)

    build_dir = resolve_project_path(args.build_dir)
    executable = build_dir / "ARPrototype3D.exe"
    build_id = str(args.build_id or _default_build_id()).strip()
    write_unity_runtime_manifest(executable, build_id=build_id)
    manifest = validate_unity_runtime(executable)

    output = args.output.expanduser()
    if not output.is_absolute():
        output = (PROJECT_ROOT / output).resolve()
    else:
        output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=str(output.parent),
    )
    os.close(descriptor)
    temporary_output = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary_output,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for path in sorted(build_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(build_dir).as_posix())
        os.replace(temporary_output, output)
    finally:
        temporary_output.unlink(missing_ok=True)

    print(
        f"Unity release asset created: {output}\n"
        f"build_id: {manifest['build_id']}\n"
        f"manifest: {RUNTIME_MANIFEST_FILENAME}"
    )
    return 0


def _default_build_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":
    raise SystemExit(main())
