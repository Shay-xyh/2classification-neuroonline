"""Download or install the Unity Windows build used by oi-mi."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.unity_runtime import validate_unity_runtime

DEFAULT_URL = "https://github.com/Omni-Intel/oi-mi/releases/latest/download/ARPrototype3D-windows-x64.zip"
DEFAULT_BUILD_NAME = "ARPrototype3D-windows-x64"
DEFAULT_DEST = PROJECT_ROOT / "unity相关" / DEFAULT_BUILD_NAME
DEFAULT_CACHE = PROJECT_ROOT / ".runtime" / "downloads" / f"{DEFAULT_BUILD_NAME}.zip"
EXPECTED_EXE = "ARPrototype3D.exe"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=os.environ.get("OI_MI_UNITY_BUILD_URL", DEFAULT_URL),
        help="Unity build zip URL. Defaults to the latest GitHub Release asset.",
    )
    parser.add_argument(
        "--from-local-zip",
        type=Path,
        default=os.environ.get("OI_MI_UNITY_BUILD_ZIP"),
        help="Install from an existing local zip instead of downloading.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help="Destination directory under the project's unity相关 runtime root.",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing extracted build.")
    args = parser.parse_args(argv)

    dest = _resolve_runtime_directory(args.dest)
    expected_exe = dest / EXPECTED_EXE
    if expected_exe.exists() and not args.force:
        try:
            manifest = validate_unity_runtime(expected_exe)
        except RuntimeError as exc:
            raise SystemExit(
                f"Installed Unity runtime is incompatible: {exc}\n"
                "Rerun with --force to replace it."
            ) from exc
        print(
            "Unity build already installed and verified: "
            f"{expected_exe} ({manifest.get('build_id', 'unknown')})"
        )
        return 0

    zip_path = _prepare_zip(args)
    _extract_zip(zip_path, dest)
    exe_path = dest / EXPECTED_EXE
    manifest = validate_unity_runtime(exe_path)
    print(
        f"Unity build installed and verified: {exe_path} "
        f"({manifest.get('build_id', 'unknown')})"
    )
    return 0


def _prepare_zip(args: argparse.Namespace) -> Path:
    DEFAULT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    local_zip = Path(args.from_local_zip).expanduser() if args.from_local_zip else None
    if local_zip is not None:
        local_zip = local_zip.resolve()
        if not local_zip.exists():
            raise SystemExit(f"Local Unity build zip does not exist: {local_zip}")
        if local_zip != DEFAULT_CACHE.resolve():
            shutil.copy2(local_zip, DEFAULT_CACHE)
        print(f"Using local Unity build zip: {DEFAULT_CACHE}")
        return DEFAULT_CACHE

    if DEFAULT_CACHE.exists() and not args.force:
        print(f"Using cached Unity build zip: {DEFAULT_CACHE}")
        return DEFAULT_CACHE

    url = str(args.url).strip()
    if not url:
        raise SystemExit("Unity build URL is empty.")

    print(f"Downloading Unity build: {url}")
    try:
        urllib.request.urlretrieve(url, DEFAULT_CACHE)
    except (urllib.error.URLError, OSError) as exc:
        raise SystemExit(
            "Failed to download Unity build. Publish the zip as a GitHub Release asset "
            f"named {DEFAULT_CACHE.name}, or pass --from-local-zip. Details: {exc}"
        ) from exc
    return DEFAULT_CACHE


def _extract_zip(zip_path: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{dest.name}-install-", dir=str(dest.parent))
    )
    extracted_root = staging_root / "payload"
    extracted_root.mkdir()
    try:
        with zipfile.ZipFile(zip_path) as archive:
            _validate_archive_paths(archive, extracted_root)
            archive.extractall(extracted_root)
    except zipfile.BadZipFile as exc:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise SystemExit(f"Invalid Unity build zip: {zip_path}") from exc
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise

    try:
        executable = _find_executable(extracted_root)
        build_root = executable.parent
        validate_unity_runtime(executable)
        _replace_directory(build_root, dest)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def _find_executable(dest: Path) -> Path:
    expected = dest / EXPECTED_EXE
    if expected.exists():
        return expected

    candidates = sorted(
        path for path in dest.rglob("*.exe")
        if path.name.lower() != "unitycrashhandler64.exe"
    )
    exact_candidates = [path for path in candidates if path.name == EXPECTED_EXE]
    if len(exact_candidates) == 1:
        return exact_candidates[0]
    if len(candidates) == 1:
        return candidates[0]

    raise SystemExit(
        f"No Unity executable found in {dest}. Expected {EXPECTED_EXE}; "
        "make sure the zip contains the full Windows build folder."
    )


def _resolve_under_project(path: Path) -> Path:
    resolved = (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise SystemExit(f"Destination must be inside the project: {resolved}") from exc
    return resolved


def _remove_directory(path: Path) -> None:
    resolved = _resolve_runtime_directory(path)
    shutil.rmtree(resolved)


def _validate_archive_paths(archive: zipfile.ZipFile, extraction_root: Path) -> None:
    root = extraction_root.resolve()
    for member in archive.infolist():
        member_path = Path(member.filename.replace("\\", "/"))
        if member_path.is_absolute() or ".." in member_path.parts:
            raise SystemExit(f"Unsafe path in Unity build zip: {member.filename}")
        target = (root / member_path).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise SystemExit(
                f"Unsafe path in Unity build zip: {member.filename}"
            ) from exc


def _replace_directory(source: Path, dest: Path) -> None:
    """Replace an installed runtime only after the candidate validates."""

    source = source.resolve()
    dest = _resolve_runtime_directory(dest)
    backup = dest.with_name(f".{dest.name}.previous")
    if backup.exists():
        _remove_directory(backup)

    had_existing = dest.exists()
    if had_existing:
        os.replace(dest, backup)
    try:
        os.replace(source, dest)
    except Exception:
        if had_existing and backup.exists() and not dest.exists():
            os.replace(backup, dest)
        raise
    if backup.exists():
        _remove_directory(backup)


def _resolve_runtime_directory(path: Path) -> Path:
    resolved = _resolve_under_project(path)
    runtime_root = (PROJECT_ROOT / "unity相关").resolve()
    try:
        resolved.relative_to(runtime_root)
    except ValueError as exc:
        raise SystemExit(
            f"Unity destination must be inside {runtime_root}: {resolved}"
        ) from exc
    if resolved == runtime_root:
        raise SystemExit(f"Unity destination cannot be the runtime root: {resolved}")
    return resolved


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
