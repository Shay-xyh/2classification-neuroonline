"""Create a local Python environment for oi-mi.

The script may be launched with any reasonably modern Python on Windows. It
locates Python 3.12, installs it with winget when possible, then creates the
project .venv from that interpreter. The default installation contains only
the Neuracle/JellyFish collection runtime; model and BrainCo dependencies are
explicit opt-ins.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"
PIP_CACHE_DIR = PROJECT_ROOT.parent / ".cache" / "pip"
REQUIRED_PYTHON = (3, 12)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    python312 = _find_or_install_python_312(allow_install=not args.no_install_python)
    _ensure_venv(python312, recreate=args.recreate_venv)
    venv_python = _venv_python()

    # Torch currently requires setuptools<82. Keep the bootstrap compatible
    # with both the collection-only default and the optional decoding extra.
    _run(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "-U",
            "pip",
            "setuptools<82",
            "wheel",
        ]
    )
    extras = []
    if args.with_brainco or args.with_all:
        extras.append("brainco")
    if args.with_decoding or args.with_all:
        extras.append("decoding")
    install_target = "." if not extras else ".[{}]".format(",".join(extras))
    _run([str(venv_python), "-m", "pip", "install", "-e", install_target])

    check_command = [
        str(venv_python),
        str(PROJECT_ROOT / "tools" / "check_environment.py"),
    ]
    if "brainco" in extras:
        check_command.append("--with-brainco")
    if "decoding" in extras:
        check_command.append("--with-decoding")
        _run(
            [
                str(venv_python),
                str(PROJECT_ROOT / "tools" / "download_cbramod_weights.py"),
            ]
        )
    _run(check_command)
    _print_next_steps()
    return 0


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-install-python",
        action="store_true",
        help="Only detect Python 3.12; do not try to install it automatically.",
    )
    parser.add_argument(
        "--recreate-venv",
        action="store_true",
        help="Delete and recreate .venv even when it already exists.",
    )
    parser.add_argument(
        "--with-decoding",
        action="store_true",
        help="Also install post-collection model, training, and realtime-decoding dependencies.",
    )
    parser.add_argument(
        "--with-brainco",
        action="store_true",
        help="Also install the optional BrainCo device backend dependencies.",
    )
    parser.add_argument(
        "--with-all",
        action="store_true",
        help="Install both decoding and BrainCo optional dependencies.",
    )
    return parser.parse_args(argv)


def _find_or_install_python_312(*, allow_install: bool) -> List[str]:
    command = _find_python_312()
    if command is not None:
        print(f"Using Python 3.12: {_describe_command(command)}")
        return command

    if not allow_install:
        raise SystemExit(_missing_python_message())

    if os.name == "nt":
        _install_python_312_windows()
        command = _find_python_312()
        if command is not None:
            print(f"Using Python 3.12: {_describe_command(command)}")
            return command
        raise SystemExit(
            "Python 3.12 was installed, but this terminal cannot find it yet. "
            "Close Git Bash, open it again, then rerun: python setup_local.py"
        )

    raise SystemExit(_missing_python_message())


def _find_python_312() -> Optional[List[str]]:
    candidates = [
        [sys.executable],
        ["py", "-3.12"],
        ["python3.12"],
        ["python"],
        ["python3"],
    ]
    seen = set()
    for command in candidates:
        key = tuple(command)
        if key in seen:
            continue
        seen.add(key)
        version = _python_version(command)
        if version is not None and version[:2] == REQUIRED_PYTHON:
            return command
    return None


def _python_version(command: Sequence[str]) -> Optional[Tuple[int, int, int]]:
    try:
        result = subprocess.run(
            list(command)
            + [
                "-c",
                "import sys; print('%d.%d.%d' % sys.version_info[:3])",
            ],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    raw = result.stdout.strip().splitlines()
    if not raw:
        return None
    try:
        major, minor, micro = raw[-1].split(".", 2)
        return int(major), int(minor), int(micro)
    except (TypeError, ValueError):
        return None


def _install_python_312_windows() -> None:
    winget = shutil.which("winget")
    if winget is None:
        raise SystemExit(
            "Python 3.12 was not found, and winget is not available to install it. "
            "Install Python 3.12 from https://www.python.org/downloads/release/python-312/ "
            "or install App Installer from Microsoft Store, then rerun setup_local.py."
        )

    print("Python 3.12 not found. Installing Python 3.12 with winget...")
    command = [
        winget,
        "install",
        "--id",
        "Python.Python.3.12",
        "--exact",
        "--source",
        "winget",
        "--scope",
        "user",
        "--accept-package-agreements",
        "--accept-source-agreements",
    ]
    try:
        _run(command)
    except subprocess.CalledProcessError:
        print("winget user-scope install failed; retrying without --scope user...")
        fallback = [part for part in command if part not in {"--scope", "user"}]
        _run(fallback)


def _ensure_venv(python312: Sequence[str], *, recreate: bool) -> None:
    existing_python = _venv_python()
    if existing_python.exists():
        existing_version = _python_version([str(existing_python)])
        if existing_version is not None and existing_version[:2] == REQUIRED_PYTHON and not recreate:
            print(f"Using existing venv: {VENV_DIR}")
            return

        print(f"Removing existing venv with incompatible Python: {VENV_DIR}")
        _remove_project_directory(VENV_DIR)

    print(f"Creating venv with Python 3.12: {VENV_DIR}")
    _run(list(python312) + ["-m", "venv", str(VENV_DIR)])


def _venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _run(command: Sequence[str]) -> None:
    printable = " ".join(str(part) for part in command)
    print(f"+ {printable}")
    PIP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PIP_CACHE_DIR"] = str(PIP_CACHE_DIR)
    subprocess.check_call(
        [str(part) for part in command],
        cwd=str(PROJECT_ROOT),
        env=environment,
    )


def _remove_project_directory(path: Path) -> None:
    resolved = path.resolve()
    project_root = PROJECT_ROOT.resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise SystemExit(f"Refusing to remove non-project directory: {resolved}") from exc
    if resolved == project_root:
        raise SystemExit(f"Refusing to remove project root: {resolved}")
    shutil.rmtree(str(resolved))


def _missing_python_message() -> str:
    if os.name == "nt":
        return (
            "Python 3.12 was not found. Install it with:\n"
            "  winget install --id Python.Python.3.12 --exact --source winget "
            "--scope user --accept-package-agreements --accept-source-agreements\n"
            "Then rerun:\n"
            "  python setup_local.py"
        )
    return (
        "Python 3.12 was not found. Install Python 3.12 with your system package "
        "manager, then rerun:\n"
        "  python3.12 setup_local.py"
    )


def _describe_command(command: Sequence[str]) -> str:
    version = _python_version(command)
    version_text = "unknown" if version is None else ".".join(str(part) for part in version)
    return f"{' '.join(command)} ({version_text})"


def _print_next_steps() -> None:
    if os.name == "nt":
        gui_command = r".venv\Scripts\python.exe cli.py gui"
    else:
        gui_command = ".venv/bin/python cli.py gui"

    print("")
    print("Setup complete.")
    print("Start the GUI with the virtual environment interpreter:")
    print(f"  {gui_command}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
