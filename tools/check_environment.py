"""Check the local Python environment for oi-mi.

This script intentionally uses only the Python standard library so it can run
before project dependencies are installed.
"""

import argparse
import importlib.util
import platform
import subprocess
import sys
from pathlib import Path

REQUIRED_MAJOR = 3
REQUIRED_MINOR = 12

COLLECTION_MODULES = (
    ("click", "click"),
    ("numpy", "numpy"),
    ("pyedflib", "pyedflib"),
    ("pyyaml", "yaml"),
    ("rich", "rich"),
    ("scipy", "scipy"),
    ("streamlit", "streamlit"),
)

DECODING_MODULES = (
    ("pylsl", "pylsl"),
    ("scikit-learn", "sklearn"),
    ("torch", "torch"),
)

BRAINCO_MODULES = (
    ("bc-ecap-sdk", "bc_ecap_sdk"),
    ("zeroconf", "zeroconf"),
)


def is_virtual_environment():
    return (
        getattr(sys, "base_prefix", sys.prefix) != sys.prefix
        or hasattr(sys, "real_prefix")
    )


def pip_version():
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
    except Exception as exc:
        return "unavailable: %s" % exc
    return completed.stdout.strip()


def missing_runtime_modules(modules):
    missing = []
    for package_name, module_name in modules:
        if importlib.util.find_spec(module_name) is None:
            missing.append(package_name)
    return missing


def suggested_commands(*, with_decoding=False, with_brainco=False):
    extras = []
    if with_brainco:
        extras.append("brainco")
    if with_decoding:
        extras.append("decoding")
    install_target = "." if not extras else ".[{}]".format(",".join(extras))
    check_flags = []
    if with_brainco:
        check_flags.append("--with-brainco")
    if with_decoding:
        check_flags.append("--with-decoding")
    check_command = "python tools/check_environment.py"
    if check_flags:
        check_command += " " + " ".join(check_flags)
    system = platform.system().lower()
    if system == "windows":
        return (
            "py -3.12 -m venv .venv",
            r".venv\Scripts\python.exe -m pip install -U pip \"setuptools<82\" wheel",
            rf".venv\Scripts\python.exe -m pip install -e \"{install_target}\"",
            check_command.replace("python", r".venv\Scripts\python.exe", 1),
            r".venv\Scripts\python.exe cli.py gui",
        )
    return (
        "python3.12 -m venv .venv",
        "source .venv/bin/activate",
        'python -m pip install -U pip "setuptools<82" wheel',
        f'pip install -e "{install_target}"',
        check_command,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-decoding",
        action="store_true",
        help="Also require post-collection model and realtime-decoding modules.",
    )
    parser.add_argument(
        "--with-brainco",
        action="store_true",
        help="Also require the optional BrainCo device modules.",
    )
    parser.add_argument(
        "--with-unity",
        action="store_true",
        help="Also validate the separate Unity runtime used by downstream car experiments.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    errors = []
    warnings = []

    print("oi-mi environment check")
    print("platform: %s" % platform.platform())
    print("python: %s" % sys.version.replace("\n", " "))
    print("executable: %s" % sys.executable)
    print("virtualenv: %s" % ("yes" if is_virtual_environment() else "no"))
    print("pip: %s" % pip_version())

    version = sys.version_info
    if (version.major, version.minor) != (REQUIRED_MAJOR, REQUIRED_MINOR):
        errors.append(
            "Python %s.%s.x is required; current interpreter is %s.%s.%s."
            % (
                REQUIRED_MAJOR,
                REQUIRED_MINOR,
                version.major,
                version.minor,
                version.micro,
            )
        )

    if not is_virtual_environment():
        warnings.append("No virtual environment is active.")

    requested_modules = list(COLLECTION_MODULES)
    if args.with_decoding:
        requested_modules.extend(DECODING_MODULES)
    if args.with_brainco:
        requested_modules.extend(BRAINCO_MODULES)
    missing = missing_runtime_modules(requested_modules)
    if missing:
        errors.append(
            "Missing requested runtime dependencies: %s. Install the matching project extras."
            % ", ".join(missing)
        )

    if args.with_unity:
        project_root = Path(__file__).resolve().parents[1]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from utils.unity_runtime import (
            DEFAULT_EXECUTABLE,
            resolve_project_path,
            validate_unity_runtime,
        )

        unity_executable = resolve_project_path(DEFAULT_EXECUTABLE)
        try:
            unity_manifest = validate_unity_runtime(unity_executable)
        except RuntimeError as exc:
            errors.append(str(exc))
        else:
            print(
                "unity runtime: %s (%s)"
                % (
                    unity_manifest.get("build_id", "unknown"),
                    unity_manifest.get("protocol_version", "unknown"),
                )
            )

    if warnings:
        print("")
        print("Warnings:")
        for warning in warnings:
            print("- %s" % warning)

    if errors:
        print("")
        print("Problems:")
        for error in errors:
            print("- %s" % error)
        print("")
        print("Suggested setup:")
        for command in suggested_commands(
            with_decoding=args.with_decoding,
            with_brainco=args.with_brainco,
        ):
            print("  %s" % command)
        return 1

    print("")
    print("Environment looks ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
