"""Build local NorthCUA wheels in dependency order.

This script exists for clean-install smoke tests and release candidates. It
does not publish anything; it only creates a local wheelhouse that can be used
with ``pip install --find-links`` or a single explicit install command.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WHEEL_TARGETS = (
    ROOT / "packages" / "northcua-protocol",
    ROOT / "packages" / "northcua-runtime-shared",
    ROOT / "packages" / "northcua-device",
    ROOT,
)


def main(argv: list[str] | None = None) -> int:
    """Build protocol, runtime-shared, device, and root wheels.

    Args:
        argv: Optional command-line arguments for tests.

    Returns:
        Process exit code.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "wheelhouse"),
        help="Directory where wheels are written. Existing wheel files are replaced.",
    )
    parser.add_argument("--keep", action="store_true", help="Keep existing wheelhouse contents before building.")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir).expanduser().resolve()
    if out_dir.exists() and not args.keep:
        for wheel in out_dir.glob("*.whl"):
            wheel.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    # setuptools' build directory is incremental. A release build after files
    # were moved or deleted can otherwise repackage stale modules from a prior
    # build even though package discovery is correct.
    _clean_build_artifacts()
    try:
        for target in WHEEL_TARGETS:
            subprocess.run(_build_command(target, out_dir), check=True, cwd=ROOT)
        print(f"Built NorthCUA wheelhouse: {out_dir}")
        return 0
    finally:
        _clean_build_artifacts()


def _build_command(target: Path, out_dir: Path) -> list[str]:
    """Return a build command that works even when `build` is not preinstalled."""

    if _has_build_module_entrypoint():
        return [sys.executable, "-m", "build", "--wheel", "--outdir", str(out_dir), str(target)]
    return [
        "uvx",
        "--from",
        "build",
        "python",
        "-m",
        "build",
        "--wheel",
        "--outdir",
        str(out_dir),
        str(target),
    ]


def _has_build_module_entrypoint() -> bool:
    """Return whether the active interpreter can run `python -m build`."""

    try:
        return importlib.util.find_spec("build.__main__") is not None
    except ModuleNotFoundError:
        return False


def _clean_build_artifacts(*, root: Path = ROOT) -> None:
    """Remove setuptools side effects from source trees after wheel builds."""

    for path in root.glob("*.egg-info"):
        shutil.rmtree(path, ignore_errors=True)
    for path in (root / "packages").glob("*/build"):
        shutil.rmtree(path, ignore_errors=True)
    for path in (root / "packages").glob("*/*.egg-info"):
        shutil.rmtree(path, ignore_errors=True)
    for path in (root / "packages").glob("*/src/*.egg-info"):
        shutil.rmtree(path, ignore_errors=True)
    shutil.rmtree(root / "build", ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
