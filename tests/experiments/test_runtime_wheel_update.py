"""Installing a candidate must replace even the same baked-in version."""

from __future__ import annotations
import os
from pathlib import Path
import subprocess
import sys
import zipfile
import pytest
from northcua.experiments.materialization import runtime_bundle


def wheel(directory, name, marker):
    normalized = name.replace("-", "_")
    path = directory / f"{normalized}-0.2.0-py3-none-any.whl"
    info = f"{normalized}-0.2.0.dist-info"
    files = {
        f"{normalized}/__init__.py": f"MARKER = {marker!r}\n",
        f"{info}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: 0.2.0\n",
        f"{info}/WHEEL": "Wheel-Version: 1.0\nGenerator: regression\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    if name == "northcua-device":
        files[f"{normalized}/cli.py"] = ""
    files[f"{info}/RECORD"] = "".join(f"{key},,\n" for key in files) + f"{info}/RECORD,,\n"
    with zipfile.ZipFile(path, "w") as archive:
        for key, value in files.items():
            archive.writestr(key, value)
    return path


@pytest.mark.skipif(sys.platform == "win32", reason="Sandbox bootstrap is Linux/bash")
def test_same_version_candidate_replaces_installed_code(tmp_path, monkeypatch):
    directory = tmp_path / "wheels"
    directory.mkdir()
    names = ("northcua", "northcua-device", "northcua-protocol", "northcua-runtime-shared")
    for name in names:
        wheel(directory, name, "old")
    environment = tmp_path / "environment"
    subprocess.run(
        ["uv", "venv", "--seed", "--python", sys.executable, str(environment)],
        check=True,
        capture_output=True,
        timeout=60,
    )
    python = str(environment / "bin" / "python")
    env = dict(os.environ, PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
    env.pop("PYTHONPATH", None)
    subprocess.run(
        [python, "-m", "pip", "install", *map(str, directory.glob("*.whl"))],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
    )
    for name in names:
        wheel(directory, name, "candidate")
    monkeypatch.setattr(runtime_bundle, "_REMOTE_WHEEL_DIR", str(directory))
    command = runtime_bundle.RuntimeWheelBundleBuilder()._install_command(python_executable=python)
    subprocess.run(
        command.command, shell=True, cwd=tmp_path, env=env, check=True, capture_output=True, text=True, timeout=30
    )
    actual = subprocess.check_output(
        [python, "-c", "import northcua; print(northcua.MARKER)"], cwd=tmp_path, env=env, text=True
    )
    assert actual.strip() == "candidate"
