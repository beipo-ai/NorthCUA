from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_module_help_does_not_emit_e2b_optional_dependency_warning() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "northcua", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    combined_output = completed.stdout + completed.stderr
    assert "E2B SDK not installed" not in combined_output


def test_controller_and_device_cli_help_are_registered() -> None:
    controller = subprocess.run(
        [sys.executable, "-m", "northcua", "controller", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    device = subprocess.run(
        [sys.executable, "-m", "northcua_device.cli", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert controller.returncode == 0
    assert "serve" in controller.stdout
    assert device.returncode == 0
    assert "connect" in device.stdout


def test_controller_tcp_proxy_help_is_registered() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "northcua.transport.tcp_proxy", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--max-connections" in completed.stdout
    assert "--write-timeout" in completed.stdout


def test_device_list_tools_does_not_emit_e2b_warning_or_default_echo_tool() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "northcua_device.cli", "list-tools", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    combined_output = completed.stdout + completed.stderr
    assert "E2B SDK not installed" not in combined_output
    assert '"device.echo"' not in completed.stdout


def test_device_list_tools_can_opt_into_agentkit_kinds() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "northcua_device.cli",
            "list-tools",
            "--platform",
            "browser",
            "--agentkit-agent-kind",
            "web",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert '"web.web_snapshot"' in completed.stdout
