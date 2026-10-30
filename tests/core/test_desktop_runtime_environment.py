from __future__ import annotations

from northcua.core.config import NorthCUAAppConfig
from northcua.core.desktop_runtime_environment import (
    DESKTOP_BACKEND_ENV,
    OSWORLD_COMMAND_TIMEOUT_ENV,
    OSWORLD_ENABLE_A11Y_ENV,
    OSWORLD_RETRY_INTERVAL_ENV,
    OSWORLD_RETRY_TIMES_ENV,
    OSWORLD_SCREENSHOT_TIMEOUT_ENV,
    OSWORLD_SERVER_PORT_ENV,
    OSWORLD_VM_IP_ENV,
    apply_app_desktop_runtime_environment,
    desktop_backend_from_environment,
    osworld_accessibility_enabled,
    restore_desktop_runtime_environment,
)


def test_desktop_runtime_environment_exports_and_restores_osworld_values() -> None:
    app_config = NorthCUAAppConfig(
        agent_config_path="agent.yaml",
        desktop={
            "backend": "osworld",
            "osworld": {
                "vm_ip": "10.0.0.5",
                "server_port": 6000,
                "command_timeout_seconds": 45,
                "screenshot_timeout_seconds": 6,
                "retry_times": 2,
                "retry_interval_seconds": 0.25,
                "enable_accessibility": False,
            },
        },
    )
    environment = {DESKTOP_BACKEND_ENV: "native", OSWORLD_VM_IP_ENV: "old-host"}

    snapshot = apply_app_desktop_runtime_environment(app_config, environment=environment)

    assert desktop_backend_from_environment(environment) == "osworld"
    assert environment[OSWORLD_VM_IP_ENV] == "10.0.0.5"
    assert environment[OSWORLD_SERVER_PORT_ENV] == "6000"
    assert environment[OSWORLD_COMMAND_TIMEOUT_ENV] == "45.0"
    assert environment[OSWORLD_SCREENSHOT_TIMEOUT_ENV] == "6.0"
    assert environment[OSWORLD_RETRY_TIMES_ENV] == "2"
    assert environment[OSWORLD_RETRY_INTERVAL_ENV] == "0.25"
    assert environment[OSWORLD_ENABLE_A11Y_ENV] == "false"
    assert not osworld_accessibility_enabled(environment)

    restore_desktop_runtime_environment(snapshot, environment=environment)

    assert environment == {DESKTOP_BACKEND_ENV: "native", OSWORLD_VM_IP_ENV: "old-host"}
