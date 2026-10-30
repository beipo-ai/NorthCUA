"""Sandbox device binding tests."""

from __future__ import annotations

from northcua.batch.contracts import BatchManifest, BatchRunConfig, BatchTaskSpec
from northcua.batch.runner.bindings import SandboxDeviceBindingFactory
from northcua.runtime.mcp.desktop_native.image_encoding import (
    SCREENSHOT_FORMAT_ENV,
    SCREENSHOT_JPEG_QUALITY_ENV,
    SCREENSHOT_WEBP_QUALITY_ENV,
)
from northcua.sandboxes.contracts import SandboxDeviceBinding, SandboxProviderSpec
from northcua.sandboxes.device.bootstrap import SandboxDeviceBootstrapper
from northcua_protocol.capabilities import Capability, PlatformKind


def _clear_screenshot_env(monkeypatch) -> None:
    for key in (SCREENSHOT_FORMAT_ENV, SCREENSHOT_JPEG_QUALITY_ENV, SCREENSHOT_WEBP_QUALITY_ENV):
        monkeypatch.delenv(key, raising=False)


def test_desktop_binding_uses_dedicated_runtime_not_agentkit_delegate() -> None:
    """Desktop workers avoid loading duplicate AgentKit desktop tool ids."""

    factory = SandboxDeviceBindingFactory()

    assert factory.agentkit_agent_kinds("desktop") == set()


def test_non_desktop_binding_exposes_agentkit_delegate() -> None:
    """Custom and web workers still use AgentKit as the low-code delegate."""

    factory = SandboxDeviceBindingFactory()

    assert factory.agentkit_agent_kinds("web") == {"web"}
    assert factory.agentkit_agent_kinds("android") == {"android"}


def test_task_scoped_device_id_includes_attempt_to_avoid_stale_retry_connections() -> None:
    """Retries get distinct target ids so readiness cannot match old workers."""

    factory = SandboxDeviceBindingFactory()
    task = BatchTaskSpec(task_id="browser-eligibility", agent_kind="desktop", query="Open Chrome")
    spec = SandboxProviderSpec(provider="e2b", template="northcua-e2b-office-web")

    first = factory.device_id_for_task("batch-1", task, spec, attempt=1)
    second = factory.device_id_for_task("batch-1", task, spec, attempt=2)

    assert first.startswith("batch-batch-1-browser-elig-")
    assert first.endswith("-a1")
    assert second.startswith("batch-batch-1-browser-elig-")
    assert second.endswith("-a2")
    assert first != second


def test_task_scoped_device_id_uses_hash_entropy_for_shared_prefixes() -> None:
    """Tasks with a common suite prefix still get distinct target devices."""

    factory = SandboxDeviceBindingFactory()
    spec = SandboxProviderSpec(provider="e2b", template="northcua-e2b-office-web")

    first = factory.device_id_for_task(
        "release",
        BatchTaskSpec(task_id="env-live-five-calc-pivot", agent_kind="desktop", query="Calc"),
        spec,
        attempt=1,
    )
    second = factory.device_id_for_task(
        "release",
        BatchTaskSpec(task_id="env-live-five-contact-sheet", agent_kind="desktop", query="Contacts"),
        spec,
        attempt=1,
    )

    assert first.startswith("batch-release-env-live-fiv-")
    assert second.startswith("batch-release-env-live-fiv-")
    assert first != second


def test_sandbox_worker_command_preserves_platform_and_gui_resource() -> None:
    """Sandbox workers must not fall back to the device CLI's macOS default."""

    bootstrapper = SandboxDeviceBootstrapper(provider=object())  # type: ignore[arg-type]
    command = bootstrapper._worker_command(
        SandboxDeviceBinding(
            device_id="batch-demo-task",
            display_name="demo:task",
            server_url="wss://controller.example/ws",
            token="secret",
            platform=PlatformKind.LINUX,
            capabilities={Capability.BROWSER_PAGE},
            python_executable="python3.12",
            log_dir="/tmp/northcua-device-logs/demo",
        )
    )

    assert "--platform linux" in command
    assert "--allowed-capability browser.page" in command
    assert "--gui-resource-id linux:batch-demo-task:gui" in command
    assert "--log-dir /tmp/northcua-device-logs/demo" in command
    assert "--config /tmp/northcua-device-worker.yaml" in command
    assert "heartbeat_interval_seconds: 30" in command
    assert "telemetry_stream_mode: terminal_sampled" in command
    assert "telemetry_sample_rate: 0.02" in command


def test_sandbox_binding_exports_desktop_coordinate_policy_from_app_config(tmp_path, monkeypatch) -> None:
    _clear_screenshot_env(monkeypatch)
    app_config = tmp_path / "app.yaml"
    app_config.write_text(
        """
agent_config_path: agent.example.yaml
desktop:
  coordinate_policy:
    input_space: normalized_1000
    clamp_to_bounds: true
""",
        encoding="utf-8",
    )
    factory = SandboxDeviceBindingFactory()
    task = BatchTaskSpec(
        task_id="desktop-task",
        agent_kind="desktop",
        query="Click center",
        metadata={"controller_app_config_path": str(app_config)},
    )
    binding = factory.for_task(
        BatchRunConfig(manifest=BatchManifest(tasks=[task])),
        task,
        sandbox_spec=SandboxProviderSpec(provider="e2b", template="northcua-e2b-desktop-full"),
        device_id="batch-demo-task",
        controller_url="wss://controller.example/ws",
        token="secret",
    )
    command = SandboxDeviceBootstrapper(provider=object())._worker_command(binding)  # type: ignore[arg-type]

    assert binding.environment == {
        "NORTHCUA_DESKTOP_COORDINATE_SPACE": "normalized_1000",
        "NORTHCUA_DESKTOP_COORDINATE_CLAMP_TO_BOUNDS": "true",
    }
    assert "export NORTHCUA_DESKTOP_COORDINATE_CLAMP_TO_BOUNDS=true" in command
    assert "export NORTHCUA_DESKTOP_COORDINATE_SPACE=normalized_1000" in command


def test_sandbox_binding_omits_default_desktop_coordinate_policy_from_metadata(tmp_path, monkeypatch) -> None:
    _clear_screenshot_env(monkeypatch)
    app_config = tmp_path / "app.yaml"
    app_config.write_text("agent_config_path: agent.example.yaml\n", encoding="utf-8")
    factory = SandboxDeviceBindingFactory()
    task = BatchTaskSpec(
        task_id="desktop-task",
        agent_kind="desktop",
        query="Click center",
        metadata={"controller_app_config_path": str(app_config)},
    )
    binding = factory.for_task(
        BatchRunConfig(manifest=BatchManifest(tasks=[task])),
        task,
        sandbox_spec=SandboxProviderSpec(provider="e2b", template="northcua-e2b-desktop-full"),
        device_id="batch-demo-task",
        controller_url="wss://controller.example/ws",
        token="secret",
    )
    command = SandboxDeviceBootstrapper(provider=object())._worker_command(binding)  # type: ignore[arg-type]

    assert binding.environment == {}
    assert "northcua.desktop.coordinate_space" not in binding.metadata
    assert "NORTHCUA_DESKTOP_COORDINATE_SPACE" not in command


def test_sandbox_binding_exports_desktop_screenshot_encoding_to_worker(monkeypatch) -> None:
    monkeypatch.setenv(SCREENSHOT_FORMAT_ENV, "jpeg")
    monkeypatch.setenv(SCREENSHOT_JPEG_QUALITY_ENV, "75")
    monkeypatch.setenv(SCREENSHOT_WEBP_QUALITY_ENV, "")
    factory = SandboxDeviceBindingFactory()
    task = BatchTaskSpec(task_id="desktop-task", agent_kind="desktop", query="Click center")

    binding = factory.for_task(
        BatchRunConfig(manifest=BatchManifest(tasks=[task])),
        task,
        sandbox_spec=SandboxProviderSpec(provider="e2b", template="northcua-e2b-desktop-full"),
        device_id="batch-demo-task",
        controller_url="wss://controller.example/ws",
        token="secret",
    )
    command = SandboxDeviceBootstrapper(provider=object())._worker_command(binding)  # type: ignore[arg-type]

    assert binding.environment == {
        SCREENSHOT_FORMAT_ENV: "jpeg",
        SCREENSHOT_JPEG_QUALITY_ENV: "75",
    }
    assert "NORTHCUA_DESKTOP_SCREENSHOT_FORMAT=jpeg" in command
    assert "NORTHCUA_DESKTOP_SCREENSHOT_JPEG_QUALITY=75" in command
    assert SCREENSHOT_WEBP_QUALITY_ENV not in command


def test_shell_capability_allowlists_shell_without_cua_or_browser_tools() -> None:
    """Evaluator shell probes enable device shell without exposing unrelated tool surfaces."""

    allowed = SandboxDeviceBindingFactory.allowed_tools_for_capabilities(
        {
            Capability.A11Y_TREE,
            Capability.DESKTOP_INPUT,
            Capability.DESKTOP_SCREENSHOT,
            Capability.DESKTOP_WINDOW,
            Capability.SHELL_EXEC,
        }
    )

    assert "shell.exec" in allowed
    assert "shell.exec_readonly" in allowed
    assert "screenshot" in allowed
    assert "click" in allowed
    assert "capture_accessibility_tree" in allowed
    assert "cua_page" not in allowed
    assert not any(tool.startswith("cua_") for tool in allowed)
