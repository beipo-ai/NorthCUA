"""Command-bus failure, validation, and artifact materialization tests."""

from __future__ import annotations

import asyncio
import json

from tests.controller.test_command_bus import (
    ArtifactKind,
    ArtifactRef,
    ArtifactScope,
    Capability,
    CommandBus,
    CommandRequest,
    CommandResult,
    CommandState,
    ControllerAuth,
    ControllerEventLog,
    ControllerSecurityConfig,
    DeviceDescriptor,
    DeviceRegistry,
    LocalControllerBlobStore,
    PlatformKind,
    RetentionPolicy,
    StorageKind,
    ToolDescriptor,
    _activate_session,
    _store,
)


def test_command_bus_sender_failure_persists_failed_result(tmp_path) -> None:
    async def _run() -> None:
        store = _store(tmp_path)
        events = ControllerEventLog(store)
        bus = CommandBus(
            store=store,
            auth=ControllerAuth(ControllerSecurityConfig(insecure_dev_no_auth=True)),
            events=events,
        )
        _activate_session(store)

        async def sender(command: CommandRequest, trace_id: str) -> CommandResult:
            del command, trace_id
            raise RuntimeError("device websocket closed")

        bus.register_sender("dev-1", sender)
        result = await bus.send(
            CommandRequest(
                command_id="cmd-fail",
                idempotency_key="idem-fail",
                session_id="sess-1",
                lease_id="lease-1",
                device_id="dev-1",
                tool_id="device.echo",
            ),
            trace_id="trace-1",
        )

        assert result.status == CommandState.FAILED
        assert (
            CommandResult.model_validate_json(store.command_row("cmd-fail")["result_json"]).status
            == CommandState.FAILED
        )

    asyncio.run(_run())


def test_command_bus_does_not_duplicate_large_inline_artifacts_to_controller_blob(tmp_path) -> None:
    async def _run() -> None:
        store = _store(tmp_path)
        events = ControllerEventLog(store)
        bus = CommandBus(
            store=store,
            auth=ControllerAuth(ControllerSecurityConfig(insecure_dev_no_auth=True)),
            events=events,
            blob_store=LocalControllerBlobStore(tmp_path / "blobs"),
            max_inline_artifact_bytes=3,
        )
        blob_root = tmp_path / "blobs"
        _activate_session(store)
        artifact = ArtifactRef.from_inline_bytes(
            artifact_id="artifact-1",
            kind=ArtifactKind.SCREENSHOT,
            data=b"larger-than-inline",
            mime_type="image/png",
            artifact_scope=ArtifactScope.SESSION,
            retention_policy=RetentionPolicy.LONG_TERM,
        )

        async def sender(command: CommandRequest, trace_id: str) -> CommandResult:
            del trace_id
            return CommandResult(
                command_id=command.command_id,
                session_id=command.session_id,
                lease_id=command.lease_id,
                device_id=command.device_id,
                status=CommandState.SUCCEEDED,
                artifacts=[artifact],
            )

        bus.register_sender("dev-1", sender)
        result = await bus.send(
            CommandRequest(
                command_id="cmd-blob",
                idempotency_key="idem-blob",
                session_id="sess-1",
                lease_id="lease-1",
                device_id="dev-1",
                tool_id="device.echo",
            ),
            trace_id="trace-1",
        )

        assert result.status == CommandState.SUCCEEDED
        assert result.artifacts[0].storage_kind == StorageKind.INLINE
        assert result.artifacts[0].inline_data == artifact.inline_data
        assert not any(blob_root.iterdir())

        persisted = CommandResult.model_validate_json(store.command_row("cmd-blob")["result_json"])
        assert persisted.artifacts[0].storage_kind == StorageKind.DEVICE_LOCAL
        assert persisted.artifacts[0].inline_data is None
        assert persisted.artifacts[0].uri.startswith("agent-artifacts://sha256/")
        assert persisted.artifacts[0].artifact_scope == ArtifactScope.SESSION
        assert persisted.artifacts[0].retention_policy == RetentionPolicy.LONG_TERM

        artifact_rows = store.artifact_rows(trace_id="trace-1")
        assert len(artifact_rows) == 1
        assert artifact_rows[0]["storage_kind"] == StorageKind.DEVICE_LOCAL.value
        artifact_json = json.loads(artifact_rows[0]["artifact_json"])
        assert "inline_data" not in artifact_json
        assert artifact_json["uri"].startswith("agent-artifacts://sha256/")

    asyncio.run(_run())


def test_command_bus_validates_tool_schema_before_dispatch(tmp_path) -> None:
    async def _run() -> None:
        store = _store(tmp_path)
        events = ControllerEventLog(store)
        registry = DeviceRegistry(store=store, events=events)
        registry.register(
            DeviceDescriptor(
                device_id="dev-1",
                platform=PlatformKind.MACOS,
                capabilities={Capability.DESKTOP_INPUT},
                tools=[
                    ToolDescriptor(
                        tool_id="desktop.click",
                        capability=Capability.DESKTOP_INPUT,
                        schema={"type": "object", "required": ["x", "y"]},
                    )
                ],
            ),
            trace_id="trace-1",
        )
        bus = CommandBus(
            store=store,
            auth=ControllerAuth(ControllerSecurityConfig(insecure_dev_no_auth=True)),
            events=events,
            registry=registry,
        )
        _activate_session(store)

        async def sender(command: CommandRequest, trace_id: str) -> CommandResult:
            raise AssertionError("sender should not receive invalid command")

        bus.register_sender("dev-1", sender)

        try:
            await bus.send(
                CommandRequest(
                    command_id="cmd-invalid",
                    idempotency_key="idem-invalid",
                    session_id="sess-1",
                    lease_id="lease-1",
                    device_id="dev-1",
                    tool_id="desktop.click",
                    parameters={"x": 1},
                ),
                trace_id="trace-1",
            )
        except ValueError as exc:
            assert "missing required" in str(exc)
        else:
            raise AssertionError("schema validation should reject missing y")

    asyncio.run(_run())


def test_command_bus_validates_tool_max_payload_before_dispatch(tmp_path) -> None:
    async def _run() -> None:
        store = _store(tmp_path)
        events = ControllerEventLog(store)
        registry = DeviceRegistry(store=store, events=events)
        registry.register(
            DeviceDescriptor(
                device_id="dev-1",
                platform=PlatformKind.MACOS,
                capabilities={Capability.DESKTOP_INPUT},
                tools=[
                    ToolDescriptor(
                        tool_id="desktop.type_text",
                        capability=Capability.DESKTOP_INPUT,
                        max_payload_bytes=5,
                    )
                ],
            ),
            trace_id="trace-1",
        )
        bus = CommandBus(
            store=store,
            auth=ControllerAuth(ControllerSecurityConfig(insecure_dev_no_auth=True)),
            events=events,
            registry=registry,
        )
        _activate_session(store)

        async def sender(command: CommandRequest, trace_id: str) -> CommandResult:
            raise AssertionError("sender should not receive oversized command")

        bus.register_sender("dev-1", sender)
        try:
            await bus.send(
                CommandRequest(
                    command_id="cmd-too-large",
                    idempotency_key="idem-too-large",
                    session_id="sess-1",
                    lease_id="lease-1",
                    device_id="dev-1",
                    tool_id="desktop.type_text",
                    parameters={"text": "this is too long"},
                ),
                trace_id="trace-1",
            )
        except ValueError as exc:
            assert "tool max_payload_bytes" in str(exc)
        else:
            raise AssertionError("tool max_payload_bytes should reject command")

    asyncio.run(_run())


def test_command_bus_rewrites_payload_artifact_ids_to_controller_catalog_ids(tmp_path) -> None:
    async def _run() -> None:
        store = _store(tmp_path)
        events = ControllerEventLog(store)
        bus = CommandBus(
            store=store,
            auth=ControllerAuth(ControllerSecurityConfig(insecure_dev_no_auth=True)),
            events=events,
        )
        _activate_session(store)
        artifact = ArtifactRef.from_inline_bytes(
            artifact_id="shot-1",
            kind=ArtifactKind.SCREENSHOT,
            data=b"png",
            mime_type="image/png",
        )

        async def sender(command: CommandRequest, trace_id: str) -> CommandResult:
            del trace_id
            return CommandResult(
                command_id=command.command_id,
                session_id=command.session_id,
                lease_id=command.lease_id,
                device_id=command.device_id,
                status=CommandState.SUCCEEDED,
                result={"screenshot_artifact_id": "shot-1"},
                artifacts=[artifact],
            )

        bus.register_sender("dev-1", sender)
        result = await bus.send(
            CommandRequest(
                command_id="cmd-artifact-ref",
                idempotency_key="idem-artifact-ref",
                session_id="sess-1",
                lease_id="lease-1",
                device_id="dev-1",
                tool_id="device.echo",
            ),
            trace_id="trace-1",
        )

        assert result.result["screenshot_artifact_id"] != "shot-1"
        assert result.result["screenshot_artifact_id_original"] == "shot-1"
        persisted = CommandResult.model_validate_json(store.command_row("cmd-artifact-ref")["result_json"])
        assert persisted.result["screenshot_artifact_id"] != "shot-1"
        assert persisted.result["screenshot_artifact_id_original"] == "shot-1"
        artifact_row = store.artifact_rows(trace_id="trace-1")[0]
        assert artifact_row["artifact_id"] == persisted.result["screenshot_artifact_id"]
        assert artifact_row["original_artifact_id"] == "shot-1"
        assert artifact_row["source"] == "command_output"

    asyncio.run(_run())


def test_command_bus_rejects_unknown_tool_ids_before_dispatch(tmp_path) -> None:
    async def _run() -> None:
        store = _store(tmp_path)
        events = ControllerEventLog(store)
        registry = DeviceRegistry(store=store, events=events)
        registry.register(
            DeviceDescriptor(
                device_id="dev-1",
                platform=PlatformKind.MACOS,
                capabilities={Capability.DESKTOP_INPUT},
                tools=[ToolDescriptor(tool_id="device.echo", capability=Capability.DESKTOP_INPUT)],
            ),
            trace_id="trace-1",
        )
        bus = CommandBus(
            store=store,
            auth=ControllerAuth(ControllerSecurityConfig(insecure_dev_no_auth=True)),
            events=events,
            registry=registry,
        )
        _activate_session(store)

        async def sender(command: CommandRequest, trace_id: str) -> CommandResult:
            raise AssertionError("sender should not be called for unknown tool")

        bus.register_sender("dev-1", sender)
        try:
            await bus.send(
                CommandRequest(
                    command_id="cmd-unknown-tool",
                    idempotency_key="idem-unknown-tool",
                    session_id="sess-1",
                    lease_id="lease-1",
                    device_id="dev-1",
                    tool_id="desktop.click",
                ),
                trace_id="trace-1",
            )
        except ValueError as exc:
            assert "unknown tool_id" in str(exc)
        else:
            raise AssertionError("unknown tool ids should be rejected before dispatch")

    asyncio.run(_run())
