"""Tests for the standalone NorthCUA protocol package."""

from __future__ import annotations

import pytest

from northcua_protocol import (
    ArtifactKind,
    ArtifactRef,
    ArtifactScope,
    Capability,
    CommandCancel,
    CommandRequest,
    CommandResult,
    DeviceDescriptor,
    DeviceHello,
    DeviceTelemetryEvent,
    Envelope,
    MessageType,
    PlatformKind,
    RetentionPolicy,
    SessionInterrupted,
    StorageKind,
    TaskCancel,
    TaskFinished,
    ToolDescriptor,
    parse_envelope,
)
from northcua_protocol.state import CommandState, SessionState, TaskState


def test_envelope_round_trips_device_hello() -> None:
    descriptor = DeviceDescriptor(
        device_id="mac-1",
        platform=PlatformKind.MACOS,
        capabilities={Capability.DESKTOP_INPUT},
        tools=[ToolDescriptor(tool_id="device.echo", capability=Capability.DESKTOP_INPUT)],
    )
    envelope = Envelope.wrap(
        MessageType.DEVICE_HELLO,
        DeviceHello(descriptor=descriptor, token="token"),
        trace_id="trace-1",
    )

    parsed = parse_envelope(envelope.model_dump_json())

    assert parsed.type == MessageType.DEVICE_HELLO
    assert isinstance(parsed.payload, DeviceHello)
    assert parsed.payload.descriptor.device_id == "mac-1"


def test_envelope_round_trips_device_telemetry() -> None:
    envelope = Envelope.wrap(
        MessageType.DEVICE_TELEMETRY,
        DeviceTelemetryEvent(
            device_id="dev-1",
            event_type="device_driver_call_finished",
            entity_type="command",
            entity_id="cmd-1",
            command_id="cmd-1",
            session_id="sess-1",
            tool_id="click",
            stage="driver",
            payload={"elapsed_ms": 123, "status": "succeeded"},
        ),
        trace_id="trace-telemetry",
    )

    parsed = parse_envelope(envelope.model_dump_json())

    assert parsed.type == MessageType.DEVICE_TELEMETRY
    assert isinstance(parsed.payload, DeviceTelemetryEvent)
    assert parsed.payload.payload["elapsed_ms"] == 123


def test_protocol_models_reject_unknown_fields() -> None:
    with pytest.raises(ValueError, match="extra_forbidden"):
        CommandRequest(
            command_id="cmd-1",
            idempotency_key="idem-1",
            session_id="sess-1",
            lease_id="lease-1",
            device_id="dev-1",
            tool_id="desktop.click",
            unexpected=True,
        )


def test_terminal_payloads_reject_non_terminal_states() -> None:
    with pytest.raises(ValueError, match="CommandResult.status must be terminal"):
        CommandResult(
            command_id="cmd-1",
            session_id="sess-1",
            lease_id="lease-1",
            device_id="dev-1",
            status=CommandState.RUNNING,
        )
    with pytest.raises(ValueError, match="TaskFinished.state must be terminal"):
        TaskFinished(task_id="task-1", trace_id="trace-1", session_id="sess-1", state=TaskState.RUNNING)
    with pytest.raises(ValueError, match="SessionLeaseReleased.state must be released"):
        from northcua_protocol import SessionLeaseReleased

        SessionLeaseReleased(
            lease_id="lease-1",
            session_id="sess-1",
            device_id="dev-1",
            state=SessionState.LEASED,
        )


def test_command_request_uses_tool_id_not_server_name() -> None:
    command = CommandRequest(
        command_id="cmd-1",
        idempotency_key="idem-1",
        session_id="sess-1",
        lease_id="lease-1",
        device_id="dev-1",
        tool_id="desktop.click",
        parameters={"x": 10, "y": 20},
    )

    assert command.tool_id == "desktop.click"
    assert command.state == CommandState.SENT
    assert command.attempt == 1
    assert command.lease_generation == 1


def test_envelope_round_trips_reserved_cancellation_and_interruption_payloads() -> None:
    interrupted = Envelope.wrap(
        MessageType.SESSION_INTERRUPTED,
        SessionInterrupted(
            lease_id="lease-1",
            session_id="sess-1",
            device_id="dev-1",
            reason="device disconnected",
        ),
        trace_id="trace-1",
    )
    command_cancel = Envelope.wrap(
        MessageType.COMMAND_CANCEL,
        CommandCancel(
            command_id="cmd-1",
            session_id="sess-1",
            lease_id="lease-1",
            device_id="dev-1",
            reason="cancel requested",
        ),
        trace_id="trace-1",
    )
    task_cancel = Envelope.wrap(
        MessageType.TASK_CANCEL,
        TaskCancel(task_id="task-1", trace_id="trace-1", session_id="sess-1", reason="submitter cancel"),
        trace_id="trace-1",
    )

    assert isinstance(parse_envelope(interrupted.model_dump_json()).payload, SessionInterrupted)
    assert isinstance(parse_envelope(command_cancel.model_dump_json()).payload, CommandCancel)
    assert isinstance(parse_envelope(task_cancel.model_dump_json()).payload, TaskCancel)


def test_inline_artifact_validates_hash_and_size() -> None:
    artifact = ArtifactRef.from_inline_bytes(
        artifact_id="art-1",
        kind=ArtifactKind.SCREENSHOT,
        data=b"image-bytes",
        mime_type="image/png",
        artifact_scope=ArtifactScope.TASK,
        retention_policy=RetentionPolicy.LONG_TERM,
    )

    assert artifact.storage_kind == StorageKind.INLINE
    assert artifact.size_bytes == len(b"image-bytes")
    assert artifact.artifact_scope == ArtifactScope.TASK
    assert artifact.retention_policy == RetentionPolicy.LONG_TERM
    blob = ArtifactRef.from_controller_blob(
        artifact_id="blob",
        kind=ArtifactKind.SCREENSHOT,
        uri=f"controller-blob://{artifact.sha256}",
        mime_type="image/png",
        sha256=artifact.sha256,
        size_bytes=artifact.size_bytes,
        artifact_scope=ArtifactScope.SESSION,
        retention_policy=RetentionPolicy.SESSION,
    )
    assert blob.storage_kind == StorageKind.CONTROLLER_BLOB
    assert blob.artifact_scope == ArtifactScope.SESSION
    assert blob.retention_policy == RetentionPolicy.SESSION

    with pytest.raises(ValueError, match="sha256 mismatch"):
        ArtifactRef(
            artifact_id="bad",
            kind=ArtifactKind.SCREENSHOT,
            storage_kind=StorageKind.INLINE,
            mime_type="image/png",
            sha256="0" * 64,
            size_bytes=len(b"image-bytes"),
            inline_data=artifact.inline_data,
        )

    with pytest.raises(ValueError, match="controller_blob uri must match sha256"):
        ArtifactRef.from_controller_blob(
            artifact_id="bad-blob",
            kind=ArtifactKind.SCREENSHOT,
            uri="controller-blob://not-the-hash",
            mime_type="image/png",
            sha256=artifact.sha256,
            size_bytes=artifact.size_bytes,
        )

    with pytest.raises(ValueError, match="size_bytes must be non-negative"):
        ArtifactRef.from_controller_blob(
            artifact_id="negative-blob",
            kind=ArtifactKind.SCREENSHOT,
            uri=f"controller-blob://{artifact.sha256}",
            mime_type="image/png",
            sha256=artifact.sha256,
            size_bytes=-1,
        )
