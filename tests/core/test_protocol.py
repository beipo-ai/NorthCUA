from __future__ import annotations

from northcua.core.enums import WebSocketMessageType
from northcua.core.protocol import HeartbeatPayload, NorthCUAWebSocketEnvelope, TaskAcceptedPayload


def test_protocol_wrap_serializes_typed_payload() -> None:
    envelope = NorthCUAWebSocketEnvelope.wrap(
        WebSocketMessageType.TASK_ACCEPTED,
        TaskAcceptedPayload(session_id="sess-1"),
    )
    parsed = NorthCUAWebSocketEnvelope.model_validate_json(envelope.model_dump_json())
    assert parsed.type == WebSocketMessageType.TASK_ACCEPTED
    assert parsed.payload.session_id == "sess-1"


def test_heartbeat_payload_round_trip() -> None:
    envelope = NorthCUAWebSocketEnvelope.wrap(
        WebSocketMessageType.HEARTBEAT,
        HeartbeatPayload(session_id="sess-1", timestamp=1.23),
    )
    assert envelope.payload.timestamp == 1.23
