"""Release-only serialization for private/public contract goldens.

Public NexAU adds stream block identifiers and raw tool output to the UMP
models.  Those fields are useful public extensions, but allowing them to leak
into compatibility goldens would make a dependency-only diff noisy.  Production
persistence, cache keys, trace history, and resume paths do not use this module;
their real runtime behavior is covered independently by integration fixtures.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from uuid import UUID

from nexau.core.messages import Message
from pydantic import BaseModel

from northcua.core.json_types import JsonObject, JsonValue


class PublicMessageExtensionPolicy(str, Enum):
    """Control whether fields introduced by public NexAU reach persistence."""

    DROP = "drop"
    PRESERVE = "preserve"


@dataclass(frozen=True, slots=True)
class NexAUGoldenSerializationPolicy:
    """Deterministic diagnostic serialization policy.

    ``DROP`` is the migration default: private ``58d3fa4`` had neither block
    ids nor ``ToolResultBlock.raw_output``.  A later product rollout may opt in
    to those fields without changing this compatibility contract.
    """

    public_extensions: PublicMessageExtensionPolicy = PublicMessageExtensionPolicy.DROP
    exclude_none: bool = True
    ensure_ascii: bool = False


def normalize_json_value(value: object) -> JsonValue:
    """Convert supported runtime values into a deterministic JSON value.

    The function deliberately preserves list order and mapping keys.  Key
    ordering is applied only when canonical JSON is rendered, so callers that
    inspect the returned payload retain the original semantic structure.
    """

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Enum):
        return normalize_json_value(value.value)
    if isinstance(value, BaseModel):
        return normalize_json_value(value.model_dump(mode="json", exclude_none=True))
    if isinstance(value, Mapping):
        return {str(key): normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [normalize_json_value(item) for item in value]
    raise TypeError(f"value is not JSON serializable at the NexAU boundary: {type(value).__name__}")


class NexAUGoldenSerializationAdapter:
    """Serialize real NexAU messages solely for compatibility goldens."""

    def __init__(self, policy: NexAUGoldenSerializationPolicy | None = None) -> None:
        self._policy = policy or NexAUGoldenSerializationPolicy()

    @property
    def policy(self) -> NexAUGoldenSerializationPolicy:
        """Return the immutable policy used by this adapter."""

        return self._policy

    def message_payload(self, message: Message) -> JsonObject:
        """Project one UMP message without replacing its NexAU type."""

        payload: JsonObject = {
            "id": str(message.id),
            "role": message.role.value,
            "content": [self._block_payload(block) for block in message.content],
            "metadata": normalize_json_value(message.metadata),
        }
        if message.created_at is not None or not self._policy.exclude_none:
            payload["created_at"] = normalize_json_value(message.created_at)
        return payload

    def messages_payload(self, messages: Sequence[Message]) -> list[JsonObject]:
        """Project messages while preserving their causal ordering."""

        return [self.message_payload(message) for message in messages]

    def canonical_json(self, messages: Sequence[Message]) -> str:
        """Return byte-stable JSON suitable for goldens and cache inputs."""

        return json.dumps(
            self.messages_payload(messages),
            ensure_ascii=self._policy.ensure_ascii,
            separators=(",", ":"),
            sort_keys=True,
        )

    def stable_digest(self, messages: Sequence[Message]) -> str:
        """Return the SHA-256 digest of the canonical message projection."""

        return hashlib.sha256(self.canonical_json(messages).encode("utf-8")).hexdigest()

    @staticmethod
    def load_message(payload: Mapping[str, object]) -> Message:
        """Load a persisted message into the installed NexAU model.

        The public model accepts both legacy UUID objects/strings and arbitrary
        external string identifiers.  No NorthCUA wrapper type is introduced.
        """

        return Message.model_validate(dict(payload))

    def _block_payload(self, block: BaseModel) -> JsonObject:
        dumped = block.model_dump(mode="json", exclude_none=self._policy.exclude_none)
        payload = {str(key): normalize_json_value(value) for key, value in dumped.items()}
        if self._policy.public_extensions == PublicMessageExtensionPolicy.DROP:
            block_type = payload.get("type")
            if block_type in {"text", "image", "reasoning"}:
                payload.pop("id", None)
            if block_type == "tool_result":
                payload.pop("raw_output", None)
        return payload


__all__ = [
    "NexAUGoldenSerializationAdapter",
    "NexAUGoldenSerializationPolicy",
    "PublicMessageExtensionPolicy",
    "normalize_json_value",
]
