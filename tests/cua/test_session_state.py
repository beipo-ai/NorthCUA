from __future__ import annotations

import asyncio
from types import SimpleNamespace

from nexau.core.messages import ImageBlock, TextBlock

from northcua.agents.desktop.services.session_state import RoundStateStore
from northcua.core.enums import ResultStatus
from northcua.core.models import CommandExecutionResultRecord
from northcua.transport.ws.result_blocks import command_result_to_tool_result_block


class _FakeSessionManager:
    def __init__(self) -> None:
        self.context: dict[str, object] | None = None
        self.storage: dict[str, object] = {}

    async def get_session(self, *, user_id: str, session_id: str) -> object | None:
        del user_id, session_id
        if self.context is None:
            return None
        return SimpleNamespace(context=self.context, storage=self.storage)

    async def update_session_state(
        self,
        *,
        user_id: str,
        session_id: str,
        context: dict[str, object],
        storage: object,
    ) -> None:
        del user_id, session_id
        self.context = context
        self.storage = storage if isinstance(storage, dict) else {}


def test_round_state_store_preserves_url_only_command_result_images() -> None:
    data_url = "data:image/png;base64,URLONLY"
    manager = _FakeSessionManager()
    store = RoundStateStore(manager, user_id="user-1")

    async def _round_trip() -> list[CommandExecutionResultRecord]:
        await store.record_command_results(
            "sess-1",
            [
                CommandExecutionResultRecord(
                    call_id="call-dict",
                    status=ResultStatus.SUCCESS,
                    result={"text": "captured", "screenshot_url": data_url},
                ),
                CommandExecutionResultRecord(
                    call_id="call-string",
                    status=ResultStatus.SUCCESS,
                    result=data_url,
                ),
            ],
        )
        consumed = await store.consume_command_results("sess-1", ["call-dict", "call-string"])
        assert consumed is not None
        return consumed

    dict_result, string_result = asyncio.run(_round_trip())

    assert dict_result.result == {"text": "captured", "screenshot_url": data_url}
    dict_block = command_result_to_tool_result_block(dict_result)
    assert isinstance(dict_block.content, list)
    assert isinstance(dict_block.content[0], TextBlock)
    assert isinstance(dict_block.content[1], ImageBlock)
    assert dict_block.content[1].url == data_url

    assert string_result.result == data_url
    string_block = command_result_to_tool_result_block(string_result)
    assert isinstance(string_block.content, list)
    assert isinstance(string_block.content[0], ImageBlock)
    assert string_block.content[0].url == data_url
