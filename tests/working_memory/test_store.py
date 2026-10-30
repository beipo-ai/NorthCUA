import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest
from nexau.core.messages import Message, Role, TextBlock, ToolResultBlock, ToolUseBlock

from northcua.agentinfra.context.adapters.ump import UMPContextProjector
from northcua.agentkernel.context.domain.enums import ContextSegmentKind, RetentionClass
from northcua.agentkernel.working_memory import MEMORY_METADATA_KEY, WorkingMemoryStore
from northcua.core.models import ToolExecutionRecord


def note(text, *, key="", call_id="note-1", tool_name="note", error=False):
    return ToolExecutionRecord(
        tool_call_id=call_id,
        tool_name=tool_name,
        tool_input={"key": key, "text": text},
        tool_output={"status": "recorded", "kind": "fact", "note_text": text},
        is_error=error,
    )


@pytest.mark.parametrize("tool_name", ["note", "web_note"])
def test_replace_replay_restart_and_task_isolation(tmp_path, tool_name):
    store = WorkingMemoryStore(tmp_path)
    first = note("old value", key="form-value", tool_name=tool_name)
    store.ingest([first], round_num=1, session_step=1)
    store.ingest(
        [note("new value", key="form-value", call_id="note-2", tool_name=tool_name)], round_num=1, session_step=2
    )
    store.ingest([first], round_num=1, session_step=1)
    reloaded = WorkingMemoryStore(tmp_path)
    assert reloaded.active_entries(1)[0]["text"] == "new value"
    assert reloaded.active_entries(2) == []
    assert WorkingMemoryStore(tmp_path / "another-session").active_entries(1) == []
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 2


def test_memory_is_pinned_without_breaking_tool_exchange_or_mutating_history(tmp_path):
    store = WorkingMemoryStore(tmp_path)
    store.ingest([note("Keep the confirmed value")], round_num=1, session_step=1)
    messages = [
        Message(role=Role.USER, content=[TextBlock(text="Complete the task")]),
        Message(role=Role.ASSISTANT, content=[ToolUseBlock(id="action", name="click", input={})]),
        Message(role=Role.TOOL, content=[ToolResultBlock(tool_use_id="action", content=[TextBlock(text="clicked")])]),
    ]
    projected = store.inject(messages, round_num=1)
    assert [m.role for m in projected] == [Role.USER, Role.USER, Role.ASSISTANT, Role.TOOL]
    assert len(messages) == 3
    assert sum(bool(m.metadata.get(MEMORY_METADATA_KEY)) for m in store.inject(projected, round_num=1)) == 1
    segments = UMPContextProjector(policy_profile="desktop", recent_iterations=1).project(projected, round_num=1)
    memory = next(segment for segment in segments if segment.kind == ContextSegmentKind.WORKING_MEMORY)
    assert memory.retention == RetentionClass.PINNED
    assert all(segment.kind != ContextSegmentKind.OPEN_TOOL_EXCHANGE for segment in segments)
    # A history window or summary can drop the entire exchange without losing memory.
    assert "confirmed value" in store.inject(messages[:1], round_num=1)[1].content[0].text


def test_projection_budget_and_failed_notes(tmp_path):
    store = WorkingMemoryStore(tmp_path)
    store.ingest([note("ignored", error=True)], round_num=1, session_step=1)
    assert not store.path.exists()
    for i in range(10):
        store.ingest([note("entry " + str(i) + ": " + "x" * 200, call_id=str(i))], round_num=1, session_step=i + 2)
    projected = store.inject([], round_num=1, max_chars=1024)
    assert len(projected[0].content[0].text) <= 1024
    assert "entry 9" in projected[0].content[0].text
    assert "omitted" in projected[0].content[0].text
    assert len(store.active_entries(1)) == 10


def test_parallel_duplicate_ingestion_is_atomic(tmp_path):
    def ingest(_):
        WorkingMemoryStore(tmp_path).ingest([note("one")], round_num=1, session_step=1)

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(ingest, range(8)))
    assert len(WorkingMemoryStore(tmp_path).active_entries(1)) == 1


def test_corruption_is_reported_and_not_silently_replaced(tmp_path):
    store = WorkingMemoryStore(tmp_path)
    store.path.write_bytes(b"corrupted memory database")
    with pytest.raises(sqlite3.DatabaseError):
        store.active_entries(1)
    assert store.path.read_bytes() == b"corrupted memory database"


def test_older_checkpoint_replay_cannot_replace_newer_memory(tmp_path):
    store = WorkingMemoryStore(tmp_path)
    store.ingest([note("latest", key="value", call_id="new")], round_num=1, session_step=10)
    store.ingest([note("stale", key="value", call_id="old")], round_num=1, session_step=2)
    assert store.active_entries(1)[0]["text"] == "latest"
