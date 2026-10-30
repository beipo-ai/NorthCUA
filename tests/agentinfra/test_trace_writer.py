from __future__ import annotations

from northcua.agentinfra.tracing.log_writer import JsonlTraceWriter


def test_jsonl_trace_writer_streams_equivalent_json_lines(tmp_path) -> None:
    writer = JsonlTraceWriter(tmp_path / "trace.jsonl")

    writer.append({"ok": True, "nested": {"value": "中文"}})

    assert writer.trace_path.read_text(encoding="utf-8").strip() == '{"ok": true, "nested": {"value": "中文"}}'
