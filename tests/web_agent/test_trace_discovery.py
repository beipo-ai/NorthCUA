from __future__ import annotations

from northcua.agents.web.trace_viewer.domain import TraceKind
from northcua.agents.web.trace_viewer.trace_discovery import discover_trace_sources


def test_discovery_finds_main_and_isolated_context_summary_traces(tmp_path) -> None:
    (tmp_path / "llm_trace.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "context_summary_llm_trace.jsonl").write_text("{}\n", encoding="utf-8")
    evaluator = tmp_path / "evaluations" / "web_round_evaluator" / "round_0001"
    evaluator.mkdir(parents=True)
    (evaluator / "context_summary_llm_trace.jsonl").write_text("{}\n", encoding="utf-8")

    sources = discover_trace_sources(tmp_path, [])

    assert [source.kind for source in sources] == [
        TraceKind.MAIN,
        TraceKind.CONTEXT_SUMMARY,
        TraceKind.CONTEXT_SUMMARY,
    ]
    assert "evaluator context summary" in sources[-1].label
