from __future__ import annotations

from fastmcp.tools.base import ToolResult

from northcua.runtime.mcp.result_adapter import normalize_tool_result


def test_normalize_tool_result_unwraps_single_result_string() -> None:
    payload = normalize_tool_result(ToolResult(content=[], structured_content={"result": "data:image/png;base64,AAAA"}))
    assert payload == "data:image/png;base64,AAAA"


def test_normalize_tool_result_unwraps_single_result_dict() -> None:
    payload = normalize_tool_result(
        ToolResult(content=[], structured_content={"result": {"text": "Clicked", "screenshot_path": "/tmp/x.png"}})
    )
    assert payload == {"text": "Clicked", "screenshot_path": "/tmp/x.png"}
