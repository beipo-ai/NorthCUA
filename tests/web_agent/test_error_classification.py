"""Focused Web tool error classification contracts."""

from northcua.agents.web.mcp.error_classification import classify_web_tool_error
from northcua_runtime_shared.validation import ToolParameterValidationError


def test_controller_oneof_validation_error_is_retryable_invalid_params() -> None:
    error = ToolParameterValidationError(
        "command parameter 'parameters' must match exactly one schema; "
        "unexpected parameters at parameters: load_state, text, url_pattern, selector"
    )

    classification = classify_web_tool_error(error)

    assert classification.code == "invalid_params"
    assert classification.reason_code == "invalid_params"
    assert classification.retryable is True
    assert classification.feedback_source == "controller_contract_validator"
    assert "exactly one" in classification.hint
    assert "null placeholders" in classification.hint
