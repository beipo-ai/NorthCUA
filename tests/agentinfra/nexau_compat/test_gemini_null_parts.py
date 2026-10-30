from tests.support.providers.gemini_null_parts import collect_gemini_null_parts


def test_gemini_null_parts_has_bounded_retries_and_clear_error():
    result = collect_gemini_null_parts("public")
    assert result.exception_type == "RuntimeError"
    assert result.exception_message == "No response content or tool calls"
    assert result.attempt_count == 3
