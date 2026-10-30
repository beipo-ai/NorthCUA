from __future__ import annotations

from northcua.policy_invocation import (
    POLICY_STEP_HEADER,
    policy_step_extra_headers,
    policy_step_ref,
    use_policy_step_ref,
)


def test_policy_step_ref_uses_existing_controller_trace_context() -> None:
    assert (
        policy_step_ref(
            {
                "northcua_session_id": "session-1",
                "agent_kind": "desktop",
                "round_num": 2,
                "session_step": 3,
            }
        )
        == "session-1:desktop:2:3"
    )
    assert policy_step_ref({"session_id": "session-1"}) == ""


def test_policy_step_header_is_task_local_and_resets() -> None:
    assert policy_step_extra_headers() == {}
    with use_policy_step_ref("session-1:web:1:1"):
        assert policy_step_extra_headers() == {POLICY_STEP_HEADER: "session-1:web:1:1"}
    assert policy_step_extra_headers() == {}


def test_unicode_and_control_characters_are_encoded_only_at_http_boundary():
    from urllib.parse import unquote
    import httpx

    logical = "任务/a%2Fb\r\nheader:desktop:1:2"
    with use_policy_step_ref(logical):
        headers = policy_step_extra_headers()
        value = headers[POLICY_STEP_HEADER]
        assert value.isascii() and "\r" not in value and "\n" not in value
        assert unquote(value) == logical
        assert httpx.Headers(headers)[POLICY_STEP_HEADER] == value
