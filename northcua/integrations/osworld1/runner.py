"""Actor-only formal OSWorld1 adapter: no environment or dispatch operations."""

from collections.abc import Awaitable, Callable

from northcua.agentinfra.runtime.protocol import DesktopRunRequest
from northcua.agents.desktop.application import DesktopApplication
from northcua.integrations.osworld2.scores import score_record


async def run_task(
    *,
    app: DesktopApplication,
    session_id: str,
    task_id: str,
    query: str,
    evaluate: Callable[[str | None], Awaitable[float]],
    max_steps: int = 100,
) -> dict:
    """Use the caller's prepared app and scorer; leave resource custody outside.

    ``evaluate`` receives DONE for completion, FAIL only for an explicit
    infeasibility decision, otherwise None (including the step limit). The
    caller owns all benchmark-specific terminal-action and scoring integration.
    This module does not create/reset/close a VM, load task data, dispatch a
    batch, repair an environment, upload traces, or stop other work.
    """
    if not task_id.strip() or not query.strip() or not 1 <= max_steps <= 100:
        raise ValueError("A task ID, original query and 1..100 step budget are required")
    config = app.app_config
    if config.prompt.system_prompt_profile != "osworld1_formal":
        raise ValueError("Select the single osworld1_formal prompt")
    if config.execution.no_tool_call_attempts != 1 or config.evaluation.enabled:
        raise ValueError("OSWorld1 uses terminal text and an external task evaluator")
    reason = ""
    raw_score = None
    result = None
    action = None
    try:
        result = await app.run_query(
            DesktopRunRequest(session_id=session_id, query=query, max_steps=max_steps, max_session_steps=100)
        )
        reason = str(result.terminal_decision.get("reason", ""))
        if str(result.terminal_status).lower() not in {"finish", "fail"} or reason in {
            "empty_response",
            "content_filter",
            "runtime_error",
        }:
            raise RuntimeError(f"Actor execution error: {reason or result.terminal_status}")
        if not result.hit_step_limit:
            if reason == "completed":
                action = "DONE"
            elif reason == "infeasible":
                action = "FAIL"
        raw_score = await evaluate(action)
        record = score_record(task_id=task_id, raw_score=raw_score, terminal_reason=reason)
    except Exception as exc:
        record = score_record(
            task_id=task_id, raw_score=raw_score, terminal_reason=reason, error=f"{type(exc).__name__}: {exc}"
        )
    record.update(session_id=session_id, evaluator_action=action)
    if result is not None:
        record.update(terminal_decision=dict(result.terminal_decision), hit_step_limit=result.hit_step_limit)
    return record
