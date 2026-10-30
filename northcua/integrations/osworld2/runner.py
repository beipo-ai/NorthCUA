"""A single prepared-task harness, with no environment or dispatch lifecycle."""

from collections.abc import Awaitable, Callable

from northcua.agentinfra.runtime.protocol import DesktopRunRequest
from northcua.agents.desktop.application import DesktopApplication
from northcua.core.enums import CoordinateSpace

from .scores import score_record


async def run_task(
    *,
    app: DesktopApplication,
    session_id: str,
    task_id: str,
    query: str,
    evaluate: Callable[[], Awaitable[float]],
    max_steps: int = 500,
) -> dict:
    """Run the original query and the caller's whole-task scorer on the same app.

    The caller supplies an already prepared, exclusive task environment and
    owns its lifetime. This function never creates, resets, releases, restarts
    or schedules an environment. A normal step-limit exit is scored as-is;
    actor/evaluator errors are excluded rather than converted to zero.
    """
    if not task_id.strip() or not query.strip() or not 1 <= max_steps <= 500:
        raise ValueError("A nonempty original query and 1..500 logical steps are required")
    config = app.app_config
    if config.prompt.system_prompt_profile != "osworld2_formal":
        raise ValueError("Select the single osworld2_formal system prompt")
    if config.desktop.coordinate_policy.input_space != CoordinateSpace.NORMALIZED_1000:
        raise ValueError("Formal OSWorld2 requires normalized_1000 coordinates")
    if config.execution.no_tool_call_attempts != 3 or config.evaluation.enabled:
        raise ValueError("Formal OSWorld2 requires three-call policy and the external whole-task evaluator")
    raw_score = None
    reason = ""
    result = None
    try:
        result = await app.run_query(
            DesktopRunRequest(session_id=session_id, query=query, max_steps=max_steps, max_session_steps=500)
        )
        reason = str(result.terminal_decision.get("reason", ""))
        if str(result.terminal_status).lower() not in {"finish", "fail"} or reason in {
            "empty_response",
            "content_filter",
            "runtime_error",
        }:
            raise RuntimeError(f"Actor execution error: {reason or result.terminal_status}")
        raw_score = await evaluate()
        record = score_record(task_id=task_id, raw_score=raw_score, terminal_reason=reason)
    except Exception as exc:
        record = score_record(
            task_id=task_id, raw_score=raw_score, terminal_reason=reason, error=f"{type(exc).__name__}: {exc}"
        )
    record["session_id"] = session_id
    if result is not None:
        record["terminal_decision"] = dict(result.terminal_decision)
        record["hit_step_limit"] = result.hit_step_limit
    return record
