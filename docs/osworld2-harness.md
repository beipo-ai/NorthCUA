# OSWorld2 formal evaluation harness

This integration exports the **actor-side harness** into the clean NorthCUA
branch. It uses NorthCUA's existing desktop loop, typed action tools, screenshot
history, normalized coordinates, context management and artifact store. It does
not bundle OSWorld2 environments, task data, websites, graders, host inventories,
VM images, setup scripts, dispatchers, queues, repair/relaunch scripts or results.
The older `integrations.osworld` lifecycle adapter is separate and unchanged.

## One formal prompt

[`system.j2`](../northcua/integrations/osworld2/system.j2) is the only OSWorld2
system prompt, imported from commit `b31035db2`. Its versioned checksum is:
SHA256 `fcb43bcb9aa142a149cf910cf99ab22c91eb98f0565dda5396b5a0ef513e63c8`.
The checksum guards this imported file against accidental edits; it is not an
independent comparison with an external campaign's original files.
There are no normal/HITL/sampling variants. The original task query is passed
verbatim; observation text only labels runtime image state. Neither adds another
behavioral prompt or the generic Desktop `[INFEASIBLE]` instruction.
The guest password mentioned in the template is the public benchmark fixture
credential, not a host, model API or production credential.

Select `prompt.system_prompt_profile: osworld2_formal`. The default Desktop/Web
prompts remain unchanged. The template is included in the wheel, not loaded from
a historical campaign directory.

## Same-context three-call policy

Set `execution.no_tool_call_attempts: 3` (ordinary Desktop defaults to 1).
The main actor's non-streaming OpenAI Chat request is frozen **after** context
preparation and serialization. Each attempt receives an independent deep copy of
identical messages, images, tools, tool choice, model parameters and cache key.
A response containing tool calls is selected immediately. Otherwise the third
response is selected. Intermediate replies are saved under the session's
`no_tool_attempts/` with raw response, request hash, logical step and selection
flag. They never reach agent history, tool execution or logical-step counters.
API exceptions and malformed responses abort this policy; `content_filter` is
returned immediately to the existing provider failure handling. None is treated
as permission to repeat an API failure three times. Configured transport retries
are separate; the formal example sets `max_retries: 0`.

The wrapper is installed on the prepared actor's own synchronous/asynchronous SDK
resources for one step and restored even on failure/cancellation. It does not
patch SDK classes, auxiliary clients, the summary model, or other agents. Nested
use of the same client is rejected. The outer step deadline allows three provider
call budgets, while each actual request retains its configured timeout (600s in
the example). Only non-streaming, single-choice OpenAI Chat is currently supported
by this policy; unsupported configurations fail explicitly.

Scope exit also closes the retry loop's lifetime flag. A synchronous SDK request
already running in a worker thread may finish after cancellation, but it cannot
start another no-tool attempt. This policy cannot be combined with
`model_response_recovery: true`; configuration validation rejects the combination.
Audit receipts use the same sanitized session directory as other runtime artifacts.

The harness retains the existing multi-tool result pairing, sliding screenshot
window, reasoning/history handling and context compaction. It does not import
campaign-bound screenshot recovery or environment repair scripts. This is a port
to this branch's public runtime, not a claim that an old worker was hot-upgraded
or that an entire benchmark was rerun on this branch.

## Use with a task already prepared by its owner

Copy [`examples/osworld2`](../examples/osworld2) into a local configuration
directory. Set model credentials via environment variables, and configure the
already prepared desktop endpoint locally. The main example preserves
`normalized_1000`, temperature 0.8, presence penalty 1, top_p 0.95, top_k 20,
min_p 0, repetition penalty 1, max output 65536 and a 600s request timeout. Choose
a model endpoint that supports these parameters; they are not silently removed.
The auxiliary configuration is independent and must match your evaluation's
contract. No task-specific query, grader, credential or host address is supplied.

The task owner binds its original user simulator using `ask_user_tool` and its
original whole-task evaluator using an async callback. This boundary deliberately
does not implement, modify or ship either backend.
`ask_user_tool` projects a typed `ToolContract`, so the callback survives normal
agent preparation and argument validation. The callback is synchronous and calls
are serialized; it must return the original simulator's string answer.
The explicitly projected `LOCAL_HANDLER` callback executes on the harness even
when desktop actions use a remote device. Mixed callback/device batches preserve
their original order and share the ordinary hooks, error results and history
replay. Unprojected or stale tools are not authorized as local callbacks merely
because they lack a device route.

```python
from pathlib import Path
from northcua.agents.desktop.application import DesktopApplication
from northcua.integrations.osworld2 import load_config, run_task, summarize_scores
from northcua.integrations.osworld2.tools import ask_user_tool

async def evaluate_prepared_task(original_query, original_user_simulator,
                                 original_whole_evaluator, unique_session_id):
    bootstrap = load_config(Path("local-config/app.yaml"))
    bootstrap.agent.tools.append(ask_user_tool(original_user_simulator))
    app = DesktopApplication(bootstrap.app, bootstrap.agent, session_id=unique_session_id)
    try:
        report = await run_task(
            app=app, session_id=unique_session_id, task_id="external-task-id",
            query=original_query, evaluate=original_whole_evaluator, max_steps=500,
        )
        return report, summarize_scores([report])
    finally:
        await app.stop()  # closes harness clients; no environment lifecycle API here
```

Use a new application/session and exclusive prepared task per independent run.
`run_task` caps a query at 500 logical steps and sets the session cap to 500 as
well. It has no continuation, automatic retry-run, reset, VM creation/release,
queue refill or evaluator-patching operation. Environment lifecycle stays with
the caller. Hitting the normal step budget calls the whole-task evaluator on the
original current submission; it is not an automatic zero or a fresh-run trigger.

## Results and evidence

Execution errors, explicit empty responses and evaluator failures have
`score=null`, `score_valid=false`; `raw_score` and the actual error are retained.
A normal evaluator zero remains a valid zero. `summarize_scores` reports the
valid denominator, exclusions, sum, mean, full-score rate and valid-zero count;
it refuses duplicate task IDs rather than double-counting attempts. All-error
means/rates are null. Selecting the current attempt and preserving its parent
lineage is the external campaign owner's responsibility. Langfuse publication
is not required by this adapter and no scores are edited remotely.

## Verification

```sh
uv run --no-sync pytest -q tests/integrations/osworld2 tests/integrations/test_osworld_runner.py
uv run --no-sync pytest -q tests/examples/test_osworld_examples.py tests/transport/test_websocket_transport_more.py
```

Tests use deterministic fixtures, real OpenAI SDK serialization with a mock HTTP
transport, and the pinned NexAU caller through the actual Desktop model-step
binding. They check same-context retries, selected-only history, auxiliary
isolation, cancellation/error restoration, cache-key boundaries, the frozen
prompt, 500-step request caps and error/zero accounting. They do not provision
VMs, call hosted models or claim fresh OSWorld2 task acceptance.
