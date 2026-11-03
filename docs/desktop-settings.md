# Recommended Desktop settings

Both recipes use the ordinary Desktop prompt and native screenshot/input tools.
For a first task, start with [first run](first-run.md). These examples adjust
model parameters, history, context and response policies; they do not select a
benchmark-specific prompt or provision a task environment.

## Choose a recipe

| Recipe | Example | Useful when |
|---|---|---|
| Bounded response recovery | [response-recovery](../examples/desktop-settings/response-recovery/) | A provider occasionally returns transient failures, empty replies or token-exhausted reasoning-only output |
| Long context and text-only response retries | [long-context](../examples/desktop-settings/long-context/) | Long tasks need context compaction and a bounded chance to obtain a tool response |

Copy a complete directory into a writable location:

```sh
mkdir -p logs
cp -R examples/desktop-settings/response-recovery logs/my-desktop
```

For the long-context recipe, copy `examples/desktop-settings/long-context` instead.
Set `NORTHCUA_MODEL`, `NORTHCUA_BASE_URL` and `OPENAI_API_KEY`. Match the
[coordinate convention](configuration.md#desktop-coordinates) to your model:
both examples use `normalized_1000`; models returning screenshot pixels need
`image_pixels`. Desktop input requires the usual [platform permissions](getting-started.md#run-desktop).

```sh
northcua config validate --app-config logs/my-desktop/app.yaml
northcua local desktop run --app-config logs/my-desktop/app.yaml --query 'Open a calculator and compute 12 * 7.' --max-steps 100
```

Use an explicit task budget appropriate for your request; the examples provide
100 and 500 default logical steps. Their sampling parameters are suggestions,
not universal model defaults. Remove provider-specific fields your endpoint does
not support. Both recipes keep the built-in outcome evaluator off initially;
[enable it](evaluation.md) when you have configured its model and budget.

## Bounded response recovery

The example uses Chat Completions, temperature `0.7`, reasoning effort `max`,
`16000` output tokens and a `120s` request timeout. It keeps 100 history turns,
three screenshot slots and supported Chat reasoning replay.

`execution.model_response_recovery: true` enables retries within one actor step,
using `llm_config.max_retries: 3` as the additional-attempt budget. Each attempt
receives independent copies of the same prepared messages and tools. Backoff is
2/4/8 seconds plus bounded jitter. Valid terminal text returns immediately;
permanent HTTP 4xx errors, cancellation and hard deadlines propagate.
SDK transport retries remain a separate network budget.

## Long context and text-only retries

The example uses `context_management.mode: enforce` with an independent summary
model. Configure `NORTHCUA_SUMMARY_MODEL`, `NORTHCUA_SUMMARY_BASE_URL` and
`NORTHCUA_SUMMARY_API_KEY`; these may reuse the primary endpoint. The copied
`summary-agent.yaml` path resolves relative to `app.yaml`.

The primary example uses temperature `0.8`, `top_p: 0.95`, presence penalty `1`,
`top_k: 20`, `min_p: 0`, repetition penalty `1`, `65536` output tokens and a `600s`
request timeout. These fields require an endpoint that supports them.

`execution.no_tool_call_attempts: 3` supports non-streaming, single-choice OpenAI
Chat requests. It selects the first tool response or the last text-only response.
Intermediate replies stay out of history, tool execution and logical-step
accounting. Receipts are retained under `no_tool_attempts/`. API/malformed-response
errors propagate; summary requests are independent. The scope closes on
cancellation so a running synchronous request cannot start another attempt.

The two response policies are mutually exclusive. Neither repeats tool execution
or restarts a task. The ordinary first-run tutorial keeps both policies off.

## Updating an older configuration

Remove `prompt.system_prompt_profile` from older app YAMLs. The Harness now uses
one ordinary Desktop prompt; benchmark-specific profiles and text-coordinate
rewrites are removed. The general 0–1000 action-coordinate mapper remains active
when `desktop.coordinate_policy.input_space: normalized_1000` is selected.
