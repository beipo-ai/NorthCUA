# Recommended Desktop settings: OSWorld2 style

These optional settings combine a Desktop prompt, normalized coordinates,
a larger step budget and bounded retries for text-only model responses. Treat
them as a starting recipe and adapt them to your model endpoint.

## Copy and configure

Copy the [example directory](../examples/desktop-settings/osworld2/):

```sh
mkdir -p logs
cp -R examples/desktop-settings/osworld2 logs/my-osworld2
```

Set the primary model's `NORTHCUA_MODEL`, `NORTHCUA_BASE_URL` and
`OPENAI_API_KEY`. The independent summary model uses `NORTHCUA_SUMMARY_MODEL`,
`NORTHCUA_SUMMARY_BASE_URL` and `NORTHCUA_SUMMARY_API_KEY`; these may point to the
same endpoint if it supports both workloads. The summary config path resolves
relative to `app.yaml`, so the copied directory can be used from another
working directory.

In `logs/my-osworld2/app.yaml`, configure `desktop.osworld.vm_ip` and
`server_port` for an available desktop HTTP endpoint; see
[environment configuration](environments.md#desktop-http-backend).

```sh
northcua config validate --app-config logs/my-osworld2/app.yaml
northcua local desktop run --app-config logs/my-osworld2/app.yaml \
  --query 'Complete the requested desktop task.' --max-steps 500
```

## Suggested settings

| Setting | Example value | Purpose |
|---|---|---|
| `prompt.system_prompt_profile` | `osworld2` | Select the bundled Desktop prompt |
| `desktop.coordinate_policy.input_space` | `normalized_1000` | Interpret model positions on a 0–1000 scale |
| `execution.max_agent_iterations_per_round` | `500` | Default logical-step budget; pass `--max-steps` explicitly when running |
| `execution.no_tool_call_attempts` | `3` | Select the first tool response, or the last of up to three text-only responses |
| `execution.model_response_recovery` | `false` | Keep the separate recovery policy off |
| `prompt.screenshot_window_images` | `3` | Sliding overlay/clean screenshot window |
| `context_management.mode` | `enforce` | Compact context with the configured summary model |
| Model parameters | Temperature `0.8`, presence penalty `1`, `top_p: 0.95`, `top_k: 20`, `min_p: 0`, repetition penalty `1` | Use an endpoint that supports these fields |
| Output/timeout | `65536` output tokens, `600s` request timeout, `max_retries: 0` | Budget for long responses; provider transport retries remain separate |

These values are explicit recommendations for this recipe, not universal model
defaults. The ordinary Desktop profile remains the starting point for general
local tasks.

## Text-only response retries

The policy supports non-streaming, single-choice OpenAI Chat requests. Each
attempt receives an independent copy of the same serialized messages, images,
tools and model parameters. A tool response is selected immediately; otherwise
the final text response is selected. Intermediate replies do not enter history,
execute tools or consume additional logical steps.

Receipts under the session's `no_tool_attempts/` record request hashes, responses
and selection decisions. API errors and malformed responses propagate;
`content_filter` goes directly to provider failure handling. The policy does not
retry tool execution or start another task. It is mutually exclusive with
`execution.model_response_recovery: true`.

The policy is scoped to the current actor step and restored on failure or
cancellation. A synchronous request already in progress may finish, but a closed
scope cannot issue another attempt. Summary-model requests are independent.

The [prompt source](../northcua/agents/desktop/prompt/profiles/osworld2/) passes
the task text through unchanged and labels each observation's image state.
It uses the standard Desktop tool surface; missing task prerequisites are
reported in the final response.

## Check the retained behavior

```sh
uv run --no-sync pytest tests/cua/test_osworld2_prompt_profile.py \
  tests/agentinfra/test_no_tool_retry.py \
  tests/agentinfra/test_response_policy_guards.py -q
```

These checks use deterministic fixtures and real SDK serialization with a mock
HTTP transport. They do not establish model capability or benchmark scores.
