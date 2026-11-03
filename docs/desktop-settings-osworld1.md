# Recommended Desktop settings: OSWorld1 style

Use these optional settings as a starting point for a screenshot-driven Desktop
agent. Adjust sampling, coordinates and budgets to your model endpoint. Start
with [first run](first-run.md) for ordinary local Desktop/Web tasks.

## Copy and configure

Copy the complete [example directory](../examples/desktop-settings/osworld1/):

```sh
mkdir -p logs
cp -R examples/desktop-settings/osworld1 logs/my-osworld1
```

Set `NORTHCUA_MODEL`, `NORTHCUA_BASE_URL` and `OPENAI_API_KEY` in your environment.
In `logs/my-osworld1/app.yaml`, set `desktop.osworld.vm_ip` and `server_port` to
an available desktop HTTP endpoint. The example uses `desktop.backend: osworld`;
[environment configuration](environments.md#desktop-http-backend) explains this
input/screenshot backend. Desktop provisioning stays with the environment owner.

```sh
northcua config validate --app-config logs/my-osworld1/app.yaml
northcua local desktop run --app-config logs/my-osworld1/app.yaml \
  --query 'Open a calculator and compute 12 * 7.' --max-steps 100
```

## Suggested settings

| Setting | Example value | Purpose |
|---|---|---|
| `prompt.system_prompt_profile` | `osworld1` | Select the bundled Desktop prompt and its text projection |
| `desktop.coordinate_policy.input_space` | `normalized_1000` | Interpret model positions on a 0–1000 scale |
| `execution.max_agent_iterations_per_round` | `100` | Default logical-step budget; pass `--max-steps` explicitly when running |
| `prompt.history_window_rounds` | `100` | Retain action history |
| `prompt.screenshot_window_images` | `3` | Sliding overlay/clean screenshot window |
| `execution.model_response_recovery` | `true` | Recover transient/empty and token-exhausted reasoning-only responses within a step |
| `execution.no_tool_call_attempts` | `1` | Return valid terminal text without text-only retries |
| `llm_config.max_retries` | `3` | Additional response-recovery attempts; SDK transport retries are separate |
| Model parameters | Chat Completions, temperature `0.7`, reasoning effort `max`, `16000` output tokens, `120s` timeout | Adapt to the model and provider |
| `llm_config.reasoning_replay_policy` | `chat_reasoning_content` | Replay Chat reasoning when supported by the endpoint |

Response recovery uses the same prepared request, with independent copies of
messages and tools for each attempt. Backoff is 2/4/8 seconds for three retries,
plus bounded jitter. Permanent HTTP 4xx errors do not retry; cancellation and
hard deadlines propagate. Recovery does not repeat tool execution, add failed
replies to history or reset the task budget. Do not combine it with
`execution.no_tool_call_attempts: 3`.

## Prompt and coordinates

The profile lives in
[`prompt/profiles/osworld1`](../northcua/agents/desktop/prompt/profiles/osworld1/).
It preserves separate system, persistent-user and observation messages.

With `normalized_1000`, the profile describes a virtual 1000 × 1000 canvas;
screenshot bytes retain their native resolution. Action dispatch converts
positions to real pixels once. Scroll distances are not positions.

This profile also applies a legacy regex projection to coordinate-like text in
task/observation/history messages. It preserves structured tool arguments,
image bytes, reasoning and original artifacts, but matching coordinate literals
in task prose can also be rewritten. The no-image fallback is 1920 × 1080.
With `image_pixels`, that text conversion and virtual canvas are disabled.
For general Desktop use, keep the default `desktop` profile and select only the
[coordinate policy](configuration.md#desktop-coordinates) your model needs.

The prompt contains OSWorld guest-account instructions when the environment
label is `osworld`; use it only with the matching prepared environment.

## Check the retained behavior

```sh
uv run --no-sync pytest tests/cua/test_osworld1_prompt_profile.py \
  tests/agentinfra/test_response_recovery.py \
  tests/agentinfra/test_response_recovery_wire.py -q
```

These deterministic checks exercise prompt rendering, coordinate projection and
response recovery. They do not establish model capability or benchmark scores.
