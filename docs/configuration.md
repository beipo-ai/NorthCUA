# Configuration

The `local` extra installs public [NexAU](https://github.com/hzhua/nexau) from
the fixed snapshot `28511d9f0362191a6c378fd441b2be55309a9e32`. The original
`nex-agi/NexAU` download address is unavailable. After updating an existing
checkout, run `uv sync --locked --extra local --extra device` to install the
matching NexAU, MCP and FastMCP versions from the updated lock file.

When installing built wheels with `uv pip` into an activated environment, also
list NexAU explicitly: uv requires direct URL dependencies to be authorized in
the install request. For a wheelhouse built from this checkout:

```sh
uv pip install --find-links wheelhouse 'northcua[local]==0.2.0' \
  'northcua-device==0.2.0' \
  'nexau @ git+https://github.com/hzhua/nexau.git@28511d9f0362191a6c378fd441b2be55309a9e32'
```

For most users, only the model name, API key and optional base URL need changing.
Copy [the env template](../.env.example), load it explicitly, then generate both
agent configurations together:

```sh
northcua config init --directory logs/my-config \
  --model "$NORTHCUA_MODEL" --base-url "$NORTHCUA_BASE_URL"
northcua config validate --app-config logs/my-config/app.example.yaml
```

The default API is OpenAI-compatible Chat. Keys remain `${env.OPENAI_API_KEY}`
references. `--api-key-env` takes a **variable name**, never the secret value.
Connection overrides are written to both Desktop and Web YAMLs. `config init`
does not overwrite existing files unless explicitly given `--force`.

`northcua config init --directory my-config` writes portable examples.
`northcua config validate --app-config my-config/app.example.yaml` loads both agent
configurations without making a model call. Environment placeholders must be populated.

| Layer | Responsibility |
|---|---|
| app YAML | Artifact root, Desktop/Web settings, observation, context, evaluation and budgets |
| agent YAML | NexAU model, provider, optional skills and model-visible MCP tools |
| controller YAML | Authentication, scheduling, execution mode, storage and artifact limits |
| device YAML | Platform, tool/capability allowlists, GUI resource identity and telemetry |
| batch/experiment/case YAML | Task inputs, fresh sandbox allocation, fixtures, retries and evaluation |

The public examples expose Desktop screenshot/mouse/keyboard actions, or Web browser
and DOM/ref actions. Desktop accessibility trees are off by default. Shell/file tools
remain available for setup and evaluation; they are not in the default desktop actor
surface. Explicitly adding tools changes the task's execution contract.

Desktop always uses screenshots returned by its GUI tools. Initial capture supplies
state before a clean tool result exists; subsequent visual decisions use the returned
state and the shared overlay/clean history window. There is no observation-mode selector.
The `observation` section configures optional `a11y` recording only. See
[context and images](context.md#images) for removed configuration fields.

## Desktop coordinates

For a Desktop model that emits coordinates normalized to **0–1000**, set the
following in the **app YAML used by the run**, for example
`logs/my-config/app.example.yaml`. Merge these fields into its existing
`desktop` section:

```yaml
desktop:
  coordinate_policy:
    input_space: normalized_1000
    clamp_to_bounds: true
```

`input_space: normalized_1000` enables this coordinate interpretation. The
default is `image_pixels`; `config init` keeps that default, so change it before
using a model that returns 0–1000 coordinates. Match the model's output convention;
a model returning screenshot pixels should keep `image_pixels`.

The model supplies `x` and `y` relative to the **latest grounding screenshot**:
`(0, 0)` is the top-left and `(500, 500)` is the center. The Harness converts
positions to screenshot pixels as `x_px = x / 1000 * width` and
`y_px = y / 1000 * height`, then applies the capture's desktop offset for input
execution. For a 1920 × 1080 capture, `(500, 500)` maps to `(960, 540)` before
that offset. This applies to click, double-click, move, drag path points and
optional scroll target positions; scroll distances retain their tool-defined units.

`clamp_to_bounds` is a separate, optional setting (default `false`). Setting it
to `true` keeps mapped positions inside the capture, including an input of
`1000` at the right or bottom edge, which maps to the last valid pixel. The
Desktop system prompt reflects the selected coordinate convention. Supply the
same app YAML to local, WebSocket or controller/batch runs using this model;
put these fields in the app YAML, not in the model's agent YAML.

**Web uses a different tool contract.** DOM/SoM refs remain element identifiers;
`web_click_point`, `web_double_click_point` and `web_drag_point` use current-viewport
CSS pixels. `desktop.coordinate_policy` does not normalize Web actions, and
`web.coordinate_policy` is not a supported configuration field.

## Desktop response retries

Desktop uses one ordinary prompt. Optional model/history/context recipes are
listed in [recommended Desktop settings](desktop-settings.md). Remove
`prompt.system_prompt_profile` from older configurations.

| App YAML setting | Default | Optional behavior |
|---|---|---|
| `execution.no_tool_call_attempts` | `1` | `3`: OpenAI Chat, non-streaming, single-choice requests can retry text-only responses with an identical serialized request, selecting a tool response immediately or the final response. Intermediate responses stay out of history and logical-step accounting. |
| `execution.model_response_recovery` | `false` | Retry transient/empty or token-exhausted reasoning-only responses inside one actor step, using `llm_config.max_retries` as the additional-attempt budget. Valid terminal text is returned immediately. |

The two optional retry policies are mutually exclusive. SDK transport retries
are separate and can add network requests; they do not reset task-step budgets.
Neither policy repeats tool execution. Default Desktop/Web tutorials keep both
policies off. The general action-coordinate mapper does not rewrite task or
history text.

## Providers

Use a fresh config directory for each connection recipe:

```sh
# Responses API; keep your OpenAI key in OPENAI_API_KEY.
northcua config init --directory logs/responses-config --model "$NORTHCUA_MODEL" \
  --base-url https://api.openai.com/v1 --api-type openai_responses

# Anthropic; set ANTHROPIC_API_KEY in the launching environment.
northcua config init --directory logs/anthropic-config --model "$NORTHCUA_MODEL" \
  --base-url https://api.anthropic.com --api-type anthropic_chat_completion \
  --api-key-env ANTHROPIC_API_KEY

# Gemini REST; set GEMINI_API_KEY in the launching environment.
northcua config init --directory logs/gemini-config --model "$NORTHCUA_MODEL" \
  --base-url https://generativelanguage.googleapis.com --api-type gemini_rest \
  --api-key-env GEMINI_API_KEY
```

Use the model ID and endpoint supported by your account. These recipes select
protocol adapters; they do not claim live acceptance of every model. The pinned
NexAU/runtime contract and the [verification record](verification.md) define the
tested boundary. Validate the generated app, then run one bounded task before scaling.

NexAU's `llm_config.api_type` supports the tested protocol adapters:

| Provider interface | API type | Typical credential |
|---|---|---|
| OpenAI-compatible Chat Completions | `openai_chat_completion` | `OPENAI_API_KEY` |
| OpenAI Responses | `openai_responses` | `OPENAI_API_KEY` |
| Anthropic Messages | `anthropic_chat_completion` | `ANTHROPIC_API_KEY` |
| Gemini REST | `gemini_rest` | `GEMINI_API_KEY` |

Set the exact model and base URL supported by your endpoint. The sample uses
`${env.NORTHCUA_MODEL}` and `${env.OPENAI_API_KEY}`; change the placeholder names when
switching providers. A model must accept images and structured tool calls for the
default agent configuration. Provider-specific reasoning, image formats and replay
settings belong in the agent config. No internal model endpoint is a project default.

```sh
northcua provider probe --help
```

Provider wire/stream tests run without network and cover tool-call IDs, reasoning,
images, retries, disconnects and cancellation. They establish protocol compatibility,
not the quality or availability of a specific hosted model. Record live-model validation
separately when credentials are available.

## Important limits

| Default | Value in generated examples | When to change |
|---|---|---|
| Actor step budget | 30 | Prefer `--max-steps 8` for a first Web task, 15 for Desktop |
| Model output tokens | 4096 | Match endpoint capability and task needs |
| Model context capacity | 128000 | Set the actual supported model limit |
| Request timeout / retries | 120 seconds / 3 | Adjust for your endpoint, not to hide invalid authentication |
| Screenshot history | 3 images | Trade visual history against model input cost |
| Context compaction | off | Configure deliberately for long tasks; see [context](context.md) |
| Evaluation | off | Use Web `--evaluate` or enable it in app YAML |
| Live evaluation probes | off | Enable explicitly to inspect the active workspace |
| External tracing | off | Local artifacts require no hosted observability service |

Runtime skill paths and asset access are documented in [Skills](skills.md).
Environment variables used by agent YAML must be present in the model/controller
process. Worker-only hosts do not need model credentials.

Generate the full current field/type/default reference without reading source:

```sh
northcua config schema --kind app > logs/app-schema.json
northcua config schema --kind batch > logs/batch-schema.json
```

`execution.max_agent_iterations_per_round` and Web `max_steps` bound work;
`max_session_steps` can additionally bound a persistent session. Model context limits,
output reserves, screenshots and semantic history are separate budgets. Evaluators
have their own step and time limits. Controller queues and create/submit/status
concurrency bound resource use independently from model-step budgets.

Use the Python models in `northcua/core/config.py`, `northcua/batch/contracts/`,
`northcua/experiments/contracts/`, `northcua_protocol/` and `northcua_device/config.py`
as the schema source of truth. Do not copy a dated production YAML as a supported API.

For local Web runs, `web.browser_channel` selects installed Chrome/Edge and
`web.browser_executable_path` selects a local binary. Remote sessions propagate only
`browser_channel`; the worker administrator can set `NORTHCUA_BROWSER_EXECUTABLE`
on the device process. Controller file paths are never sent as executable selections.

## Local browser environment

The local Playwright backend accepts `NORTHCUA_PLAYWRIGHT_BROWSER` (default
`chromium`), `NORTHCUA_PLAYWRIGHT_HEADLESS` (default `1`), and
`NORTHCUA_PLAYWRIGHT_VIEWPORT` (default `1280x720`). Use `0`, `false` or `no`
for a visible browser. The previous `NEXCUA_` spellings remain fallback aliases;
a canonical `NORTHCUA_` value takes precedence. The existing policy request
correlation header `X-NexCUA-Controller-Step-Ref` is retained for endpoint compatibility.
These are protocol compatibility names, not project branding.
