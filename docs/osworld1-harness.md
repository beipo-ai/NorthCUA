# OSWorld1 formal harness

This is an **actor-only integration**, not an OSWorld distribution or evaluation
launcher. It publishes the formal GUI actor's prompt, request/history policy,
coordinate projection and bounded response recovery in the clean NorthCUA
runtime. It does not ship tasks, task-specific evaluators, repairs, images,
websites, setup assets, credentials, results, tracing databases or dispatch/stop
scripts. The separate OSWorld2 integration is unchanged.

## One prompt

`northcua/integrations/osworld1/prompt.j2` is the **only OSWorld1 formal prompt
source**. It contains the frozen system, persistent-user and observation
sections in one file. These still render to their original message roles;
combining source files must not combine or reorder API messages. Select
`prompt.system_prompt_profile: osworld1_formal`. There is no sampling/skill
variant, second "optimized" prompt, or OSWorld2 ask-user instruction.

The versioned content is imported from commit `648804d19`; its SHA test detects
accidental changes to this export. It is not independent verification against
an external campaign's training or evaluation files. The public disposable benchmark account/password
mentioned inside it is benchmark instruction text, not a deployment credential.
Do not use this profile to control an untrusted personal desktop.

## Actor configuration and retained improvements

The two YAML files in `examples/osworld1/` contain only portable harness settings.
Model, API endpoint and key are environment references. They are not launch
scripts and do not provision a desktop or select task data.

| Contract | Behavior |
|---|---|
| Budget | 100 logical actor steps; model calls/retries do not reset this budget |
| Sampling example | Chat Completions, temperature 0.7, reasoning effort max, 16000 output tokens, 120s SDK timeout, max_retries 3 |
| History | 100 turns, three-image sliding overlay/clean window, action history and Chat reasoning replay |
| Tools | Existing DesktopNativeActionExecutor; no shell/browser/ask_user tools added to the model-visible surface |
| Tool errors | Existing stop policy feeds failed tool results into the next step instead of immediately killing the task |
| Response recovery | `execution.model_response_recovery: true`: initial call plus up to `llm_config.max_retries` same-request retries after transient/empty responses or token-exhausted reasoning-only output |
| Backoff | 2, 4, 8 seconds for three retries, plus bounded jitter; max delay 30s plus jitter |
| Deadline | Historical 3600s guard per outer attempt; configured SDK timeout/retries remain separate |
| Terminal text | A valid text-only response is not retried merely for lacking tools; OSWorld2's three-call rule is off |
| Scoring | External original whole-task scorer; preserve fractional/zero scores, exclude execution/evaluator errors |

Recovery applies only to the prepared actor for its current call. It does not
patch the global SDK, retry auxiliary models, execute tools twice, change model
parameters or add failed replies to history. Cancellation and hard deadlines
propagate. Exhaustion stays an execution error in this clean runtime, not a
synthetic success or an evaluator zero. The older private runner's empty-result
sentinel is not a reliable completion signal and is not exported as one.

Each recovery attempt gets a deep copy of the prepared messages and tools, so
mutations from a failed attempt do not reach later attempts or caller history.
An explicit HTTP status takes priority over error-text matching: 429 and 5xx can
retry; permanent 4xx errors cannot. SDK transport retries remain separate and can
increase the number of network requests. Do not combine this policy with
`no_tool_call_attempts: 3`; the configuration validator rejects that combination.

### Coordinate contract

For `normalized_1000`, retain the historical virtual `1000 x 1000` prompt canvas
and pixel wording used in training; this is intentional, not a claim that image
bytes were resized. Actual screenshots remain at their native resolution.
The existing coordinate mapper converts model positions to real pixels exactly
once on action dispatch. Scroll wheel deltas are not positions.

OSWorld1-only text projection also rewrites source-resolution literals and
coordinate-like literals in persistent/observation/history text. It never
rewrites structured tool-call arguments, image bytes or reasoning blocks. This
is a legacy regex contract, not semantic parsing: it can also rewrite matching
coordinate literals in task prose. It is preserved for historical comparability,
not silently "fixed" during export. The historical no-image fallback is
1920x1080. `image_pixels` uses the same prompt file without that text conversion
or virtual resolution. Choose the coordinate policy explicitly for the model.

## API boundary

`northcua.integrations.osworld1.run_task` accepts an already prepared
`DesktopApplication`, a unique session/task identity, the original query and an
async `evaluate(action)` callback. It calls the same app for the whole task.
The callback receives `"DONE"` for completion, `"FAIL"` for an explicit
infeasibility decision, otherwise `None` (including a normal step limit).
The caller integrates those terminal hints with its own original evaluator.
Tool errors, actor/API errors and a budget limit must not be silently equated
with a model's explicit `[INFEASIBLE]` claim.

The adapter neither creates, resets, closes nor releases the caller's environment.
It does not change VM routing, fetch task JSON, run setup or schedule a batch.
The pre-existing generic `integrations.osworld.run_osworld_task` example is a
different lifecycle API; it does not select this formal profile automatically.

Reasoning replay is selected with the clean runtime's existing
YAML field `llm_config.reasoning_replay_policy: chat_reasoning_content`
(loaded into `LLMConfig.extra_params` by the config loader).
Model/provider-specific overrides belong in the caller's untracked config;
do not substitute OSWorld2's prompt or model-specific rules.

## Verification

Run the deterministic tests (no model/API/VM required):

```sh
uv run --no-sync pytest tests/integrations/osworld1 tests/integrations/osworld2 \
  tests/cua/test_stop_policy.py tests/core/test_coordinates.py -q
```

Tests cover the single packaged template, role preservation, pixel versus
normalized projection, unmodified tool arguments/images, reasoning replay,
bounded retry/backoff/exhaustion/cancellation, original query and budget,
terminal hints and error score exclusion. These tests are not a new hosted
model or fresh-VM benchmark run. Environment validation remains the caller's
responsibility; no environment or production deployment acceptance is claimed.
