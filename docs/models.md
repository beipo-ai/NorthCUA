# Models and endpoints

NorthCUA Harness connects a model to Desktop/Web tools; model weights are not an
installation dependency. The same runtime can use a hosted endpoint or a model
served locally with a supported API.

## Published model family

Public model cards, configurations and license files were checked on **2026-11-03**.
Nominal parameter counts follow the cards and report; active parameters do not
represent the memory required to load all model weights.

| Model | Total / active parameters | Architecture | Checked revision |
|---|---|---|---|
| [NorthCUA-Nano](https://huggingface.co/china-beipo/NorthCUA-Nano) | 9B / 9B | Dense | [`e8c1a7ea38a2fb23f133ad005bd33b3f93518793`](https://huggingface.co/china-beipo/NorthCUA-Nano/tree/e8c1a7ea38a2fb23f133ad005bd33b3f93518793) |
| [NorthCUA-Mini](https://huggingface.co/china-beipo/NorthCUA-Mini) | 35B / 3B | MoE | [`cafee9af6a950bffea2a4a17544f253abf80d44b`](https://huggingface.co/china-beipo/NorthCUA-Mini/tree/cafee9af6a950bffea2a4a17544f253abf80d44b) |
| [NorthCUA-Pro](https://huggingface.co/china-beipo/NorthCUA-Pro) | 397B / 17B | MoE | [`2f1f0cbf4ed3f1243c37838889c446f50bc3d32c`](https://huggingface.co/china-beipo/NorthCUA-Pro/tree/2f1f0cbf4ed3f1243c37838889c446f50bc3d32c) |

All three cards and checkpoint license files specify Apache-2.0. Availability
and metadata are checked separately from inference and benchmark acceptance.
The report's `beipo-ai` links currently resolve to the `china-beipo` resources.

## NorthCUA-Pro research model

The [31 October 2026 Technical Report](paper/main.pdf) names
[beipo-ai/NorthCUA-Pro](https://huggingface.co/beipo-ai/NorthCUA-Pro). That address
now resolves to the public [china-beipo/NorthCUA-Pro model resource](https://huggingface.co/china-beipo/NorthCUA-Pro).
On **2026-11-03**, its card, configuration, index, license and all 122 weight-shard
URLs were publicly accessible without authentication. No weight contents were
downloaded for this check; this supersedes the 31 October HTTP 401 observation.

| Published metadata | Verified value |
|---|---|
| Pinned checkpoint | [`2f1f0cbf4ed3f1243c37838889c446f50bc3d32c`](https://huggingface.co/china-beipo/NorthCUA-Pro/tree/2f1f0cbf4ed3f1243c37838889c446f50bc3d32c) |
| Model family | Qwen3.5 MoE; `Qwen3_5MoeForConditionalGeneration` |
| Card's parameter count | 397B total / 17B active |
| Weight index | 122 safetensors shards; 793,604,721,632 bytes of tensors |
| Model license | [Apache-2.0, shipped with the checkpoint](https://huggingface.co/china-beipo/NorthCUA-Pro/blob/2f1f0cbf4ed3f1243c37838889c446f50bc3d32c/LICENSE) |

The checkpoint is a separate distribution with its own license and notices.
Its public availability does not mean this Harness has accepted its inference,
tool-call compatibility or benchmark performance. The card itself distinguishes
the exported checkpoint from the report's experiment results.

Reported benchmark results belong to the report's specified checkpoints and
protocols. Installing this Harness does not reproduce those experiments or
substitute a different checkpoint. Follow the publisher's model card for serving
requirements and model-specific settings; record the checkpoint revision.

**Serve the model separately from the Harness.** The weight tensors alone occupy
about 793.6 GB; the no-key tutorial does not download them. The export records
Transformers 5.20.2, while this Harness's locked runtime is a separate environment.
Use an inference engine that supports the checkpoint's architecture and expose
an image-capable, structured-tool API compatible with the chosen provider recipe.
This repository does not claim a tested server-launch recipe for this checkpoint.
Start with one bounded task and inspect the actual tool calls and outcome before
batching. See [resource status](resources.md).

## Minimal endpoint configuration

Use a model that supports **image inputs and structured function/tool calls**.
The default recipe is OpenAI-compatible Chat Completions. A text-only endpoint
cannot operate the visual Desktop/Web agent.

Copy [`.env.example`](../.env.example) and change the model ID and API key. Keep
the default OpenAI URL, or change the base URL to your provider/local server:

```dotenv
NORTHCUA_MODEL='your-vision-model'
OPENAI_API_KEY='your-api-key'
NORTHCUA_BASE_URL='https://api.openai.com/v1'
```

Load those variables using the appropriate [Bash or PowerShell recipe](first-run.md#2a-use-your-model-edit-only-model-connection-settings).
The CLI does not automatically source `.env`. Follow that guide's **explicit
`--model` and `--base-url` flags** when generating the two agents together;
setting `NORTHCUA_BASE_URL` alone does not override a template's literal URL.
The Bash/zsh and PowerShell commands differ, so use the recipe for your shell.
Then validate the generated app:

```sh
northcua config validate --app-config logs/my-config/app.example.yaml
```

Generated credential fields retain environment references. `--api-key-env` takes
the **name** of a variable, never its secret value. Do not commit local configs,
`.env`, keys or raw task traces.

| API interface | `--api-type` | Credential variable |
|---|---|---|
| OpenAI-compatible Chat | `openai_chat_completion` | `OPENAI_API_KEY` |
| OpenAI Responses | `openai_responses` | `OPENAI_API_KEY` |
| Anthropic Messages | `anthropic_chat_completion` | `ANTHROPIC_API_KEY` |
| Gemini REST | `gemini_rest` | `GEMINI_API_KEY` |

Provider-specific endpoint and reasoning options are documented in
[configuration](configuration.md#providers). Support for a protocol is not a
claim that every endpoint or model has passed live task acceptance.

## Match the Desktop coordinate convention

If the model outputs **0–1000 relative positions**, merge this into the app YAML
used by the task, for example `logs/my-config/app.example.yaml`:

```yaml
desktop:
  coordinate_policy:
    input_space: normalized_1000
    clamp_to_bounds: true
```

`(500, 500)` means the center of the latest grounding screenshot. The Harness
maps positions to actual pixels exactly once. `clamp_to_bounds` is optional;
it keeps edge positions inside the capture. A model that outputs screenshot
pixels should keep the default `image_pixels`.

This setting does not normalize Web point tools: those use viewport CSS pixels;
DOM/SoM refs remain identifiers. Read the [full coordinate contract](configuration.md#desktop-coordinates).

## Start small, then scale

Validate configuration, run one task with an explicit step budget, and inspect
its actual outcome and artifacts. Then use [local batch](local-batch.md) or the
[recommended Desktop settings](desktop-settings-osworld2.md). Their example sampling and
retry parameters are explicit experiment recipes, not universal model defaults.

Without credentials, the [no-key tutorial](first-run.md#2b-verify-locally-without-a-model-key)
checks the real runtime with a deterministic local model. Its success is an
integration result, not autonomous model capability.
