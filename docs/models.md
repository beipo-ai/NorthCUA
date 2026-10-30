# Models and endpoints

NorthCUA Harness connects a model to Desktop/Web tools; model weights are not an
installation dependency. The same runtime can use a hosted endpoint or a model
served locally with a supported API.

## NorthCUA-Pro research model

The [31 October 2026 Technical Report](paper/main.pdf) names
[beipo-ai/NorthCUA-Pro](https://huggingface.co/beipo-ai/NorthCUA-Pro) as its Hugging
Face resource. At the release check on **2026-10-31**, both the model page and its
public API returned HTTP 401. Public access, checkpoint revision and weight
license are therefore **not confirmed by this release**. Do not treat the
Harness's Apache-2.0 license as a model-weight license. See [resource status](resources.md).

Reported benchmark results belong to the report's specified checkpoints and
protocols. Installing this Harness does not reproduce those experiments or
substitute a different checkpoint. Once weights are accessible, follow the
publisher's model card for the exact architecture, serving requirements,
license and model-specific settings; record the checkpoint revision.

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
[optional benchmark profiles](osworld2-harness.md). Their example sampling and
retry parameters are explicit experiment recipes, not universal model defaults.

Without credentials, the [no-key tutorial](first-run.md#2b-verify-locally-without-a-model-key)
checks the real runtime with a deterministic local model. Its success is an
integration result, not autonomous model capability.
