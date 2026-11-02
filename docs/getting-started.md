# Installation and first task

For the complete path including model templates, expected outputs and local
credential-free checks, start with [From zero to a verified task](first-run.md).

Install the NorthCUA Harness provided by the NorthCUA project. Its packages and command
remain named `northcua`; the project branding does not change installation or imports.

Before installation, read the [known NexAU/Transformers dependency risk](security/dependencies.md).
The pinned upstream constraint is retained under an explicit, expiring policy.

Use Python 3.12+ and [uv 0.13.6+](https://docs.astral.sh/uv/). Start in the repository checkout
shown in the [README](../README.md#quick-start), or in an extracted source archive.

```sh
uv sync --locked --extra local --extra device
uv run --no-sync playwright install chromium
export OPENAI_API_KEY='your-key'
export NORTHCUA_MODEL='your-vision-model'
uv run --no-sync northcua config validate --app-config config/app.example.yaml
```

The agent needs a model that accepts images and structured tool calls. The sample uses
OpenAI-compatible Chat; [configuration](configuration.md#providers) covers other interfaces.
`config validate` resolves both agent configurations without making a model call.

## Choose the installed components

| Extra | Purpose |
|---|---|
| `local` | The pinned public NexAU runtime for model execution |
| `device` | The `northcua-device` worker command in a source checkout |
| `e2b` | Cloud sandbox provisioning adapters |
| `viewer` | Optional Streamlit trace viewer; use with `local` |
| `dev` | Tests, lint, type checking and wheel-build tools |

The base package includes controller and runtime bridges. The device distribution depends
on those bridges, but a remote tool-only device does not need NexAU. When using `uv sync`,
include all extras you want to retain. Examples use `uv run --no-sync` after installation
so a later command does not change the selected environment.

A browser binary is a separate installation. On Linux, use
`uv run --no-sync playwright install --with-deps chromium` to install system dependencies.
For the bare commands in operational guides, activate the virtual environment first:

```sh
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
```

## Run Web

```sh
uv run --no-sync northcua local web run --app-config config/app.example.yaml \
  --url https://example.com --query 'What is the main heading?' --max-steps 8
```

Use `--headed` to inspect the browser. Each run creates its own browser context; personal
browser profiles are not reused. Set `web.browser_channel: chrome` or `msedge` to select
an installed channel, or keep the default Playwright-managed browser. The sample permits
localhost URLs; private-network and file URLs require explicit configuration.

## Run Desktop

```sh
uv run --no-sync northcua local desktop run --app-config config/app.example.yaml \
  --query 'Open a calculator and compute 12 * 7.' --max-steps 15
```

Use an interactive desktop and run one agent per physical display. macOS requires Screen
Recording and Accessibility permissions for the host launching Python. Linux uses X11
and `DISPLAY` (Xvfb for isolated CI tests); a Wayland-only session is outside the native
backend contract. Windows requires an unlocked interactive session. See the
[verification matrix](verification.md#execution-matrix) for platform evidence.

## Read the outcome

The CLI prints the task's terminal status and artifact locations. A step-budget limit or
a failed actor produces a nonzero exit status. Evaluation has a separate verdict.
Enable it with `--evaluate` for Web or `evaluation.enabled: true` in app configuration.
[The evaluation guide](evaluation.md) explains live probes and how to open saved traces.

Run artifacts default to `logs/` under the working directory. Agent-config paths are
resolved relative to app YAML. Use `--log-dir /absolute/path` to choose another root.
Prompt templates and example configurations are packaged resources, so wheel installs
work outside the source tree.

## Install the four release wheels

The project supports source and wheel installation; it does not assume that version
0.2.0 has already been published to PyPI. Build from the checkout:

```sh
uv sync --locked --extra local --extra device --extra dev
uv run --no-sync python -m scripts.build_wheelhouse
```

Copy all four wheels from this build into a fresh Python environment on the destination:

```sh
python -m pip install /path/to/wheelhouse/*.whl
python -m pip install 'northcua[local]==0.2.0' --find-links /path/to/wheelhouse
python -m playwright install chromium
northcua config init --directory my-config
northcua config validate --app-config my-config/app.example.yaml
```

Set the model environment variables before validation. The `local` extra fetches public
NexAU at the pinned Git commit, so Git and network access are needed. Keep all four wheels
from one release together. Use `my-config/app.example.yaml` for subsequent run commands.

Continue with [execution modes](running.md), [context and memory](context.md), or
[batch execution](batch.md).
