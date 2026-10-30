# From zero to a verified task

Choose one path: use a hosted/local vision model for your own work, or use the
credential-free tutorial to verify installation first. The tutorial runs real
NexAU and real GUI/browser tools against a **scripted model endpoint**; it does
not measure autonomous model quality. No E2B, NorthJetty, private dataset or
remote filesystem is needed.

## 1. Install once

Use Python **3.12** (the tested baseline), Git and [uv 0.13.6+](https://docs.astral.sh/uv/).
Clone the public repository; skip the first two commands if you already have a
checkout or an extracted source archive. These commands work in Bash and PowerShell:

```sh
git clone --branch main https://github.com/china-qijizhifeng/NorthCUA-Harness.git
cd NorthCUA-Harness
uv sync --locked --python 3.12 --extra local --extra device
uv run --no-sync playwright install chromium
```

Activate the environment in every terminal you use. On Linux/macOS with Bash or zsh:

```sh
source .venv/bin/activate
northcua --help
```

On Windows with PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
northcua --help
```

If your Windows shell prevents activation, run `uv run --no-sync northcua ...`
instead of the bare `northcua ...` commands below; changing the machine's script
execution policy is not required.

On Linux install browser system libraries with
`uv run --no-sync playwright install --with-deps chromium`. Local browser tasks
are separate from the macOS/Linux-only local batch tutorial. See [installation](getting-started.md)
for wheel installation and [troubleshooting](troubleshooting.md) for diagnostics.

All following terminals use the same checkout and activated environment. Commands
are also available as `python -m northcua ...`. Installation requires downloading
public packages and a browser; tutorial execution subsequently needs only loopback.

## 2A. Use your model: edit only model connection settings

The required model settings are the **model ID and API key**. Keep the template's
OpenAI base URL or set your endpoint's URL. The model must accept images and
structured tool calls. See [model configuration and availability](models.md)
before choosing an endpoint; a listed research model is not proof that its weights
are currently downloadable. [Project resources](resources.md) tracks public links.

On Linux/macOS, copy [`.env.example`](../.env.example), edit those values, then
load it explicitly. NorthCUA does not automatically source `.env`:

```sh
cp .env.example .env
# Edit .env before the next command. Keep it private.
set -a
. ./.env
set +a
northcua config init --directory logs/my-config \
  --model "$NORTHCUA_MODEL" --base-url "$NORTHCUA_BASE_URL"
```

On Windows PowerShell, set the same values in the launching terminal. The API-key
prompt avoids putting the secret into a command or a configuration file:

```powershell
$env:NORTHCUA_MODEL = 'your-vision-model'
$env:OPENAI_API_KEY = [System.Net.NetworkCredential]::new('', (Read-Host 'Model API key' -AsSecureString)).Password
$env:NORTHCUA_BASE_URL = 'https://api.openai.com/v1'
northcua config init --directory logs/my-config --model "$env:NORTHCUA_MODEL" --base-url "$env:NORTHCUA_BASE_URL"
```

Then validate and run one bounded Web task in either shell:

```sh
northcua config validate --app-config logs/my-config/app.example.yaml
northcua local web run --app-config logs/my-config/app.example.yaml --url https://example.com --query 'Report the main heading.' --max-steps 8 --session-id first-web
```

`config init` creates both agents with the same connection settings and keeps the
key as an environment reference. It refuses existing files unless `--force` is
explicitly supplied. The selected model must accept images and structured tool
calls. Do not put an API key in `--api-key-env`: that option takes a variable name.
[Configuration recipes](configuration.md) cover Responses, Anthropic and Gemini.

For Desktop, grant the launching terminal Screen Recording and Accessibility on
macOS; Linux needs X11 and `DISPLAY`; Windows needs an unlocked interactive desktop.
Run only one Desktop task on a physical display.

If your Desktop model returns **0–1000 relative coordinates**, first edit the
existing `desktop` section in `logs/my-config/app.example.yaml`:

```yaml
desktop:
  coordinate_policy:
    input_space: normalized_1000
    clamp_to_bounds: true
```

The model returns relative positions and the Harness maps them to pixels. Keep
the default `image_pixels` for a model that returns screenshot pixels. See
[coordinate configuration](configuration.md#desktop-coordinates)
for the convention and Web's separate CSS-pixel/ref tools. Then run:

```sh
northcua local desktop run --app-config logs/my-config/app.example.yaml --query 'Open a calculator and compute 12 * 7.' --max-steps 15 --session-id first-desktop
```

Desktop has one observation path: GUI tools return screenshots. No observation
mode, driver or private service needs configuring. Read the terminal result and
inspect its artifact path; a model claiming completion is not by itself proof
that the requested outcome happened.

## 2B. Verify locally without a model key

In terminal A, keep this service running:

```sh
northcua demo serve --directory logs/tutorial
```

Wait for a JSON line with `"ready": true`. This creates app/agent configuration,
`batch.yaml`, `demo.json` and a loopback-only service at `http://127.0.0.1:8787`.
It refuses a nonempty directory so it cannot overwrite earlier evidence. Choose
another directory for a new tutorial run. No API environment variables are needed.

In terminal B:

```sh
northcua config validate --app-config logs/tutorial/app.example.yaml
northcua local web run --app-config logs/tutorial/app.example.yaml --url http://127.0.0.1:8787/task/first-web --query 'Task ID: first-web. Activate the button and verify the receipt.' --session-id first-web --max-steps 6 --evaluate
```

Success means all of the following: the command exits 0; the actor finishes; the
evaluation report says `success`; `logs/tutorial/events.jsonl` contains a
`web_action` for `first-web` and an evaluation with `verified: true`. The check
matches the task ID in the query against that page's task-specific receipt.
Keep `Task ID: first-web` aligned with `/task/first-web` when using the scripted endpoint.
The evaluator reads the changed page in the **same live browser session**.
Reopening the page does not retain the receipt.

## 3. Verify native Desktop input

This section is optional for Web-only setup. It sends real keyboard input to a
disposable fixture window; it is not a headless or simulated Desktop test.
Stop terminal A with Ctrl-C, then start a fresh fixture directory with a native form:

```sh
northcua demo serve --directory logs/tutorial-desktop --desktop
```

This needs Tk (`python -c 'import tkinter'`) and the desktop permissions above.
A small **NorthCUA tutorial — disposable form** window must be visible. Leave that
window available and avoid other mouse/keyboard work during this task. The demo
focuses only its own form before the actor types.

In terminal B:

```sh
northcua local desktop run --app-config logs/tutorial-desktop/app.example.yaml --query 'Fill the disposable tutorial form and press Enter.' --session-id first-desktop --max-steps 6
```

Check the receipt in Bash/zsh:

```sh
python - <<'PY'
import json
from pathlib import Path
p = Path('logs/tutorial-desktop')
expected = json.loads((p / 'demo.json').read_text())['desktop_text']
assert (p / 'desktop-result.txt').read_text() == expected
print('PASS: native form contains this run’s expected text')
PY
```

Or in PowerShell:

```powershell
$demo = Get-Content -Raw logs/tutorial-desktop/demo.json | ConvertFrom-Json
$actual = [System.IO.File]::ReadAllText('logs/tutorial-desktop/desktop-result.txt')
if ($actual -cne $demo.desktop_text) { throw 'The native form receipt does not match this run' }
Write-Output 'PASS: native form contains this run''s expected text'
```

The actor goes through NexAU → note/type/keypress → native tools. Check the saved
screenshots and terminal status too. Desktop evaluation is off in this smoke
configuration; the native form receipt is the task-specific outcome check.

## 4. Find evidence and stop

The CLI prints the artifact directory. Demo agent artifacts live under the
generated directory's `artifacts/`; `events.jsonl` records only fixture activity,
not a replacement for the full trace. [Results and evaluation](evaluation.md)
explain actor status, evaluation verdict and the optional viewer.

Ctrl-C the demo terminal to close the fixture server and its disposable window.
Keep output directories for inspection. Continue with [local batch and
recovery](local-batch.md), [service deployment](running.md), or
[Agent skills](skills.md).
