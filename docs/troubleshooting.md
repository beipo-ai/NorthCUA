# Diagnose a run

Start with the failed command's exit status and printed artifact path. Keep the
original evidence; rerun in a new output directory after correcting the cause.
Report Python/OS, source commit, installation method, mode and redacted config.

| Symptom | Check and action |
|---|---|
| `northcua` missing or import failure | Activate the environment used by `uv sync`; compare `python -c 'import sys; print(sys.executable)'` and `python -m northcua --help`. Reinstall from the locked source or matching wheels, not another private checkout. |
| NexAU not installed | Source: `uv sync --locked --extra local --extra device`. Keep all desired extras on later sync commands; use `uv run --no-sync` to run. |
| Playwright executable missing | In that environment run `python -m playwright install chromium`; Linux may need `--with-deps`. A Python package install does not install the browser binary. |
| Missing model/key variable | Source your private `.env` explicitly. `config validate` requires referenced variables but makes no provider call. Never paste the key into an issue. |
| 401/403 or model not found | Check key, exact provider model ID and API base URL. A Chat endpoint commonly ends in `/v1`; do not append `/chat/completions` to `base_url`. |
| Unsupported images/tool calls | Select a model/endpoint supporting both. Check `api_type` and the provider's supported parameters; lowering an unrelated screenshot budget will not fix protocol incompatibility. |
| Blank Desktop screenshots / no input | Confirm the interactive session and permissions of the terminal launching Python. On macOS grant Screen Recording and Accessibility; Linux native input needs X11. |
| Demo native window fails to open | Check `python -c 'import tkinter'`; install a Python distribution/Tk package for your OS. Only use `--desktop` with an interactive display (or Linux Xvfb). |
| Demo address already in use | Stop your previous demo or use another `--port`; use its generated URLs. The demo refuses nonempty output directories to protect earlier evidence. |
| Task waits for a device | Check controller and worker logs, endpoint, authentication, platform and agent kind. Local batch requires the exact running controller's absolute SQLite path in its manifest. |
| Local batch worker fails | Inspect `sandbox-logs/<task>/attempt-*/worker.log`. Ensure the worker uses an installed environment and its browser cache is available; the local provider uses the runner's Python. |
| A batch is drained or paused | Inspect `batch status` and the saved state. Wait for the earlier runner to exit, then `batch resume`; do not start two runners in one output directory. |
| Resume rejects manifest/config identity | Use the original saved batch. Start a new batch ID/output root for different tasks or controller identity; do not force edits into SQLite. |
| Actor finished but evaluation failed | Read both reports. Live probes need `evaluation.tools.allow_client_live_probe`; artifact-only evaluation cannot recreate vanished page state. |
| Cleanup failed | Retain the allocation ID and logs. Local supervisor receipts live in the OS temp `northcua-local-<uid>/local-<uuid>/` directory. Require `finished.json` to contain `cleanup_confirmed: true`; its existence or a requested stop alone is not confirmation. Inspect `cleanup_error`, `remaining_pids` and `unreadable_pids`. A killed supervisor may leave workers needing manual, ownership-verified cleanup. |

## Useful diagnostics

```sh
northcua config validate --app-config logs/my-config/app.example.yaml
northcua config schema --kind app > logs/app-schema.json
northcua config schema --kind batch > logs/batch-schema.json
northcua-device doctor --help
northcua provider probe --help
northcua batch status --batch-id local-web --output-dir logs/batch-tutorial/batches
northcua batch report --batch-id local-web --output-dir logs/batch-tutorial/batches
```

Provider probes may call the configured endpoint; configuration validation does
not. Use [the local scripted tutorial](first-run.md#2b-verify-locally-without-a-model-key)
to separate browser/runtime setup from hosted-model connectivity.

On macOS, a virtual environment whose `.pth` files have a filesystem `hidden`
flag can lose editable imports because Python skips those files. Inspect flags
with `ls -lO .venv/lib/python3.12/site-packages/*.pth`. Recreate the environment
in a directory not being modified by a hiding/sync utility; do not work around
this by changing the project's Python import paths. For example, set
`UV_PROJECT_ENVIRONMENT=/absolute/path/to/a/new/venv` when running `uv sync`, then
activate that environment explicitly in every terminal.

## Installer version on macOS

Use `uv --version` to confirm **0.13.6 or newer**. Older uv can interpret
PyObjC's Darwin version markers as strings, omit six frameworks and still
report a compatible environment. Update uv using its installation method,
then repeat the locked install with all your selected extras. The project
enforces this minimum for source installation; CI uses 0.13.6.
