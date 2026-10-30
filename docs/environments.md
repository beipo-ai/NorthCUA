# Public environments

## E2B

The two [template definitions](../examples/templates/) install the four locally built
release wheels. Desktop extends E2B's public `desktop` template; Web uses the public
Python 3.12 base. No historical NorthCUA account-specific template is required.
The upstream desktop substrate can also be rebuilt from the
[E2B desktop template source and guide](https://github.com/e2b-dev/desktop/tree/main/template).

```sh
uv sync --locked --extra local --extra dev --extra e2b
uv run --no-sync python -m scripts.build_wheelhouse
export E2B_API_KEY='your-key'
uv run --no-sync northcua sandbox template build --help
uv run --no-sync northcua sandbox template build --spec examples/templates/web.yaml
uv run --no-sync northcua sandbox template build --spec examples/templates/desktop.yaml
```

Run builds from the repository root so the `wheelhouse/` copy is in the build context.
The cloud account needs permission to build the chosen aliases. Template build checks
validate metadata without requiring a running display; a live canary still must verify
screenshots, input, worker readiness and cleanup. A successful image build is not that canary.

For wheel-only controller installations set `NORTHCUA_RUNTIME_WHEELHOUSE` to a directory
containing exactly one wheel for each of the four distributions. High-level experiments
using `runtime.mode: upload_wheels` stage that release without needing source files or
`pyproject.toml` under site-packages. Dependency installation happens inside the sandbox;
its network/package access must be configured by the deployment.

Current native templates use Linux X11. Template app catalogs are task-specific:
install the application, declare its launch/readiness requirements, and provide its
fixtures and oracle. The bundled catalog does not advertise uninstalled office or
proprietary applications. The [verification record](verification.md) states whether a
real cloud build/run has been completed for this candidate.

## OSWorld and OSWorld2

Install the desired upstream benchmark and configure its DesktopEnv provider separately.
NorthCUA accepts local upstream task JSON and constructor kwargs for that environment:

```sh
python -m northcua.integrations.osworld \
  --task /path/to/benchmark/task.json \
  --environment-config /path/to/desktop-env.json \
  --app-config examples/osworld/app.osworld.example.yaml \
  --output-dir logs/osworld --max-steps 50
```

An environment config is the JSON equivalent of arguments to the selected benchmark's
`DesktopEnv(...)`, for example its provider name, VM path and headless setting. Repeat
`--task` to run additional tasks. Each task creates a new environment. For concurrency,
use separate processes and provider resources; do not share a physical display.

The adapter resets from the complete upstream task JSON, copies its VM IP/server port
into NorthCUA's OSWorld HTTP backend, runs the Desktop actor, records a literal DONE or
FAIL only when the actor makes that terminal decision, invokes `env.evaluate()`, saves
the score and then releases the environment. Step exhaustion is not an infeasibility
claim. Infrastructure failure is reported separately from an oracle score.

The API boundary follows upstream
[DesktopEnv reset/step/evaluate](https://github.com/xlang-ai/OSWorld/blob/main/desktop_env/desktop_env.py).
Use the matching OSWorld2 environment for OSWorld2 task definitions; this adapter does
not reimplement benchmark metrics or claim a benchmark score without a real run.
No Langfuse dataset, private source checkout path or hosted task inventory is required.
