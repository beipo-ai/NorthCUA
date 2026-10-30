# Tests and release gates

The public onboarding workflow has real browser integration coverage:

```sh
NORTHCUA_RUN_BROWSER_TESTS=1 uv run --no-sync pytest tests/integration/test_local_batch.py -q
```

It runs three real browser tasks through both controller executor modes, requires
live evaluation receipts, checks worker cleanup, then resumes without repeating
successful tasks. Unit checks cover configuration generation, ownership boundaries,
supervisor expiry/stop/parent death and a failing evaluator receipt. CI runs these
alongside existing browser and native checks. The fixture model is scripted.

```sh
uv sync --locked --extra local --extra device --extra dev --extra e2b --extra viewer
uv run --no-sync pytest -q
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pyright --pythonpath .venv/bin/python
uv run --no-sync python -m scripts.build_wheelhouse
```

The default suite is deterministic. It covers config/contracts, four provider protocols,
context/history, agent/evaluator lifecycle, controller/device transport, deadlines,
claims/resume, prepared environments, artifact custody and sandbox cleanup. Provider
wire/stream fixtures do not call paid APIs. Fake sandboxes do not prove cloud execution.

## Real browser checks

```sh
uv run --no-sync playwright install chromium
NORTHCUA_RUN_BROWSER_TESTS=1 uv run --no-sync pytest tests/integration/test_web_modes.py -q
uv run --no-sync python -m tests.support.web_agent_playwright_e2e \
  --local-fixtures --fail-on-error --output logs/browser-report.json
```

The integration model is a deterministic local HTTP endpoint used through the real NexAU
provider path. The actor changes page memory and the evaluator must see that same state;
a replacement browser/reload cannot pass. The backend suite exercises local pages,
refs, form state, tabs, screenshots, live probes and actionability failures. Cleanup or
probe failures fail the test. An installed Chrome/Edge channel can be selected explicitly;
managed Playwright binaries are the default CI path.

The Web suite checks application-local, CLI, local WebSocket serve and both controller
executor modes. Native Desktop input can be tested explicitly:

```sh
NORTHCUA_RUN_NATIVE_TESTS=1 uv run --no-sync pytest tests/integration/test_native_desktop.py -q
```

Native smoke tests require an interactive display. Linux CI uses Xvfb. macOS/Windows
physical input checks are separate manual/opt-in checks and require a usable session. Never count
`--help`, metadata-only doctor output or mocked backend tests as native input acceptance.

## CI/CD

CI is configured for pull requests and pushes to main, plus manual dispatch and
reusable release calls. Defining a job is not evidence that a hosted
runner executed it; consult the verification record for observed results.

CI checks locked installation, unit/contract tests, format/lint, scoped type checking,
builds all four distributions, tests installed wheels outside the source tree, and runs
real browser/native integration on Linux. CI artifacts include JUnit reports and the
wheelhouse. The wheel lane executes the five Web entry-point checks and both
local batch executor modes outside the checkout.
Release automation builds a versioned candidate and checks the artifact, and includes a source archive;
publication is a deliberate maintainer action, not a side effect of tests.

A release reviewer checks package resources, direct dependencies, docs/CLI consistency,
license/citation, absence of private defaults, failure/cancellation behaviour and the
paper map. Runtime and release signoffs name their tested scope. Live model providers,
E2B, Windows and macOS device modes only receive signoff after their own evidence exists.
The candidate's actual state is recorded in [verification](verification.md).

## Updating tests

Change tests for removed APIs only when the API was deliberately removed. Preserve
mixed-file tests for surviving functionality. Add failure-oriented tests for new runtime
ownership or persistence behaviour. Tool schema snapshot changes need review of the
underlying schema; never update opaque hashes merely to silence a failure.
