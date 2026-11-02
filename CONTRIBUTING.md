# Contributing to NorthCUA

The current release provides the NorthCUA Harness. Contributions should make
Desktop/Web tasks easier to run, inspect or reproduce. The
[architecture](docs/architecture.md) explains ownership boundaries; the
[clean scope](docs/clean-scope.md) describes what belongs in this release.
Project discussions follow the [community conduct guidelines](CODE_OF_CONDUCT.md).

## Development setup

Use Python 3.12+, uv 0.13.6+ and the committed dependency lock:

```sh
uv sync --locked --extra local --extra device --extra dev --extra e2b --extra viewer
uv run --no-sync pytest -q
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pyright --pythonpath .venv/bin/python
```

[Testing](docs/testing.md) lists opt-in browser/native checks and wheel-only validation.
The default suite does not call paid model endpoints or create cloud sandboxes.
Run `uvx --from pre-commit==4.7.0 pre-commit run --all-files` for repository hooks.
This tool runs separately from the project environment; see [release preparation](docs/releasing.md)
for permanent Git-hook installation.

## Before a pull request

Describe the user-visible problem and the final behavior. Include the commands you ran
and their outcomes. Identify tests that use mocks or deterministic model endpoints, and
name any real deployment modes still unverified.

Changes to resource ownership need failure/cancellation coverage. Tool schema changes need
a review of the model-visible contract, not just regenerated snapshots. When changing a
paper mechanism, update its operational guide and the [paper map](docs/paper-map.md).
Keep operational examples small, public and reproducible. Historical deployment IDs and
one-off migration scripts do not belong in supported examples.

NexAU adapters preserve documented message, formatting and runtime behavior. Removing one
requires evidence that its callers and behavioral contract are covered by the replacement.
Avoid broad refactors mixed with unrelated feature changes.

## Report a problem

Include this repository's commit, Python/OS versions, installation method, execution mode,
redacted configuration and a minimal task. Link a small relevant trace if possible.
Remove credentials, screenshots of personal data and private file contents first.
Use [the security policy](SECURITY.md) for sensitive vulnerabilities.

## Contact

For contribution questions, contact the project maintainers:

- [zhangchaoyun@qijizhifeng.com](mailto:zhangchaoyun@qijizhifeng.com)
- [libingxuan@qijizhifeng.com](mailto:libingxuan@qijizhifeng.com)

## Release review

Independent runtime and release reviews check lifecycle/cleanup, installability, public
dependencies, docs/CLI consistency and the exact evidence behind capability claims.
A signoff states its scope. Local tests, remote CI, hosted-model evaluation and cloud/OS
acceptance are separate gates. Update the [verification record](docs/verification.md)
from actual output and preserve earlier dated evidence when a new result supersedes it.

Contributions are made under the repository's [Apache-2.0 license](LICENSE). Preserve
applicable copyright and third-party notices.
