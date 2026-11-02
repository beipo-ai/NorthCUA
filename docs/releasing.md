# Preparing a source release

The canonical repository is [beipo-ai/NorthCUA](https://github.com/beipo-ai/NorthCUA).
The four distributions share one version. Source publication, package publication,
model-weight publication and website deployment are separate operations.

## Validate the candidate

Use Python 3.12 and uv 0.13.6+ in the checkout:

```sh
uv sync --locked --extra local --extra device --extra dev --extra e2b --extra viewer
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pyright --pythonpath .venv/bin/python
uv run --no-sync pytest -q
uv run --no-sync python -m scripts.audit_dependencies
uvx --from pre-commit==4.7.0 pre-commit run --all-files
uv run --no-sync python -m scripts.build_wheelhouse
```

Check the exact candidate commit in [GitHub Actions](https://github.com/beipo-ai/NorthCUA/actions/workflows/ci.yml).
Require all five jobs: publication security, contracts, browser/native integration,
Linux wheels and macOS wheels. A cancelled or older successful run does not count
for a new candidate. The reusable CI also gates the release-candidate workflow.
See [testing](testing.md) for installed-wheel and optional platform checks.

Have another reviewer follow the [operator skill](../skills/northcua/SKILL.md)
from a fresh environment. Verify a real browser task, three-task local batch,
completed resume without replay and all worker cleanup receipts. Preserve the
distinction between a scripted local model and actual hosted-model acceptance.

## Review the public contents

- Scan fetched Git history and a tracked export, including files removed from HEAD.
  CI uses the checksum-pinned Gitleaks binary with redacted reports; there is no
  blanket secret allowlist. Review unexpected findings instead of suppressing them.
- Inspect PDF text/metadata, report and logo hashes, media, attribution, package
  metadata and wheel contents. Text scans do not prove that pixels or audio are
  free of private information. Keep the audit's actual coverage explicit.
- Exclude `.env`, private configs, inventories, raw traces, build environments and
  unrelated website code. Only synthetic credentials belong in security fixtures.
- Check public model-card URLs, pinned revisions and license files separately.
  Do not present report scores as measurements of an untested exported checkpoint.
- Keep English/Chinese README instructions aligned, and resolve local links and
  anchors. Include license/notices, citation, contribution, conduct and private
  security-reporting contacts.

The maintainer has accepted a narrow [Transformers risk exception](security/dependencies.md)
while retaining public NexAU. Do not describe a passing policy gate as zero known
vulnerabilities. Revisit the decision before the recorded expiration date.

The [security policy](../SECURITY.md) describes runtime boundaries and reporting.

## Publish deliberately

Confirm the canonical URL, repository visibility and anonymous access before an
announcement. Repository description, topics, badges and clone commands should
point at the same project. A migration-only repository should direct readers here
and retain no active code branch; back it up before replacing its default branch.

If a version tag is published, it must exactly match the package version, for
example `v0.2.0`. The release-candidate workflow produces checked wheels, source,
documentation and a checksum manifest. It does not upload to PyPI, publish model
weights or deploy a website. Review those artifacts before distributing them.

Pre-commit runs in its own tool environment so its environment builder does not
conflict with NexAU's runtime dependencies. For permanent local Git hooks, use
`uv tool install pre-commit==4.7.0` followed by `pre-commit install`; reinstall hooks
if the tool environment is removed. Do not use dependency overrides to force
an incompatible runtime installation.

Record the executed checks and limitations in [verification](verification.md).
Retain failed diagnostic runs and their fixes rather than counting them as passes.
