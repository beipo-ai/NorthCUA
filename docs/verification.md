# NorthCUA Harness verification

The [0.2.0 release record](release-0.2.0.md) records the original execution,
distribution and independent review evidence, with a [3 November publication recheck](release-0.2.0.md#3-november-publication-recheck).
The [core Harness scope update](release-0.2.0.md#core-harness-scope-update)
records the adapter removal and resource migration. The subsequent
[ordinary-prompt and short-showcase update](release-0.2.0.md#one-desktop-prompt-and-short-showcases)
records the latest scope cleanup.
The [canonical CI](https://github.com/beipo-ai/NorthCUA/actions/workflows/ci.yml)
shows the status for each subsequent commit. Follow [release preparation](releasing.md)
before publishing another candidate.

## Execution matrix

Use the [release execution matrix](release-0.2.0.md#execution-matrix). Reproduce
checks with [testing](testing.md), [first run](first-run.md) and
[local batch](local-batch.md). A scripted endpoint verifies real tool integration;
it does not establish the capability of a hosted research model.

## Historical audits

These records preserve implementation provenance from the former repository;
their revisions, earlier report snapshots and publication status are historical.

- [Initial clean-scope audit, 3 October](archive/verification-2026-10-03.md)
- [Operator, native and batch audit, 13 October](archive/verification-2026-10-13.md)
- [NexAU source migration, 31 October](archive/nexau-migration-2026-10-31.md)
- [Desktop prompt/response-policy intake, 31 October](archive/agent-intake-2026-10-31.md)
