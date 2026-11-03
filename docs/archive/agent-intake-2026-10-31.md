# Agent update intake — 2026-10-31

This historical record identifies the source of retained Desktop prompt and
response-policy improvements. The current release keeps the core Harness and
recommended settings only. Benchmark task runners, score callbacks/aggregation,
user-simulator bindings and the DesktopEnv lifecycle launcher have been removed.
Use [recommended settings](../../examples/desktop-settings/README.md) and the
[current verification guide](../verification.md) for the maintained workflow.

## Source and retained mechanisms

- `b31035db2`: Desktop prompt and bounded same-context text-only response retries.
- `648804d19`: Desktop prompt, coordinate text projection and response recovery.

The core retains independent request copies, bounded retries, selected-only
history, cancellation/deadline propagation, safe artifact namespaces, explicit
HTTP error classification and mutually exclusive retry policies. Local extension
tool dispatch remains a generic Harness capability with test-only callback fixtures.

Response recovery lives under `northcua/agentinfra/providers/response_recovery.py`.
The subsequent scope cleanup removed all benchmark-specific prompts, their
selectors and coordinate-text rewriting. Recommended settings now use the
ordinary Desktop prompt; the general action-coordinate mapper is retained.

Earlier acceptance runs in this audit predate the removal and do not certify the
current source. Refer to the canonical CI and current verification record.
