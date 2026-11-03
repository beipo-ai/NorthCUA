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

Prompt resources now live under `northcua/agents/desktop/prompt/profiles/`;
response recovery lives under `northcua/agentinfra/providers/response_recovery.py`.
The current profile names are `osworld1` and `osworld2`. Earlier `_formal`
configuration names are no longer accepted. The OSWorld2 prompt no longer refers
to the removed user-simulator tool.

Earlier acceptance runs in this audit predate the removal and do not certify the
current source. Refer to the canonical CI and current verification record.
