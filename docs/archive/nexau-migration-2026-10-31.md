# NexAU dependency migration verification

> Historical preparation audit. These observations and revision identifiers belong
> to the earlier source repository. For this publication, use the
> [0.2.0 release verification](../release-0.2.0.md); old dependency/platform results
> and publication status below are not claims about the new release.

Verification date: **2026-10-31**, NorthCUA **0.2.0**, on Python **3.12.13** /
macOS arm64. This is a scoped dependency migration on top of repository revision
`5d5576d00a829f38c75f41410f8e98629e8625f3` and the current branding/documentation
changes. It does not supersede the separate platform results in the
[earlier audit](verification-2026-10-13.md).

## Dependency source

The `local` extra now uses the project owner's confirmed public source:
[hzhua/nexau](https://github.com/hzhua/nexau), pinned to
`28511d9f0362191a6c378fd441b2be55309a9e32` (NexAU 0.4.1).
The previous download repository is unavailable, and its old commit is not
part of the new repository's history. Both the published dependency declaration
and the uv source/lock use the new full commit; no third-party mirror is required.

The new NexAU requires MCP 2. NorthCUA and its device distribution now declare
`mcp>=2,<3` and `fastmcp>=4.1,<5`; the tested lock resolves MCP **2.3.0** and
FastMCP **4.1.4**. The compatible AG-UI protocol pin remains **0.1.17**.
Use the updated checkout and run:

```sh
uv sync --locked --extra local --extra device
```

For wheel installation with uv, include the NexAU URL explicitly, as shown in
[configuration](../configuration.md). uv rejects an indirect URL dependency unless
the installation request also names that source.

## Observed checks

| Gate | Result and boundary |
|---|---|
| Fresh locked source install | Passed with local, device, dev, E2B and viewer extras; installed NexAU provenance matches the full new commit. |
| Default regression suite | **2,846 passed, 9 skipped**, with one documentation-link failure caused by stale logo paths in the validation copy. Restoring the actual current documentation and rerunning all nine public release/document contracts passed. No behavioral test remains failing. |
| Static checks | Ruff lint and formatting passed; configured Pyright reported zero errors/warnings; `uv lock --check` passed. |
| Source real browser integration | **7 passed**: local actor/evaluator state, WebSocket serving, both controller execution modes, local worker batches and completed-batch resume. |
| Distribution | Four wheels built and installed in a fresh environment outside the checkout. CLI, configuration, worker metadata/tools, controller startup, packaged runtime resources and NexAU provenance checks passed. |
| Installed-wheel browser integration | **7 passed**, outside the source tree, covering the same Web/controller/batch paths and resume behavior. |
| Independent dependency review | Scoped **SIGNOFF**; reviewer independently passed 55 related tests and the lock check, and inspected the provider fixture changes and installation/integration reports. |

Browser checks use real Playwright/browser actions with a deterministic local
model endpoint and task-specific outcome evaluation. They establish executable
integration, not hosted-model capability or benchmark performance.

The nine default-suite skips are the opt-in browser/native tests. The seven
browser cases were run separately above. Physical native Desktop input was not
rerun during this dependency migration; neither were live E2B/OSWorld provisioning,
Windows/Linux platform acceptance or hosted-model tasks. No remote CI run or
package publication is claimed.

## Compatibility changes reviewed

- The new NexAU calls OpenAI Responses through `create(stream=True)`, for both
  synchronous and asynchronous streaming. The test transport and recorded wire
  fixture now reflect that interface. An independent comparison of all 64 cases
  found exactly eight streaming cases with that operation/flag change; all other
  request fields and cases are unchanged. Stream completion, disconnect and
  cancellation assertions remain active.
- FastMCP 4 wraps invalid tool arguments in `ToolError`. The server-boundary test
  expects that wrapper and still checks the error text and that the handler was
  not called. Direct validation tests retain their validation-error assertions.
- NexAU now validates native MCP transports eagerly. Routing-only tests attach
  NorthCUA's legacy visibility/snapshot metadata after validating the other
  agent fields. This mirrors the production boundary, which retains that
  metadata separately and clears NexAU's native MCP list before agent creation.
  The helper is test-only and is never used to start native NexAU MCP clients.

Local audit outputs are retained under ignored `logs/nexau-migration/20261031/`.
They are verification artifacts, not runtime dependencies. Public reproduction
commands live in [testing](../testing.md), [first run](../first-run.md) and
[local batch](../local-batch.md).
