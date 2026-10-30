# Agent update intake — 2026-10-31

> Historical preparation audit. These observations and revision identifiers belong
> to the earlier source repository. For this publication, use the
> [0.2.0 release verification](../release-0.2.0.md); old dependency/platform results
> and publication status below are not claims about the new release.

Reviewed sources on `codex/northcua-clean-open-source`:

- [`b31035db2`](https://github.com/china-qijizhifeng/NorthCUA/commit/b31035db295039a834788ac2b15f0050c19585a9): OSWorld2 actor harness and same-context no-tool retries.
- [`648804d19`](https://github.com/china-qijizhifeng/NorthCUA/commit/648804d1986255cbab4b614c05e98663fb87bf10): OSWorld1 actor harness, one versioned prompt and response recovery.

Both commits were already on the remote clean branch. The local branch was
fast-forwarded to `648804d19`, preserving its earlier staged and working changes.
The corrections below are additional working-tree changes; this record does not
claim they have been committed or pushed.

## Retained functionality

| Area and implementation | Intake decision |
|---|---|
| `providers/no_tool_retry.py`, Desktop model binding, shared single-model runner | Keep the optional three-call OpenAI Chat policy. Freeze serialized requests, select the first tool response or final text response, retain intermediate receipts, and keep retries outside history/tool execution/logical-step accounting. Unsupported streaming/multiple-choice/provider configurations fail explicitly. |
| OSWorld1 `recovery.py` and shared single-model runner | Keep bounded transient/empty-response and token-exhausted reasoning-only recovery, 2/4/8-second backoff plus jitter, cancellation/deadline propagation and valid terminal text. SDK retries remain a separate budget. |
| `prompt/builders.py`, `core/config_base.py`, OSWorld1/2 prompt builders/templates | Keep separate, explicitly selected profiles and a single versioned template per benchmark. Default Desktop/Web behavior and the single tool-returned-image path remain the local starting point. |
| OSWorld1 `coordinates.py` | Keep the historical virtual 1000×1000 canvas and regex text projection only in `osworld1_formal` with normalized coordinates. Preserve structured tool arguments, image bytes, reasoning and original artifacts; action dispatch maps coordinates once. The regex can alter coordinate-like task prose and is deliberately not a general Desktop normalization rule. |
| OSWorld1 `runner.py` | Keep the prepared-app adapter, original query, 100-step cap and explicit DONE/FAIL hints. A normal budget limit is distinct from infeasibility and execution failure. The caller retains environment and evaluator ownership. |
| OSWorld2 `runner.py`, `tools.py`, `config.py` | Keep the prepared-app adapter, 500-step cap, original task-owner `ask_user` callback and independent summary-model configuration. Resolve relative summary config paths from the app config directory. |
| OSWorld2 `scores.py`, shared by both adapters | Keep real zero/fractional scores; exclude execution/evaluator errors with `score=null` and an explicit invalid flag. Report the valid denominator and reject duplicate task IDs before aggregation. |
| Five YAML examples, two guides, examples index and package-data entries | Keep portable environment references and versioned model/sampling recipes. Ship both templates in wheels and check their installed checksums. Link the optional integrations from the documentation index and configuration guide. |
| The five incoming test files | Keep the 64 original harness tests, and strengthen the OSWorld1 wire test with a failed-attempt input mutation. Add boundary and complete-query regressions for the fixes below. |

## Problems corrected before acceptance

| Finding | Correction and regression evidence |
|---|---|
| `ask_user_tool` returned a raw NexAU tool, rejected by runtime preparation | Define typed parameters and project a `ToolContract`; test actual `prepare_agent`, parameter validation and the original callback answer. |
| Even after preparation, all paused calls went to the device, so `ask_user` never executed | Carry explicit placement in runtime projection metadata. Dispatch only current `LOCAL_HANDLER` projections on the harness; preserve mixed callback/device order and the existing hooks, results and history replay. Test local/remote custody, stale projections, callback errors/cancellation and a complete Desktop query where the answer reaches the next HTTP model request. No global temporary MCP registration is used. |
| Retry receipts used an unsanitized session path | Reuse `sanitize_session_id`, matching the existing artifact namespace. Test traversal and absolute logical session IDs in disposable directories. |
| Cancelling an async bridge left its synchronous worker sending further no-tool attempts | Close a scope-owned thread-safe lifetime flag on exit and check it before/after each request. Test cancelling a real `asyncio.to_thread` bridge while its first request is blocked. An already-running SDK request may finish; no follow-up attempt is started by the closed loop. |
| Permanent HTTP errors could retry when their text contained “timeout” or “rate limit” | Make an explicit HTTP status authoritative: retry 429/5xx, not permanent 4xx. Test misleading error text for 400/401/403/404/422. |
| Two response policies could multiply each other's requests | Reject the combination in app configuration validation. Defaults remain one no-tool attempt and response recovery disabled. |
| OSWorld1 retries shared mutable messages/tools | Deep-copy a prepared baseline for each attempt. The real SDK/HTTP wire regression deliberately mutates a failed attempt and verifies identical subsequent requests and unchanged caller history. |

## Scope decisions

No incoming actor functionality was silently removed. Benchmark-specific behavior
stays opt-in. In particular, the historical OSWorld1 text rewrite and the example
sampling values are retained for comparison, without becoming general defaults.
The template checksum tests prove stability of these imported files; they do not
independently establish identity with an external training/campaign snapshot.

Environment creation/reset/release, original tasks and graders, task repairs,
VM inventories, dispatch/stop scripts, NorthJetty and campaign results stay outside
these adapters. The existing generic OSWorld lifecycle adapter remains separate.

The previous dependency repair is preserved: public NexAU
`28511d9f0362191a6c378fd441b2be55309a9e32` from `hzhua/nexau`, MCP 2 and FastMCP 4.
Only the two template package-data entries were taken from the incoming manifest
delta; the outdated dependency declarations were not restored.

## Verification

On Python 3.12/macOS arm64:

- Final default suite: **2,928 passed, 9 skipped**, 196.56 seconds. Opt-in platform checks remain separate.
- Focused harness/dispatch regression: **88 passed**. Independent reviewer additionally ran the 17 callback/guard cases, including the complete Desktop query.
- Ruff lint/format, configured Pyright and `uv lock --check`: passed.
- Source Web tasks: **5 passed**; both local batch execution modes and resume: **2 passed**. These use real managed Chromium, deterministic local model endpoints and live outcome checks.
- Four wheels rebuilt; wheel-only configuration, worker/controller startup, runtime bundle and installed template checks passed. The same **17 callback/guard tests passed outside the checkout** against the rebuilt wheels.
- Installed-wheel Web/controller/batch tests: **7 passed**, 41.35 seconds, outside the source checkout, with real Chromium and both batch execution modes/resume.

The first browser runs requested an uninstalled managed-browser revision. Those
failures are retained as diagnostics, not acceptance results. Accepted runs use
the explicitly selected managed headless Chromium revision 1234; local workers
receive its path through `NORTHCUA_BROWSER_EXECUTABLE`.

The complete Desktop callback test uses an enforced in-memory display backend,
a physical-capture failure guard, and only note/question fixture actions. An
initial incorrectly isolated diagnostic run was stopped and is not counted as
simulated or physical-platform acceptance.

This is code/integration acceptance, not a new hosted-model benchmark, VM/cloud
provisioning run, native-platform certification, remote CI result or publication.
Local evidence and exact source/wheel hashes are retained under ignored
`logs/agent-intake/20261031/`. Independent reviewer scope and final artifact
checks are recorded there as well.
