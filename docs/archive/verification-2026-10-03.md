# NorthCUA Harness verification

> Historical preparation audit. These observations and revision identifiers belong
> to the earlier source repository. For this publication, use the
> [0.2.0 release verification](../release-0.2.0.md); old dependency/platform results
> and publication status below are not claims about the new release.

Historical record. See [the current verification](verification-2026-10-13.md) for the later
onboarding, skill, local batch and distribution audit.

Verification date: **2026-10-03**. NexCUA repository, NorthCUA Harness candidate: **0.2.0**, branch
`codex/northcua-clean-open-source`. The latest audit follows candidate
`6c7fa33f9f1deedf2326aa830db3ce68ecde3f61` and simplifies Desktop to one
**tool-returned-screenshot path**. The clean implementation originated from main
`315508f3c08a7dec7f753691f38a609c9b8d0601`.

The local completion gates below passed after this simplification. Earlier test results
and reviewer signoffs do not substitute for these runs. The matrix distinguishes real
task execution, deterministic regression coverage and outstanding deployment acceptance.

## Execution matrix

| Path or gate | Latest result | What this establishes |
|---|---|---|
| Full default suite | **2815 passed, 7 skipped** in 180.83s | Runtime, provider streams, context/memory, evaluator, transport, batch/recovery and release regressions. The seven opt-in tests executed separately below. Two warnings come from intentional fork-safety tests. |
| Web: local application, CLI, local WebSocket server, controller in-process and controller process | **5 passed** in 25.94s | Actual Chrome and pinned public NexAU through a deterministic local HTTP model; shared actor/evaluator live page state, image budgets and working-memory projection. |
| The same five Web paths from final rebuilt wheels | **5 passed** in 25.51s | Execution outside the checkout in a separate environment. One warning concerns the copied test's unregistered `integration` marker; all five tests executed. |
| Native macOS Desktop | **2 passed** in 7.35s | Actual input and screenshots on a disposable Tk window, plus DesktopApplication → NexAU → note/type/keypress → observed text. Every actor model request must contain exactly one image at budget 1, including the request following a text-only note. The session ID contains Chinese and a slash. Evaluation is disabled in this native smoke test. |
| Four distributions | **Four 0.2.0 wheels built** | Packaged configs, LICENSE and NOTICE are present. Every packaged Python source matches the checkout; deleted observation modules are absent. |
| Wheel-only deployment | Passed before and after installing pinned public NexAU; final rebuild smoke also passed | Config generation, worker CLI/doctor, controller startup and runtime bundling outside the checkout. Dependency resolution is separate from the source lockfile. |
| Local Desktop WebSocket server | Controlled regression tests passed | Admission, physical-display exclusivity, ACK failure, cancellation, terminal errors and shutdown. No complete physical task through this server was accepted in this audit. |
| Remote Desktop / controller device input | Contract, transport and failure tests passed | The shared observation/command chain is covered; a complete physical task through each remote Desktop mode remains unverified. |
| Batch and OSWorld adapter | Deterministic regressions and fixtures passed | Claims, preparation, deadlines, retries, recovery, artifact custody, cleanup, task/oracle mapping and terminal behavior. No real cloud provisioning or upstream OSWorld VM acceptance. |
| Static and release checks | Passed | Ruff lint/format, configured Pyright scope, lockfile, workflow actionlint, whitespace, public package/config contracts, documentation links and paper/artwork hashes. |

The source run used **Python 3.12.13 on macOS arm64**, the committed lockfile,
NexAU `c98f7fb73f41f338e462d0585c62c1e62362b693` and **Chrome 145 / Playwright
browser revision 1208**. Installed-wheel dependencies were resolved separately.
No hosted-model credentials or cloud accounts were used. The deterministic HTTP model
checks real execution wiring and evidence delivery, not autonomous model capability.
See [testing](../testing.md) for commands and opt-in requirements.

## Single Desktop observation path

GUI actions return post-action screenshots. The runtime uses their clean images as the
next state and projects history through the report's overlay/clean window. Initial
capture supplies an image when no clean tool result exists; notes preserve the current
grounding image. Optional accessibility recording and evaluator evidence remain available.

The removed code includes the pre-round probe policy, separate user-observation
projection and factory, configurable observation chains, an empty window/control chain,
the duplicate legacy collector, unused probe adapters, generic image allocation and
export-only prompt interfaces. Orphaned driver comparison helpers and unused Desktop
wait/clipboard settings were also removed. The [configuration migration](../context.md#images)
identifies obsolete keys, which now fail validation instead of being silently ignored.

Tests for deleted modes were removed; tests of surviving behavior were migrated and
extended. Real-provider image assertions cover the note-only boundary. Native fixture
focus is explicitly confirmed after application initialization, and the test waits for
its own window's output before asserting the result. This addresses the focus/timing
failures encountered during this audit without changing production clipboard handling.

## Independent reviewer signoff

Both independent, read-only review agents examined this simplification:

- **Runtime: scoped SIGNOFF.** No unresolved P1/P2 in the reviewed image/state chain,
  removed APIs, local/remote observation adapters or source persistence. Independently
  passed **303 focused tests**, followed by **19 configuration-snapshot/tool-surface
  checks** with overlapping coverage. Verified the real Web and native execution logs.
- **Release: scoped SIGNOFF.** Independently passed **26 release/configuration/observation
  boundary checks**. Source and packaged Desktop/Web configurations and the OSWorld
  example load successfully. The runtime import scan found no references to deleted
  modules; documentation distinguishes removed modes from retained capabilities.

The final full-suite, native and rebuilt-wheel results above satisfy the local completion
conditions. The signatures cover this change and its stated checks, not all deployment
platforms. Earlier lifecycle/packaging fixes and their evidence are recorded in the
preceding candidate and [changelog](../../CHANGELOG.md).

## Remaining acceptance gates

**Windows physical execution, hosted-model providers, real E2B creation/template builds
and an upstream OSWorld VM run remain deferred** at the project owner's request.
Linux X11 runtime acceptance and remote GitHub Actions results were not observed in this
local audit. Physical Desktop serve/remote modes need complete task runs as noted above.

GitHub Actions defines locked tests, Linux browser/X11 tasks, Linux/macOS wheel
installations and Linux wheel-only Web tasks. Its static validation passed; a hosted run
is still required. Pushes to main and the clean-candidate branch, pull requests and manual
runs are configured. Release automation depends on CI and produces wheels, source,
documentation and checksums. **No package or GitHub release has been published.**

The bundled 97-page report is a separately identified research draft. Its pending
benchmark tables are not populated from repository smoke tests. See the
[paper map](../paper-map.md) and [clean scope](../clean-scope.md) for implementation boundaries.
