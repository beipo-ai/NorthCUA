# NorthCUA Harness verification

> Historical preparation audit. These observations and revision identifiers belong
> to the earlier source repository. For this publication, use the
> [0.2.0 release verification](../release-0.2.0.md); old dependency/platform results
> and publication status below are not claims about the new release.

The [2026-10-31 agent update intake](agent-intake-2026-10-31.md) records the
OSWorld actor profiles, response-policy corrections, callback dispatch and
their final regression/installed-package checks.

For the current NexAU source and dependency pin, see the scoped
[2026-10-31 migration verification](nexau-migration-2026-10-31.md). The results below
are the historical 2026-10-13 environment; they are not new-platform acceptance
for that dependency migration.

Verification date: **2026-10-13**. Candidate **0.2.0**, implementation revision
**`df8fb92fdfb4edb603ef5d22c7a1e321da0df4e2`**, branch
`codex/northcua-clean-open-source`. Later documentation-only changes record these
results and clarify evaluator/cleanup reporting; they do not change the tested
runtime. The [2026-10-03 audit](verification-2026-10-03.md) is retained as history.

This audit covers the from-zero operator skill, model templates, local Desktop/Web
tutorials, real local browser batches, recovery and distributable wheels. All
task checks below use **real GUI/browser tools and public NexAU with a scripted
local model endpoint**. They establish executable workflows and observable
outcomes, not hosted-model capability or benchmark performance.

## Execution matrix

| Gate | Observed result | Evidence boundary |
|---|---|---|
| Complete default suite | **2,846 passed, 9 skipped**, 185.72s | Nine browser/native tests require opt-in execution. Two warnings come from intentional fork-safety tests. |
| Static checks | Passed | Ruff lint/format (1,183 files), configured Pyright (zero errors/warnings), actionlint, lockfile, whitespace and public release/config/document contracts. |
| Skill: from-zero Web | Independent fresh environment: passed | Current skill and linked guide, locked install, Chromium setup, CLI actor, correct page receipt, screenshot/DOM, live evaluator and service cleanup. |
| Skill: native macOS Desktop | Independent fresh environment: passed | Four actor steps through note/type/keypress; native form contains this run's exact expected text, saved screenshot confirms it, window/server stop and ports close. |
| Native input regressions | Independent **2 passed**, 8.35s | Physical input and DesktopApplication → NexAU → native tools on disposable windows, including screenshot/history assertions. Actual saved output and screenshots inspected; display released. |
| Local browser batch | Independent fresh environment: **3/3 passed** | Real controller, workers, browser actions, per-task receipts and live evaluation; one attempt per task. |
| Completed-batch resume | Passed | Status and fixture events unchanged; no repeated click, attempt or evaluation. |
| Actual drain and recovery | Passed | One active task finishes; two queued tasks remain pending; the first runner exits before resume; the two remaining tasks finish once. All six tasks across both batches have verified outcomes and cleanup. |
| Installed-wheel Web and batch | **7 passed**, 39.28s | Local application, CLI, WebSocket serving, controller in-process/process and local batch in-process/process; executed outside the source checkout. Both batches also resume without repeat work. |
| Distribution installation | **Four wheels built; base and local-extra installation passed** | Fresh pip resolution, config generation/validation, worker CLI/doctor, controller startup, runtime bundle, public NexAU, `pip check` and matching packaged sources. |
| Local process failure paths | Independent **39 focused tests** and four separate reproductions passed | Supervisor loss, detached descendants, fast orphan creation and synchronous launch failure; truthful failure reporting and credential-file cleanup. |

The source baseline is **Python 3.12.13 on macOS arm64**, the committed lockfile
and NexAU `c98f7fb73f41f338e462d0585c62c1e62362b693`. The installed-wheel environment
resolved independently: **Playwright 1.63.0**, **AG-UI protocol 0.1.17**, with an
explicit managed headless **Chromium 151.0.7922.34 / revision 1234** executable.
The newer matching browser revision 1243 was not accepted in this audit: its
download was stopped after the named managed-browser test completed. This is not
a claim that every browser release/channel has passed.

The accepted root wheel SHA-256 is
`8bd2ba236941a2d6f4c11436690eccedd2bcb5eb969e0cda524d51b2274afc00`.
All four accepted wheels and their checksum manifest are retained in the local
ignored `wheelhouse/` directory. No package publication is implied.

## Independent reviewer signoff

Independent review agents worked on the frozen implementation and retained their
own commands, failures, outputs and scope statements:

- **Operator skill / native use: scoped SIGNOFF.** Followed the current skill from installation,
  ran real Web and native Desktop tutorials, inspected receipts and screenshots,
  then passed both opt-in native regressions and stopped their services/windows.
  Credential-free scripted tutorial success is identified as such.
- **Batch use: scoped SIGNOFF.** Fresh installation, one bounded Web task, three
  tasks, completed resume, drain/recovery and all six live evaluator/cleanup
  receipts passed. No author success logs were used as acceptance evidence.
- **Lifecycle: scoped SIGNOFF.** No remaining blocker in the supported trusted
  worker/Playwright ownership scope. A detached child created immediately before
  worker exit is reclaimed; abnormal worker exit still produces an unconfirmed
  receipt. Killing the supervisor itself correctly raises and may require manual
  ownership-verified worker cleanup; it is not counted as automatic recovery.
- **Release: scoped SIGNOFF.** Rebuilt all four distributions, checked public
  dependency provenance/configuration/docs, passed 40 focused checks and seven
  installed task tests. All 37 allocations from that review, including failed
  diagnostic runs, have confirmed cleanup and no remaining owned worker/supervisor.

Detailed local evidence is retained under `logs/onboarding-dev/`,
`logs/reviewer-skill/run-20261013-final-df8fb92fd/`,
`logs/reviewer-batch-final/`, `logs/reviewer-lifecycle/` and
`logs/reviewer-release/`. These ignored directories are audit artifacts, not
dependencies for running the project. Reproduce the public workflows with
[first run](../first-run.md), [local batch](../local-batch.md) and [testing](../testing.md).

## Failures found and resolved

- Fresh pip installation originally selected AG-UI 1.0, incompatible with the
  pinned NexAU imports. The shipped local extra now constrains the independently
  tested compatible version; the wheel CI lane requests that extra too.
- Native tutorial completion originally hid lost modifier/Return input. Quartz
  flags and native outcome checks fix this path. The focus guard now distinguishes
  the system cursor overlay from another application's obstructing window.
- Initial cleanup missed detached processes and could fabricate success after
  supervisor loss. Process identities, inherited ownership markers, repeated
  discovery and atomic, explicitly confirmed receipts address the supported
  worker path. Opaque new process candidates fail closed instead of being killed.
- The release artifact initially omitted hidden `.env.example`; the curated
  artifact now includes it. A run-wide page receipt was replaced with an exact
  task-specific receipt, including prefix-collision rejection.

System **Chrome 154** intermittently crashed during independent wheel testing
with SIGBUS/crashpad errors. Those strict one-attempt tests remain recorded as
failures, including runs where a retry later completed the task. The seven-case
acceptance above used the explicitly named managed headless browser instead.

Batch summary `evaluation_status`/`evaluation_attempts` can remain blank/zero on
this runtime path. Reviewers inspected each session's actual `evaluation_report`
and evaluator `final_report.json`, live snapshot and task-specific receipt.
The [local batch guide](../local-batch.md) documents those authoritative outputs.

## Remaining acceptance gates

**Hosted-model task capability, real E2B provisioning/template builds, upstream
OSWorld VM execution, Windows physical input and Linux X11 runtime execution were
not accepted in this local audit.** Physical Desktop WebSocket/remote modes also
need their own complete task runs. Contract tests or the earlier audit are not
substitutes for those deployment results.

CI defines locked tests, Linux browser/X11 checks, Linux/macOS wheel installs and
Linux installed-wheel Web/batch tasks. Its static checks passed; **no remote
GitHub Actions execution was observed**. Release automation produces a candidate
artifact with source, wheels, docs, skills and checksums. No PyPI or GitHub release
has been published by this work.

The bundled research report remains the separately identified 2026-10-03 draft.
Repository smoke tests do not populate its pending benchmark tables. See the
[paper map](../paper-map.md) and [clean scope](../clean-scope.md) for what is released.
