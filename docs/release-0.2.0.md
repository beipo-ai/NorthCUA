# 0.2.0 release verification

Verification date: **31 October 2026**. Python **3.12.13**, macOS arm64,
uv **0.13.6**. This source publication starts a new Git history in
[NorthCUA-Harness](https://github.com/china-qijizhifeng/NorthCUA-Harness).
The previous repository remains intact. Earlier implementation audits are
[archived separately](verification.md#historical-audits).

## Execution matrix

| Gate | Observed result | Scope |
|---|---|---|
| Default regression suite | **2,930 passed, 9 skipped**; 199.21s | Opt-in browser/native checks are separate below. Two expected fork-safety deprecation warnings. |
| Static and repository checks | **Passed** | Ruff lint/format, configured scoped Pyright, lock check, actionlint, all pre-commit hooks and Git whitespace checks. |
| Release/asset contracts after final packaging changes | **19 passed** | Four-package metadata, config mirrors, local document links, report/brand/figure/recording provenance. |
| Formal actor dispatch/recovery guards | **17 passed** | Harness-local callback placement, tool contracts, bounded response policies, cancellation and retry baselines. |
| From-zero operator skill | Independent **PASS** | Fresh environment, CLI/config generation and validation, matching managed Chromium 145 / revision 1208. |
| Source Web entry points and local batch | **7 passed**; 49.22s | Local application, CLI, WebSocket, both controller executors, real browser, live evaluation, both local batch modes and completed resume. |
| Browser-tool regressions | **4/4 passed**, **3/3 live probes** | Local fixtures: refs/forms/tabs/screenshots and actionability. Redaction scan passed. |
| Native macOS Desktop | **2/2 passed**; final run 7.46s | Physical click/type/Return/capture on disposable windows; DesktopApplication → NexAU → native tools, saved form result and durable note. No owned test service left running. |
| Independent local batch tutorial | **3/3 succeeded, one attempt each** | Live evaluator receipts; resume did not replay actions/evaluation; all three worker cleanup receipts confirmed, no remaining owned processes. |
| Installed distributions | **Four wheels built; base/local smoke and three `pip check` runs passed** | Completely fresh environment outside the checkout; exact package/license/resources/NexAU provenance. |
| Installed-wheel Web/controller/batch | **7 passed**; 53.15s | Five Web entry points and both three-task local batch modes/resume, executed from site-packages. |
| Remote CI | **All four hosted jobs passed** in the [runtime acceptance run](https://github.com/china-qijizhifeng/NorthCUA-Harness/actions/runs/41110377739) | Linux contracts, real browser/X11, Linux/macOS wheels and installed tasks. The final report-only identity update is checked separately below; [current main status](https://github.com/china-qijizhifeng/NorthCUA-Harness/actions/workflows/ci.yml) remains authoritative for later revisions. |

Task checks above use **real browser/native tools and public NexAU with a
scripted local model endpoint**. They establish executable integration and
observable outcomes, not hosted-model capability or benchmark performance.
The full regression was followed by the final installer fix and focused
release/actor/native reruns; application runtime code did not change.

## Independent review

Separate AI reviewers inspected their own scope and retained independent
commands, outputs, file hashes and limitations. A signoff is scoped to those
observations, rather than a blanket model/platform certification.

- **Paper, brand and public claims — SIGNOFF.** Exact supplied 123-page report,
  its title/team/citation, slogan, first-page logo/letter outlines, three current
  mechanism figures, font notices, demo attribution and resource availability.
  All 13 brand/figure/demo assets match their manifests. Diagram rendering and the
  Smartest poster were checked against their original sources.
- **Operator experience and agent skill — SIGNOFF.** Independently followed the
  first-run/skill workflow, completed Web and the local three-task batch,
  checked resume and cleanup, then reviewed the final English/Chinese README,
  model/endpoint recipes, links and installer minimum. No local link/anchor issue
  remained in the reviewed documentation.

- **Distribution and repository — SIGNOFF.** Independently rebuilt all four
  final wheels, installed base and local extras into a new environment, passed
  both wheel smoke checks and three strict `pip check` runs, verified source
  provenance and the two packaged OSWorld prompts, then ran seven installed
  Web/controller/batch cases outside the checkout. The source allowlist contains
  no virtual environments, private inventories, raw traces, website checkout,
  build trees or real credentials; the private-key test fixture is explicitly fake.
- **Website — SIGNOFF.** Latest report, exact logos, headline model scores and
  evaluation protocols, citations and availability labels align with the report.
  The separate website passed its root/subpath build and link/asset audits,
  9 unit checks, 38 browser cases (4 declared GPU-scope skips), a final 9-case
  targeted three-engine regression and full decode of all seven recordings.
  Actual in-app browser interactions were checked. Its online advisory request
  timed out; no completed vulnerability-scan or public-hosting claim is made.

## Release checks that caught and closed issues

- Entry-point review found a broken tutorial anchor and a model-base-URL recipe
  that failed to pass an explicit CLI override. Both were corrected and rechecked.
- Distribution review found an old report-team attribution in NOTICE and missing
  exact-file pre-commit exceptions for the two intentional demo videos. All four
  package notices now agree; large-file checks remain enabled elsewhere.
- An EOF hook changed the supplied mark's bytes. The asset contract detected it;
  the original bytes were restored and source assets/licenses explicitly preserved.
- Old **uv 0.10.9** omitted six PyObjC frameworks on Darwin while its own check
  reported compatibility; `pip check` correctly rejected that environment.
  Official **uv 0.13.6** evaluates those markers correctly. The project enforces
  that minimum, CI pins it, and the existing lock naturally installs all six
  matching **12.1** frameworks. No third-party metadata override or manual
  application dependency workaround is used. The source environment now passes
  independent `pip check`. Failed diagnostic environments are not acceptance runs.

- The first hosted browser lane encountered GitHub's `ubuntu-latest` migration
  to Ubuntu 26.04; locked Playwright 1.58 correctly rejected that unsupported OS.
  Linux source/wheel/release jobs now use **Ubuntu 24.04**, retaining the tested
  application lock. Official SHA-pinned Node 24 actions replace deprecated
  Node 20 actions. The failed first run is not counted as acceptance.

- The first Linux contract run passed 2,928 tests but exposed two schema/risk
  rejection tests that unnecessarily initialized a native desktop. Those tests
  now inject a recording MCP server at the construction boundary and explicitly
  assert zero dispatch; real contract validation and risk checks remain active.
  Production code is unchanged, and real native input has separate coverage.

- A hosted integration retry exposed a missing parent directory for pytest's
  explicit `reports/web-modes` temporary root. The browser job now creates
  `reports/` before any task test. Tests and their failure conditions remain
  enabled; failed setup-only runs are not acceptance.

## Report, assets and availability

[paper/manifest.json](paper/manifest.json) identifies the unchanged **123-page
31 October NorthCUA Technical Report**, SHA-256
`bf2ce9e665cd7eb2212c0d5635d276aaba46232ad89f9f34f7ce6b2367244611`.
A final source recheck identified a newer compilation of the same report. Its
introduction/conclusion wording changed and references reflowed; the independent
paper reviewer confirmed identical logo glyphs, mechanism figures, headline
metrics and protocols. The latest supplied PDF is bundled, with its new
identity reflected in all manifests; release intent in the report is not
substituted for verified model-download availability.

Canonical logos use the exact blue Beipo mark and embedded TeX Gyre Adventor
letter outlines from that report. Current figures and historical recordings have
separate provenance and are bundled in the repository.

The official website returned **404** and the Hugging Face page/API **401** at
the check. Model downloads, revision and weight license are not claimed.
[Resource status](resources.md) provides identities and local fallbacks.

## Acceptance boundaries and reproduction

**Hosted-model capability, real E2B provisioning, upstream OSWorld VM execution,
Windows physical input, and complete native Desktop WebSocket/remote task runs
were not accepted in this local audit.** Linux X11 and hosted package installs passed their own hosted runtime-candidate
jobs, as linked above; their result is independent from the macOS checks.
No PyPI package or public model release is implied by a source upload.

Reproduce the supported paths with [first run](first-run.md),
[local batch](local-batch.md) and [testing](testing.md). NexAU is pinned to
`28511d9f0362191a6c378fd441b2be55309a9e32`; the lock records the remaining
source environment. Preserve code/model/environment revisions, task and evaluator
settings, task-specific receipts and cleanup outcomes with experiment artifacts.
Raw local traces and credentials are excluded from the publication.
