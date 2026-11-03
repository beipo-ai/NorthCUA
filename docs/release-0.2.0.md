# 0.2.0 release verification

Verification date: **31 October 2026**. Python **3.12.13**, macOS arm64,
uv **0.13.6**. This source publication uses a clean Git history, now hosted in
[beipo-ai/NorthCUA](https://github.com/beipo-ai/NorthCUA).
The previous repositories remain intact. Earlier implementation audits are
[archived separately](verification.md#historical-audits).

The current scope is recorded in the [core Harness update](#core-harness-scope-update).
The original release matrix below is retained as historical evidence.

## Execution matrix

| Gate | Observed result | Scope |
|---|---|---|
| Default regression suite | **2,930 passed, 9 skipped**; 199.21s | Opt-in browser/native checks are separate below. Two expected fork-safety deprecation warnings. |
| Static and repository checks | **Passed** | Ruff lint/format, configured scoped Pyright, lock check, actionlint, all pre-commit hooks and Git whitespace checks. |
| Release/asset contracts after final packaging changes | **19 passed** | Four-package metadata, config mirrors, local document links, report/brand/figure/recording provenance. |
| Actor dispatch/recovery guards | **17 passed** | Harness-local callback placement, tool contracts, bounded response policies, cancellation and retry baselines. |
| From-zero operator skill | Independent **PASS** | Fresh environment, CLI/config generation and validation, matching managed Chromium 145 / revision 1208. |
| Source Web entry points and local batch | **7 passed**; 49.22s | Local application, CLI, WebSocket, both controller executors, real browser, live evaluation, both local batch modes and completed resume. |
| Browser-tool regressions | **4/4 passed**, **3/3 live probes** | Local fixtures: refs/forms/tabs/screenshots and actionability. Redaction scan passed. |
| Native macOS Desktop | **2/2 passed**; final run 7.46s | Physical click/type/Return/capture on disposable windows; DesktopApplication → NexAU → native tools, saved form result and durable note. No owned test service left running. |
| Independent local batch tutorial | **3/3 succeeded, one attempt each** | Live evaluator receipts; resume did not replay actions/evaluation; all three worker cleanup receipts confirmed, no remaining owned processes. |
| Installed distributions | **Four wheels built; base/local smoke and three `pip check` runs passed** | Completely fresh environment outside the checkout; exact package/license/resources/NexAU provenance. |
| Installed-wheel Web/controller/batch | **7 passed**; 53.15s | Five Web entry points and both three-task local batch modes/resume, executed from site-packages. |
| Remote CI | **All four hosted jobs passed** in the previous repository's [runtime acceptance run](https://github.com/china-qijizhifeng/NorthCUA-Harness/actions/runs/41110377739) | Linux contracts, real browser/X11, Linux/macOS wheels and installed tasks. The final report-only identity update is checked separately below; [current main status](https://github.com/beipo-ai/NorthCUA/actions/workflows/ci.yml) remains authoritative for later revisions. |

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

The Hugging Face page/API returned **401** at the check. Model downloads,
revision and weight license are not claimed.
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

## 3 November publication recheck

The canonical source is now [beipo-ai/NorthCUA](https://github.com/beipo-ai/NorthCUA).
This recheck keeps the 31 October execution record above as historical evidence.

- **Local default regression after dependency updates:** 2,940 passed, 9 opt-in skips, 185.57s with Python
  3.12 and uv 0.13.6. The 19 local-process ownership/cleanup tests also passed
  independently after the fixture fix described below. Lint, format, scoped
  typing, lock, actionlint, 29 release tests and isolated pre-commit checks passed.
- **Updated-runtime task checks:** all seven real Web/controller/batch tests
  passed in 45.44s. Both physical-input/native Desktop tests passed in 7.61s
  after explicitly activating their own disposable macOS fixture before publishing
  click coordinates. The click receipt still gates all keyboard input; no
  production Desktop code changed. The preceding focus-only failure is retained
  as diagnostic evidence, not a pass.
- **Independent onboarding:** a fresh locked source installation, configuration
  generation, real Web task and a three-task local-process batch succeeded.
  All tasks finished in one attempt; completed resume left events unchanged;
  all three cleanup receipts confirmed zero remaining worker PIDs. Only a
  deterministic local model was used, and no physical desktop was operated.
- **Source/history/distribution inspection:** Gitleaks found no secrets in the
  reviewed reachable history, tracked export or four rebuilt wheels. Additional
  review covered source blobs, package metadata, the report's 123 pages of text,
  PDF attachment/action checks, unchanged asset hashes and media metadata.
  Local OCR sampled 52 video frames at five-second intervals plus five images;
  this is not an exhaustive video-pixel or audio audit. No real credential or
  private infrastructure endpoint was identified in the reviewed material.
- **Dependency policy:** compatible affected packages were upgraded and the
  source/device distributions now require Pillow 13+. E2B and viewer minima
  match compatible releases. Development hooks run in a separate tool environment.
  Both PyPI and OSV audited all **337 locked public-package versions**, with
  zero unreviewed findings or lookup errors. Six Transformers advisory groups
  remain under the maintainer's [explicit, expiring exception](security/dependencies.md);
  this is not a zero-vulnerability claim. NexAU's public pin is unchanged.
- **Model publication:** all three model cards/configurations/license files are
  now public. The Pro index and all 122 fixed-revision shard URLs were checked
  without authentication or weight-body downloads. See [models](models.md) for
  exact revisions and independent-serving requirements; inference and reported
  benchmark results were not accepted by this check.

The first organization-repository CI run at `966d18f` failed a test that sampled
a detached child's PID after successful cleanup had already removed it. The
fixture now publishes the PID atomically and waits for the observer to capture
its identity before allowing early exit. Cleanup receipts, empty remaining-PID
checks and the unrelated-process survival assertion are retained. Runtime code
was not changed. That failed run and its cancelled macOS job do not count as passes.

The current CI adds a fifth, blocking **publication-security** job: a
checksum-pinned Gitleaks release scans full fetched history, merge-parent diffs
and a tracked source export with redacted reports; the dependency gate retains
raw findings and blocks new or expired risks. Follow the exact candidate's
[canonical CI result](https://github.com/beipo-ai/NorthCUA/actions/workflows/ci.yml)
for the hosted verdict. [Release preparation](releasing.md) defines the remaining
publication steps and distinguishes source, model, website and PyPI publication.


## Core Harness scope update

**3 November 2026.** This update narrows the distribution to core Desktop/Web
Harness behavior and optional recommended settings. It removes benchmark task
runners, DesktopEnv lifecycle launchers, external scoring callbacks/aggregation
and user-simulator bindings. The Desktop HTTP screenshot/input backend, ordinary
task entry points, generic local extension tools, built-in outcome evaluators and
recoverable batches remain available.

This earlier verification preceded the subsequent removal of all benchmark-specific
prompts and their selectors. Use the current
[recommended settings](desktop-settings.md) and remove
`prompt.system_prompt_profile` from an older local configuration. Shared response
recovery lives in the provider module. Independent summary-model paths resolve relative to the app YAML.

| Check | Observed result |
|---|---|
| Default regression | **2,906 passed, 9 skipped**; 188.66s; two existing fork-safety warnings |
| Targeted core/profile/configuration checks | **71 passed**; 3.41s |
| Source real-browser/controller/local batches and resume | **7 passed**; 44.04s |
| Physical input and Desktop actor in the owned disposable window | **2 passed**; 8.33s |
| Four newly built wheels | No removed integration modules; both relocated prompt resources and core recovery module present |
| Fresh base/local wheel installation outside the checkout | Both config/worker/controller/bundle smoke checks and dependency compatibility check passed |
| Installed-wheel real-browser/controller/local batches | **7 passed**; 43.26s; two marker-registration warnings in the copied test files |
| Static, lock and documentation checks | Ruff lint/format, scoped Pyright, lock check, whitespace and skill validation passed; final release/onboarding checks **19 passed** |

Independent code/distribution review passed **92 focused tests**, checked copied
configs outside the checkout, inspected all four wheel archives and scanned the
unpacked wheels for credentials with zero findings. Independent documentation
review checked local links/anchors and ran four CLI configuration validations
plus four core-loader/summary-model reads outside the checkout. Both gave scoped
signoff for this removal and resource migration.

These runs use scripted local model endpoints for tool integration, not hosted
research weights or benchmark acceptance. The retained NexAU version and
[accepted upstream dependency risk](security/dependencies.md) are unchanged.
The [canonical CI](https://github.com/beipo-ai/NorthCUA/actions/workflows/ci.yml)
records hosted checks for each subsequent committed revision.


## One Desktop prompt and short showcases

The subsequent 3 November cleanup removes every OSWorld-specific prompt,
the prompt-profile selector, benchmark-conditioned instructions in the ordinary
Desktop template and legacy coordinate-text rewriting. The general 0–1000
action mapper, response recovery and context compaction remain. Recommended
recipes use the native Desktop backend and one ordinary prompt; older configs
must remove `prompt.system_prompt_profile`.

The bundled showcases are now the maintainer-supplied 30-second Ruby Heart and
Blender UFO edits. Both READMEs use native GitHub video attachments while
retaining byte-identical MP4s and checksummed result frames for offline viewing.
The source/controller attribution and accelerated-edit boundaries are described
in [showcases](showcase.md). The report, research figures and brand files are unchanged.

The local focused prompt/configuration/retry/release checks passed **111 tests**.
Each clip was independently confirmed to contain 900 frames at 30fps/1080p with
H.264/AAC and to decode fully; independent frame re-extraction matched the
result PNGs. Both native players were clicked and observed advancing on the
actual GitHub branch README while signed out. The canonical CI remains the
revision-specific record for full regression and installed-runtime gates.
