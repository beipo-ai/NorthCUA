# Clean release scope

This NorthCUA release contains the NorthCUA Harness and is based on main `315508f3c08a7dec7f753691f38a609c9b8d0601` in a separate
worktree. The original working checkout and historical research artifacts are not
modified by the cleanup.

| Disposition | Scope |
|---|---|
| Keep and validate | Desktop/Web agents, typed tools, observations, visual history, semantic compaction, ordinary skills, working memory, artifact/live evaluators, controller/device, generic external-task APIs, batch and artifact/reporting code |
| Remove | Private driver/PiP and companion control UI; training/data conversion/generation pipelines; environment-builder actor; CUA-Gym/WebHarbor/NorthJetty workflows; reusable warm pools; dated experiment wrappers, inventory acceptance suites and private/public NexAU migration release administration |
| Replace | Private model defaults → public env-based configs; dated template catalog → public Desktop/Web substrates; source-only install paths → packaged resources/wheelhouse; Benchmark adapters → ordinary Desktop entry points and recommended settings; historical docs → user workflows and paper map |
| Test-only | Deterministic model/provider streams, fixture pages, fake provider/task scenarios and regression schemas; these are not live experiment claims |

## Decisions applied

- Breaking CLI/config/Python changes are allowed. Old outputs may be inspected; old runs
  are not promised to resume in place.
- Desktop targets Linux X11, macOS and Windows. Controller/batch's first-release host is Linux.
- Desktop actor defaults to screenshot/input; accessibility is optional. Web includes DOM/ref actions.
- Setup and evaluators may use constrained shell/file tools; they are not default Desktop actor tools.
- Public NexAU is pinned. OpenAI-compatible Chat, Responses, Anthropic and Gemini adapters remain.
- Every fresh/prepared environment belongs to one attempt; no task state is reused through a warm pool.
- Ordinary skills remain; recursive skill/checkpoint-search research systems are not shipped.
- License: Apache-2.0, as selected by the project owner.

## Retained because removal is not yet justified

| Area | Why it remains | Boundary |
|---|---|---|
| NexAU creation/history/provider adapters | Focused regressions depend on their semantics | Pin and contract tests; no private dependency |
| AgentKit and shared runtime abstractions | Used by both agents, evaluators and device transport | Public extension mechanism, not a third bundled agent |
| External/managed task custody and policy-route interfaces | Generic task/model-routing integrations | No compatibility promise to private RL clients |
| Case recipes and script evaluation | Needed for task setup and externally supplied oracles | Users supply app catalogs and task data |
| Legacy browser/shell MCP contracts | Still used by evaluator and generic tool integration paths | Not default Desktop actor tools |
| E2B SDK compatibility and reliable cleanup | Runtime cancellation/late-create tests depend on them | Cloud canary remains a separate release gate |
| Protocol extension values | Used by generic plugin contracts | No bundled mobile-agent support is advertised |
| Fake provider and deterministic fixture traces | Exercise failure/recovery without cloud credentials | Always identified as simulated, never proof of live task success |

Further removal needs dependency and behaviour evidence. An uncertain component is
kept with a stated boundary rather than silently deleted or presented as supported
in every environment. The detailed original file inventory and cleanup audit are
maintained alongside the implementation review outside the distributable packages.

## Follow-up audit, 2026-10-03

A second review traced surviving entry points and failure paths. It removed the duplicate
CLI registry/adapter chain and export-only visual planner, replaced historical test-template
identifiers with synthetic fixture names, and deleted the unused root configuration stub
and environment-builder contract kind. Active CLI handlers and provider adapters remain.

The review also found behavior that a deletion audit alone cannot establish: WebSocket
admission/shutdown races, a missing Desktop failure terminal, logical session-ID mismatch,
and remote cleanup reporting/cache retention. These are repaired and covered by targeted
regressions; see [verification](verification.md) for current run and review status.

The subsequent Desktop simplification keeps one tool-returned-image path. It deletes
the pre-round probe policy, user-observation projection, their strategy factory and
configuration types, custom observation-chain configuration, the empty window/control
compatibility chain, and the duplicate legacy collector. Unused window/control/text
probe adapters and export-only prompt interfaces are removed with their callers.
Unused Desktop wait/clipboard configuration fields and orphaned driver test helpers
are also removed; their presence did not correspond to active runtime behavior.
Both agents use the paper's sliding overlay/clean allocator; the generic-budget branch
and its selector are removed. Initial capture, action screenshots, history budgets,
optional accessibility records and evaluator screenshots remain active functionality.

“Clean” describes a maintained, bounded implementation, not proof that every retained line
is necessary or every environment has been exercised. The retained-uncertainty table above
remains deliberate. New deletion proposals should trace actual callers and preserve the
behavioral coverage of surviving paths.

## Desktop settings and response policies

The optional [OSWorld1-style](desktop-settings-osworld1.md) and
[OSWorld2-style](desktop-settings-osworld2.md) settings retain Desktop prompt,
coordinate/history projection, context compaction and bounded response policies.
These mechanisms belong to the core Harness and use its ordinary task entry points.
Benchmark launchers, DesktopEnv lifecycle adapters, external score callbacks,
score aggregation and user-simulator bindings are outside this release. The
OSWorld HTTP screenshot/input backend remains available for a prepared desktop.
