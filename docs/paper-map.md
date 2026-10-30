# Paper → implementation → evidence

This repository releases the **NorthCUA Harness** described in §3 of the
[NorthCUA Technical Report](paper/main.pdf), **31 October 2026, 123 pages**.
The broader research system also includes Experience Flywheel and Staged Learning.
The report contains model results; the runtime tests here measure different
contracts and do not reproduce those model experiments.

The [paper manifest](paper/manifest.json) identifies the exact supplied Compact
PDF by SHA-256. The PDF is distributed without content edits. The current logo
and mechanism figures are sourced from that same report snapshot.

## Harness mechanisms

| Current report section | Role in the clean implementation | Code and evidence |
|---|---|---|
| §3.1 Unified Execution Across Desktop and Web | Common session/round/step progression with domain-specific observations and actions | `northcua/agentkit/runtime/`, `northcua/agents/`; hierarchy, runner and transport tests |
| §3.1.2 Tool Integration and Dispatch | Typed tool contracts, local/MCP dispatch and post-action feedback | `northcua_runtime_shared/tool_contracts/`, `northcua/runtime/mcp/`; schema and execution regressions |
| §3.1.3 Desktop and Web Instantiations | Native/OSWorld Desktop tools; Playwright screenshots, DOM/ref and tab state | Backend tests and opt-in real-browser/native checks |
| §3.2.1 Action-Aware Visual Context | Shared overlay/clean allocation; one total Web image budget including SoM | `northcua/agentkernel/prompt/`, Web `prompt/visual_window/message_projector.py`; final-message image-count tests |
| §3.2.2 Task-Preserving Context Compaction | Complete-exchange segmentation, pinned task/current state, summary projection and checkpoints | `northcua/agentinfra/context/`; context, history and overflow regressions |
| §3.2.3 Structured Working Memory | Idempotent durable note events, replacement by key, round-scoped active view | `northcua/agentkernel/working_memory.py`; persistence and real-provider projection tests |
| §3.3 Agentic Outcome Evaluation | Recorded-evidence retrieval, authorized live probes and a separate final report | Shared/Desktop/Web evaluator modules; lifecycle, round and same-session integration tests |
| §3.3.2 Task-Grounded, Multidimensional Rubrics | Structured criteria, outcome, execution assessment and evidence references | Evaluator contracts, final-report validation and prompt regressions |
| §3.4.1 Remote Execution with Centralized Agent Control | Controller runners with leased device tools; in-process and process executors | Controller/device/protocol packages; transport, process and real Web mode checks |
| §3.4.2 Isolated Environments with Pipelined Preparation | Fresh environments and task/attempt-bound prepared reservations | Batch preparation, assignment and cleanup failure tests |
| §3.4.3 Completion-Oriented Scheduling and Recovery | Claims, deadlines, backpressure, reconciliation, retries and artifact custody | Batch, controller and sandbox lifecycle regressions |

The [system tour](system-tour.md) follows this argument with the current paper's figures.
Exact executed modes and reviewer scope are in [verification](verification.md).
A test listed here is evidence for its checked contract; it is not a measurement of
long-horizon capability or deployment throughput.

## Research scope and release scope

| Broader research component | Relationship to this repository |
|---|---|
| Experience Flywheel: grounding examples and intent/environment/verifier expansion | Supplies tasks and data in the research system; data production and generation pipelines are removed here. |
| Recursive skill-guided exploration and checkpoint-assisted collection | Research-system hard-task collection methods; not bundled implementations. Ordinary skill loading remains available. |
| Mid-training, long-horizon SFT and agentic RL | Training systems and model checkpoints are outside this release. Generic external/managed task interfaces remain for integrations. |
| Benchmark/model capability panels | Reported model results are in the current report; this repository's scripted-model integration tests are not a benchmark reproduction. |

## Provenance and maintenance

The source was prepared from the original NorthCUA clean branch; historical implementation lineage is retained in the archived verification records, while this publication starts a new Git history.
Historical implementation references in the research materials identify other revisions;
they are not the release's execution evidence. The clean branch adds durable working
memory, repairs evaluator resource ownership and makes Web image budgeting explicit.
These changes should be attributed to the release rather than retroactively to historical
research runs.

When changing a mechanism, update its operational guide, behavioral tests and this map.
When reporting an experiment, retain the exact paper/code/NexAU revisions, task source,
model/provider settings, observation/action profile, budgets, evaluator configuration and
artifacts. Use [CITATION.cff](../CITATION.cff) for citation metadata.
