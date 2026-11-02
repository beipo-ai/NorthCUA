# NorthCUA documentation

This repository provides the **NorthCUA Harness** for Desktop and Web agents.
It connects a request to actions, observations, outcome checks and a durable execution
record. Start with one local task, then choose the deployment and evaluation path your
workload needs. Commands, Python imports and package names use `northcua`.

## Start here

| Your goal | Guide |
|---|---|
| Start from zero, with model templates or no-key local checks | [First run](first-run.md) |
| Let an external Agent operate the project | [Operator skill](../skills/northcua/SKILL.md) and [skill guide](skills.md) |
| Run three real local browser tasks and resume | [Local batch tutorial](local-batch.md) |
| Understand what the harness provides | [System tour](system-tour.md) |
| Install and run a first Desktop/Web task | [Getting started](getting-started.md) |
| Choose a model or check research-model availability | [Model setup](models.md) and [resource status](resources.md) |
| Choose a provider, tool surface and budgets | [Configuration](configuration.md) |
| Inspect what happened and judge the result | [Evaluation and artifacts](evaluation.md) |
| Find and fix a failure | [Troubleshooting](troubleshooting.md) |

## Run and scale

[Execution modes](running.md) covers Python, CLI, local WebSocket serving and controller/device
deployment. [Batch execution](batch.md) adds attempt isolation, preparation, persistent
recovery and reporting. [Environments](environments.md) covers public E2B templates and
upstream OSWorld integration.

For an already prepared benchmark desktop, the optional [OSWorld1 actor profile](osworld1-harness.md)
and [OSWorld2 actor profile](osworld2-harness.md) expose versioned prompts and bounded
response policies. Their environment and evaluator backends remain caller-owned;
the first-run tutorials above remain the local starting point.

## Understand and extend

[Context and memory](context.md) explains visual windows, compaction and notes.
[Architecture](architecture.md) connects those mechanisms to the code and package boundaries.
[The paper map](paper-map.md) links the research design to concrete implementations and
regression coverage. [Contributing](../CONTRIBUTING.md) describes the development workflow.

[Recorded project workflows](showcase.md) are bundled separately from executable tutorials.

## Check the evidence

[Testing](testing.md) gives reproducible commands. [Release verification](release-0.2.0.md) records
what actually ran and the scope of independent reviewer signoffs.
[Clean scope](clean-scope.md) explains removals, retained compatibility and open decisions.
[Security](../SECURITY.md) describes operational boundaries and reporting.
[Release preparation](releasing.md) covers candidate gates, source inspection
and the separate publication steps.

Use `northcua config schema --kind app` or `--kind batch` for complete generated
JSON schemas and `--help` for optional CLI flags. Tutorials use supported public examples. Deterministic tests and research
illustrations are labeled so they can be distinguished from real-model evaluations.
