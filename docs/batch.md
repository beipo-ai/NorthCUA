# Batch execution and recovery

Start with [three real local Web tasks](local-batch.md) if you have no cloud
environment. It uses `local-process` and the same durable scheduler. `fake` is a
simulation; it does not execute browser or Desktop actions.

## Manifest fields

| Field | Meaning |
|---|---|
| `batch_id` | Stable identity within the output root; change it for a new workload |
| `defaults.controller_url` | Endpoint used by the batch submitter |
| `defaults.device_controller_url` | Optional worker-facing endpoint; leave empty for local use |
| `defaults.concurrency` | Active task limit; start at 1–2 |
| `defaults.sandbox` | Provider, template, platform and attempt lifetime |
| `defaults.max_retries` | Extra whole-task attempts after retryable failures |
| `defaults.timeout_seconds` | Controller task wait deadline |
| `tasks[].task_id` | Unique stable task identity, used by resume |
| `tasks[].agent_kind`, `query`, `url` | Desktop/Web kind, requested task, optional initial Web URL |
| `tasks[].max_steps` | Actor budget, independent of evaluator and infra timeouts |
| `tasks[].sandbox` | Complete per-task sandbox override; not a partial merge |

Use `northcua config schema --kind batch` for the full field reference. YAML/JSON
manifests contain defaults and tasks; JSONL contains one task per line and uses
default provider settings, so prefer YAML when learning local batching.

## Operate and recover

`status` reads persisted task states; `report` assembles the summary and failure
breakdown. `drain` stops new admissions while admitted tasks finish. `pause`
pauses scheduling; `cancel` requests cancellation. Wait for the current runner
to exit before `resume`. Keep controller identity, task IDs and saved configuration
stable. Resume reconciles previously submitted tasks rather than blindly rerunning
them. A new task list belongs in a new batch.

Infrastructure errors, task failure and evaluation failure are different outcomes.
Read `last_failure_category`, `error`, attempts and evaluator status in the report
before changing a retry budget. A successful actor with pending/failed evaluation
is not a verified successful task. Do not manually reset attempts in SQLite.

## Cloud environments

Batch manifests specify tasks, the controller endpoint, sandbox provider/template,
step/time budgets and concurrency. Published examples cover [web](../examples/batch-web.yaml)
and [desktop](../examples/batch-desktop.yaml); edit their endpoint and task first.

```sh
northcua batch validate --manifest examples/batch-web.yaml
northcua batch run --manifest examples/batch-web.yaml --dry-run --output-dir logs/batch
northcua batch run --manifest examples/batch-web.yaml \
  --controller-url wss://your-controller.example/ws --token "$NORTHCUA_CONTROLLER_TOKEN" \
  --parallelism 2 --output-dir logs/batch
northcua batch status --batch-id web-example --output-dir logs/batch
northcua batch resume --batch-id web-example --output-dir logs/batch \
  --token "$NORTHCUA_CONTROLLER_TOKEN"
```

Build the [public sandbox templates](environments.md) before a real E2B run. Set the
provider's API key in the provisioning process. A real high-level experiment/case run
also requires `NORTHCUA_ALLOW_LIVE_SANDBOXES=1`. Secrets are not saved in run configuration
unless the explicit token-persistence option is used.

Each task attempt gets a fresh environment. The prepared-sandbox queue may provision
and initialize an upcoming **specific task/attempt** while other tasks run; the prepared
environment is consumed once. There is no reusable warm-pool allocation strategy.
Use `northcua batch run --help` for preparation, claim-wait, create/submit/status
concurrency and backpressure options. Start at concurrency 1 and increase after a canary.

The SQLite state repository owns claims, heartbeats, retry budgets, prepared reservations
and terminal status. Resume first reconciles known controller tasks. A finished controller
task is not blindly resubmitted because a local process crashed before saving its result.
Infra failures, evaluator failures, task failures and cleanup failures have distinct
records. Cancellation, pause and drain are explicit operations.

```sh
northcua batch drain --batch-id web-example --output-dir logs/batch
northcua batch report --batch-id web-example --output-dir logs/batch
```

Maintenance operations default to a plan/dry run. Inspect the scope before applying
cleanup. Legacy reusable environments are excluded from automatic cleanup ownership.
Old run directories are evidence, not supported in-place resume inputs for 0.2.0.

`experiment run --spec` adds fixture staging, managed-controller lifecycle and reporting.
`case run --spec` adds app catalog and setup/evaluation recipes. The bundled catalog only
names the two public substrates; provide a task-specific catalog for application
requirements. Historical private template inventories are not shipped as defaults.
