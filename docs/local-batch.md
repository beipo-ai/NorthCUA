# A real local batch, then resume

This tutorial uses the existing BatchRunner, controller, SQLite state, device
transport and evaluator. Each attempt starts a new **local browser worker** and
browser context. The `local-process` provider is for trusted small Web batches on
macOS/Linux. It is not an OS sandbox: it shares the host filesystem, user and
network. Desktop runs separately, one task per display.

Install using [first run](first-run.md). Activate the same environment in all
terminals. No hosted-model key, cloud account or remote path is required below.

## Start the fixture and controller

Terminal A (keep open):

```sh
northcua demo serve --directory logs/batch-tutorial --response-delay 1
```

Wait for `ready: true`. Terminal B (keep open):

```sh
northcua controller serve --app-config logs/batch-tutorial/app.batch.yaml \
  --host 127.0.0.1 --port 8765 --executor-mode in_process \
  --runner-workers 2 --worker-queue-max 4 \
  --db-path logs/batch-tutorial/controller.sqlite \
  --blob-dir logs/batch-tutorial/blobs \
  --artifact-root-dir logs/batch-tutorial/controller-artifacts \
  --insecure-dev-no-auth
```

The generated `batch.yaml` already has three tasks, their fixture URLs and the
absolute controller DB path. Wait for the controller to start before running it.
For another port, pass `--controller-port` to `demo serve` and the same `--port`
to the controller. Only loopback connections may use the no-auth option.

## Validate, plan, execute

Terminal C:

```sh
northcua batch validate --manifest logs/batch-tutorial/batch.yaml
northcua batch run --manifest logs/batch-tutorial/batch.yaml \
  --dry-run --output-dir logs/batch-tutorial/plan
northcua batch run --manifest logs/batch-tutorial/batch.yaml \
  --output-dir logs/batch-tutorial/batches
northcua batch status --batch-id local-web --output-dir logs/batch-tutorial/batches
northcua batch report --batch-id local-web --output-dir logs/batch-tutorial/batches
```

Use a separate output root for a dry run. A successful run has **three succeeded
tasks**, each with one attempt, and three real page-action receipts in
`events.jsonl`. Each actor writes a note and clicks the page; each evaluator must
observe its own receipt in the original browser. Inspect individual task errors
and cleanup errors even when the runner command itself returns normally.

Persistent state and reports are under `batches/local-web/`: `manifest.json`,
`run_config.json`, `batch_state.sqlite`, `batch_summary.json`, telemetry and
`sandbox-logs/`. Controller task artifacts are in `controller-artifacts/`.
Do not edit SQLite directly or delete evidence to make a failed run appear green.

Read each controller session's `result.json` → `evaluation_report.overall_verdict`
and `evaluations/web_round_evaluator/round_0001/final_report.json` for the live
evaluator verdict. The batch summary's `evaluation_status` / `evaluation_attempts`
projection can remain empty / zero for this runtime path; it is not the outcome
check. Keep actor status, evaluator verdict and cleanup confirmation separate.

## Resume without repeating successful tasks

Keep the same controller and fixture alive:

```sh
northcua batch resume --batch-id local-web --output-dir logs/batch-tutorial/batches
```

After a completed batch this should preserve three successes and their attempt
counts. It must not produce another round of page clicks. Check `batch status`
and the count of `web_action` events before and after.

For an actual drain/recovery exercise, create a fresh fixture directory, set
`--response-delay 3`, and use `--concurrency 1` on the first `batch run`. While
the first task is running, in terminal D:

```sh
northcua batch drain --batch-id local-web --output-dir logs/batch-tutorial/batches
```

Drain lets admitted work finish and leaves queued work for `batch resume`.
The drained runner exits **1** when queued work remains: the batch is deliberately
incomplete, not a failed drain operation. Confirm task states before resuming.
Wait for the original runner to exit before resuming. Resume uses the saved
manifest and reconciles known controller tasks; do not change its task list or
reuse its batch ID for a different experiment. `pause` and `cancel` have different
semantics; see [batch operations](batch.md).

Prefer drain to abruptly killing the runner. If the runner dies, a local
supervisor stops its workers and browsers; incomplete attempts may need a retry.
This does not imply that arbitrary external website actions can be safely replayed.
Completed tasks are preserved by the existing reconciliation path.

## Switch to your real model and tasks

Keep the `local-process` manifest structure. Start the controller with the model
configuration from [first run](first-run.md#2a-use-your-model-edit-only-model-connection-settings)
instead of `app.batch.yaml`; change task URLs, queries and budgets. Set
`sandbox.metadata.controller_db` to that controller's absolute DB path. The demo
service is only needed if you still use its fixture pages.

The normal model template disables evaluation by default. To retain live outcome
checks, set both `evaluation.enabled: true` and
`evaluation.tools.allow_client_live_probe: true` in the controller's app YAML.
The evaluator uses the configured model too; provision an appropriate task budget
and inspect its report separately from actor completion.

| Setting | Starting value | Meaning |
|---|---|---|
| `defaults.concurrency` | 2 | Active tasks; start with 1 for a new endpoint |
| `defaults.max_retries` | 1 | One extra whole-task attempt after a retryable failure |
| `defaults.timeout_seconds` | 120 | Controller-task wait limit |
| `sandbox.resources.timeout_seconds` | 300 | Local worker lifetime, renewed by the lease mechanism |
| `tasks[].max_steps` | 6 for this demo | Actor decision budget; use a suitable budget for real work |

The local provider accepts only Web/browser, fresh per-task allocations. It
rejects guest shell commands, uploads, post-run hooks, prepared queues and
CPU/memory/disk isolation requests. It uses the running Python installation;
it does not install packages or build templates. Use the documented E2B path
when you need separately provisioned environments.

## Shut down

Wait for batch completion or drain it. Stop the controller, then the fixture
service with Ctrl-C. Each released local allocation has a cleanup receipt under
the OS temporary directory `northcua-local-<uid>/local-<uuid>/finished.json`;
diagnostic files may remain there, but worker and browser processes must end.
The receipt must include `cleanup_confirmed: true`; a supervisor exit without
that confirmation is a cleanup error, not success. Detached browser descendants
are tracked by process identity and an inherited allocation marker. Cleanup scans
again during shutdown to find children whose parent has already exited. Unexpected
worker exits and unreadable newly appeared processes leave cleanup unconfirmed.
This applies to the installed worker and Playwright, which inherit the marker;
arbitrary daemon commands are unsupported.
The provider never kills a process based only on a PID read from old state.
See [troubleshooting](troubleshooting.md) if cleanup was not confirmed.
