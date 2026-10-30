# Evaluation and artifacts

Actor completion and evaluator verdict are separate records. Enable evaluation in app
YAML, or override it for a local web run with `--evaluate` / `--no-evaluate`.

```yaml
evaluation:
  enabled: true
  max_steps: 12
  force_finalizer_after_steps: 8
  round_timeout_seconds: 300
  fail_open: true
  tools:
    allow_client_live_probe: true
    allow_client_mutating_tools: false
```

The finalizer budget is clamped to leave a final reporting step. The same configured
step limits apply to Desktop and Web evaluators. Artifact inspection is available
without a live environment; live browser tools are bound to the actor's original
browser/context/page. Read-only probes are the default. Mutating evaluator tools require
explicit opt-in because they change the state being judged.

Web finalization order is: terminal evidence → actor result → evaluator → evaluated
result → browser release. A release failure is recorded and cannot be reported as a
successful cleanup. Evaluator exceptions preserve actor evidence and release the browser.
With fail-open enabled they are recorded as evaluation errors; with fail-open disabled
the run fails. Cancelling an actor does not start a new evaluator.

A Web session writes `result.json` for its latest round and
`rounds/round_NNNN/result.json` for each individual round. Evaluation files live under
`evaluations/web_round_evaluator/round_NNNN/`. Re-evaluation selects the requested round;
a missing round is an explicit error.

```sh
northcua local web evaluate --app-config config/app.example.yaml \
  --artifact-dir logs/example --round-num 1
northcua local desktop evaluate --app-config config/app.example.yaml \
  --artifact-dir logs/example
```

Offline evaluation uses saved evidence and does not open a replacement browser. A live
probe cannot truthfully reproduce a page's expired in-memory state from artifacts alone.

Common artifacts include step traces, model calls, DOM snapshots, screenshots, action
overlays, terminal verification, evaluator bundles/reports, working memory, summary
checkpoints and timing/usage telemetry. The controller catalogs references and checksums
rather than treating file paths as evidence of successful upload. Batch completion saves
required artifacts before releasing the attempt environment.

The optional Web trace viewer reads saved traces:

```sh
uv sync --locked --extra local --extra device --extra viewer
uv run --no-sync northcua-web-viewer --artifact-root logs/example
```

For transport bottlenecks use `northcua-transport-report --help`.
