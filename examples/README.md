# Examples

Start with [the from-zero tutorial](../docs/first-run.md). `northcua demo serve`
generates a complete local app/model configuration and a three-task real browser
manifest; [local batch](../docs/local-batch.md) runs it through the controller.
The demo model is scripted and clearly labeled; browser/device execution is real.
Copy [the model env template](../.env.example) to switch to your own endpoint.

- `skills/local-form/`: minimal runtime skill; see [loading instructions](../docs/skills.md).

- `batch-web.yaml` and `batch-desktop.yaml`: fresh attempts against your controller.
- `batch-fake.yaml`: deterministic scheduling demo; not a live agent benchmark.
- `templates/web.yaml` and `templates/desktop.yaml`: public E2B substrates built from release wheels.
- `osworld/`: configure the native HTTP backend for a public OSWorld DesktopEnv.
- `fixtures/web-e2b-three/`: small local pages used for deterministic browser checks.

Replace endpoint/credential settings and build the templates before a live run.
Historical private inventories and dated experiment recipes are not public examples.
See [the environment guide](../docs/environments.md) and [batch guide](../docs/batch.md).

OSWorld2 formal harness configuration (no environment or dispatch code): [osworld2](../docs/osworld2-harness.md).
OSWorld1 GUI actor configuration, with legacy prompt/coordinate projection: [osworld1](../docs/osworld1-harness.md).
