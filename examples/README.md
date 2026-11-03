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
- `desktop-settings/`: optional Desktop prompt, coordinate, history and response-policy recipes.
- `fixtures/web-e2b-three/`: small local pages used for deterministic browser checks.

Replace endpoint/credential settings and build the templates before a live run.
Historical private inventories and dated experiment recipes are not public examples.
See [the environment guide](../docs/environments.md) and [batch guide](../docs/batch.md).

See [recommended Desktop settings](desktop-settings/README.md) for the optional recipes.
