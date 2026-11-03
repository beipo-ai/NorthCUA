# Public environments

## E2B

The two [template definitions](../examples/templates/) install the four locally built
release wheels. Desktop extends E2B's public `desktop` template; Web uses the public
Python 3.12 base. No historical NorthCUA account-specific template is required.
The upstream desktop substrate can also be rebuilt from the
[E2B desktop template source and guide](https://github.com/e2b-dev/desktop/tree/main/template).

```sh
uv sync --locked --extra local --extra dev --extra e2b
uv run --no-sync python -m scripts.build_wheelhouse
export E2B_API_KEY='your-key'
uv run --no-sync northcua sandbox template build --help
uv run --no-sync northcua sandbox template build --spec examples/templates/web.yaml
uv run --no-sync northcua sandbox template build --spec examples/templates/desktop.yaml
```

Run builds from the repository root so the `wheelhouse/` copy is in the build context.
The cloud account needs permission to build the chosen aliases. Template build checks
validate metadata without requiring a running display; a live canary still must verify
screenshots, input, worker readiness and cleanup. A successful image build is not that canary.

For wheel-only controller installations set `NORTHCUA_RUNTIME_WHEELHOUSE` to a directory
containing exactly one wheel for each of the four distributions. High-level experiments
using `runtime.mode: upload_wheels` stage that release without needing source files or
`pyproject.toml` under site-packages. Dependency installation happens inside the sandbox;
its network/package access must be configured by the deployment.

Current native templates use Linux X11. Template app catalogs are task-specific:
install the application, declare its launch/readiness requirements, and provide its
fixtures and oracle. The bundled catalog does not advertise uninstalled office or
proprietary applications. The [verification record](verification.md) states whether a
real cloud build/run has been completed for this candidate.

## Desktop HTTP backend

For an already running desktop exposing the OSWorld-compatible HTTP screenshot
and input service, configure the app YAML:

```yaml
desktop:
  backend: osworld
  osworld:
    vm_ip: 127.0.0.1
    server_port: 5000
  coordinate_policy:
    input_space: normalized_1000
```

Replace the host and port with the endpoint you own; match the coordinate policy
to the model's output convention. Run the ordinary Desktop entry point:

```sh
northcua local desktop run --app-config logs/my-config/app.example.yaml \
  --query 'Open a calculator and compute 12 * 7.' --max-steps 50
```

The backend provides screenshots, input and optional accessibility records.
The environment owner prepares the desktop and manages its lifecycle. Task JSON,
benchmark launchers and scorers are not part of this backend. [Recommended Desktop settings](desktop-settings.md) provide model, history,
context and response-policy recipes for the same ordinary Harness.
