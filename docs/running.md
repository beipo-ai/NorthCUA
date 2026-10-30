# Execution modes

| Mode | Purpose | Entry point |
|---|---|---|
| Local Desktop/Web | Run agent and tools on one host | `northcua local desktop/web run` |
| Local websocket server | Submit repeated requests to a running application | `northcua local desktop/web serve` |
| Controller in-process | Central model execution, tools on a leased worker | `northcua controller serve --executor-mode in_process` |
| Controller process | Isolate each agent execution in a worker process | `northcua controller serve --executor-mode process` |
| Batch | Fresh or task-prepared sandbox attempts with persistent recovery | `northcua batch run/resume` |
| Artifact evaluation | Evaluate saved evidence without recreating the task | `northcua local desktop/web evaluate` |

Desktop backends target Linux X11, macOS and Windows. Controller/batch orchestration
is supported on Linux for this release. Platform implementation and completed physical
validation are separate; consult the [verification record](verification.md).

## Local WebSocket server

In a source checkout, activate `.venv` first (`source .venv/bin/activate`), or prefix
commands with `uv run --no-sync`. Wheel installations use the active Python environment.

```sh
northcua local web serve --app-config config/app.example.yaml --session-id web-server
# Desktop: northcua local desktop serve --app-config config/app.example.yaml
```

The server uses `websocket.host` and `websocket.port` from app configuration. The default
is loopback. This local task protocol is separate from the controller submission protocol.
For example, a Web client can send and receive typed envelopes:

```python
import asyncio
import websockets
from northcua.core.enums import WebSocketMessageType
from northcua.core.protocol import NorthCUAWebSocketEnvelope, WebTaskRequestPayload

async def main():
    async with websockets.connect("ws://127.0.0.1:8765", proxy=None) as socket:
        request = WebTaskRequestPayload(
            session_id="my-task", request="Report the main heading.",
            url="https://example.com", max_steps=8,
        )
        await socket.send(NorthCUAWebSocketEnvelope.wrap(
            WebSocketMessageType.TASK_REQUEST, request,
        ).model_dump_json())
        while True:
            response = NorthCUAWebSocketEnvelope.model_validate_json(await socket.recv())
            print(response)
            if response.type in {WebSocketMessageType.TASK_FINISHED, WebSocketMessageType.ERROR}:
                break

asyncio.run(main())
```

A session is reserved before task acceptance. Concurrent Web requests for the same session
are rejected. Local Desktop serving permits one task across sessions because they share
one physical display. Stop closes admission, cancels pending/active work and drains cached
facades. These guarantees cover one server process; do not launch separate Desktop servers
against the same display.

## Controller and worker

Run these in separate terminals for a loopback development setup:

```sh
northcua controller serve --app-config config/app.example.yaml \
  --host 127.0.0.1 --port 8765 --executor-mode in_process --insecure-dev-no-auth
northcua-device connect --server ws://127.0.0.1:8765 --platform browser \
  --agentkit-agent-kind web --device-id browser-1 --insecure-dev-no-auth
northcua submit web --controller ws://127.0.0.1:8765 \
  --target-device-id browser-1 --url https://example.com \
  --query 'Report the main heading.' --wait
```

For a native worker choose `--platform linux`, `macos` or `windows` and use
`northcua submit desktop`. For process isolation set `--executor-mode process`.
The controller configuration owns the provider credentials; tools execute on the worker.

For a network deployment use the generated controller/device configs with matching
non-placeholder device and submitter tokens, and expose a reachable `wss://` endpoint.
The no-auth flag is a loopback development convenience. Enforce the deployment's
capability/tool allowlist and configure artifact storage before accepting tasks.

Local websocket `serve` uses the same application facade as local `run`. Web evaluation
is executed within the active browser lifetime in all entry points. A session can have
multiple rounds, but each new task round has a separate active working-memory scope.
Model/tool traces and round results remain accessible after the round ends.

## Python

```python
import asyncio
from northcua.agents.web import WebAgentApplication
from northcua.agentinfra.runtime.protocol import WebRunRequest

async def main():
    app = WebAgentApplication.from_app_config("config/app.example.yaml", session_id="example")
    try:
        return await app.run_query(WebRunRequest(
            session_id="example", query="Report the main heading", url="https://example.com", max_steps=8,
        ))
    finally:
        await app.stop()

asyncio.run(main())
```

The generic external-task and managed-task APIs remain available for integrations.
They carry task IDs, leases and artifact custody; compatibility with older private
training clients is not part of the clean release contract.
