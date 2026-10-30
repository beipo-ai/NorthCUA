"""No-LLM process worker used by controller concurrency simulations."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os

from northcua.controller.execution.processes.contracts import (
    WorkerArtifactManifest,
    WorkerCommandDispatchRequest,
    WorkerEnvelope,
    WorkerError,
    WorkerHello,
    WorkerJobFailureKind,
    WorkerMessageType,
    WorkerRunRequest,
    WorkerTerminalResult,
)
from northcua.controller.execution.processes.ipc import read_envelope, write_envelope
from northcua.controller.execution.processes.worker_client import WorkerIPCClient, worker_heartbeat
from northcua_protocol.enums import MessageType
from northcua_protocol.payloads import TaskFinished
from northcua_protocol.state import TaskState


async def _run_worker(socket_path: str, *, nonce: str) -> int:
    reader, writer = await asyncio.open_unix_connection(socket_path)
    await write_envelope(
        writer,
        WorkerEnvelope(
            message_type=WorkerMessageType.HELLO,
            payload_json=WorkerHello(nonce=nonce, worker_pid=os.getpid()).model_dump_json(),
        ),
    )
    run_envelope = await read_envelope(reader)
    request = WorkerRunRequest.model_validate_json(run_envelope.payload_json)
    ipc = WorkerIPCClient(reader=reader, writer=writer, trace_id=request.request.trace_id)
    await ipc.start()
    await ipc.send_payload(WorkerMessageType.READY, "{}")
    heartbeat_task = (
        None
        if _metadata_bool(request.request.metadata, "fake_disable_heartbeat")
        else asyncio.create_task(_heartbeat_loop(ipc, interval_seconds=request.heartbeat_interval_seconds))
    )
    try:
        if _metadata_bool(request.request.metadata, "fake_worker_error"):
            await ipc.send_payload(
                WorkerMessageType.WORKER_ERROR,
                WorkerError(
                    failure_kind=WorkerJobFailureKind.TASK_FAILED,
                    message="fake worker reported error",
                    retryable=False,
                ).model_dump_json(),
            )
            return 1
        if _metadata_bool(request.request.metadata, "fake_spoof_progress"):
            await ipc.send_payload(
                WorkerMessageType.PROGRESS,
                json.dumps(
                    {
                        "message_type": MessageType.TASK_FINISHED.value,
                        "payload_json": TaskFinished(
                            task_id=request.request.task_id,
                            trace_id=request.request.trace_id,
                            session_id=request.session_id,
                            device_id=request.lease.device_id,
                            lease_id=request.lease.lease_id,
                            state=TaskState.FINISHED,
                        ).model_dump_json(),
                    }
                ),
            )
            await asyncio.sleep(0.1)
            return 1
        command_results: list[dict[str, object]] = []
        for index in range(_metadata_int(request.request.metadata, "fake_command_count", default=0)):
            response = await ipc.send_command(
                WorkerCommandDispatchRequest(
                    request_id=f"fake-command-{index}",
                    tool_id="fake.tool",
                    parameters={"index": index},
                    command_id=f"fake-command-{index}",
                    timeout_seconds=1.0,
                    metadata_overrides={"worker_job_id": request.job_id},
                )
            )
            if response.result is None:
                raise RuntimeError(response.error or "fake command dispatch failed")
            command_results.append(_command_result_payload(response.result.result))
        await asyncio.sleep(_metadata_float(request.request.metadata, "fake_worker_seconds", default=0.01))
        terminal = WorkerTerminalResult(
            finished=TaskFinished(
                task_id=(
                    "wrong-task"
                    if _metadata_bool(request.request.metadata, "fake_bad_terminal")
                    else request.request.task_id
                ),
                trace_id=request.request.trace_id,
                session_id=request.session_id,
                device_id=request.lease.device_id,
                lease_id=request.lease.lease_id,
                state=TaskState.FINISHED,
                result={"status": "ok", "worker_mode": "fake", "command_results": command_results},
            ),
            artifact_manifest=WorkerArtifactManifest(),
        )
        await ipc.send_payload(WorkerMessageType.TERMINAL, terminal.model_dump_json())
        return 0
    finally:
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task
        await ipc.close()


async def _heartbeat_loop(ipc: WorkerIPCClient, *, interval_seconds: float) -> None:
    while True:
        await ipc.send_heartbeat(worker_heartbeat(phase="fake"))
        await asyncio.sleep(interval_seconds)


def _metadata_int(metadata: dict[str, object], key: str, *, default: int) -> int:
    value = metadata.get(key)
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return default


def _metadata_float(metadata: dict[str, object], key: str, *, default: float) -> float:
    value = metadata.get(key)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        with contextlib.suppress(ValueError):
            return float(value)
    return default


def _metadata_bool(metadata: dict[str, object], key: str) -> bool:
    value = metadata.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False


def _command_result_payload(result: object) -> dict[str, object]:
    return result if isinstance(result, dict) else {"value": result}


def main() -> int:
    """Run the fake process worker.

    Returns:
        Process exit code from the async worker entrypoint.
    """

    parser = argparse.ArgumentParser(description="NorthCUA fake controller process worker")
    parser.add_argument("--socket", required=True)
    parser.add_argument("--nonce", required=True)
    args = parser.parse_args()
    return asyncio.run(_run_worker(args.socket, nonce=args.nonce))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
