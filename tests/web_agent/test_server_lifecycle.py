"""Exercise connection interleavings at the server's real admission boundary."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from northcua.agents.web.server import WebAgentWebSocketServer
from northcua.core.enums import WebSocketMessageType
from northcua.core.protocol import NorthCUAWebSocketEnvelope, WebTaskRequestPayload


class Socket:
    def __init__(self, *, block_ack=False, fail_ack=False):
        self.types = []
        self.ack_entered = asyncio.Event()
        self.release_ack = asyncio.Event()
        self.block_ack = block_ack
        self.fail_ack = fail_ack

    async def send(self, message):
        kind = NorthCUAWebSocketEnvelope.model_validate_json(message).type
        self.types.append(kind)
        if kind == WebSocketMessageType.TASK_ACCEPTED:
            self.ack_entered.set()
            if self.fail_ack:
                raise OSError("connection closed before ACK")
            if self.block_ack:
                await self.release_ack.wait()


def _web_server():
    facade = SimpleNamespace(session_id="same", drain_background_tasks=AsyncMock())
    server = WebAgentWebSocketServer(facade=facade, host="127.0.0.1", port=0)
    server._run_web_task = AsyncMock()
    request = NorthCUAWebSocketEnvelope.wrap(
        WebSocketMessageType.TASK_REQUEST, WebTaskRequestPayload(session_id="same", request="Task")
    )
    return server, facade, request


@pytest.fixture(params=["web", "desktop"])
def setup_server(request):
    if request.param == "web":
        return _web_server

    def factory():
        from northcua.agents.desktop.server import NorthCUAWebSocketServer
        from northcua.core.protocol import DesktopTaskRequestPayload
        from tests.transport.websocket_server_test_utils import _DummyFacade, _DummyTransport

        facade = _DummyFacade(session_id="same")
        server = NorthCUAWebSocketServer(facade=facade, execution_transport=_DummyTransport(), host="127.0.0.1", port=0)
        server._run_pending_rounds = AsyncMock()
        server._run_web_task = server._run_pending_rounds
        facade.drain_background_tasks = AsyncMock()
        envelope = NorthCUAWebSocketEnvelope.wrap(
            WebSocketMessageType.TASK_REQUEST, DesktopTaskRequestPayload(session_id="same", request="Task")
        )
        return server, facade, envelope

    return factory


@pytest.mark.asyncio
async def test_same_session_is_reserved_before_sending_ack(setup_server):
    server, facade, request = setup_server()
    first, second = Socket(block_ack=True), Socket()
    pending = asyncio.create_task(server._dispatch_envelope(first, request))
    await first.ack_entered.wait()
    await server._dispatch_envelope(second, request)
    assert second.types == [WebSocketMessageType.ERROR]
    first.release_ack.set()
    await pending
    await asyncio.gather(*server._background_tasks)
    server._run_web_task.assert_awaited_once()
    await server.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_ack_failure_or_cancellation_releases_reservation(cancel, setup_server):
    server, facade, request = setup_server()
    socket = Socket(block_ack=cancel, fail_ack=not cancel)
    pending = asyncio.create_task(server._dispatch_envelope(socket, request))
    await socket.ack_entered.wait()
    if cancel:
        pending.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else OSError):
        await pending
    assert not server._active_sessions and not server._session_connections and not server._background_tasks
    if hasattr(facade, "rounds"):
        assert not facade.rounds._queued
    await server._dispatch_envelope(Socket(), request)
    if hasattr(facade, "rounds"):
        assert len(facade.rounds._queued) == 1
    await asyncio.gather(*server._background_tasks)
    server._run_web_task.assert_awaited_once()
    await server.stop()


@pytest.mark.asyncio
async def test_shutdown_rejects_new_work_during_facade_drain(setup_server):
    server, facade, request = setup_server()
    draining, released = asyncio.Event(), asyncio.Event()

    async def drain():
        draining.set()
        await released.wait()

    facade.drain_background_tasks.side_effect = drain
    stopping = asyncio.create_task(server.stop())
    await draining.wait()
    socket = Socket()
    await server._dispatch_envelope(socket, request)
    assert socket.types == [WebSocketMessageType.ERROR]
    released.set()
    await stopping
    server._run_web_task.assert_not_awaited()
    assert not server._active_sessions and not server._background_tasks


@pytest.mark.asyncio
async def test_ack_in_progress_cannot_launch_after_shutdown(setup_server):
    server, facade, request = setup_server()
    socket = Socket(block_ack=True)
    pending = asyncio.create_task(server._dispatch_envelope(socket, request))
    await socket.ack_entered.wait()
    await server.stop()
    socket.release_ack.set()
    with pytest.raises(asyncio.CancelledError):
        await pending
    server._run_web_task.assert_not_awaited()
    assert not server._active_sessions and not server._session_connections and not server._background_tasks


@pytest.mark.asyncio
async def test_shutdown_drains_every_facade_and_clears_cache_on_error(setup_server):
    server, facade, request = setup_server()
    facade.drain_background_tasks.side_effect = RuntimeError("first drain failed")
    other = SimpleNamespace(drain_background_tasks=AsyncMock())
    server._facades["other"] = other
    with pytest.raises(ExceptionGroup):
        await server.stop()
    other.drain_background_tasks.assert_awaited_once()
    assert not server._facades and not server._active_sessions and not server._session_connections


@pytest.mark.asyncio
async def test_local_desktop_rejects_other_session_on_same_display():
    from northcua.agents.desktop.server import NorthCUAWebSocketServer
    from northcua.core.protocol import DesktopTaskRequestPayload
    from tests.transport.websocket_server_test_utils import _DummyFacade, _DummyTransport

    server = NorthCUAWebSocketServer(
        facade=_DummyFacade(), execution_transport=_DummyTransport(), host="127.0.0.1", port=0
    )
    server._active_sessions.add("first")
    socket = Socket()
    await server._dispatch_envelope(
        socket,
        NorthCUAWebSocketEnvelope.wrap(
            WebSocketMessageType.TASK_REQUEST, DesktopTaskRequestPayload(session_id="other", request="Task")
        ),
    )
    assert socket.types == [WebSocketMessageType.ERROR]
    await server.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["hydrate", "run"])
async def test_desktop_acknowledged_failure_has_terminal_with_logical_id(failure):
    from northcua.agents.desktop.server import NorthCUAWebSocketServer
    from northcua.core.protocol import DesktopTaskRequestPayload
    from tests.transport.websocket_server_test_utils import _DummyFacade, _DummyTransport, _DummyWebSocket

    facade = _DummyFacade(session_id="任务/a")
    if failure == "hydrate":
        facade.hydrate_state = AsyncMock(side_effect=RuntimeError("provider unavailable"))
    else:
        facade.run_round = AsyncMock(side_effect=RuntimeError("provider unavailable"))
    server = NorthCUAWebSocketServer(facade=facade, execution_transport=_DummyTransport(), host="127.0.0.1", port=0)
    socket = _DummyWebSocket()
    await server._dispatch_envelope(
        socket,
        NorthCUAWebSocketEnvelope.wrap(
            WebSocketMessageType.TASK_REQUEST, DesktopTaskRequestPayload(session_id="任务/a", request="Task")
        ),
    )
    await asyncio.gather(*server._background_tasks)
    await asyncio.sleep(0)
    messages = [NorthCUAWebSocketEnvelope.model_validate_json(item) for item in socket.messages]
    assert [m.type for m in messages] == [WebSocketMessageType.TASK_ACCEPTED, WebSocketMessageType.TASK_FINISHED]
    assert all(m.payload.session_id == "任务/a" for m in messages)
    assert messages[-1].payload.terminal_decision["reason"] == "runtime_error"
    assert not server._active_sessions and not server._session_connections
    await server.stop()
