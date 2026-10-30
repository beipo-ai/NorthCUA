from __future__ import annotations

import asyncio
import socket

import pytest

from northcua.transport.tcp_proxy import (
    AsyncTcpProxy,
    TcpProxyConfig,
    _close_writer,
    _configure_writer,
)


async def _start_server(handler):
    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    assert server.sockets
    return server, int(server.sockets[0].getsockname()[1])


@pytest.mark.asyncio
async def test_proxy_forwards_many_connections_without_per_connection_threads() -> None:
    async def echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while chunk := await reader.read(65536):
                writer.write(chunk)
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    target, target_port = await _start_server(echo)
    proxy = AsyncTcpProxy(
        TcpProxyConfig(
            listen_host="127.0.0.1",
            target_port=target_port,
            stats_interval_seconds=0,
            max_connections=128,
            fd_reserve=32,
        )
    )
    await proxy.start()

    async def round_trip(index: int) -> None:
        reader, writer = await asyncio.open_connection("127.0.0.1", proxy.bound_port)
        payload = f"connection-{index}:".encode() + bytes([index % 251]) * 32768
        writer.write(payload)
        await writer.drain()
        assert await reader.readexactly(len(payload)) == payload
        writer.close()
        await writer.wait_closed()

    try:
        await asyncio.gather(*(round_trip(index) for index in range(96)))
    finally:
        await proxy.stop()
        target.close()
        await target.wait_closed()

    assert proxy.stats.accepted == 96
    assert proxy.stats.closed == 96
    assert proxy.stats.rejected == 0
    assert proxy.stats.connect_failures == 0
    assert proxy.stats.relay_failures == 0
    assert proxy.stats.client_to_target_bytes == proxy.stats.target_to_client_bytes


@pytest.mark.asyncio
async def test_proxy_applies_backpressure_without_truncating_payload() -> None:
    received = bytearray()

    async def slow_sink(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await asyncio.sleep(0.1)
            while chunk := await reader.read(8192):
                received.extend(chunk)
                await asyncio.sleep(0.001)
            writer.write(b"done")
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    target, target_port = await _start_server(slow_sink)
    proxy = AsyncTcpProxy(
        TcpProxyConfig(
            listen_host="127.0.0.1",
            target_port=target_port,
            stats_interval_seconds=0,
            write_buffer_high_bytes=16384,
            max_connections=16,
            fd_reserve=32,
        )
    )
    await proxy.start()
    payload = bytes(range(251)) * 32768
    reader, writer = await asyncio.open_connection("127.0.0.1", proxy.bound_port)
    try:
        writer.write(payload)
        await writer.drain()
        writer.write_eof()
        assert await reader.readexactly(4) == b"done"
    finally:
        writer.close()
        await writer.wait_closed()
        await proxy.stop()
        target.close()
        await target.wait_closed()

    assert received == payload
    assert proxy.stats.relay_failures == 0


@pytest.mark.asyncio
async def test_stop_drains_existing_connection_and_rejects_new_accepts() -> None:
    release = asyncio.Event()

    async def held_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await release.wait()
            writer.write(await reader.read())
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    target, target_port = await _start_server(held_connection)
    proxy = AsyncTcpProxy(
        TcpProxyConfig(
            listen_host="127.0.0.1",
            target_port=target_port,
            stats_interval_seconds=0,
            drain_timeout_seconds=2,
            max_connections=16,
            fd_reserve=32,
        )
    )
    await proxy.start()
    listen_port = proxy.bound_port
    reader, writer = await asyncio.open_connection("127.0.0.1", listen_port)
    writer.write(b"finish-before-stop")
    writer.write_eof()
    for _ in range(100):
        if proxy.stats.active == 1:
            break
        await asyncio.sleep(0.001)
    assert proxy.stats.active == 1
    stop_task = asyncio.create_task(proxy.stop())
    await asyncio.sleep(0.05)
    assert not stop_task.done()
    with pytest.raises(OSError):
        await asyncio.open_connection("127.0.0.1", listen_port)

    release.set()
    assert await reader.read() == b"finish-before-stop"
    writer.close()
    await writer.wait_closed()
    await stop_task
    target.close()
    await target.wait_closed()

    assert proxy.stats.active == 0
    assert proxy.stats.closed == 1


@pytest.mark.asyncio
async def test_canceled_stop_caller_does_not_abandon_shared_drain() -> None:
    release = asyncio.Event()

    async def held_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await release.wait()
            writer.write(await reader.read())
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    target, target_port = await _start_server(held_connection)
    proxy = AsyncTcpProxy(
        TcpProxyConfig(
            listen_host="127.0.0.1",
            target_port=target_port,
            stats_interval_seconds=0,
            drain_timeout_seconds=2,
            max_connections=16,
            fd_reserve=32,
        )
    )
    await proxy.start()
    reader, writer = await asyncio.open_connection("127.0.0.1", proxy.bound_port)
    writer.write(b"finish-after-cancel")
    writer.write_eof()
    for _ in range(100):
        if proxy.stats.active == 1:
            break
        await asyncio.sleep(0.001)

    first_stop = asyncio.create_task(proxy.stop())
    await asyncio.sleep(0.01)
    first_stop.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_stop
    assert proxy._stopped is False  # noqa: SLF001 - shutdown completion contract

    release.set()
    assert await reader.read() == b"finish-after-cancel"
    writer.close()
    await writer.wait_closed()
    await proxy.stop()
    target.close()
    await target.wait_closed()

    assert proxy._stopped is True  # noqa: SLF001
    assert proxy.stats.active == 0
    assert proxy.stats.closed == 1


class _RecordingSocket:
    def __init__(self) -> None:
        self.options: list[tuple[int, int, int]] = []

    def setsockopt(self, level: int, option: int, value: int) -> None:
        self.options.append((level, option, value))


class _RecordingTransport:
    def __init__(self) -> None:
        self.limits: tuple[int, int] | None = None
        self.aborted = False

    def set_write_buffer_limits(self, *, high: int, low: int) -> None:
        self.limits = (high, low)

    def abort(self) -> None:
        self.aborted = True


class _FakeWriter:
    def __init__(self, *, close_blocks: bool = False) -> None:
        self.socket = _RecordingSocket()
        self.transport = _RecordingTransport()
        self.closed = False
        self.close_blocks = close_blocks

    def get_extra_info(self, name: str):
        return self.socket if name == "socket" else None

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        if self.close_blocks:
            await asyncio.Event().wait()


class _BlockedDrainWriter(_FakeWriter):
    def __init__(self) -> None:
        super().__init__()
        self.chunks: list[bytes] = []

    def write(self, chunk: bytes) -> None:
        self.chunks.append(chunk)

    async def drain(self) -> None:
        await asyncio.Event().wait()

    def can_write_eof(self) -> bool:
        return False


def test_proxy_configures_keepalive_and_bounded_write_buffer() -> None:
    writer = _FakeWriter()
    config = TcpProxyConfig(write_buffer_high_bytes=65536)

    _configure_writer(writer, config)  # type: ignore[arg-type]

    assert (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) in writer.socket.options
    assert (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1) in writer.socket.options
    assert writer.transport.limits == (65536, 16384)


@pytest.mark.asyncio
async def test_proxy_close_timeout_aborts_stuck_transport() -> None:
    writer = _FakeWriter(close_blocks=True)

    timed_out = await _close_writer(writer, timeout_seconds=0.01)  # type: ignore[arg-type]

    assert timed_out is True
    assert writer.closed is True
    assert writer.transport.aborted is True


@pytest.mark.asyncio
async def test_proxy_write_timeout_terminates_stuck_relay() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(b"blocked-peer")
    writer = _BlockedDrainWriter()
    proxy = AsyncTcpProxy(
        TcpProxyConfig(
            write_timeout_seconds=0.01,
            stats_interval_seconds=0,
            max_connections=16,
            fd_reserve=32,
        )
    )

    with pytest.raises(TimeoutError, match="write drain timed out"):
        await asyncio.wait_for(
            proxy._relay(reader, writer, client_to_target=True),  # noqa: SLF001
            timeout=0.1,
        )

    assert writer.chunks == [b"blocked-peer"]
    assert proxy.stats.write_timeouts == 1


@pytest.mark.asyncio
async def test_proxy_idle_timeout_tracks_both_directions() -> None:
    received = bytearray()

    async def sink(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while chunk := await reader.read(1024):
                received.extend(chunk)
        finally:
            writer.close()
            await writer.wait_closed()

    target, target_port = await _start_server(sink)
    proxy = AsyncTcpProxy(
        TcpProxyConfig(
            listen_host="127.0.0.1",
            target_port=target_port,
            idle_timeout_seconds=0.08,
            stats_interval_seconds=0,
            max_connections=16,
            fd_reserve=32,
        )
    )
    await proxy.start()
    _reader, writer = await asyncio.open_connection("127.0.0.1", proxy.bound_port)
    payload = b"x" * 1024
    try:
        for _ in range(8):
            writer.write(payload)
            await writer.drain()
            await asyncio.sleep(0.03)
        writer.write_eof()
        for _ in range(100):
            if len(received) == 8 * len(payload):
                break
            await asyncio.sleep(0.005)
    finally:
        writer.close()
        await writer.wait_closed()
        await proxy.stop()
        target.close()
        await target.wait_closed()

    assert received == payload * 8
    assert proxy.stats.idle_timeouts == 0


@pytest.mark.asyncio
async def test_proxy_concurrent_start_is_linearized() -> None:
    proxy = AsyncTcpProxy(
        TcpProxyConfig(
            listen_host="127.0.0.1",
            stats_interval_seconds=0,
            max_connections=16,
            fd_reserve=32,
        )
    )
    results = await asyncio.gather(proxy.start(), proxy.start(), return_exceptions=True)
    assert sum(result is None for result in results) == 1
    errors = [result for result in results if isinstance(result, RuntimeError)]
    assert len(errors) == 1
    assert "already running" in str(errors[0])
    await proxy.stop()


@pytest.mark.asyncio
async def test_proxy_stop_closes_listener() -> None:
    proxy = AsyncTcpProxy(
        TcpProxyConfig(
            listen_host="127.0.0.1",
            stats_interval_seconds=0,
            max_connections=16,
            fd_reserve=32,
        )
    )
    await proxy.start()
    port = proxy.bound_port
    await proxy.stop()

    with pytest.raises(OSError):
        await asyncio.open_connection("127.0.0.1", port)


@pytest.mark.asyncio
async def test_proxy_rejects_are_synchronous_and_untracked() -> None:
    proxy = AsyncTcpProxy(TcpProxyConfig(max_connections=1, fd_reserve=32, stats_interval_seconds=0))
    first = _FakeWriter()
    rejected = [_FakeWriter() for _ in range(50)]

    proxy._accept_client(asyncio.StreamReader(), first)  # type: ignore[arg-type]  # noqa: SLF001
    for writer in rejected:
        proxy._accept_client(asyncio.StreamReader(), writer)  # type: ignore[arg-type]  # noqa: SLF001

    assert proxy.stats.active == 1
    assert proxy.stats.rejected == 50
    assert all(writer.transport.aborted for writer in rejected)
    assert len(proxy._connections) == 1  # noqa: SLF001
    for task in list(proxy._connections):  # noqa: SLF001
        task.cancel()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_proxy_stops_accepting_before_capacity_consumes_process_fds() -> None:
    release = asyncio.Event()

    async def held_connection(_reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await release.wait()
        writer.close()
        await writer.wait_closed()

    target, target_port = await _start_server(held_connection)
    proxy = AsyncTcpProxy(
        TcpProxyConfig(
            listen_host="127.0.0.1",
            target_port=target_port,
            stats_interval_seconds=0,
            max_connections=1,
            backlog=16,
            fd_reserve=32,
        )
    )
    await proxy.start()
    first_reader, first_writer = await asyncio.open_connection("127.0.0.1", proxy.bound_port)
    del first_reader
    for _ in range(100):
        if proxy.stats.active == 1:
            break
        await asyncio.sleep(0.001)

    second_reader, second_writer = await asyncio.open_connection("127.0.0.1", proxy.bound_port)
    del second_reader
    await asyncio.sleep(0.05)
    assert proxy.stats.active == 1
    assert proxy.stats.accepted == 1
    assert proxy._accept_reader_registered is False  # noqa: SLF001

    release.set()
    first_writer.close()
    await first_writer.wait_closed()
    for _ in range(100):
        if proxy.stats.accepted == 2:
            break
        await asyncio.sleep(0.005)
    assert proxy.stats.accepted == 2

    second_writer.close()
    await second_writer.wait_closed()
    await proxy.stop()
    target.close()
    await target.wait_closed()


@pytest.mark.asyncio
async def test_forced_stop_cancels_per_connection_idle_watchers() -> None:
    release = asyncio.Event()

    async def held_connection(_reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await release.wait()
        writer.close()
        await writer.wait_closed()

    target, target_port = await _start_server(held_connection)
    proxy = AsyncTcpProxy(
        TcpProxyConfig(
            listen_host="127.0.0.1",
            target_port=target_port,
            stats_interval_seconds=0,
            idle_timeout_seconds=0,
            drain_timeout_seconds=0.02,
            max_connections=8,
            fd_reserve=32,
        )
    )
    await proxy.start()
    connections = [await asyncio.open_connection("127.0.0.1", proxy.bound_port) for _ in range(3)]
    for _ in range(100):
        if proxy.stats.active == 3:
            break
        await asyncio.sleep(0.001)

    await proxy.stop()
    await asyncio.sleep(0)
    pending_proxy_tasks = [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task() and task.get_name().startswith("northcua-tcp-proxy-")
    ]
    assert pending_proxy_tasks == []

    for _reader, writer in connections:
        writer.close()
        await writer.wait_closed()
    release.set()
    target.close()
    await target.wait_closed()


@pytest.mark.asyncio
async def test_proxy_rejects_start_when_fd_limit_cannot_cover_connections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("northcua.transport.tcp_proxy.resource.getrlimit", lambda _resource: (100, 100))
    proxy = AsyncTcpProxy(
        TcpProxyConfig(
            listen_host="127.0.0.1",
            max_connections=64,
            fd_reserve=32,
            stats_interval_seconds=0,
        )
    )

    with pytest.raises(RuntimeError, match="file-descriptor limit is too low"):
        await proxy.start()
