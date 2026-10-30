from __future__ import annotations

from northcua.batch.contracts import BatchTaskSpec
from northcua.batch.request_adapters import WebBatchRequestAdapter


def test_web_batch_adapter_preserves_headed_metadata() -> None:
    task = BatchTaskSpec(
        task_id="visible-web",
        agent_kind="web",
        query="Open the page",
        url="https://example.com",
        metadata={"headed": True},
    )

    payload = WebBatchRequestAdapter().project(task)

    assert payload["metadata"]["headed"] is True
