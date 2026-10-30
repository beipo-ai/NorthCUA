"""Failure category inference coverage."""

from northcua.telemetry.contracts import FailureCategory
from northcua.telemetry.failure_categories import failure_category_from_message


def test_opening_handshake_failure_is_controller_submit() -> None:
    assert failure_category_from_message("timed out during opening handshake") == FailureCategory.CONTROLLER_SUBMIT


def test_controller_task_failure_is_controller_submit() -> None:
    assert (
        failure_category_from_message("controller task failed before terminal result")
        == FailureCategory.CONTROLLER_SUBMIT
    )


def test_lost_submitter_mapping_is_controller_submit() -> None:
    assert failure_category_from_message("queued task lost its submitter mapping") == FailureCategory.CONTROLLER_SUBMIT


def test_idle_socket_zero_byte_read_is_controller_submit() -> None:
    assert (
        failure_category_from_message("0 bytes read on a total of 4 expected bytes")
        == FailureCategory.CONTROLLER_SUBMIT
    )


def test_controller_connect_refused_is_controller_submit() -> None:
    assert (
        failure_category_from_message("ConnectionRefusedError: [Errno 111] Connect call failed ('127.0.0.1', 9192)")
        == FailureCategory.CONTROLLER_SUBMIT
    )
