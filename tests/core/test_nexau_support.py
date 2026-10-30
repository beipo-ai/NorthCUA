from __future__ import annotations

import logging

import pytest

from northcua.core.nexau_support import rethrow_missing_nexau, suppress_nexau_optional_dependency_noise


def test_rethrow_missing_nexau_wraps_clear_install_message() -> None:
    with pytest.raises(ModuleNotFoundError, match=r"uv sync --extra local"):
        rethrow_missing_nexau(ModuleNotFoundError("No module named 'nexau'", name="nexau"))


def test_suppress_nexau_optional_dependency_noise_raises_logger_level() -> None:
    logger = logging.getLogger("nexau.archs.sandbox.e2b_sandbox")
    executor_logger = logging.getLogger("nexau.archs.main_sub.execution.executor")
    original_level = logger.level
    original_filters = list(executor_logger.filters)
    try:
        logger.setLevel(logging.NOTSET)
        suppress_nexau_optional_dependency_noise()
        assert logger.level == logging.ERROR
        warning_record = logging.LogRecord(
            name="nexau.archs.main_sub.execution.executor",
            level=logging.WARNING,
            pathname=__file__,
            lineno=1,
            msg="⚠️ Structured tool call mode enabled but no structured tool definitions were provided.",
            args=(),
            exc_info=None,
        )
        assert executor_logger.filter(warning_record) is False
    finally:
        logger.setLevel(original_level)
        executor_logger.filters[:] = original_filters
