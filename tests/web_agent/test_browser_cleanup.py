from unittest.mock import AsyncMock
import pytest
from northcua.agents.web.backend.playwright import PlaywrightWebActionBackend
from northcua.core.config import WebAgentConfig


@pytest.mark.asyncio
async def test_context_close_failure_still_releases_browser_and_driver():
    backend = PlaywrightWebActionBackend(config=WebAgentConfig(), session_id="cleanup")
    context, browser, driver = AsyncMock(), AsyncMock(), AsyncMock()
    context.close.side_effect = RuntimeError("context crashed")
    backend._context, backend._browser, backend._playwright = context, browser, driver
    result = await backend.close()
    browser.close.assert_awaited_once()
    driver.stop.assert_awaited_once()
    with pytest.raises(Exception):
        result.require_success()
    assert backend._context is backend._browser is backend._playwright is None
