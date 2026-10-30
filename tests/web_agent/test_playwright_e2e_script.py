from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

from tests.support.web_agent_playwright_e2e import LocalFixtureServer, _redaction_scan, _tab_labels


def test_local_fixture_server_serves_shared_absolute_fixture_routes() -> None:
    fixture_root = Path(__file__).resolve().parents[2] / "examples" / "fixtures" / "web-e2b-three"

    with LocalFixtureServer(fixture_root) as server:
        with urlopen(f"{server.base_url}/license.html", timeout=5) as response:
            license_body = response.read().decode("utf-8")
        with urlopen(f"{server.base_url}/details.html", timeout=5) as response:
            details_body = response.read().decode("utf-8")

    assert "Renewal Details" in license_body
    assert "09:30 UTC" in details_body


def test_playwright_e2e_redaction_scan_rejects_image_data_and_paths() -> None:
    report = {
        "cases": [
            {
                "tool_payload": {
                    "image": "data:image/png;base64,abc",
                    "path": "/tmp/web/screenshots/tool.png",
                }
            }
        ]
    }

    scan = _redaction_scan(report)

    assert scan["ok"] is False
    assert scan["findings"]


def test_tab_labels_ignores_titles_and_urls() -> None:
    labels = _tab_labels(
        {
            "tabs": [
                {
                    "label": "",
                    "title": "NorthCUA Launch Details",
                    "url": "http://127.0.0.1:1/details.html",
                },
                {
                    "label": "details",
                    "title": "NorthCUA Launch Details",
                    "url": "http://127.0.0.1:1/details.html",
                },
            ]
        }
    )

    assert labels == ["", "details"]
