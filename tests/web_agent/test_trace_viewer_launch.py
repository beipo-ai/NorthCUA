from pathlib import Path
from unittest.mock import patch

import pytest


def test_installed_viewer_renders_without_package_relative_imports(tmp_path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    from northcua.agents.web.trace_viewer import launcher

    path = Path(launcher.__file__).with_name("app.py")
    with patch("sys.argv", [str(path), "--artifact-root", str(tmp_path)]):
        result = AppTest.from_file(str(path)).run(timeout=15)
    assert not result.exception
    assert result.title[0].value == "NorthCUA Web / Evaluator LLM Trace Viewer"
    assert result.sidebar.text_input[0].value == str(tmp_path)


def test_viewer_console_routes_arguments_to_streamlit():
    pytest.importorskip("streamlit")
    from northcua.agents.web.trace_viewer import launcher

    with patch("sys.argv", ["northcua-web-viewer", "--artifact-root", "logs/test"]):
        with patch("streamlit.web.cli.main") as main:
            launcher.main()
            main.assert_called_once()
            import sys

            assert sys.argv[-3:] == ["--", "--artifact-root", "logs/test"]
