"""Load portable harness configs without overriding model or sampling values."""

from pathlib import Path

from northcua.core.config_loader import NorthCUABootstrapConfig, NorthCUAConfigLoader


def load_config(path: str | Path) -> NorthCUABootstrapConfig:
    bootstrap = NorthCUAConfigLoader.from_app_config(path)
    summary = bootstrap.app.context_management.summary.model
    if summary.agent_config_path:
        summary_path = Path(summary.agent_config_path).expanduser()
        if not summary_path.is_absolute():
            summary_path = bootstrap.app_config_path.parent / summary_path
        summary.agent_config_path = str(summary_path.resolve(strict=True))
    return bootstrap
