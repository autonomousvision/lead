"""Shared fixtures for unit tests."""

import pytest


@pytest.fixture(autouse=True)
def _isolate_lead_config_env(monkeypatch) -> None:
    """Keep ambient LEAD_CONFIG overrides out of the tests."""
    monkeypatch.delenv("LEAD_CONFIG", raising=False)
