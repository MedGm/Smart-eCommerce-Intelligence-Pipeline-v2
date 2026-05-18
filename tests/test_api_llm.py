"""Tests for LLM API endpoints (mocked Gemini)."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from src.api.main import app

    return TestClient(app)


def test_summary_executive(client):
    with patch("src.llm.summarizer.generate_summary", return_value="exec summary"):
        r = client.post("/api/summary", json={"type": "executive"})
    assert r.status_code == 200
    assert r.json()["result"] == "exec summary"


def test_summary_strategy(client):
    with patch("src.llm.summarizer.generate_strategy_report", return_value="strategy"):
        r = client.post("/api/summary", json={"type": "strategy"})
    assert r.status_code == 200
    assert r.json()["result"] == "strategy"


def test_summary_profile(client):
    with (
        patch("src.mcp.architecture.MCPClient.get_top_products", return_value="[]"),
        patch("src.llm.summarizer.generate_product_profile", return_value="profile"),
    ):
        r = client.post("/api/summary", json={"type": "profile"})
    assert r.status_code == 200
    assert r.json()["result"] == "profile"


def test_chat_streams(client):
    with patch("src.llm.summarizer.chat_with_data", return_value="hello world"):
        r = client.post(
            "/api/chat",
            json={"query": "what are top products?", "history": []},
        )
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
