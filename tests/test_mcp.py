"""Tests for MCP architecture (responsible design)."""

import json

import pytest
from src.mcp.architecture import AnalyticsReaderServer, MCPClient


@pytest.fixture
def analytics_dir(tmp_path, monkeypatch):
    """Set up a temporary analytics directory with sample files."""
    analytics = tmp_path / "analytics"
    analytics.mkdir()
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    # Create sample analytics files
    (analytics / "topk_products.csv").write_text("title,score\nProduct A,0.9\nProduct B,0.7\n")
    (analytics / "model_metrics.json").write_text(json.dumps({"model": "RF", "f1": 0.85}))
    return analytics


def test_allowed_files_list(analytics_dir):
    server = AnalyticsReaderServer()
    files = server.list_available_files()
    assert "topk_products.csv" in files
    assert "model_metrics.json" in files


def test_read_allowed_file(analytics_dir):
    server = AnalyticsReaderServer()
    content = server.read_analytics_file("topk_products.csv")
    assert content is not None
    assert "Product A" in content


def test_read_denied_file(analytics_dir):
    """Files not in the allowed list should be denied."""
    # Create a file that is NOT in the allowed list
    (analytics_dir / "secret.csv").write_text("secret data")
    server = AnalyticsReaderServer()
    content = server.read_analytics_file("secret.csv")
    assert content is None


def test_mcp_client_routes_correctly(analytics_dir):
    client = MCPClient()
    # Should route through AnalyticsReaderServer
    content = client.get_analytics("topk_products.csv")
    assert content is not None
    assert "Product A" in content

    # List should work
    files = client.list_analytics()
    assert isinstance(files, list)
    assert len(files) > 0


def test_mcp_client_denied_access(analytics_dir):
    client = MCPClient()
    content = client.get_analytics("nonexistent.csv")
    assert content is None
