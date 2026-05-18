"""Tests for PRISM FastAPI analytics endpoints."""
import duckdb
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from src.api.main import app
    return TestClient(app)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_stats_empty_warehouse(client, tmp_path):
    """Stats returns zeros when warehouse missing."""
    r = client.get("/api/stats")
    assert r.status_code == 200
    data = r.json()
    assert "n_products" in data
    assert data["n_products"] == 0


def test_topk_empty(client, tmp_path):
    r = client.get("/api/topk")
    assert r.status_code == 200
    assert r.json() == []


def test_clusters_empty(client, tmp_path):
    r = client.get("/api/clusters")
    assert r.status_code == 200
    assert r.json() == []


def test_rules_empty(client, tmp_path):
    r = client.get("/api/rules")
    assert r.status_code == 200
    assert r.json() == []


def test_stats_with_warehouse(client, tmp_path):
    """Stats reads product count from warehouse.duckdb."""
    conn = duckdb.connect(str(tmp_path / "warehouse.duckdb"))
    conn.execute(
        "CREATE TABLE products AS SELECT 'p1' AS product_id, 'keyboards' AS category, "
        "'Keychron K2' AS title, 'Keychron' AS shop_name, 100.0 AS price"
    )
    conn.close()
    r = client.get("/api/stats")
    assert r.status_code == 200
    assert r.json()["n_products"] == 1
