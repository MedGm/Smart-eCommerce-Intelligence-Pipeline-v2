import shutil
from pathlib import Path

import pandas as pd


def _make_parquet(tmp_dir: Path) -> Path:
    """Create a minimal cleaned_products.parquet for testing."""
    df = pd.DataFrame([
        {
            "source_platform": "shopify",
            "shop_name": "Ruggable",
            "product_id": "1",
            "product_url": "https://ruggable.com/p/1",
            "title": "Blue Rug",
            "description": "Nice rug",
            "category": "Rugs",
            "brand": "Ruggable",
            "price": 89.0,
            "old_price": 120.0,
            "availability": "instock",
            "rating": 4.8,
            "review_count": 200,
            "geography": "US",
            "scraped_at": "2026-05-17T00:00:00Z",
            "dq_score": 0.9,
        }
    ])
    path = tmp_dir / "cleaned_products.parquet"
    df.to_parquet(path, index=False)
    return path


def test_load_products_local_returns_dataframe(tmp_path, monkeypatch):
    monkeypatch.delenv("MINIO_ENDPOINT", raising=False)
    monkeypatch.setenv("DUCKDB_PATH", str(tmp_path / "test.duckdb"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    (tmp_path / "processed").mkdir()
    shutil.copy(_make_parquet(tmp_path), tmp_path / "processed" / "cleaned_products.parquet")

    from src.storage.duckdb_client import load_products
    df = load_products(source="local")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["title"] == "Blue Rug"


def test_query_runs_sql(tmp_path, monkeypatch):
    monkeypatch.delenv("MINIO_ENDPOINT", raising=False)
    monkeypatch.setenv("DUCKDB_PATH", str(tmp_path / "test.duckdb"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    (tmp_path / "processed").mkdir()
    shutil.copy(_make_parquet(tmp_path), tmp_path / "processed" / "cleaned_products.parquet")

    from src.storage.duckdb_client import load_products, query
    load_products(source="local")
    result = query("SELECT COUNT(*) AS n FROM products", duckdb_path=str(tmp_path / "test.duckdb"))
    assert result.iloc[0]["n"] == 1


def test_load_products_auto_uses_local_when_no_minio(tmp_path, monkeypatch):
    monkeypatch.delenv("MINIO_ENDPOINT", raising=False)
    monkeypatch.setenv("DUCKDB_PATH", str(tmp_path / "test.duckdb"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    (tmp_path / "processed").mkdir()
    shutil.copy(_make_parquet(tmp_path), tmp_path / "processed" / "cleaned_products.parquet")

    from src.storage.duckdb_client import load_products
    df = load_products(source="auto")
    assert len(df) == 1
