"""
DuckDB analytics layer.
Reads cleaned Parquet from MinIO (s3://) or local filesystem,
exposes a SQL query interface via a persistent warehouse.duckdb file.
"""
from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd

from src.config import data_dir, get_logger, processed_dir

logger = get_logger(__name__)


def _warehouse_path() -> Path:
    env = os.environ.get("DUCKDB_PATH", "")
    return Path(env) if env else data_dir() / "warehouse.duckdb"


def _conn(duckdb_path: str | None = None) -> duckdb.DuckDBPyConnection:
    path = duckdb_path or str(_warehouse_path())
    conn = duckdb.connect(path)
    endpoint = os.environ.get("MINIO_ENDPOINT", "")
    if endpoint:
        host = endpoint.replace("http://", "").replace("https://", "")
        access_key = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
        secret_key = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
        conn.execute(f"""
            INSTALL httpfs; LOAD httpfs;
            SET s3_endpoint='{host}';
            SET s3_access_key_id='{access_key}';
            SET s3_secret_access_key='{secret_key}';
            SET s3_use_ssl=false;
            SET s3_url_style='path';
        """)
    return conn


def load_products(source: str = "auto") -> pd.DataFrame:
    """
    Load cleaned products into DuckDB `products` table and return as DataFrame.

    source="auto"  — MinIO if MINIO_ENDPOINT set, else local Parquet
    source="minio" — s3://processed/cleaned_products.parquet
    source="local" — data/processed/cleaned_products.parquet
    """
    minio_configured = bool(os.environ.get("MINIO_ENDPOINT"))

    if source == "minio" or (source == "auto" and minio_configured):
        parquet_path = "s3://processed/cleaned_products.parquet"
    else:
        parquet_path = str(processed_dir() / "cleaned_products.parquet")

    conn = _conn()
    conn.execute(
        f"CREATE OR REPLACE TABLE products AS SELECT * FROM read_parquet('{parquet_path}')"
    )
    logger.info("Loaded products table from %s into %s", parquet_path, _warehouse_path())
    return conn.execute("SELECT * FROM products").df()


def query(sql: str, duckdb_path: str | None = None) -> pd.DataFrame:
    """Run arbitrary SQL against warehouse.duckdb and return as DataFrame."""
    return _conn(duckdb_path).execute(sql).df()
