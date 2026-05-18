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
    try:
        conn.execute(
            f"CREATE OR REPLACE TABLE products AS SELECT * FROM read_parquet('{parquet_path}')"
        )
        logger.info("Loaded products table from %s into %s", parquet_path, _warehouse_path())
        result = conn.execute("SELECT * FROM products").df()
        _warehouse_path().chmod(0o666)  # world-rw so non-root containers (Superset) can open it
        return result
    finally:
        conn.close()


def rebuild_warehouse() -> None:
    """
    Full warehouse rebuild: sync analytics CSVs from MinIO (if available),
    then load all tables into warehouse.duckdb.

    Tables created:
      products           ← cleaned_products.parquet
      topk_products      ← analytics/topk_products.csv
      topk_per_category  ← analytics/topk_per_category.csv
      topk_per_shop      ← analytics/topk_per_shop.csv
      clusters           ← analytics/clusters.csv
      association_rules  ← analytics/association_rules.csv
    """
    from src.config import analytics_dir

    minio_configured = bool(os.environ.get("MINIO_ENDPOINT"))

    # Sync analytics CSVs from MinIO → local
    if minio_configured:
        try:
            from src.storage.minio_client import _client
            c = _client()
            a_dir = analytics_dir()
            a_dir.mkdir(parents=True, exist_ok=True)
            for fname in [
                "topk_products.csv", "topk_per_category.csv", "topk_per_shop.csv",
                "clusters.csv", "association_rules.csv",
                "model_metrics.json", "model_metrics_xgboost.json", "cluster_metrics.json",
            ]:
                try:
                    c.download_file("processed", f"analytics/{fname}", str(a_dir / fname))
                    logger.info("Downloaded analytics/%s", fname)
                except Exception as e:
                    logger.warning("Could not download analytics/%s: %s", fname, e)
            # Also sync cleaned parquet
            p_dir = processed_dir()
            p_dir.mkdir(parents=True, exist_ok=True)
            try:
                c.download_file("processed", "cleaned_products.parquet",
                                str(p_dir / "cleaned_products.parquet"))
                logger.info("Downloaded cleaned_products.parquet")
            except Exception as e:
                logger.warning("Could not download cleaned_products.parquet: %s", e)
        except Exception as e:
            logger.warning("MinIO sync failed: %s", e)

    a_dir = analytics_dir()
    conn = _conn()
    try:
        # products table from parquet
        parquet = processed_dir() / "cleaned_products.parquet"
        if parquet.exists():
            conn.execute(
                f"CREATE OR REPLACE TABLE products AS SELECT * FROM read_parquet('{parquet}')"
            )
            logger.info("Loaded products (%d rows)", conn.execute("SELECT count(*) FROM products").fetchone()[0])

        # Analytics CSV tables
        csv_tables = [
            ("topk_products",     "topk_products.csv"),
            ("topk_per_category", "topk_per_category.csv"),
            ("topk_per_shop",     "topk_per_shop.csv"),
            ("clusters",          "clusters.csv"),
            ("association_rules", "association_rules.csv"),
        ]
        for table, fname in csv_tables:
            path = a_dir / fname
            if path.exists():
                conn.execute(
                    f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_csv_auto('{path}')"
                )
                n = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                logger.info("Loaded %s (%d rows)", table, n)
            else:
                logger.warning("Missing %s — table %s skipped", fname, table)

        _warehouse_path().chmod(0o666)
        logger.info("Warehouse rebuilt: %s", _warehouse_path())
    finally:
        conn.close()


def query(sql: str, duckdb_path: str | None = None) -> pd.DataFrame:
    """Run arbitrary SQL against warehouse.duckdb and return as DataFrame."""
    conn = _conn(duckdb_path)
    try:
        return conn.execute(sql).df()
    finally:
        conn.close()
