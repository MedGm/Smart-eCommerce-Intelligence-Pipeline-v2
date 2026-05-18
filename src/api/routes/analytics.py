"""Analytics REST endpoints — reads from warehouse.duckdb (read-only)."""
from __future__ import annotations

import json

import duckdb
from fastapi import APIRouter

from src.config import analytics_dir, data_dir

router = APIRouter()


def _conn() -> duckdb.DuckDBPyConnection:
    path = data_dir() / "warehouse.duckdb"
    return duckdb.connect(str(path), read_only=True)


@router.get("/stats")
def get_stats() -> dict:
    n_products = 0
    top_categories: list[dict] = []
    best_shop = ""

    try:
        conn = _conn()
        n_products = conn.execute("SELECT count(*) FROM products").fetchone()[0]
        top_categories = (
            conn.execute(
                "SELECT category, count(*) AS n FROM products "
                "GROUP BY category ORDER BY n DESC LIMIT 5"
            )
            .fetchdf()
            .to_dict("records")
        )
        best_shop_row = conn.execute(
            "SELECT shop_name, avg(score) AS avg_score FROM topk_products "
            "GROUP BY shop_name ORDER BY avg_score DESC LIMIT 1"
        ).fetchone()
        best_shop = best_shop_row[0] if best_shop_row else ""
        conn.close()
    except Exception:
        pass

    metrics: dict = {}
    for fname in ["model_metrics.json", "model_metrics_xgboost.json", "cluster_metrics.json"]:
        p = analytics_dir() / fname
        if p.exists():
            metrics.update(json.loads(p.read_text()))

    return {
        "n_products": n_products,
        "top_categories": top_categories,
        "best_shop": best_shop,
        "model_metrics": metrics,
    }


@router.get("/topk")
def get_topk() -> list[dict]:
    try:
        conn = _conn()
        rows = conn.execute(
            "SELECT title, shop_name, category, price, score "
            "FROM topk_products ORDER BY score DESC LIMIT 50"
        ).fetchdf().to_dict("records")
        conn.close()
        return rows
    except Exception:
        return []


@router.get("/clusters")
def get_clusters() -> list[dict]:
    try:
        conn = _conn()
        rows = conn.execute(
            "SELECT cluster, count(*) AS count FROM clusters GROUP BY cluster ORDER BY cluster"
        ).fetchdf().to_dict("records")
        conn.close()
        return rows
    except Exception:
        return []


@router.get("/rules")
def get_rules() -> list[dict]:
    try:
        conn = _conn()
        rows = conn.execute(
            "SELECT antecedents, consequents, support, confidence, lift "
            "FROM association_rules ORDER BY lift DESC LIMIT 30"
        ).fetchdf().to_dict("records")
        conn.close()
        return rows
    except Exception:
        return []
