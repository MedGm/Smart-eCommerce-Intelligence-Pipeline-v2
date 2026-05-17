"""
Auto-create PRISM BI dashboard in Superset via REST API.
Run: docker compose run --rm app python scripts/setup_superset_dashboard.py
"""
import json
import sys
import requests

BASE = "http://superset:8088"
CREDS = {"username": "admin", "password": "admin", "provider": "db", "refresh": True}


def login() -> requests.Session:
    s = requests.Session()
    token = s.post(f"{BASE}/api/v1/security/login", json=CREDS).json()["access_token"]
    s.headers.update({"Authorization": f"Bearer {token}"})
    csrf = s.get(f"{BASE}/api/v1/security/csrf_token/").json()["result"]
    s.headers.update({"X-CSRFToken": csrf, "Referer": BASE})
    return s


def get_db_id(s: requests.Session) -> int:
    for db in s.get(f"{BASE}/api/v1/database/").json().get("result", []):
        if "duckdb" in db.get("backend", "").lower():
            return db["id"]
    sys.exit("DuckDB not found. Add it in Superset → Settings → Databases first.")


def upsert_dataset(s: requests.Session, db_id: int, table: str) -> int:
    for ds in s.get(f"{BASE}/api/v1/dataset/").json().get("result", []):
        if ds["table_name"] == table and ds["database"]["id"] == db_id:
            return ds["id"]
    r = s.post(f"{BASE}/api/v1/dataset/", json={"database": db_id, "schema": "main", "table_name": table})
    r.raise_for_status()
    return r.json()["id"]


def chart(s: requests.Session, name: str, viz: str, ds_id: int, params: dict) -> int:
    params["viz_type"] = viz
    r = s.post(f"{BASE}/api/v1/chart/", json={
        "slice_name": name,
        "viz_type": viz,
        "datasource_id": ds_id,
        "datasource_type": "table",
        "params": json.dumps(params),
    })
    r.raise_for_status()
    cid = r.json()["id"]
    print(f"  ✓ {name} (id={cid})")
    return cid


def metric(col: str, agg: str = "COUNT", label: str | None = None) -> dict:
    return {
        "aggregate": agg,
        "column": {"column_name": col},
        "expressionType": "SIMPLE",
        "label": label or f"{agg}({col})",
    }


def build_position(chart_ids: list[int]) -> str:
    layout: dict = {
        "ROOT_ID": {"type": "ROOT", "id": "ROOT_ID", "children": ["GRID_ID"]},
        "GRID_ID": {"type": "GRID", "id": "GRID_ID", "children": []},
        "HEADER_ID": {"type": "HEADER", "id": "HEADER_ID", "meta": {"text": "PRISM — Product Intelligence"}},
    }
    row_ids = []
    for i, pair in enumerate([chart_ids[j: j + 2] for j in range(0, len(chart_ids), 2)]):
        row_id = f"ROW-{i}"
        width = 12 // len(pair)
        cell_ids = [f"CHART-{cid}" for cid in pair]
        layout[row_id] = {
            "type": "ROW", "id": row_id,
            "children": cell_ids,
            "meta": {"background": "BACKGROUND_TRANSPARENT"},
        }
        for cid in pair:
            layout[f"CHART-{cid}"] = {
                "type": "CHART", "id": f"CHART-{cid}", "children": [],
                "meta": {"chartId": cid, "width": width, "height": 50},
            }
        row_ids.append(row_id)
    layout["GRID_ID"]["children"] = row_ids
    return json.dumps(layout)


def main() -> None:
    print("Logging in…")
    s = login()

    print("Finding DuckDB…")
    db_id = get_db_id(s)
    print(f"  db_id={db_id}")

    print("Creating datasets…")
    topk_id  = upsert_dataset(s, db_id, "topk_products")
    prod_id  = upsert_dataset(s, db_id, "products")
    clust_id = upsert_dataset(s, db_id, "clusters")
    rules_id = upsert_dataset(s, db_id, "association_rules")
    shop_id  = upsert_dataset(s, db_id, "topk_per_shop")

    print("Creating charts…")
    ids = []

    # 1 — Top products table (score column is correct name)
    ids.append(chart(s, "Top-K Products", "table", topk_id, {
        "adhoc_filters": [],
        "all_columns": ["title", "shop_name", "category", "price", "score", "dq_score"],
        "order_by_cols": ['["score", false]'],
        "page_length": 25,
        "include_search": True,
    }))

    # 2 — Avg score by category (use table viz — no datetime needed)
    ids.append(chart(s, "Avg Score by Category", "table", topk_id, {
        "adhoc_filters": [],
        "all_columns": ["category"],
        "metrics": [metric("score", "AVG", "Avg Score")],
        "groupby": ["category"],
        "order_by_cols": ['["Avg Score", false]'],
        "page_length": 20,
    }))

    # 3 — Products per shop (table)
    ids.append(chart(s, "Products per Shop", "table", prod_id, {
        "adhoc_filters": [],
        "groupby": ["shop_name"],
        "metrics": [metric("product_id", "COUNT", "Products")],
        "order_by_cols": ['["Products", false]'],
        "page_length": 20,
    }))

    # 4 — Category breakdown (table)
    ids.append(chart(s, "Products by Category", "table", prod_id, {
        "adhoc_filters": [],
        "groupby": ["category"],
        "metrics": [metric("product_id", "COUNT", "Products")],
        "order_by_cols": ['["Products", false]'],
        "page_length": 20,
    }))

    # 5 — Cluster membership (table)
    ids.append(chart(s, "Cluster Membership", "table", clust_id, {
        "adhoc_filters": [],
        "groupby": ["cluster"],
        "metrics": [metric("cluster", "COUNT", "Products")],
        "order_by_cols": ['["Products", false]'],
        "page_length": 20,
    }))

    # 6 — Association rules
    ids.append(chart(s, "Association Rules (by Lift)", "table", rules_id, {
        "adhoc_filters": [],
        "all_columns": ["antecedents", "consequents", "support", "confidence", "lift"],
        "order_by_cols": ['["lift", false]'],
        "page_length": 25,
        "include_search": True,
    }))

    # 7 — Avg price by category
    ids.append(chart(s, "Avg Price by Category", "table", topk_id, {
        "adhoc_filters": [],
        "groupby": ["category"],
        "metrics": [metric("price", "AVG", "Avg Price")],
        "order_by_cols": ['["Avg Price", false]'],
        "page_length": 20,
    }))

    # 8 — Top shops by avg score
    ids.append(chart(s, "Top Shops by Score", "table", shop_id, {
        "adhoc_filters": [],
        "groupby": ["shop_name"],
        "metrics": [metric("score", "AVG", "Avg Score")],
        "order_by_cols": ['["Avg Score", false]'],
        "page_length": 20,
    }))

    print(f"  chart_ids={ids}")

    print("Creating dashboard…")
    r = s.post(f"{BASE}/api/v1/dashboard/", json={
        "dashboard_title": "PRISM — Product Intelligence",
        "published": True,
        "position_json": build_position(ids),
    })
    r.raise_for_status()
    dash_id = r.json()["id"]
    print(f"\nDone! → http://localhost:8088/superset/dashboard/{dash_id}/")


if __name__ == "__main__":
    main()
