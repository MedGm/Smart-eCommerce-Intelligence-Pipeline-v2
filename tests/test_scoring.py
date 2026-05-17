"""Tests for Top-K scoring."""

import pandas as pd
from src.scoring.topk import WEIGHTS, compute_score, topk_overall, topk_per_shop


def test_compute_score():
    df = pd.DataFrame(
        {
            "rating": [5.0, 0.0],
            "review_count": [100, 0],
            "is_in_stock": [True, False],
            "discount_pct": [0.2, 0.0],
        }
    )
    s = compute_score(df)
    assert len(s) == 2
    assert s.iloc[0] >= s.iloc[1]


def test_weights_sum_to_one():
    # Float precision: 0.35 + 0.3 + 0.2 + 0.15 can be 0.999... in some runtimes
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


def test_topk_per_shop_keeps_group_columns():
    df = pd.DataFrame(
        {
            "source_platform": ["shopify", "shopify", "woocommerce"],
            "shop_name": ["A", "A", "B"],
            "product_id": ["1", "2", "3"],
            "score": [0.9, 0.5, 0.8],
        }
    )
    out = topk_per_shop(df, k=1)
    assert "shop_name" in out.columns
    assert "source_platform" in out.columns
    assert set(out["shop_name"]) == {"A", "B"}


def test_compute_score_sparse_signal_fallback():
    df = pd.DataFrame(
        {
            "rating": [0.0, 0.0],
            "review_count": [0, 0],
            "is_in_stock": [True, False],
            "availability": ["instock", "outofstock"],
            "discount_pct": [0.2, 0.0],
        }
    )
    s = compute_score(df)
    assert s.iloc[0] > s.iloc[1]


def test_topk_overall_diversifies_shops_with_cap():
    df = pd.DataFrame(
        {
            "shop_name": ["A"] * 8 + ["B"] * 2 + ["C"] * 2,
            "score": [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.80, 0.79, 0.70, 0.69],
        }
    )
    out = topk_overall(df, k=6, max_per_shop_ratio=0.5)
    counts = out["shop_name"].value_counts().to_dict()
    assert set(out["shop_name"]) >= {"A", "B", "C"}
    assert counts.get("A", 0) <= 3
