"""Tests for feature engineering."""

import pandas as pd
from src.features.build_features import (
    build_features,
    discount_pct,
    is_in_stock,
    price_zscore_by_category,
    rating_weighted_reviews,
)


def test_discount_pct_basic():
    df = pd.DataFrame({"price": [80.0, 100.0], "old_price": [100.0, 100.0]})
    result = discount_pct(df)
    assert abs(result[0] - 0.2) < 1e-6
    assert abs(result[1] - 0.0) < 1e-6


def test_discount_pct_no_old_price():
    df = pd.DataFrame({"price": [80.0], "old_price": [0.0]})
    result = discount_pct(df)
    assert result[0] == 0.0


def test_price_zscore_by_category():
    df = pd.DataFrame(
        {
            "price": [10.0, 20.0, 30.0, 100.0],
            "category": ["A", "A", "A", "B"],
        }
    )
    result = price_zscore_by_category(df)
    assert len(result) == 4
    # Within category A, mean=20, std=10 → first is -1.0
    assert abs(result.iloc[0] - (-1.0)) < 1e-6


def test_rating_weighted_reviews():
    df = pd.DataFrame({"rating": [5.0, 0.0], "review_count": [100, 0]})
    result = rating_weighted_reviews(df)
    assert result.iloc[0] > result.iloc[1]


def test_is_in_stock():
    df = pd.DataFrame({"availability": ["in stock", "out of stock", "available"]})
    result = is_in_stock(df)
    assert result.tolist() == [True, False, True]


def test_is_in_stock_handles_schema_and_compact_flags():
    df = pd.DataFrame(
        {
            "availability": [
                "instock",
                "outofstock",
                "https://schema.org/InStock",
                "https://schema.org/OutOfStock",
            ]
        }
    )
    result = is_in_stock(df)
    assert result.tolist() == [True, False, True, False]


def test_build_features_adds_columns():
    df = pd.DataFrame(
        {
            "source_platform": ["shopify"],
            "shop_name": ["TestShop"],
            "product_id": ["p1"],
            "title": ["Test Product"],
            "description": ["A great product"],
            "category": ["Electronics"],
            "brand": ["Brand"],
            "price": [99.99],
            "old_price": [129.99],
            "availability": ["in stock"],
            "rating": [4.5],
            "review_count": [10],
        }
    )
    result = build_features(df)
    for col in [
        "discount_pct",
        "price_zscore_by_category",
        "rating_weighted_reviews",
        "is_in_stock",
        "description_length",
        "title_length",
        "shop_product_count",
        "category_frequency",
        "popularity_proxy",
    ]:
        assert col in result.columns, f"Missing feature column: {col}"
