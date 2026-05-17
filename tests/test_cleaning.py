"""Tests for preprocessing cleaning."""

import pandas as pd
from src.preprocessing.clean import (
    canonicalize_product_urls,
    clean,
    clean_categories,
    remove_duplicates,
    standardize_prices,
)


def test_remove_duplicates():
    df = pd.DataFrame(
        [
            {"source_platform": "s", "shop_name": "h", "product_id": "1", "title": "A"},
            {"source_platform": "s", "shop_name": "h", "product_id": "1", "title": "B"},
        ]
    )
    out = remove_duplicates(df)
    assert len(out) == 1


def test_standardize_prices():
    df = pd.DataFrame({"price": ["10.5", "invalid", -1], "old_price": [20, 30, 40]})
    out = standardize_prices(df)
    assert out["price"].iloc[0] == 10.5
    assert pd.isna(out["price"].iloc[1])
    assert pd.isna(out["price"].iloc[2]) or out["price"].iloc[2] < 0 == False


def test_clean_pipeline():
    df = pd.DataFrame(
        [
            {
                "source_platform": "s",
                "shop_name": "h",
                "product_id": "1",
                "title": "A",
                "price": 10,
                "old_price": 20,
                "rating": 4,
                "review_count": 5,
            },
            {
                "source_platform": "s",
                "shop_name": "h",
                "product_id": "1",
                "title": "A",
                "price": 10,
                "old_price": 20,
                "rating": 4,
                "review_count": 5,
            },
        ]
    )
    out = clean(df)
    assert len(out) == 1


def test_canonicalize_product_urls():
    df = pd.DataFrame(
        {
            "product_url": [
                "HTTPS://Example.com/products/item-1/?utm_source=ads&b=2&a=1#section",
                "",
            ]
        }
    )
    out = canonicalize_product_urls(df)
    assert out["product_url"].iloc[0] == "https://example.com/products/item-1?a=1&b=2"
    assert pd.isna(out["product_url"].iloc[1])


def test_clean_categories_aliases():
    df = pd.DataFrame({"category": [" Gift Card ", "uncategorised", "Storage & Brewing", ""]})
    out = clean_categories(df)
    assert out["category"].iloc[0] == "gift cards"
    assert out["category"].iloc[1] == "uncategorized"
    assert out["category"].iloc[2] == "storage and brewing"
    assert pd.isna(out["category"].iloc[3])


def test_clean_categories_shop_all_prefix_removed():
    df = pd.DataFrame({"category": ["Shop All Matching Sets"]})
    out = clean_categories(df)
    assert out["category"].iloc[0] == "matching sets"


def test_clean_categories_uses_path_when_category_looks_like_title():
    df = pd.DataFrame(
        {
            "title": ["Peachy Keen Chiffon Maxi Dress - Pink Combo"],
            "category": ["Peachy Keen Chiffon Maxi Dress - Pink Combo"],
            "category_path_raw": [
                "women > Women's Dresses > Peachy Keen Chiffon Maxi Dress - Pink/combo"
            ],
        }
    )
    out = clean_categories(df)
    assert out["category"].iloc[0] == "women s dresses"


def test_clean_categories_path_skips_title_like_leaf():
    df = pd.DataFrame(
        {
            "title": ["Prisma Monochrome Tufted Rug"],
            "category": ["Prisma Monochrome Rug"],
            "category_path_raw": ["Home > Indoor Rugs > Tufted Rugs > Prisma Monochrome Rug"],
        }
    )
    out = clean_categories(df)
    assert out["category"].iloc[0] == "tufted rugs"
