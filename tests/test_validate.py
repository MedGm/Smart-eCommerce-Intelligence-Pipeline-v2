"""Tests for preprocessing validation and DQ counters."""

import json

import pandas as pd
from src.preprocessing.validate import (
    add_extraction_status_columns,
    add_record_dq_score,
    build_dq_counters,
    validate_required,
    write_dq_counters,
)


def test_validate_required_drops_empty_strings():
    df = pd.DataFrame(
        [
            {
                "source_platform": "shopify",
                "shop_name": "A",
                "product_id": "1",
                "product_url": "https://example.com/p/1",
                "title": "Item 1",
            },
            {
                "source_platform": "shopify",
                "shop_name": "A",
                "product_id": "2",
                "product_url": "   ",
                "title": "Item 2",
            },
        ]
    )
    out = validate_required(df)
    assert len(out) == 1
    assert out.iloc[0]["product_id"] == "1"


def test_build_dq_counters_core_metrics():
    df = pd.DataFrame(
        [
            {
                "source_platform": "shopify",
                "shop_name": "A",
                "product_id": "1",
                "price": 10,
                "category": "uncategorized",
                "rating": 0,
            },
            {
                "source_platform": "shopify",
                "shop_name": "A",
                "product_id": "1",
                "price": None,
                "category": None,
                "rating": None,
            },
        ]
    )
    counters = build_dq_counters(df)
    assert counters["rows_total"] == 2
    assert counters["missing_price"] == 1
    assert counters["missing_category"] == 2
    assert counters["missing_rating"] == 2
    assert counters["duplicates_core_key"] == 2


def test_write_dq_counters(tmp_path):
    out_path = tmp_path / "dq_counters.json"
    counters = {"rows_total": 3, "missing_price": 1}
    write_dq_counters(counters, out_path)

    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded == counters


def test_add_extraction_status_columns():
    df = pd.DataFrame(
        [
            {
                "price": 10.0,
                "price_parse_failed": False,
                "rating": 4.5,
                "rating_parse_failed": False,
                "category": "audio",
                "category_parse_failed": False,
            },
            {
                "price": None,
                "price_parse_failed": True,
                "rating": None,
                "rating_parse_failed": False,
                "category": None,
                "category_parse_failed": False,
                "taxonomy_breadcrumb_present": False,
                "taxonomy_jsonld_category_present": False,
                "taxonomy_jsonld_breadcrumb_present": False,
                "taxonomy_product_type_present": False,
                "taxonomy_tags_present": False,
                "taxonomy_url_hint_present": False,
            },
        ]
    )

    out = add_extraction_status_columns(df)
    assert out.loc[0, "price_status"] == "found"
    assert out.loc[1, "price_status"] == "extraction_failed"
    assert out.loc[0, "rating_status"] == "found"
    assert out.loc[1, "rating_status"] == "not_present"
    assert out.loc[0, "category_status"] == "found"
    assert out.loc[1, "category_status"] == "missing"


def test_add_extraction_status_evidence_aware_failures():
    df = pd.DataFrame(
        [
            {
                "source_platform": "shopify",
                "product_url": "https://example.com/products/a",
                "title": "Great product",
                "description": "Now with 5 stars and customer reviews",
                "price": None,
                "price_parse_failed": False,
                "rating": None,
                "rating_parse_failed": False,
                "review_count": 0,
                "category": "uncategorized",
                "category_parse_failed": False,
                "taxonomy_tags_present": True,
            },
            {
                "source_platform": "woocommerce",
                "product_url": "https://example.com/product-category/headphones/item",
                "title": "Item",
                "description": "",
                "price": None,
                "price_parse_failed": False,
                "rating": None,
                "rating_parse_failed": False,
                "review_count": 0,
                "category": "uncategorized",
                "category_parse_failed": False,
                "taxonomy_jsonld_category_present": True,
            },
        ]
    )

    out = add_extraction_status_columns(df)
    assert out.loc[0, "price_status"] == "extraction_failed"
    assert out.loc[0, "rating_status"] == "extraction_failed"
    assert out.loc[0, "category_status"] == "inferred"
    assert out.loc[0, "category_source"] == "weak_signal_inference"
    assert out.loc[1, "category_status"] == "extraction_failed"
    assert out.loc[1, "category_source"] == "taxonomy_evidence_parse_failed"


def test_add_record_dq_score_range():
    df = pd.DataFrame(
        [
            {
                "source_platform": "shopify",
                "shop_name": "A",
                "product_id": "1",
                "product_url": "https://example.com/p/1",
                "title": "Valid title",
                "description": "This is a sufficiently long product description.",
                "price_status": "found",
                "rating_status": "found",
                "category_status": "found",
            },
            {
                "source_platform": "shopify",
                "shop_name": "A",
                "product_id": "2",
                "product_url": "https://example.com/p/2",
                "title": "Bad",
                "description": "short",
                "price_status": "extraction_failed",
                "rating_status": "not_present",
                "category_status": "missing",
            },
        ]
    )

    out = add_record_dq_score(df)
    assert "dq_score" in out.columns
    assert out["dq_score"].between(0, 100).all()
    assert out.loc[0, "dq_score"] > out.loc[1, "dq_score"]
