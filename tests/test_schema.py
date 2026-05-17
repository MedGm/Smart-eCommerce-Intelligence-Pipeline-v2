"""Tests for ProductRecord schema."""

from src.scraping.base import ProductRecord


def test_product_record_to_dict():
    r = ProductRecord(
        source_platform="shopify",
        shop_name="TestShop",
        product_id="p1",
        product_url="https://example.com/p1",
        title="Product",
        description="Desc",
        category="Electronics",
        brand="Brand",
        price=99.99,
        old_price=129.99,
        availability="in stock",
        rating=4.5,
        review_count=10,
        geography="US",
        scraped_at="2024-01-01T00:00:00",
        taxonomy_breadcrumb_present=True,
        taxonomy_breadcrumb_count=3,
        taxonomy_jsonld_category_present=True,
        taxonomy_jsonld_breadcrumb_present=False,
        taxonomy_product_type_present=True,
        taxonomy_tags_present=False,
        taxonomy_url_hint_present=True,
        taxonomy_sources_detected="breadcrumb_html|product_type|url_hint",
        taxonomy_evidence_strength="high",
    )
    d = r.to_dict()
    assert d["source_platform"] == "shopify"
    assert d["price"] == 99.99
    assert d["taxonomy_evidence_strength"] == "high"


def test_product_record_from_dict():
    d = {
        "source_platform": "woocommerce",
        "shop_name": "WC",
        "product_id": "42",
        "product_url": "https://wc.com/42",
        "title": "T",
        "description": "D",
        "category": None,
        "brand": None,
        "price": 19.99,
        "old_price": None,
        "availability": None,
        "rating": 4.0,
        "review_count": 0,
        "geography": None,
        "scraped_at": "2024-01-01",
        "taxonomy_breadcrumb_present": False,
        "taxonomy_breadcrumb_count": None,
        "taxonomy_jsonld_category_present": False,
        "taxonomy_jsonld_breadcrumb_present": False,
        "taxonomy_product_type_present": False,
        "taxonomy_tags_present": True,
        "taxonomy_url_hint_present": True,
        "taxonomy_sources_detected": "tags|url_hint",
        "taxonomy_evidence_strength": "low",
    }
    r = ProductRecord.from_dict(d)
    assert r.product_id == "42"
    assert r.source_platform == "woocommerce"
    assert r.taxonomy_sources_detected == "tags|url_hint"
