from unittest.mock import patch

import requests
from src.scraping.enrich_bs4 import enrich_product
from src.scraping.html_fallback import extract_product_fields_from_html
from src.scraping.shopify import ShopifyScraper
from src.scraping.woocommerce import WooCommerceScraper


class _Resp:
    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code


def test_extract_fields_from_jsonld_graph_and_offers_list():
    html = """
    <html><head>
      <script type=\"application/ld+json\">
      {
        \"@context\": \"https://schema.org\",
        \"@graph\": [
          {\"@type\": \"BreadcrumbList\"},
          {
            \"@type\": \"Product\",
            \"description\": \"Hydrating serum\",
            \"category\": \"Skincare\",
            \"aggregateRating\": {\"ratingValue\": \"4.8\", \"reviewCount\": \"123\"},
            \"offers\": [{\"price\": \"29.90\", \"availability\": \"https://schema.org/InStock\"}]
          }
        ]
      }
      </script>
    </head><body></body></html>
    """

    fields = extract_product_fields_from_html(html)

    assert fields["description"] == "Hydrating serum"
    assert fields["category"] == "Skincare"
    assert fields["rating"] == 4.8
    assert fields["review_count"] == 123
    assert fields["price"] == 29.9
    assert fields["availability"] == "https://schema.org/InStock"
    assert fields["taxonomy_jsonld_category_present"] is True
    assert fields["taxonomy_jsonld_breadcrumb_present"] is True
    assert fields["taxonomy_evidence_strength"] == "high"


def test_extract_fields_from_selector_fallbacks():
    html = """
    <html><head>
      <meta itemprop=\"description\" content=\"Premium backpack\" />
      <meta itemprop=\"price\" content=\"$1,299.50\" />
      <meta property=\"product:category\" content=\"Travel\" />
      <meta itemprop=\"availability\" content=\"in stock\" />
    </head><body></body></html>
    """

    fields = extract_product_fields_from_html(html)

    assert fields["description"] == "Premium backpack"
    assert fields["price"] == 1299.5
    assert fields["category"] == "Travel"
    assert fields["availability"] == "in stock"
    assert fields["taxonomy_evidence_strength"] in {"none", "low"}


@patch("src.scraping.shopify.requests.get")
def test_shopify_html_fallback_uses_resilient_extractor(mock_get, tmp_path):
    html = """
    <html><head>
      <meta name=\"description\" content=\"Fallback desc\" />
      <script type=\"application/ld+json\">
      {"@type":"Product","aggregateRating":{"ratingValue":"4.2","reviewCount":"11"}}
      </script>
    </head><body></body></html>
    """
    mock_get.return_value = _Resp(text=html, status_code=200)

    scraper = ShopifyScraper(output_dir=tmp_path, store_url="https://example.com", shop_name="Demo")
    fields = scraper._fetch_product_html_fallback("demo-product")

    assert fields["description"] == "Fallback desc"
    assert fields["rating"] == 4.2
    assert fields["review_count"] == 11


@patch("src.scraping.enrich_bs4.time.sleep")
@patch("src.scraping.enrich_bs4.requests.get")
def test_enrich_product_fills_missing_fields(mock_get, mock_sleep):
    html = """
    <html><head>
      <meta property=\"og:price:amount\" content=\"19.99\" />
      <script type=\"application/ld+json\">
      {"@type":"Product","category":"Accessories","aggregateRating":{"ratingValue":"4.6","reviewCount":"57"}}
      </script>
    </head><body></body></html>
    """
    mock_get.return_value = _Resp(text=html, status_code=200)

    product = {
        "product_url": "https://example.com/products/item",
        "description": "",
        "availability": None,
        "price": None,
        "rating": None,
        "review_count": None,
        "category": None,
    }

    enriched = enrich_product(product, delay=0)

    assert enriched["price"] == 19.99
    assert enriched["rating"] == 4.6
    assert enriched["review_count"] == 57
    assert enriched["category"] == "Accessories"
    mock_sleep.assert_called_once()


@patch("src.scraping.shopify.requests.get", side_effect=requests.RequestException("boom"))
def test_shopify_html_fallback_handles_request_error(_mock_get, tmp_path):
    scraper = ShopifyScraper(output_dir=tmp_path, store_url="https://example.com", shop_name="Demo")
    assert scraper._fetch_product_html_fallback("demo-product") == {}


def test_woocommerce_parser_fallback_fields(tmp_path):
    scraper = WooCommerceScraper(output_dir=tmp_path, site_url="https://wc.example", shop_name="WC")
    product = {
        "id": 10,
        "url": "https://wc.example/p/demo",
        "title": "Demo Product",
        "summary": "<p>Short <b>summary</b></p>",
        "tags": [{"name": "Fitness"}],
        "is_in_stock": True,
        "rating_value": "4.7",
        "reviews_count": "42",
    }

    assert scraper._product_url(product) == "https://wc.example/p/demo"
    assert scraper._title(product) == "Demo Product"
    assert scraper._description(product) == "Short summary"
    assert scraper._infer_category(product) == "Fitness"
    assert scraper._availability(product) == "instock"
    assert scraper._rating_info(product) == (4.7, 42)


def test_woocommerce_infer_category_from_attributes(tmp_path):
    scraper = WooCommerceScraper(output_dir=tmp_path, site_url="https://wc.example", shop_name="WC")
    product = {
        "attributes": [
            {
                "name": "collection",
                "options": ["Outdoor"],
            }
        ]
    }
    assert scraper._infer_category(product) == "Outdoor"


def test_woocommerce_taxonomy_evidence(tmp_path):
    scraper = WooCommerceScraper(output_dir=tmp_path, site_url="https://wc.example", shop_name="WC")
    product = {
        "categories": [{"name": "Headphones"}],
        "tags": [{"name": "Audio"}],
        "attributes": [{"name": "collection", "options": ["Studio"]}],
    }
    evidence = scraper._taxonomy_evidence(product, "https://wc.example/product-category/audio/x")
    assert evidence["taxonomy_jsonld_category_present"] is True
    assert evidence["taxonomy_tags_present"] is True
    assert evidence["taxonomy_product_type_present"] is True
    assert evidence["taxonomy_url_hint_present"] is True
    assert evidence["taxonomy_evidence_strength"] == "high"
