import tempfile
from pathlib import Path
from unittest.mock import patch


def test_playwright_not_called_when_json_returns_slugs():
    from src.scraping.shopify import ShopifyScraper

    with tempfile.TemporaryDirectory() as tmp:
        scraper = ShopifyScraper(
            output_dir=Path(tmp), store_url="https://example.myshopify.com", shop_name="Test"
        )
        fake_slugs = [
            {"slug": "product-1", "collection": "all"},
            {"slug": "product-2", "collection": "all"},
        ]
        with (
            patch.object(scraper, "_extract_product_slugs_json_listing", return_value=fake_slugs),
            patch.object(scraper, "_extract_product_slugs_playwright", return_value=[]) as mock_pw,
            patch.object(scraper, "_fetch_product_json", return_value=None),
            patch.object(scraper, "_fetch_product_html_fallback", return_value={}),
        ):
            scraper.scrape()
        mock_pw.assert_not_called()


def test_playwright_called_when_json_returns_empty():
    from src.scraping.shopify import ShopifyScraper

    with tempfile.TemporaryDirectory() as tmp:
        scraper = ShopifyScraper(
            output_dir=Path(tmp), store_url="https://example.myshopify.com", shop_name="Test"
        )
        with (
            patch.object(scraper, "_extract_product_slugs_json_listing", return_value=[]),
            patch.object(scraper, "_extract_product_slugs_playwright", return_value=[]) as mock_pw,
            patch.object(scraper, "_fetch_product_json", return_value=None),
            patch.object(scraper, "_fetch_product_html_fallback", return_value={}),
        ):
            scraper.scrape()
        mock_pw.assert_called_once()
