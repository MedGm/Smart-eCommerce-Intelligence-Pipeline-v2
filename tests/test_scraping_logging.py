import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch


def test_shopify_scraper_uses_logger_not_print(caplog):
    from src.scraping.shopify import ShopifyScraper

    with tempfile.TemporaryDirectory() as tmp:
        scraper = ShopifyScraper(
            output_dir=Path(tmp), store_url="https://example.com", shop_name="TestShop"
        )
        with (
            patch.object(scraper, "_extract_product_slugs_json_listing", return_value=[]),
            patch.object(scraper, "_extract_product_slugs_playwright", return_value=[]),
            patch("builtins.print") as mock_print,
            caplog.at_level(logging.INFO, logger="src.scraping.shopify"),
        ):
            scraper.scrape()

        assert any("TestShop" in r.getMessage() for r in caplog.records), (
            f"Expected logger output with 'TestShop', got: {[r.getMessage() for r in caplog.records]}"
        )
        mock_print.assert_not_called()


def test_woocommerce_scraper_uses_logger_not_print(caplog):
    from src.scraping.woocommerce import WooCommerceScraper

    with tempfile.TemporaryDirectory() as tmp:
        scraper = WooCommerceScraper(
            output_dir=Path(tmp), site_url="https://example.com", shop_name="TestWC"
        )
        resp = MagicMock(status_code=200)
        resp.json.return_value = []
        with (
            patch.object(scraper, "_session_get_with_retry", return_value=resp),
            patch("builtins.print") as mock_print,
            caplog.at_level(logging.INFO, logger="src.scraping.woocommerce"),
        ):
            scraper.scrape()

        assert any("TestWC" in r.getMessage() for r in caplog.records), (
            f"Expected logger output with 'TestWC', got: {[r.getMessage() for r in caplog.records]}"
        )
        mock_print.assert_not_called()
