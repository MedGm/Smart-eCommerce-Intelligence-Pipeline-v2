import logging
from pathlib import Path
import tempfile


def test_shopify_scraper_uses_logger_not_print(caplog):
    from src.scraping.shopify import ShopifyScraper
    with tempfile.TemporaryDirectory() as tmp:
        # Use a valid store_url to trigger the logging path
        scraper = ShopifyScraper(
            output_dir=Path(tmp),
            store_url="https://example.com",
            shop_name="TestShop"
        )
        with caplog.at_level(logging.DEBUG, logger="src.scraping.shopify"):
            scraper.scrape()
        # Should have log records containing the shop name
        assert any("TestShop" in r.getMessage() for r in caplog.records), \
            f"Expected logger output with 'TestShop', got records: {[r.getMessage() for r in caplog.records]}"


def test_woocommerce_scraper_uses_logger_not_print(caplog):
    from src.scraping.woocommerce import WooCommerceScraper
    with tempfile.TemporaryDirectory() as tmp:
        # Use a valid site_url to trigger the logging path
        scraper = WooCommerceScraper(
            output_dir=Path(tmp),
            site_url="https://example.com",
            shop_name="TestWC"
        )
        with caplog.at_level(logging.DEBUG, logger="src.scraping.woocommerce"):
            scraper.scrape()
        # Should have log records containing the shop name
        assert any("TestWC" in r.getMessage() for r in caplog.records), \
            f"Expected logger output with 'TestWC', got records: {[r.getMessage() for r in caplog.records]}"
