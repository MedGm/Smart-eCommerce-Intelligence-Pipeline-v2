import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch


def test_fetch_product_json_retries_on_429():
    from src.scraping.shopify import ShopifyScraper

    with tempfile.TemporaryDirectory() as tmp:
        scraper = ShopifyScraper(
            output_dir=Path(tmp), store_url="https://example.myshopify.com", shop_name="Test"
        )
        resp_429 = MagicMock()
        resp_429.status_code = 429

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.json.return_value = {
            "product": {"id": 1, "title": "T", "variants": [], "images": [], "body_html": ""}
        }

        with (
            patch("requests.get", side_effect=[resp_429, resp_200]) as mock_get,
            patch("time.sleep"),
        ):
            result = scraper._fetch_product_json("test-slug")

        assert mock_get.call_count == 2, f"Expected 2 calls (1 retry), got {mock_get.call_count}"
        assert result is not None, "Expected product data on retry success"


def test_get_with_retry_gives_up_after_max_retries():
    from src.scraping.shopify import ShopifyScraper

    with tempfile.TemporaryDirectory() as tmp:
        scraper = ShopifyScraper(
            output_dir=Path(tmp), store_url="https://example.myshopify.com", shop_name="Test"
        )
        resp_429 = MagicMock()
        resp_429.status_code = 429

        with patch("requests.get", return_value=resp_429) as mock_get, patch("time.sleep"):
            result = scraper._get_with_retry("https://example.com/product.json", max_retries=3)

        assert result is None, "Expected None after exhausting retries"
        assert mock_get.call_count == 3, f"Expected 3 attempts, got {mock_get.call_count}"
