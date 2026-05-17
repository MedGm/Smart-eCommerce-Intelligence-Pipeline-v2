import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile

from src.scraping.base import ProductRecord


def _record():
    return ProductRecord(
        source_platform="shopify",
        shop_name="TestShop",
        product_id="1",
        product_url="https://example.com/p/1",
        title="Test",
        description="Desc",
        category="Cat",
        brand="Brand",
        price=9.99,
        old_price=None,
        availability="instock",
        rating=4.5,
        review_count=10,
        geography="US",
        scraped_at="2026-05-17T00:00:00Z",
    )


def test_save_uploads_to_minio_when_configured(monkeypatch):
    monkeypatch.setenv("MINIO_ENDPOINT", "http://localhost:9000")

    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("DATA_DIR", tmp)  # make data_dir() point to tmp
        from src.scraping.base import BaseScraper
        scraper = BaseScraper.__new__(BaseScraper)
        scraper.output_dir = Path(tmp)
        scraper.run_id = "20260517T130000Z"

        with patch("src.storage.minio_client.upload_file") as mock_upload, \
             patch("src.storage.minio_client.is_minio_configured", return_value=True):
            path = scraper.save([_record()], filename="ruggable.json")

        assert mock_upload.called
        call_args = mock_upload.call_args_list[0][0]
        assert call_args[0] == path           # local path
        assert call_args[1] == "raw-data"     # bucket
        assert "ruggable" in call_args[2]     # key contains store name


def test_save_no_upload_when_not_configured(monkeypatch):
    monkeypatch.delenv("MINIO_ENDPOINT", raising=False)

    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("DATA_DIR", tmp)
        from src.scraping.base import BaseScraper
        scraper = BaseScraper.__new__(BaseScraper)
        scraper.output_dir = Path(tmp)
        scraper.run_id = "20260517T130000Z"

        with patch("src.storage.minio_client.upload_file") as mock_upload, \
             patch("src.storage.minio_client.is_minio_configured", return_value=False):
            scraper.save([_record()], filename="ruggable.json")

        mock_upload.assert_not_called()


def test_save_local_file_always_written(monkeypatch):
    monkeypatch.delenv("MINIO_ENDPOINT", raising=False)

    with tempfile.TemporaryDirectory() as tmp:
        from src.scraping.base import BaseScraper
        scraper = BaseScraper.__new__(BaseScraper)
        scraper.output_dir = Path(tmp)
        scraper.run_id = None

        path = scraper.save([_record()], filename="ruggable.json")

        assert path.exists()
        data = json.loads(path.read_text())
        assert data[0]["title"] == "Test"
