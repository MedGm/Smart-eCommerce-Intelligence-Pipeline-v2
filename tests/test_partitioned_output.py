"""
TDD tests for partitioned timestamped output paths (Task 3).
Verifies that BaseScraper.save() writes to run_id subdirectory when run_id is set,
and falls back to flat path when run_id is None (backward compat).
"""

import json
import tempfile
from pathlib import Path

from src.scraping.base import ProductRecord


def _make_record(platform="shopify") -> ProductRecord:
    return ProductRecord(
        source_platform=platform,
        shop_name="TestShop",
        product_id="1",
        product_url="https://example.com/products/test",
        title="Test Product",
        description="A test product",
        category="Testing",
        brand="TestBrand",
        price=9.99,
        old_price=None,
        availability="instock",
        rating=4.5,
        review_count=10,
        geography="US",
        scraped_at="2026-05-17T13:00:00+00:00",
    )


def test_save_writes_to_timestamped_subdir():
    from src.scraping.base import BaseScraper

    with tempfile.TemporaryDirectory() as tmp:
        run_id = "20260517T130000Z"
        scraper = BaseScraper.__new__(BaseScraper)
        scraper.output_dir = Path(tmp)
        scraper.run_id = run_id

        record = _make_record()
        path = scraper.save([record], filename="ruggable.json")

        # Must be inside run_id subdir
        assert path.parent.name == "ruggable"
        assert path.name == f"{run_id}.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert len(data) == 1
        assert data[0]["title"] == "Test Product"


def test_save_without_run_id_falls_back_to_flat():
    """Backward compat: if run_id is None, save flat (old behaviour)."""
    from src.scraping.base import BaseScraper

    with tempfile.TemporaryDirectory() as tmp:
        scraper = BaseScraper.__new__(BaseScraper)
        scraper.output_dir = Path(tmp)
        scraper.run_id = None

        record = _make_record()
        path = scraper.save([record], filename="ruggable.json")
        assert path == Path(tmp) / "ruggable.json"
        assert path.exists()
