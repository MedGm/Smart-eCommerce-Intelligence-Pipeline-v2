"""
Shared product schema and base scraper interface.
Single schema for both Shopify and WooCommerce; maps to dossier fields.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.config import data_dir as _data_dir


@dataclass
class ProductRecord:
    source_platform: str
    shop_name: str
    product_id: str
    product_url: str
    title: str
    description: str
    category: str | None
    brand: str | None
    price: float | None
    old_price: float | None
    availability: str | None
    rating: float | None
    review_count: int | None
    geography: str | None
    scraped_at: str
    taxonomy_breadcrumb_present: bool | None = None
    taxonomy_breadcrumb_count: int | None = None
    taxonomy_jsonld_category_present: bool | None = None
    taxonomy_jsonld_breadcrumb_present: bool | None = None
    taxonomy_product_type_present: bool | None = None
    taxonomy_tags_present: bool | None = None
    taxonomy_url_hint_present: bool | None = None
    taxonomy_sources_detected: str | None = None
    taxonomy_evidence_strength: str | None = None
    category_path_raw: str | None = None
    category_leaf_raw: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ProductRecord":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})


class BaseScraper:
    """Base class for Shopify and WooCommerce adapters."""

    def __init__(self, output_dir: Path, run_id: str | None = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id

    def scrape(self) -> list[ProductRecord]:
        """Override: fetch products and return list of ProductRecord."""
        raise NotImplementedError

    def save(self, records: list[ProductRecord], filename: str = "products.json") -> Path:
        if self.run_id:
            stem = Path(filename).stem        # "ruggable"
            suffix = Path(filename).suffix    # ".json"
            dest_dir = self.output_dir / stem
            dest_dir.mkdir(parents=True, exist_ok=True)
            path = dest_dir / f"{self.run_id}{suffix}"
        else:
            path = self.output_dir / filename
        data = [r.to_dict() for r in records]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Upload to MinIO when configured — silent no-op if unavailable
        from src.storage.minio_client import is_minio_configured, upload_file
        if is_minio_configured():
            try:
                key = str(path.relative_to(_data_dir()))
            except ValueError:
                key = path.name
            upload_file(path, bucket="raw-data", key=key)

        return path
