"""
Static scraping enrichment using requests + BeautifulSoup.

Visits individual Shopify product pages (already collected by Playwright)
and extracts structured fields: description, availability, rating from HTML.
Demonstrates scraping statique (dossier: requests + BeautifulSoup).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from src.scraping.html_fallback import extract_product_fields_from_html

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}
TIMEOUT = 12


def enrich_product(product: dict, delay: float = 1.0) -> dict:
    """Fetch a Shopify product page with requests+BS4 and fill missing fields."""
    url = product.get("product_url", "")
    if not url:
        return product

    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    except requests.RequestException as exc:
        print(f"  BS4 enrich: error fetching {url}: {exc}")
        return product

    if resp.status_code != 200:
        return product

    soup = BeautifulSoup(resp.text, "html.parser")
    fallback_fields = extract_product_fields_from_html(str(soup))

    if not product.get("description") and fallback_fields.get("description"):
        product["description"] = fallback_fields["description"]

    if not product.get("availability") and fallback_fields.get("availability"):
        product["availability"] = fallback_fields["availability"]

    if product.get("price") is None and fallback_fields.get("price") is not None:
        product["price"] = fallback_fields["price"]

    if product.get("rating") is None and fallback_fields.get("rating") is not None:
        product["rating"] = fallback_fields["rating"]

    if not product.get("review_count") and fallback_fields.get("review_count") is not None:
        product["review_count"] = fallback_fields["review_count"]

    if not product.get("category") and fallback_fields.get("category"):
        product["category"] = fallback_fields["category"]

    time.sleep(delay)
    return product


def enrich_raw_file(path: Path, max_products: int = 20, delay: float = 1.0) -> list[dict]:
    """Load a raw JSON file, enrich up to max_products, save back."""
    with open(path, encoding="utf-8") as f:
        products = json.load(f)

    enriched = 0
    for p in products:
        if enriched >= max_products:
            break
        if p.get("source_platform") != "shopify":
            continue
        before = dict(p)
        enrich_product(p, delay=delay)
        if p != before:
            enriched += 1
            print(f"  Enriched: {p.get('title', '?')[:50]}")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(products, f, indent=2, ensure_ascii=False)
    print(f"BS4 enrichment done: {enriched} products enriched in {path}")
    return products


if __name__ == "__main__":
    import os

    data_dir = Path(os.environ.get("DATA_DIR", "data"))
    raw = data_dir / "raw" / "shopify" / "products.json"
    if raw.exists():
        enrich_raw_file(raw, max_products=20, delay=1.5)
    else:
        print(f"No raw file at {raw}. Run scrapers first.")
