"""
WooCommerce scraper adapter (multi-store, Store API).

Fixes applied vs v1:
- Prices divided by 10^currency_minor_unit (typically cents → dollars).
- HTML stripped from descriptions via BeautifulSoup.
- Multi-store support via constructor params.

Dossier tools: requests, BeautifulSoup, WooCommerce REST API.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from src.config import get_logger
from src.scraping.base import BaseScraper, ProductRecord
from src.scraping.html_fallback import (
    extract_product_fields_from_html,
    extract_woocommerce_taxonomy_from_html,
)

logger = get_logger(__name__)


class WooCommerceScraper(BaseScraper):
    """Scrape product data from a WooCommerce site using the Store API."""

    def __init__(
        self,
        output_dir: Path,
        site_url: str = "",
        shop_name: str = "Unknown",
        geography: str | None = None,
        run_id: str | None = None,
    ):
        super().__init__(output_dir, run_id=run_id)
        self.site_url = site_url.rstrip("/")
        self.shop_name = shop_name
        self.geography = geography
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36"
                )
            }
        )
        self._currency_minor_unit: int | None = None

    def _api_url(self, per_page: int, page: int) -> str:
        return f"{self.site_url}/wp-json/wc/store/v1/products?per_page={per_page}&page={page}"

    def _detect_currency_minor_unit(self, product: dict) -> int:
        """Detect how many decimal places to shift prices (usually 2 for USD/EUR)."""
        if self._currency_minor_unit is not None:
            return self._currency_minor_unit
        prices = product.get("prices") or {}
        unit = prices.get("currency_minor_unit")
        if unit is not None:
            try:
                self._currency_minor_unit = int(unit)
                return self._currency_minor_unit
            except (ValueError, TypeError):
                pass
        self._currency_minor_unit = 2
        return 2

    def _parse_price(self, product: dict) -> tuple[float | None, float | None]:
        """Extract current and old price, converting from minor units to dollars."""
        prices = product.get("prices") or {}
        divisor = 10 ** self._detect_currency_minor_unit(product)

        def _to_float(value: object) -> float | None:
            if value is None:
                return None
            try:
                return float(value) / divisor
            except (TypeError, ValueError):
                return None

        current = _to_float(prices.get("price"))
        regular = _to_float(prices.get("regular_price"))
        sale = _to_float(prices.get("sale_price"))

        price = current or sale or regular
        old_price = regular if sale and regular and sale < regular else None
        return price, old_price

    def _infer_category(self, product: dict) -> str | None:
        categories = product.get("categories") or []
        if categories:
            name = categories[0].get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
        tags = product.get("tags") or []
        if tags:
            name = tags[0].get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
        attrs = product.get("attributes") or []
        if attrs:
            first = attrs[0]
            if isinstance(first, dict):
                options = first.get("options") or []
                if options:
                    option = options[0]
                    if isinstance(option, str) and option.strip():
                        return option.strip()
        return None

    def _taxonomy_evidence(self, product: dict, product_url: str) -> dict:
        categories = product.get("categories") or []
        tags = product.get("tags") or []
        attrs = product.get("attributes") or []

        categories_present = bool(categories)
        tags_present = bool(tags)
        attrs_present = bool(attrs)
        url_hint_present = any(
            token in product_url.lower() for token in ["/product-category/", "/category/"]
        )

        sources: list[str] = []
        if categories_present:
            sources.append("taxonomy_categories")
        if attrs_present:
            sources.append("taxonomy_attributes")
        if tags_present:
            sources.append("tags")
        if url_hint_present:
            sources.append("url_hint")

        if categories_present:
            strength = "high"
        elif attrs_present:
            strength = "medium"
        elif tags_present or url_hint_present:
            strength = "low"
        else:
            strength = "none"

        return {
            "taxonomy_breadcrumb_present": categories_present,
            "taxonomy_breadcrumb_count": len(categories) if categories_present else None,
            "taxonomy_jsonld_category_present": categories_present,
            "taxonomy_jsonld_breadcrumb_present": False,
            "taxonomy_product_type_present": attrs_present,
            "taxonomy_tags_present": tags_present,
            "taxonomy_url_hint_present": url_hint_present,
            "taxonomy_sources_detected": "|".join(sorted(sources)) if sources else None,
            "taxonomy_evidence_strength": strength,
        }

    def _session_get_with_retry(
        self, url: str, max_retries: int = 3, backoff_base: float = 1.5, **kwargs
    ):
        for attempt in range(max_retries):
            try:
                resp = self.session.get(url, **kwargs)
                if resp.status_code in (429, 503):
                    if attempt < max_retries - 1:
                        wait = backoff_base**attempt
                        logger.warning(
                            "HTTP %d from %s, retry %d/%d in %.1fs",
                            resp.status_code,
                            url,
                            attempt + 1,
                            max_retries,
                            wait,
                        )
                        time.sleep(wait)
                    continue
                return resp
            except requests.RequestException as exc:
                logger.warning("Request failed %s: %s (attempt %d)", url, exc, attempt + 1)
                if attempt < max_retries - 1:
                    time.sleep(backoff_base**attempt)
        return None

    def _strip_html(self, text: str) -> str:
        """Remove HTML tags from description text."""
        if not text:
            return ""
        cleaned = BeautifulSoup(text, "html.parser").get_text(separator=" ").strip()
        cleaned = " ".join(cleaned.split())
        return cleaned[:1000]

    def _fetch_product_html(self, url: str) -> str | None:
        """Fetch raw HTML from a product permalink for taxonomy enrichment."""
        if not url:
            return None
        resp = self._session_get_with_retry(url, timeout=15)
        if resp is not None and resp.status_code == 200:
            return resp.text
        return None

    def _enrich_from_html(self, product_url: str, product_title: str | None = None) -> dict:
        """
        Fetch the product page HTML and extract taxonomy evidence + category.

        Cascade priority (implemented inside extract_product_fields_from_html):
          1. JSON-LD BreadcrumbList leaf
          2. HTML woocommerce-breadcrumb / generic breadcrumb links
          3. meta[property='product:category'] / [itemprop='category']
          4. URL hint

        Returns empty dict if HTML cannot be fetched or yields no useful data.
        """
        html = self._fetch_product_html(product_url)
        if not html:
            return {}
        # Keep generic extraction for non-taxonomy fields (price/availability/description)
        generic_fields = extract_product_fields_from_html(html, product_url)
        # Override taxonomy resolution with WooCommerce-specific ranked cascade
        woo_taxonomy_fields = extract_woocommerce_taxonomy_from_html(
            html,
            product_url=product_url,
            product_title=product_title,
        )
        return {**generic_fields, **woo_taxonomy_fields}

    def _availability(self, product: dict) -> str | None:
        stock_status = product.get("stock_status")
        if isinstance(stock_status, str) and stock_status:
            return stock_status
        in_stock = product.get("is_in_stock")
        if in_stock is True:
            return "instock"
        if in_stock is False:
            return "outofstock"
        return None

    def _rating_info(self, product: dict) -> tuple[float | None, int | None]:
        avg = product.get("average_rating")
        if avg in (None, "", "0"):
            avg = product.get("rating") or product.get("rating_value")

        count = product.get("review_count")
        if count in (None, "", 0, "0"):
            count = product.get("rating_count") or product.get("reviews_count")

        try:
            rating = float(avg) if avg not in (None, "", "0") else None
        except (TypeError, ValueError):
            rating = None
        try:
            reviews = int(count) if count not in (None, 0, "0") else None
        except (TypeError, ValueError):
            reviews = None
        return rating, reviews

    def _product_url(self, product: dict) -> str:
        url = product.get("permalink") or product.get("link") or product.get("url") or ""
        if isinstance(url, str):
            return url.strip()
        return ""

    def _title(self, product: dict) -> str:
        title = product.get("name") or product.get("title") or ""
        if isinstance(title, str):
            return title.strip()
        return ""

    def _description(self, product: dict) -> str:
        raw_desc = product.get("description") or product.get("short_description") or ""
        if not raw_desc:
            short = product.get("summary") or ""
            raw_desc = short
        if isinstance(raw_desc, str):
            return self._strip_html(raw_desc)
        return ""

    def scrape(self) -> list[ProductRecord]:
        if not self.site_url:
            logger.warning("WooCommerceScraper: no site_url configured, skipping.")
            return []

        logger.info("WooCommerceScraper: starting %s (%s)", self.shop_name, self.site_url)

        per_page = 40
        max_pages = 25
        records: list[ProductRecord] = []
        now = datetime.now(timezone.utc).isoformat()

        for page in range(1, max_pages + 1):
            url = self._api_url(per_page=per_page, page=page)
            resp = self._session_get_with_retry(url, timeout=15)
            if resp is None:
                logger.warning(
                    "  [%s] Page %d failed after retries, stopping.", self.shop_name, page
                )
                break

            if resp.status_code != 200:
                logger.warning(
                    "  [%s] Page %d status %d, stopping.", self.shop_name, page, resp.status_code
                )
                break

            try:
                data = resp.json()
            except ValueError:
                logger.warning("  [%s] Invalid JSON on page %d, stopping.", self.shop_name, page)
                break

            if not isinstance(data, list) or not data:
                break

            for product in data:
                product_id = str(product.get("id"))
                product_url = self._product_url(product)
                title = self._title(product)
                description = self._description(product)

                if not product_id or not title or not product_url:
                    continue

                category = self._infer_category(product)
                price, old_price = self._parse_price(product)
                availability = self._availability(product)
                rating, review_count = self._rating_info(product)
                taxonomy = self._taxonomy_evidence(product, product_url)

                # Fix 2: When API gives no category, enrich from product page HTML.
                # Fetches and parses JSON-LD BreadcrumbList + HTML breadcrumb markup.
                category_path_raw: str | None = None
                category_leaf_raw: str | None = None
                if category is None:
                    html_fields = self._enrich_from_html(product_url, product_title=title)
                    if html_fields:
                        category = html_fields.get("category") or category
                        category_path_raw = html_fields.get("category_path_raw")
                        category_leaf_raw = html_fields.get("category_leaf_raw")
                        # Merge taxonomy evidence: HTML source wins when API had nothing
                        if html_fields.get("taxonomy_evidence_strength", "none") != "none":
                            for key in (
                                "taxonomy_breadcrumb_present",
                                "taxonomy_breadcrumb_count",
                                "taxonomy_jsonld_category_present",
                                "taxonomy_jsonld_breadcrumb_present",
                                "taxonomy_product_type_present",
                                "taxonomy_tags_present",
                                "taxonomy_url_hint_present",
                                "taxonomy_sources_detected",
                                "taxonomy_evidence_strength",
                            ):
                                if key in html_fields:
                                    taxonomy[key] = html_fields[key]

                record = ProductRecord(
                    source_platform="woocommerce",
                    shop_name=self.shop_name,
                    product_id=product_id,
                    product_url=product_url,
                    title=title,
                    description=description,
                    category=category,
                    brand=self.shop_name,
                    price=price,
                    old_price=old_price,
                    availability=availability,
                    rating=rating,
                    review_count=review_count,
                    geography=self.geography,
                    scraped_at=now,
                    category_path_raw=category_path_raw,
                    category_leaf_raw=category_leaf_raw,
                    taxonomy_breadcrumb_present=taxonomy["taxonomy_breadcrumb_present"],
                    taxonomy_breadcrumb_count=taxonomy["taxonomy_breadcrumb_count"],
                    taxonomy_jsonld_category_present=taxonomy["taxonomy_jsonld_category_present"],
                    taxonomy_jsonld_breadcrumb_present=taxonomy[
                        "taxonomy_jsonld_breadcrumb_present"
                    ],
                    taxonomy_product_type_present=taxonomy["taxonomy_product_type_present"],
                    taxonomy_tags_present=taxonomy["taxonomy_tags_present"],
                    taxonomy_url_hint_present=taxonomy["taxonomy_url_hint_present"],
                    taxonomy_sources_detected=taxonomy["taxonomy_sources_detected"],
                    taxonomy_evidence_strength=taxonomy["taxonomy_evidence_strength"],
                )
                records.append(record)

            logger.info(
                "  [%s] Page %d: %d items (total: %d)",
                self.shop_name,
                page,
                len(data),
                len(records),
            )
            if len(data) < per_page:
                break

        logger.info("WooCommerceScraper: %s done — %d products", self.shop_name, len(records))
        return records
