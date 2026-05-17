#!/usr/bin/env python3
"""
Quick validator for Shopify / WooCommerce target stores.

What it does:
- Checks site reachability
- Detects likely platform
- Tests common Shopify/WooCommerce endpoints
- Gives a lightweight score
- Exports results to CSV

Usage:
    python validate_targets.py --input targets.txt --output target_results.csv

targets.txt format:
    https://example1.com
    https://example2.com
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from dataclasses import asdict, dataclass
from urllib.parse import urljoin

import requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}

TIMEOUT = 12


@dataclass
class ValidationResult:
    url: str
    reachable: bool
    homepage_status: int | None
    detected_platform: str
    shopify_collections_all_ok: bool
    shopify_products_hint: bool
    woocommerce_store_api_ok: bool
    woocommerce_shop_ok: bool
    pagination_hint: bool
    anti_bot_hint: bool
    field_coverage_hint: str
    score: float
    keep: bool
    notes: str


def fetch(url: str) -> tuple[requests.Response | None, str | None]:
    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=TIMEOUT,
            allow_redirects=True,
        )
        return response, None
    except requests.RequestException as exc:
        return None, str(exc)


def normalize_url(url: str) -> str:
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url.rstrip("/")


def detect_platform(html: str, headers: dict[str, str]) -> str:
    html_lower = html.lower()
    server_header = headers.get("server", "").lower()
    x_powered_by = headers.get("x-powered-by", "").lower()

    shopify_markers = [
        "cdn.shopify.com",
        "shopify.theme",
        "shopify",
        "/products/",
        "/collections/",
    ]
    woo_markers = [
        "woocommerce",
        "wp-content",
        "wp-json",
        "wordpress",
        "wc-blocks",
    ]

    shopify_score = sum(1 for m in shopify_markers if m in html_lower)
    woo_score = sum(1 for m in woo_markers if m in html_lower)

    if "shopify" in server_header or "shopify" in x_powered_by:
        shopify_score += 1
    if "wordpress" in x_powered_by:
        woo_score += 1

    if shopify_score > woo_score and shopify_score >= 2:
        return "Shopify"
    if woo_score > shopify_score and woo_score >= 2:
        return "WooCommerce"
    return "Unknown"


def has_pagination_hint(html: str) -> bool:
    patterns = [
        r'rel=["\']next["\']',
        r"page=\d+",
        r"/page/\d+",
        r"pagination",
        r"next page",
    ]
    html_lower = html.lower()
    return any(re.search(p, html_lower) for p in patterns)


def has_anti_bot_hint(html: str, status_code: int) -> bool:
    html_lower = html.lower()
    markers = [
        "captcha",
        "cloudflare",
        "attention required",
        "verify you are human",
        "access denied",
        "blocked",
    ]
    return status_code in {403, 429, 503} or any(m in html_lower for m in markers)


def infer_field_coverage(html: str) -> str:
    html_lower = html.lower()
    found = 0
    signals = [
        "price",
        "title",
        "description",
        "availability",
        "rating",
        "review",
        "product",
    ]
    for signal in signals:
        if signal in html_lower:
            found += 1

    if found >= 5:
        return "Good"
    if found >= 3:
        return "Medium"
    return "Poor"


def test_shopify(url: str) -> tuple[bool, bool, str]:
    notes = []
    collections_ok = False
    products_hint = False

    collection_url = urljoin(url + "/", "collections/all")
    resp, err = fetch(collection_url)
    if resp is not None and resp.status_code == 200:
        collections_ok = True
        notes.append("collections/all OK")
        if "/products/" in resp.text.lower():
            products_hint = True
            notes.append("product links found in collections page")
    elif err:
        notes.append(f"collections/all error: {err}")
    elif resp is not None:
        notes.append(f"collections/all status={resp.status_code}")

    return collections_ok, products_hint, "; ".join(notes)


def test_woocommerce(url: str) -> tuple[bool, bool, str]:
    notes = []
    store_api_ok = False
    shop_ok = False

    api_url = urljoin(url + "/", "wp-json/wc/store/v1/products?per_page=5")
    resp, err = fetch(api_url)
    if resp is not None and resp.status_code == 200:
        ctype = resp.headers.get("Content-Type", "").lower()
        body = resp.text[:200].strip()
        if "application/json" in ctype or body.startswith("[") or body.startswith("{"):
            store_api_ok = True
            notes.append("WooCommerce Store API OK")
    elif err:
        notes.append(f"store api error: {err}")
    elif resp is not None:
        notes.append(f"store api status={resp.status_code}")

    shop_url = urljoin(url + "/", "shop/")
    resp2, err2 = fetch(shop_url)
    if resp2 is not None and resp2.status_code == 200:
        shop_ok = True
        notes.append("/shop/ OK")
    elif err2:
        notes.append(f"/shop/ error: {err2}")
    elif resp2 is not None:
        notes.append(f"/shop/ status={resp2.status_code}")

    return store_api_ok, shop_ok, "; ".join(notes)


def compute_score(
    reachable: bool,
    platform: str,
    shopify_collections_ok: bool,
    woocommerce_store_api_ok: bool,
    pagination_hint: bool,
    anti_bot_hint: bool,
    field_coverage: str,
) -> float:
    score = 0.0

    if reachable:
        score += 1.0

    if platform in {"Shopify", "WooCommerce"}:
        score += 1.5

    if shopify_collections_ok or woocommerce_store_api_ok:
        score += 2.0

    if pagination_hint:
        score += 1.0

    if not anti_bot_hint:
        score += 1.5

    if field_coverage == "Good":
        score += 2.0
    elif field_coverage == "Medium":
        score += 1.0

    # small bonus for likely easier extraction
    if woocommerce_store_api_ok:
        score += 1.0

    return min(score, 10.0)


def validate_site(url: str) -> ValidationResult:
    normalized = normalize_url(url)
    response, error = fetch(normalized)

    if response is None:
        return ValidationResult(
            url=normalized,
            reachable=False,
            homepage_status=None,
            detected_platform="Unknown",
            shopify_collections_all_ok=False,
            shopify_products_hint=False,
            woocommerce_store_api_ok=False,
            woocommerce_shop_ok=False,
            pagination_hint=False,
            anti_bot_hint=False,
            field_coverage_hint="Poor",
            score=0.0,
            keep=False,
            notes=f"Homepage unreachable: {error}",
        )

    html = response.text
    platform = detect_platform(html, response.headers)
    pagination_hint = has_pagination_hint(html)
    anti_bot_hint = has_anti_bot_hint(html, response.status_code)
    field_coverage = infer_field_coverage(html)

    shopify_collections_ok = False
    shopify_products_hint = False
    woocommerce_store_api_ok = False
    woocommerce_shop_ok = False
    notes = []

    if platform == "Shopify":
        shopify_collections_ok, shopify_products_hint, shopify_notes = test_shopify(normalized)
        notes.append(shopify_notes)
    elif platform == "WooCommerce":
        woocommerce_store_api_ok, woocommerce_shop_ok, woo_notes = test_woocommerce(normalized)
        notes.append(woo_notes)
    else:
        # Test both lightly if unknown
        shopify_collections_ok, shopify_products_hint, shopify_notes = test_shopify(normalized)
        woocommerce_store_api_ok, woocommerce_shop_ok, woo_notes = test_woocommerce(normalized)
        if shopify_collections_ok:
            platform = "Shopify?"
        elif woocommerce_store_api_ok:
            platform = "WooCommerce?"
        notes.extend([shopify_notes, woo_notes])

    score = compute_score(
        reachable=True,
        platform=platform.replace("?", ""),
        shopify_collections_ok=shopify_collections_ok,
        woocommerce_store_api_ok=woocommerce_store_api_ok,
        pagination_hint=pagination_hint,
        anti_bot_hint=anti_bot_hint,
        field_coverage=field_coverage,
    )

    keep = score >= 7.0 and not anti_bot_hint

    return ValidationResult(
        url=normalized,
        reachable=True,
        homepage_status=response.status_code,
        detected_platform=platform,
        shopify_collections_all_ok=shopify_collections_ok,
        shopify_products_hint=shopify_products_hint,
        woocommerce_store_api_ok=woocommerce_store_api_ok,
        woocommerce_shop_ok=woocommerce_shop_ok,
        pagination_hint=pagination_hint,
        anti_bot_hint=anti_bot_hint,
        field_coverage_hint=field_coverage,
        score=score,
        keep=keep,
        notes=" | ".join(n for n in notes if n.strip()),
    )


def load_targets(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    return [line for line in lines if line and not line.startswith("#")]


def save_results(path: str, results: list[ValidationResult]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate target Shopify/WooCommerce stores.")
    parser.add_argument("--input", required=True, help="Path to input file with one URL per line")
    parser.add_argument("--output", required=True, help="CSV output path")
    parser.add_argument("--sleep", type=float, default=1.0, help="Delay between sites in seconds")
    args = parser.parse_args()

    targets = load_targets(args.input)
    if not targets:
        raise ValueError("No targets found in input file.")

    results: list[ValidationResult] = []

    for idx, url in enumerate(targets, start=1):
        print(f"[{idx}/{len(targets)}] Validating {url}")
        result = validate_site(url)
        results.append(result)
        print(
            f"  platform={result.detected_platform}, score={result.score:.1f}, keep={result.keep}"
        )
        time.sleep(args.sleep)

    save_results(args.output, results)
    print(f"\nSaved results to: {args.output}")


if __name__ == "__main__":
    main()
