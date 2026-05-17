#!/usr/bin/env python3
"""
WooCommerce REST API v3 example (authenticated).

Demonstrates the WooCommerce REST API v3 approach described in the dossier.
Requires WOOCOMMERCE_URL, WOOCOMMERCE_KEY, WOOCOMMERCE_SECRET in .env.
The main pipeline uses the public Store API (/wp-json/wc/store/v1/) instead.

Usage:
    python scripts/woocommerce_rest_example.py
"""

import os

import requests
from dotenv import load_dotenv

load_dotenv()

SITE_URL = os.environ.get("WOOCOMMERCE_URL", "").rstrip("/")
KEY = os.environ.get("WOOCOMMERCE_KEY", "")
SECRET = os.environ.get("WOOCOMMERCE_SECRET", "")


def main():
    if not SITE_URL or not KEY or not SECRET:
        print(
            "WooCommerce REST API v3 example:\n"
            "  Set WOOCOMMERCE_URL, WOOCOMMERCE_KEY, WOOCOMMERCE_SECRET in .env.\n"
            "  Main pipeline uses the public Store API (/wp-json/wc/store/v1/)."
        )
        return

    url = f"{SITE_URL}/wp-json/wc/v3/products"
    params = {"per_page": 5}
    resp = requests.get(url, auth=(KEY, SECRET), params=params, timeout=15)
    if resp.status_code != 200:
        print(f"Error: status {resp.status_code}\n{resp.text[:300]}")
        return

    products = resp.json()
    for p in products:
        print(
            f"  [{p.get('id')}] {p.get('name')}  |  "
            f"price={p.get('price')}  |  "
            f"status={p.get('status')}  |  "
            f"stock={p.get('stock_status')}"
        )
    print(f"\nFetched {len(products)} products via WooCommerce REST API v3.")


if __name__ == "__main__":
    main()
