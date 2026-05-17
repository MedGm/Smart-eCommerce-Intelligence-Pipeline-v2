#!/usr/bin/env python3
"""
Shopify Storefront API example (GraphQL, token-based).

Demonstrates the Storefront API approach described in the dossier.
Requires SHOPIFY_STORE and SHOPIFY_STOREFRONT_TOKEN in .env.

Usage:
    python scripts/shopify_storefront_example.py
"""

import os

import requests
from dotenv import load_dotenv

load_dotenv()

STORE = os.environ.get("SHOPIFY_STORE", "").rstrip("/")
TOKEN = os.environ.get("SHOPIFY_STOREFRONT_TOKEN", "")

QUERY = """{
  products(first: 5) {
    edges {
      node {
        id
        title
        descriptionHtml
        priceRange {
          minVariantPrice { amount currencyCode }
          maxVariantPrice { amount currencyCode }
        }
        availableForSale
        totalInventory
      }
    }
  }
}"""


def main():
    if not STORE or not TOKEN:
        print(
            "Shopify Storefront API example:\n"
            "  Set SHOPIFY_STORE and SHOPIFY_STOREFRONT_TOKEN in .env to use.\n"
            "  Main pipeline uses Playwright storefront scraping instead."
        )
        return

    url = f"{STORE}/api/2024-10/graphql.json"
    headers = {
        "Content-Type": "application/json",
        "X-Shopify-Storefront-Access-Token": TOKEN,
    }
    resp = requests.post(url, json={"query": QUERY}, headers=headers, timeout=15)
    if resp.status_code != 200:
        print(f"Error: status {resp.status_code}\n{resp.text[:300]}")
        return

    data = resp.json()
    products = data.get("data", {}).get("products", {}).get("edges", [])
    for edge in products:
        node = edge["node"]
        price = node.get("priceRange", {}).get("minVariantPrice", {})
        print(
            f"  {node['title']}  |  "
            f"{price.get('amount', '?')} {price.get('currencyCode', '')}  |  "
            f"available={node.get('availableForSale')}"
        )
    print(f"\nFetched {len(products)} products via Storefront API.")


if __name__ == "__main__":
    main()
