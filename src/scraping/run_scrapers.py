"""
Orchestrator: run all Shopify and WooCommerce scrapers using the A2A CoordinatorAgent.
"""

from __future__ import annotations

import sys

from src.config import get_logger
from src.scraping.agents import CoordinatorAgent
from src.scraping.stores import load_stores

logger = get_logger(__name__)


def run():
    logger.info("Initializing A2A Scraping Coordinator.")
    shopify_stores, woocommerce_stores = load_stores()
    coordinator = CoordinatorAgent(max_workers=3)
    logger.info(
        f"Targeting {len(shopify_stores)} Shopify stores and {len(woocommerce_stores)} WooCommerce stores."
    )
    records = coordinator.orchestrate(shopify_stores, woocommerce_stores)
    logger.info("A2A Orchestration completed successfully.")
    return records


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
