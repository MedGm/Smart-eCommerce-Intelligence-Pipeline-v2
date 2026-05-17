"""
Orchestrator: run all Shopify and WooCommerce scrapers using the A2A CoordinatorAgent.
"""

from __future__ import annotations

import sys

from src.config import get_logger
from src.scraping.agents import CoordinatorAgent
from src.scraping.stores import SHOPIFY_STORES, WOOCOMMERCE_STORES

logger = get_logger(__name__)


def run():
    logger.info("Initializing A2A Scraping Coordinator.")

    # 3 workers by default gives good parallelization for our 8 targets
    coordinator = CoordinatorAgent(max_workers=3)

    logger.info(
        f"Targeting {len(SHOPIFY_STORES)} Shopify stores and {len(WOOCOMMERCE_STORES)} WooCommerce stores."
    )

    # The CoordinatorAgent handles parallelization, LLM planning, and aggregation
    records = coordinator.orchestrate(SHOPIFY_STORES, WOOCOMMERCE_STORES)

    logger.info("A2A Orchestration completed successfully.")
    return records


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
