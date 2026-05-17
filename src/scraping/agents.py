from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from src.config import data_dir, get_logger
from src.scraping.base import ProductRecord
from src.scraping.shopify import ShopifyScraper
from src.scraping.woocommerce import WooCommerceScraper

logger = get_logger(__name__)


@dataclass
class ScrapingTask:
    platform: str
    store_info: dict


class WorkerAgent:
    """
    A2A Worker Agent.
    Receives a list of ScrapingTasks from the Coordinator, instantiates the
    correct scraper (Shopify/WooCommerce), and executes the extraction.
    """

    def __init__(self, agent_id: str, run_id: str | None = None):
        self.agent_id = agent_id
        self.run_id = run_id
        self.raw_dir = data_dir() / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def execute_batch(self, tasks: list[ScrapingTask]) -> list[ProductRecord]:
        logger.info(f"WorkerAgent [{self.agent_id}]: Starting batch of {len(tasks)} tasks.")
        results = []

        for task in tasks:
            store = task.store_info
            safe_name = store["name"].lower().replace(" ", "_").replace("'", "")
            logger.info(f"WorkerAgent [{self.agent_id}]: Scraping {safe_name} ({task.platform})")

            if task.platform == "shopify":
                shopify_dir = self.raw_dir / "shopify"
                shopify_dir.mkdir(exist_ok=True)
                scraper = ShopifyScraper(
                    output_dir=shopify_dir,
                    store_url=store["url"],
                    shop_name=store["name"],
                    geography=store.get("geography"),
                    collections=store.get("collections", ["all"]),
                    max_collection_pages=store.get("max_collection_pages", 20),
                    run_id=self.run_id,
                )
                records = scraper.scrape()
                if records:
                    scraper.save(records, f"{safe_name}.json")
                    results.extend(records)

            elif task.platform == "woocommerce":
                wc_dir = self.raw_dir / "woocommerce"
                wc_dir.mkdir(exist_ok=True)
                scraper = WooCommerceScraper(
                    output_dir=wc_dir,
                    site_url=store["url"],
                    shop_name=store["name"],
                    geography=store.get("geography"),
                    run_id=self.run_id,
                )
                records = scraper.scrape()
                if records:
                    scraper.save(records, f"{safe_name}.json")
                    results.extend(records)

        logger.info(
            f"WorkerAgent [{self.agent_id}]: Finished batch. Extracted {len(results)} products."
        )
        return results


class CoordinatorAgent:
    """
    A2A Coordinator Agent.
    Distributes stores across WorkerAgents using round-robin, then spawns them
    concurrently to execute the plan.
    """

    def __init__(self, max_workers: int = 3, run_id: str | None = None):
        self.max_workers = max_workers
        self.run_id = run_id

    def plan_distribution(
        self, shopify_stores: list[dict], wc_stores: list[dict]
    ) -> dict[str, list[ScrapingTask]]:
        """Distribute stores across workers using round-robin."""
        all_tasks = [ScrapingTask("shopify", s) for s in shopify_stores] + [
            ScrapingTask("woocommerce", s) for s in wc_stores
        ]
        return self._round_robin_plan(all_tasks)

    def _round_robin_plan(self, tasks: list[ScrapingTask]) -> dict[str, list[ScrapingTask]]:
        plan = {f"worker_{i}": [] for i in range(self.max_workers)}
        for i, task in enumerate(tasks):
            worker_id = f"worker_{i % self.max_workers}"
            plan[worker_id].append(task)
        logger.info(
            f"CoordinatorAgent: Generated round-robin plan across {self.max_workers} workers."
        )
        return plan

    def orchestrate(self, shopify_stores: list[dict], wc_stores: list[dict]) -> list[ProductRecord]:
        logger.info("CoordinatorAgent: Starting orchestration.")

        # 1. Planning phase
        distribution_plan = self.plan_distribution(shopify_stores, wc_stores)

        all_records = []

        # 2. Execution phase (Concurrent Workers)
        logger.info(
            f"CoordinatorAgent: Dispatching tasks to {len(distribution_plan)} WorkerAgents concurrently."
        )
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_worker = {}
            for worker_id, tasks in distribution_plan.items():
                if not tasks:
                    continue
                agent = WorkerAgent(agent_id=worker_id, run_id=self.run_id)
                future = executor.submit(agent.execute_batch, tasks)
                future_to_worker[future] = worker_id

            for future in as_completed(future_to_worker):
                worker_id = future_to_worker[future]
                try:
                    records = future.result()
                    all_records.extend(records)
                except Exception as exc:
                    logger.error(
                        f"CoordinatorAgent: WorkerAgent [{worker_id}] generated an exception: {exc}"
                    )

        # 3. Aggregation phase
        self._aggregate_results(all_records)
        logger.info(
            f"CoordinatorAgent: Orchestration complete. Total products aggregated: {len(all_records)}"
        )
        return all_records

    def _aggregate_results(self, records: list[ProductRecord]):
        """Save aggregated platform-level JSON files."""
        raw_dir = data_dir() / "raw"

        # Shopify
        shopify_records = [r.to_dict() for r in records if r.source_platform == "shopify"]
        if shopify_records:
            if self.run_id:
                shopify_agg_dir = raw_dir / "shopify" / self.run_id
                shopify_agg_dir.mkdir(parents=True, exist_ok=True)
                agg_path = shopify_agg_dir / "products.json"
            else:
                agg_path = raw_dir / "shopify" / "products.json"
            with open(agg_path, "w", encoding="utf-8") as f:
                json.dump(shopify_records, f, indent=2, ensure_ascii=False)

        # WooCommerce
        wc_records = [r.to_dict() for r in records if r.source_platform == "woocommerce"]
        if wc_records:
            if self.run_id:
                wc_agg_dir = raw_dir / "woocommerce" / self.run_id
                wc_agg_dir.mkdir(parents=True, exist_ok=True)
                agg_path = wc_agg_dir / "products.json"
            else:
                agg_path = raw_dir / "woocommerce" / "products.json"
            with open(agg_path, "w", encoding="utf-8") as f:
                json.dump(wc_records, f, indent=2, ensure_ascii=False)
