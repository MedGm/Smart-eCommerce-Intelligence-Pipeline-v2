"""Tests for A2A scraping orchestration."""

from unittest.mock import patch

from src.scraping.agents import CoordinatorAgent, ScrapingTask, WorkerAgent


def test_coordinator_round_robin():
    coordinator = CoordinatorAgent(max_workers=2)

    shopify = [{"url": "s1", "name": "S1"}, {"url": "s2", "name": "S2"}]
    wc = [{"url": "w1", "name": "W1"}]

    plan = coordinator._round_robin_plan(
        [ScrapingTask("shopify", s) for s in shopify] + [ScrapingTask("woocommerce", s) for s in wc]
    )

    assert len(plan) == 2
    assert "worker_0" in plan
    assert "worker_1" in plan

    # Task 0 -> worker_0, Task 1 -> worker_1, Task 2 -> worker_0
    assert len(plan["worker_0"]) == 2
    assert len(plan["worker_1"]) == 1
    assert plan["worker_0"][0].store_info["url"] == "s1"
    assert plan["worker_1"][0].store_info["url"] == "s2"
    assert plan["worker_0"][1].store_info["url"] == "w1"


@patch("src.scraping.agents.WorkerAgent.execute_batch")
def test_coordinator_orchestrate(mock_execute):
    # Mock the worker to return dummy records
    mock_execute.return_value = []

    coordinator = CoordinatorAgent(max_workers=2)
    shopify = [{"url": "s1", "name": "S1"}]
    wc = [{"url": "w1", "name": "W1"}]

    # Block writing to disk during test
    with patch("src.scraping.agents.CoordinatorAgent._aggregate_results"):
        records = coordinator.orchestrate(shopify, wc)

    assert isinstance(records, list)
    assert mock_execute.call_count == 2  # 2 tasks assigned to workers


@patch("src.scraping.agents.ShopifyScraper.scrape")
def test_worker_agent_shopify(mock_scrape, tmp_path):
    mock_scrape.return_value = []

    # Override raw_dir temporarily
    with patch("src.scraping.agents.data_dir", return_value=tmp_path):
        agent = WorkerAgent(agent_id="test_worker")

        task = ScrapingTask(platform="shopify", store_info={"url": "test", "name": "TestStore"})

        records = agent.execute_batch([task])
        assert records == []
        assert mock_scrape.called
