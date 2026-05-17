def test_plan_distribution_uses_round_robin():
    from src.scraping.agents import CoordinatorAgent

    coordinator = CoordinatorAgent(max_workers=2)
    shopify = [{"store_url": "https://a.com", "shop_name": "A"}]
    wc = [{"site_url": "https://b.com", "shop_name": "B"}]
    plan = coordinator.plan_distribution(shopify, wc)
    total_tasks = sum(len(v) for v in plan.values())
    assert total_tasks == 2
    assert set(plan.keys()) == {"worker_0", "worker_1"}


def test_no_llm_import_in_plan_distribution():
    """Confirm _init_llm is removed and plan_distribution doesn't attempt LLM."""
    from src.scraping.agents import CoordinatorAgent

    # _init_llm should not exist
    assert not hasattr(CoordinatorAgent, "_init_llm"), "_init_llm should be removed (dead code)"
