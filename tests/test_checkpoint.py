import json
from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock


def test_completed_store_is_skipped(tmp_path):
    """WorkerAgent skips a store already in the checkpoint file."""
    from src.scraping.agents import WorkerAgent, ScrapingTask

    run_id = "20260517T000000Z"
    checkpoint_path = tmp_path / "raw" / run_id / "checkpoint.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(json.dumps({"completed": ["ruggable"]}))

    with patch("src.scraping.agents.data_dir", return_value=tmp_path):
        worker = WorkerAgent(agent_id="worker_0", run_id=run_id)
        task = ScrapingTask("shopify", {"url": "https://ruggable.com", "name": "ruggable"})

        scraper_calls = []
        with patch("src.scraping.agents.ShopifyScraper") as MockScraper:
            MockScraper.return_value.scrape.side_effect = lambda: scraper_calls.append(1) or []
            worker.execute_batch([task])

    assert len(scraper_calls) == 0, "Should have skipped ruggable — already in checkpoint"


def test_completed_store_written_to_checkpoint(tmp_path):
    """After scraping, store name is written to checkpoint."""
    from src.scraping.agents import WorkerAgent, ScrapingTask

    run_id = "20260517T000000Z"
    with patch("src.scraping.agents.data_dir", return_value=tmp_path):
        worker = WorkerAgent(agent_id="worker_0", run_id=run_id)
        task = ScrapingTask("shopify", {"url": "https://ruggable.com", "name": "TestStore"})

        mock_record = MagicMock()
        mock_record.source_platform = "shopify"
        mock_record.to_dict.return_value = {}

        with patch("src.scraping.agents.ShopifyScraper") as MockScraper:
            MockScraper.return_value.scrape.return_value = [mock_record]
            MockScraper.return_value.save.return_value = tmp_path / "test.json"
            worker.execute_batch([task])

    checkpoint_path = tmp_path / "raw" / run_id / "checkpoint.json"
    assert checkpoint_path.exists(), "Checkpoint file not created"
    data = json.loads(checkpoint_path.read_text())
    assert "TestStore" in data["completed"]


def test_no_run_id_no_checkpoint(tmp_path):
    """Without run_id, no checkpoint is read or written — backward compat."""
    from src.scraping.agents import WorkerAgent, ScrapingTask

    with patch("src.scraping.agents.data_dir", return_value=tmp_path):
        worker = WorkerAgent(agent_id="worker_0", run_id=None)
        task = ScrapingTask("shopify", {"url": "https://ruggable.com", "name": "TestStore"})

        with patch("src.scraping.agents.ShopifyScraper") as MockScraper:
            MockScraper.return_value.scrape.return_value = []
            worker.execute_batch([task])

    # No checkpoint directory should be created
    checkpoint_path = tmp_path / "raw" / "checkpoint.json"
    assert not checkpoint_path.exists()
