"""Smoke test: pipeline stages can be invoked without crashing."""

import json

import pytest


@pytest.fixture
def data_dir(tmp_path):
    (tmp_path / "raw" / "shopify").mkdir(parents=True)
    (tmp_path / "raw" / "woocommerce").mkdir(parents=True)
    (tmp_path / "processed").mkdir(parents=True)
    (tmp_path / "analytics").mkdir(parents=True)
    return tmp_path


def test_preprocessing_run_empty_data(data_dir, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    from src.preprocessing.run import run

    df = run()
    assert (data_dir / "processed" / "cleaned_products.parquet").exists()
    assert (data_dir / "processed" / "dq_counters.json").exists()
    assert (data_dir / "processed" / "run_metadata.json").exists()
    assert (data_dir / "processed" / "field_failure_samples.json").exists()
    assert len(df) == 0


def test_preprocessing_writes_dq_counters(data_dir, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    raw_file = data_dir / "raw" / "shopify" / "sample.json"
    raw_file.write_text(
        json.dumps(
            [
                {
                    "source_platform": "shopify",
                    "shop_name": "Demo",
                    "product_id": "1",
                    "product_url": "https://example.com/p/1?utm_source=x",
                    "title": "Valid Item",
                    "price": "19.99",
                    "category": "Gift Card",
                },
                {
                    "source_platform": "shopify",
                    "shop_name": "Demo",
                    "product_id": "2",
                    "product_url": "",
                    "title": "Invalid Item",
                    "price": None,
                    "category": None,
                },
            ]
        ),
        encoding="utf-8",
    )

    from src.preprocessing.run import run

    df = run()
    counters_path = data_dir / "processed" / "dq_counters.json"
    metadata_path = data_dir / "processed" / "run_metadata.json"
    failure_samples_path = data_dir / "processed" / "field_failure_samples.json"

    assert counters_path.exists()
    assert metadata_path.exists()
    assert failure_samples_path.exists()
    counters = json.loads(counters_path.read_text(encoding="utf-8"))
    assert counters["rows_input"] == 2
    assert counters["rows_after_validate"] == 1
    assert counters["rows_dropped_required"] == 1
    assert "missing_price" in counters
    assert counters["schema_version"]
    assert counters["extraction_version"]
    assert len(df) == 1


def test_scoring_requires_features(data_dir, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    from src.scoring.topk import run

    run()
    # Should not raise; may write nothing if no features.parquet
