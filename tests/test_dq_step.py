import pandas as pd
import pytest
from pathlib import Path


def _write_valid_parquet(path: Path) -> None:
    df = pd.DataFrame([
        {
            "product_id": str(i),
            "title": f"Product {i}",
            "source_platform": "shopify" if i % 2 == 0 else "woocommerce",
            "dq_score": 0.8,
            "price": 29.99,
            "availability": "instock",
        }
        for i in range(20)
    ])
    df.to_parquet(path, index=False)


def _write_invalid_parquet(path: Path) -> None:
    """Missing required columns — product_id and title absent."""
    df = pd.DataFrame([{"price": 9.99}] * 5)
    df.to_parquet(path, index=False)


def _write_empty_parquet(path: Path) -> None:
    pd.DataFrame(
        columns=["product_id", "title", "source_platform", "dq_score"]
    ).to_parquet(path, index=False)


def test_validate_passes_on_valid_data(tmp_path):
    parquet = tmp_path / "cleaned_products.parquet"
    _write_valid_parquet(parquet)

    from src.pipeline.dq_step import validate_cleaned_products
    assert validate_cleaned_products(str(parquet)) is True


def test_validate_fails_on_missing_required_columns(tmp_path):
    parquet = tmp_path / "cleaned_products.parquet"
    _write_invalid_parquet(parquet)

    from src.pipeline.dq_step import validate_cleaned_products
    assert validate_cleaned_products(str(parquet)) is False


def test_validate_fails_on_empty_dataframe(tmp_path):
    parquet = tmp_path / "cleaned_products.parquet"
    _write_empty_parquet(parquet)

    from src.pipeline.dq_step import validate_cleaned_products
    assert validate_cleaned_products(str(parquet)) is False


def test_run_or_raise_raises_on_invalid_data(tmp_path, monkeypatch):
    parquet = tmp_path / "cleaned_products.parquet"
    _write_invalid_parquet(parquet)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    from src.pipeline.dq_step import run_or_raise
    with pytest.raises(RuntimeError, match="DQ validation failed"):
        run_or_raise()


def test_run_or_raise_noop_on_valid_data(tmp_path, monkeypatch):
    parquet = tmp_path / "processed" / "cleaned_products.parquet"
    parquet.parent.mkdir()
    _write_valid_parquet(parquet)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    from src.pipeline.dq_step import run_or_raise
    run_or_raise()  # must not raise
