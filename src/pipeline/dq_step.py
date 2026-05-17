"""
Great Expectations DQ gate for cleaned_products.parquet.
Validates 8 expectations; raises RuntimeError on failure (KFP hard-stop).
"""
from __future__ import annotations

from pathlib import Path

from src.config import get_logger, processed_dir

logger = get_logger(__name__)


def validate_cleaned_products(parquet_path: str | None = None) -> bool:
    """Run GE expectations. Returns True if all pass, False otherwise."""
    import great_expectations as gx
    import pandas as pd

    path = Path(parquet_path) if parquet_path else processed_dir() / "cleaned_products.parquet"
    if not path.exists():
        logger.error("Parquet not found: %s", path)
        return False

    df = pd.read_parquet(path)

    context = gx.get_context(mode="ephemeral")
    source = context.data_sources.add_pandas("source")
    asset = source.add_dataframe_asset("products")
    batch_def = asset.add_batch_definition_whole_dataframe("batch")

    suite = context.suites.add(gx.ExpectationSuite(name="products_suite"))
    # Row count > 0 (use between with min=1, max unbounded)
    suite.add_expectation(
        gx.expectations.ExpectTableRowCountToBeBetween(min_value=1)
    )
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="product_id"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="title"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="source_platform"))
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="product_id", mostly=0.99)
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="title", mostly=0.99)
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="dq_score", min_value=0.0, max_value=1.0, mostly=0.95
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="source_platform", value_set=["shopify", "woocommerce"]
        )
    )

    validation_def = context.validation_definitions.add(
        gx.ValidationDefinition(
            name="validate_cleaned_products",
            data=batch_def,
            suite=suite,
        )
    )
    result = validation_def.run(batch_parameters={"dataframe": df})

    if not result.success:
        failed = [r for r in result.results if not r.success]
        logger.error(
            "DQ validation failed — %d/%d expectations failed:",
            len(failed),
            len(result.results),
        )
        for r in failed:
            logger.error("  FAIL: %s", r.expectation_config.type)
    else:
        logger.info("DQ validation passed (%d expectations)", len(result.results))

    return bool(result.success)


def run_or_raise(parquet_path: str | None = None) -> None:
    """Raises RuntimeError on DQ failure. Entry point for KFP step."""
    if not validate_cleaned_products(parquet_path):
        raise RuntimeError(
            "DQ validation failed — pipeline stopped. "
            "Check logs for failed expectations."
        )


if __name__ == "__main__":
    run_or_raise()
