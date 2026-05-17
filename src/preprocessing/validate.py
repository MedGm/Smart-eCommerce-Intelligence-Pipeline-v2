"""
Validation and logging of invalid/missing rows.
"""

import json

import pandas as pd

from src.config import get_logger, processed_dir

logger = get_logger(__name__)


def validate_required(df: pd.DataFrame, required: list[str] | None = None) -> pd.DataFrame:
    """Keep rows that have non-null values for required columns. Log drops."""
    required = required or ["source_platform", "shop_name", "product_id", "product_url", "title"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    for col in required:
        if out[col].dtype == object:
            out[col] = out[col].fillna("").astype(str).str.strip().replace("", pd.NA)

    before = len(df)
    out = out.dropna(subset=required)
    dropped = before - len(out)
    if dropped:
        logger.warning("Dropped %d rows with missing required fields.", dropped)
        log_invalid_rows(df, required)
    return out


def log_invalid_rows(df: pd.DataFrame, required: list[str]) -> None:
    """Write invalid rows to a log CSV for inspection."""
    invalid_check = df[required].copy()
    for col in required:
        if invalid_check[col].dtype == object:
            invalid_check[col] = (
                invalid_check[col].fillna("").astype(str).str.strip().replace("", pd.NA)
            )
    mask = invalid_check.isna().any(axis=1)
    invalid = df[mask]
    if invalid.empty:
        return
    missing_fields = (
        invalid_check[mask]
        .isna()
        .apply(
            lambda row: ",".join([field for field, is_missing in row.items() if is_missing]),
            axis=1,
        )
    )
    invalid = invalid.copy()
    invalid["missing_required_fields"] = missing_fields
    log_path = processed_dir() / "invalid_rows.csv"
    invalid.to_csv(log_path, index=False)
    logger.info("Wrote %d invalid rows to %s", len(invalid), log_path)


def build_dq_counters(df: pd.DataFrame) -> dict:
    """Compute standardized data-quality counters for monitoring."""
    counters: dict[str, int | float] = {
        "rows_total": int(len(df)),
    }
    if df.empty:
        counters.update(
            {
                "missing_price": 0,
                "missing_category": 0,
                "missing_rating": 0,
                "duplicates_core_key": 0,
            }
        )
        return counters

    if "price" in df.columns:
        counters["missing_price"] = int(df["price"].isna().sum())
    else:
        counters["missing_price"] = 0

    if "category" in df.columns:
        cat = df["category"].fillna("uncategorized").astype(str).str.strip().str.lower()
        counters["missing_category"] = int(cat.isin(["", "none", "nan", "uncategorized"]).sum())
    else:
        counters["missing_category"] = 0

    if "rating" in df.columns:
        counters["missing_rating"] = int(df["rating"].fillna(0).eq(0).sum())
    else:
        counters["missing_rating"] = 0

    core_cols = [c for c in ["source_platform", "shop_name", "product_id"] if c in df.columns]
    if len(core_cols) == 3:
        counters["duplicates_core_key"] = int(df.duplicated(subset=core_cols, keep=False).sum())
    else:
        counters["duplicates_core_key"] = 0

    return counters


def add_extraction_status_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Annotate key fields with extraction status labels for observability.

    Status vocabulary:
    - found: value present and parsed
    - not_present / missing: source likely did not expose value
    - extraction_failed: non-empty raw value was present but parsing failed
    """
    out = df.copy()

    title = out.get("title", pd.Series("", index=out.index)).fillna("").astype(str)
    description = out.get("description", pd.Series("", index=out.index)).fillna("").astype(str)
    text_blob = (title + " " + description).str.lower()
    product_url = (
        out.get("product_url", pd.Series("", index=out.index)).fillna("").astype(str).str.lower()
    )
    platform = (
        out.get("source_platform", pd.Series("", index=out.index))
        .fillna("")
        .astype(str)
        .str.lower()
    )

    # -------- Price status --------
    if "price" in out.columns:
        parse_failed = (
            out.get("price_parse_failed", pd.Series(False, index=out.index))
            .fillna(False)
            .astype(bool)
        )
        has_price = out["price"].notna()
        has_price_evidence = (
            platform.isin(["shopify", "woocommerce"])
            & product_url.str.contains(r"/products?/", regex=True)
        ) | text_blob.str.contains(r"\$|usd|eur|gbp|price|sale", regex=True)

        status = pd.Series("not_present", index=out.index)
        source = pd.Series("none", index=out.index)
        confidence = pd.Series("low", index=out.index)

        status.loc[has_price] = "found"
        source.loc[has_price] = "direct_numeric"
        confidence.loc[has_price] = "high"

        status.loc[parse_failed] = "extraction_failed"
        source.loc[parse_failed] = "parse_error"
        confidence.loc[parse_failed] = "low"

        miss_with_evidence = ~has_price & ~parse_failed & has_price_evidence
        status.loc[miss_with_evidence] = "extraction_failed"
        source.loc[miss_with_evidence] = "expected_but_missing"
        confidence.loc[miss_with_evidence] = "low"

        out["price_status"] = status
        out["price_source"] = source
        out["price_confidence"] = confidence

    # -------- Rating status --------
    if "rating" in out.columns:
        parse_failed = (
            out.get("rating_parse_failed", pd.Series(False, index=out.index))
            .fillna(False)
            .astype(bool)
        )
        has_rating = out["rating"].fillna(0).gt(0)
        review_count = (
            pd.to_numeric(out.get("review_count", pd.Series(0, index=out.index)), errors="coerce")
            .fillna(0)
            .astype(int)
        )
        has_rating_evidence = review_count.gt(0) | text_blob.str.contains(
            r"review|rated|stars?|out of 5|aggregate rating", regex=True
        )

        status = pd.Series("not_present", index=out.index)
        source = pd.Series("none", index=out.index)
        confidence = pd.Series("low", index=out.index)

        status.loc[has_rating] = "found"
        source.loc[has_rating] = "direct_numeric"
        confidence.loc[has_rating] = "high"

        status.loc[parse_failed] = "extraction_failed"
        source.loc[parse_failed] = "parse_error"
        confidence.loc[parse_failed] = "low"

        miss_with_evidence = ~has_rating & ~parse_failed & has_rating_evidence
        status.loc[miss_with_evidence] = "extraction_failed"
        source.loc[miss_with_evidence] = "expected_but_missing"
        confidence.loc[miss_with_evidence] = "low"

        out["rating_status"] = status
        out["rating_source"] = source
        out["rating_confidence"] = confidence

    # -------- Category status --------
    if "category" in out.columns:
        parse_failed = (
            out.get("category_parse_failed", pd.Series(False, index=out.index))
            .fillna(False)
            .astype(bool)
        )
        cat = out["category"].fillna("").astype(str).str.strip().str.lower()
        has_category = cat.ne("") & cat.ne("uncategorized")

        taxonomy_breadcrumb_present = (
            out.get("taxonomy_breadcrumb_present", pd.Series(False, index=out.index))
            .fillna(False)
            .astype(bool)
        )
        taxonomy_jsonld_category_present = (
            out.get("taxonomy_jsonld_category_present", pd.Series(False, index=out.index))
            .fillna(False)
            .astype(bool)
        )
        taxonomy_jsonld_breadcrumb_present = (
            out.get("taxonomy_jsonld_breadcrumb_present", pd.Series(False, index=out.index))
            .fillna(False)
            .astype(bool)
        )
        taxonomy_product_type_present = (
            out.get("taxonomy_product_type_present", pd.Series(False, index=out.index))
            .fillna(False)
            .astype(bool)
        )
        taxonomy_tags_present = (
            out.get("taxonomy_tags_present", pd.Series(False, index=out.index))
            .fillna(False)
            .astype(bool)
        )
        taxonomy_url_hint_present = (
            out.get("taxonomy_url_hint_present", pd.Series(False, index=out.index))
            .fillna(False)
            .astype(bool)
        ) | product_url.str.contains(r"/collections?/|/product-category/", regex=True)

        high_evidence = (
            taxonomy_breadcrumb_present
            | taxonomy_jsonld_category_present
            | taxonomy_jsonld_breadcrumb_present
        )
        medium_evidence = taxonomy_product_type_present
        low_evidence = taxonomy_tags_present | taxonomy_url_hint_present
        any_evidence = high_evidence | medium_evidence | low_evidence

        status = pd.Series("missing", index=out.index)
        source = pd.Series("none", index=out.index)
        confidence = pd.Series("low", index=out.index)

        status.loc[has_category] = "found"
        source.loc[has_category & high_evidence] = "direct_taxonomy"
        confidence.loc[has_category & high_evidence] = "high"
        source.loc[has_category & ~high_evidence & medium_evidence] = "typed_taxonomy"
        confidence.loc[has_category & ~high_evidence & medium_evidence] = "medium"
        source.loc[has_category & ~high_evidence & ~medium_evidence & low_evidence] = (
            "weak_taxonomy"
        )
        confidence.loc[has_category & ~high_evidence & ~medium_evidence & low_evidence] = "low"
        source.loc[has_category & ~any_evidence] = "direct_field_only"
        confidence.loc[has_category & ~any_evidence] = "medium"

        status.loc[parse_failed] = "extraction_failed"
        source.loc[parse_failed] = "parse_error"
        confidence.loc[parse_failed] = "low"

        inferred_mask = ~has_category & ~parse_failed & ~high_evidence & low_evidence
        status.loc[inferred_mask] = "inferred"
        source.loc[inferred_mask] = "weak_signal_inference"
        confidence.loc[inferred_mask] = "low"

        extraction_failed_mask = ~has_category & ~parse_failed & (high_evidence | medium_evidence)
        status.loc[extraction_failed_mask] = "extraction_failed"
        source.loc[extraction_failed_mask] = "taxonomy_evidence_parse_failed"
        confidence.loc[extraction_failed_mask] = "low"

        out["category_status"] = status
        out["category_source"] = source
        out["category_confidence"] = confidence

    return out


def add_record_dq_score(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-record DQ score (0-100) from completeness and extraction quality."""
    out = df.copy()

    required_cols = ["source_platform", "shop_name", "product_id", "product_url", "title"]
    required_present = pd.Series(1.0, index=out.index)
    for col in required_cols:
        if col not in out.columns:
            required_present *= 0.0
            continue
        if out[col].dtype == object:
            present = out[col].fillna("").astype(str).str.strip().ne("")
        else:
            present = out[col].notna()
        required_present *= present.astype(float)

    def _map_status(col: str, mapping: dict[str, float], default: float) -> pd.Series:
        if col not in out.columns:
            return pd.Series(default, index=out.index)
        return out[col].map(mapping).fillna(default)

    price_q = _map_status(
        "price_status",
        {"found": 1.0, "not_present": 0.4, "extraction_failed": 0.0},
        0.4,
    )
    rating_q = _map_status(
        "rating_status",
        {"found": 1.0, "not_present": 0.3, "extraction_failed": 0.0},
        0.3,
    )
    category_q = _map_status(
        "category_status",
        {
            "found": 1.0,
            "missing": 0.3,
            "not_present": 0.3,
            "inferred": 0.6,
            "extraction_failed": 0.0,
        },
        0.3,
    )

    title_q = pd.Series(0.0, index=out.index)
    if "title" in out.columns:
        title_q = out["title"].fillna("").astype(str).str.strip().str.len().ge(4).astype(float)

    description_q = pd.Series(0.0, index=out.index)
    if "description" in out.columns:
        description_q = (
            out["description"].fillna("").astype(str).str.strip().str.len().ge(20).astype(float)
        )

    score = (
        0.35 * required_present
        + 0.20 * price_q
        + 0.15 * category_q
        + 0.15 * rating_q
        + 0.10 * title_q
        + 0.05 * description_q
    )
    out["dq_score"] = (score * 100).round(2)
    return out


def write_dq_counters(counters: dict, output_path=None) -> None:
    """Persist data-quality counters in processed layer for downstream observability."""
    path = output_path or (processed_dir() / "dq_counters.json")
    path.write_text(json.dumps(counters, indent=2), encoding="utf-8")
    logger.info("Wrote DQ counters to %s", path)
