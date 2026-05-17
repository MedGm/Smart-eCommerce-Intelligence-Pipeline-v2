"""
Run full preprocessing: load raw JSON from data/raw, clean, validate, transform, write processed.
Reproducible: same raw input -> same processed output.
"""

import json
from datetime import datetime, timezone

import pandas as pd

from src.config import data_dir, get_logger, processed_dir
from src.preprocessing.clean import clean
from src.preprocessing.transform import fill_missing, harmonize_categories
from src.preprocessing.validate import (
    add_extraction_status_columns,
    add_record_dq_score,
    build_dq_counters,
    validate_required,
    write_dq_counters,
)

logger = get_logger(__name__)

PREPROCESS_SCHEMA_VERSION = "2.2.0"
EXTRACTION_VERSION = "fallback-v3-taxonomy-evidence"


def load_raw(root=None) -> pd.DataFrame:
    """Load all raw product JSONs from raw/shopify and raw/woocommerce."""
    root = root or data_dir()
    raw = root / "raw"
    rows = []
    for platform in ("shopify", "woocommerce"):
        platform_dir = raw / platform
        if not platform_dir.exists():
            continue
        for path in platform_dir.glob("**/*.json"):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    rows.extend(data)
                else:
                    rows.append(data)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def run():
    root = data_dir()
    p_dir = processed_dir()
    p_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw(root)
    run_ts = datetime.now(timezone.utc).isoformat()
    if df.empty:
        logger.warning("No raw data found. Run scrapers first. Writing empty cleaned output.")
        df = pd.DataFrame(
            columns=[
                "source_platform",
                "shop_name",
                "product_id",
                "product_url",
                "title",
                "description",
                "category",
                "brand",
                "price",
                "old_price",
                "availability",
                "rating",
                "review_count",
                "geography",
                "scraped_at",
                "price_status",
                "rating_status",
                "category_status",
                "dq_score",
            ]
        )
        dq_counters = build_dq_counters(df)
        dq_counters.update(
            {
                "rows_input": 0,
                "rows_after_clean": 0,
                "rows_after_validate": 0,
                "rows_dropped_required": 0,
                "rows_dropped_dedup": 0,
                "retention_after_validate": 0.0,
            }
        )
    else:
        rows_input = len(df)
        rows_before_clean = len(df)
        df = clean(df)
        rows_after_clean = len(df)
        rows_dropped_dedup = max(0, rows_before_clean - rows_after_clean)

        dq_counters = build_dq_counters(df)

        rows_before_validate = len(df)
        df = validate_required(df)
        rows_after_validate = len(df)

        dq_counters.update(
            {
                "rows_input": int(rows_input),
                "rows_after_clean": int(rows_after_clean),
                "rows_after_validate": int(rows_after_validate),
                "rows_dropped_required": int(max(0, rows_before_validate - rows_after_validate)),
                "rows_dropped_dedup": int(rows_dropped_dedup),
                "retention_after_validate": round(float(rows_after_validate / rows_input), 6)
                if rows_input
                else 0.0,
            }
        )

        df = add_extraction_status_columns(df)
        df = harmonize_categories(df)
        df = fill_missing(df)
        df = add_record_dq_score(df)

        status_cols = ["price_status", "rating_status", "category_status"]
        for status_col in [c for c in status_cols if c in df.columns]:
            prefix = status_col.replace("_status", "")
            counts = df[status_col].value_counts(dropna=False)
            for status_value, count in counts.items():
                dq_counters[f"{prefix}_{str(status_value)}"] = int(count)

            if "source_platform" in df.columns:
                grouped = (
                    df.groupby("source_platform", dropna=False)[status_col]
                    .value_counts(dropna=False)
                    .reset_index(name="count")
                )
                for _, row in grouped.iterrows():
                    platform = str(row["source_platform"]).lower().replace(" ", "_")
                    status_value = str(row[status_col]).lower().replace(" ", "_")
                    dq_counters[f"{prefix}_{platform}_{status_value}"] = int(row["count"])

        evidence_bool_cols = [
            "taxonomy_breadcrumb_present",
            "taxonomy_jsonld_category_present",
            "taxonomy_jsonld_breadcrumb_present",
            "taxonomy_product_type_present",
            "taxonomy_tags_present",
            "taxonomy_url_hint_present",
        ]
        for col in [c for c in evidence_bool_cols if c in df.columns]:
            as_bool = df[col].fillna(False).astype(bool)
            dq_counters[f"category_evidence_{col}_true"] = int(as_bool.sum())
            dq_counters[f"category_evidence_{col}_false"] = int((~as_bool).sum())
            if "source_platform" in df.columns:
                grouped = (
                    pd.DataFrame(
                        {
                            "source_platform": df["source_platform"].fillna("unknown").astype(str),
                            "flag": as_bool,
                        }
                    )
                    .groupby("source_platform", dropna=False)["flag"]
                    .sum()
                )
                for platform, count in grouped.items():
                    platform_key = str(platform).lower().replace(" ", "_")
                    dq_counters[f"category_evidence_{platform_key}_{col}_true"] = int(count)

        if "taxonomy_evidence_strength" in df.columns:
            strength_counts = (
                df["taxonomy_evidence_strength"]
                .fillna("none")
                .astype(str)
                .str.lower()
                .value_counts()
            )
            for strength, count in strength_counts.items():
                dq_counters[f"category_evidence_strength_{strength}"] = int(count)

            if "source_platform" in df.columns:
                grouped_strength = (
                    df.assign(
                        taxonomy_evidence_strength=df["taxonomy_evidence_strength"]
                        .fillna("none")
                        .astype(str)
                        .str.lower()
                    )
                    .groupby("source_platform", dropna=False)["taxonomy_evidence_strength"]
                    .value_counts(dropna=False)
                    .reset_index(name="count")
                )
                for _, row in grouped_strength.iterrows():
                    platform = str(row["source_platform"]).lower().replace(" ", "_")
                    strength = str(row["taxonomy_evidence_strength"]).lower().replace(" ", "_")
                    dq_counters[f"category_evidence_strength_{platform}_{strength}"] = int(
                        row["count"]
                    )

        if "dq_score" in df.columns and not df.empty:
            dq_counters["dq_score_mean"] = round(float(df["dq_score"].mean()), 4)
            dq_counters["dq_score_p25"] = round(float(df["dq_score"].quantile(0.25)), 4)
            dq_counters["dq_score_p50"] = round(float(df["dq_score"].quantile(0.50)), 4)
            dq_counters["dq_score_p75"] = round(float(df["dq_score"].quantile(0.75)), 4)

        sample_cols = [
            c
            for c in [
                "source_platform",
                "shop_name",
                "product_id",
                "product_url",
                "price_status",
                "price_source",
                "category_status",
                "category_source",
                "taxonomy_breadcrumb_present",
                "taxonomy_breadcrumb_count",
                "taxonomy_jsonld_category_present",
                "taxonomy_jsonld_breadcrumb_present",
                "taxonomy_product_type_present",
                "taxonomy_tags_present",
                "taxonomy_url_hint_present",
                "taxonomy_sources_detected",
                "taxonomy_evidence_strength",
                "category_path_raw",
                "category_leaf_raw",
                "rating_status",
                "rating_source",
                "dq_score",
            ]
            if c in df.columns
        ]
        rating_failed = (
            df["rating_status"].eq("extraction_failed")
            if "rating_status" in df.columns
            else pd.Series(False, index=df.index)
        )
        category_failed = (
            df["category_status"].eq("extraction_failed")
            if "category_status" in df.columns
            else pd.Series(False, index=df.index)
        )
        price_failed = (
            df["price_status"].eq("extraction_failed")
            if "price_status" in df.columns
            else pd.Series(False, index=df.index)
        )
        samples = {
            "rating_extraction_failed": df[rating_failed][sample_cols]
            .head(12)
            .to_dict(orient="records"),
            "category_extraction_failed": df[category_failed][sample_cols]
            .head(12)
            .to_dict(orient="records"),
            "price_extraction_failed": df[price_failed][sample_cols]
            .head(12)
            .to_dict(orient="records"),
            "lowest_dq_score": df.sort_values("dq_score", ascending=True)[sample_cols]
            .head(12)
            .to_dict(orient="records")
            if "dq_score" in df.columns
            else [],
        }

    dq_counters["schema_version"] = PREPROCESS_SCHEMA_VERSION
    dq_counters["extraction_version"] = EXTRACTION_VERSION
    dq_counters["run_ts_utc"] = run_ts

    run_metadata = {
        "schema_version": PREPROCESS_SCHEMA_VERSION,
        "extraction_version": EXTRACTION_VERSION,
        "run_ts_utc": run_ts,
        "rows_output": int(len(df)),
    }

    out_path = p_dir / "cleaned_products.parquet"
    df.to_parquet(out_path, index=False)
    write_dq_counters(dq_counters, p_dir / "dq_counters.json")
    (p_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    (p_dir / "field_failure_samples.json").write_text(
        json.dumps(samples if not df.empty else {}, indent=2), encoding="utf-8"
    )
    logger.info("Preprocessing done: %d rows -> %s", len(df), out_path)
    return df


if __name__ == "__main__":
    run()
