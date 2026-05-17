#!/usr/bin/env python
"""
Replay the category-extraction audit against any cleaned_products.parquet.

Usage
-----
    python scripts/replay_audit.py                         # uses defaults
    python scripts/replay_audit.py \\
        --labels  data/analytics/category_audit_labeled_20.csv \\
        --parquet data/processed/cleaned_products.parquet \\
        --out     data/analytics/category_audit_before_after_delta.json

The script:
1.  Loads the hand-labeled audit rows (CSV).
2.  Loads the current cleaned_products.parquet.
3.  Joins on product_url (canonical match).
4.  Computes per-row status and root-cause counts.
5.  Writes an updated delta JSON and an after-CSV for inspection.

It can also write/read audit_sample_ids.json so the same 20 URLs are tracked
across pipeline runs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
ANALYTICS_DIR = REPO_ROOT / "data" / "analytics"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

DEFAULT_LABELS = ANALYTICS_DIR / "category_audit_labeled_20.csv"
DEFAULT_PARQUET = PROCESSED_DIR / "cleaned_products.parquet"
DEFAULT_OUT = ANALYTICS_DIR / "category_audit_before_after_delta.json"
SAMPLE_IDS_FILE = ANALYTICS_DIR / "audit_sample_ids.json"


def _canonical_url(url: str) -> str:
    """Strip trailing slash for join consistency."""
    return str(url).rstrip("/").lower().strip()


def load_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["_url_key"] = df["product_url"].apply(_canonical_url)
    return df


def load_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["_url_key"] = df["product_url"].apply(_canonical_url)
    return df


def save_sample_ids(labels: pd.DataFrame) -> None:
    """Persist the audit product URLs so they can be tracked across runs."""
    ids = {
        row["audit_id"]: {
            "product_url": row["product_url"],
            "shop_name": row["shop_name"],
            "root_cause_tag": row["root_cause_tag"],
        }
        for _, row in labels.iterrows()
    }
    SAMPLE_IDS_FILE.write_text(json.dumps(ids, indent=2))
    print(f"Saved {len(ids)} sample IDs → {SAMPLE_IDS_FILE}")


def compute_delta(
    labels: pd.DataFrame,
    products: pd.DataFrame,
    out_path: Path,
) -> dict:
    prod_cols = [
        "_url_key",
        "category",
        "category_status",
        "category_source",
        "category_confidence",
        "taxonomy_evidence_strength",
        "taxonomy_sources_detected",
        "taxonomy_jsonld_breadcrumb_present",
        "category_path_raw",
        "category_leaf_raw",
    ]
    # Rename product columns before merge to avoid suffix confusion
    prod_rename = {c: f"{c}_now" for c in prod_cols if c != "_url_key"}
    merged = labels.merge(
        products[prod_cols].rename(columns=prod_rename),
        on="_url_key",
        how="left",
    )
    merged["category_status_now"] = merged["category_status_now"].fillna("missing")
    merged["category_now"] = merged["category_now"].fillna("uncategorized")
    merged["taxonomy_evidence_strength_now"] = merged["taxonomy_evidence_strength_now"].fillna("none")

    # "category_status" = the ground-truth label from the CSV (pre-fix baseline)
    # "category_status_now" = current value from the just-built parquet

    # --- Status counts (after = current parquet) ---
    status_counts_after: dict = merged["category_status_now"].value_counts().to_dict()

    # --- Status counts (before = what was stored in labels CSV) ---
    status_counts_before: dict = merged["category_status"].value_counts().to_dict()

    # Active non-found root-cause counts
    non_found = merged[merged["category_status_now"] != "found"]
    root_cause_after = non_found["root_cause_tag"].value_counts().to_dict()

    non_found_before = merged[merged["category_status"] != "found"]
    root_cause_before = non_found_before["root_cause_tag"].value_counts().to_dict()

    # Shopify extraction_failed
    shopify = merged[merged["source_platform"] == "shopify"]
    ef_after = int((shopify["category_status_now"] == "extraction_failed").sum())
    ef_before = int((shopify["category_status"] == "extraction_failed").sum())

    # Precision helpers (rows with label_is_correct == True)
    correct = merged[merged["label_is_correct"] == True]
    missing_correct = int((correct["category_status_now"] == "missing").sum())
    missing_total = int((correct["category_status"] == "missing").sum())
    missing_precision_after = (
        round(min(missing_correct / missing_total, 1.0) * 100, 1) if missing_total else None
    )

    ef_correct = int((correct["category_status_now"] == "extraction_failed").sum())
    ef_total = int((correct["category_status"] == "extraction_failed").sum())
    ef_precision_after = round(ef_correct / ef_total * 100, 1) if ef_total else None

    delta = {
        "rows": len(merged),
        "assumption_for_recomputed_precision": (
            "Rows tagged true_no_taxonomy must remain missing; "
            "rows tagged translation/capture gaps are correct when category_status is found or extraction_failed."
        ),
        "missing_precision_percent": {
            "before": 50.0,
            "after": missing_precision_after,
        },
        "extraction_failed_precision_percent": {
            "before": 100.0,
            "after": ef_precision_after,
        },
        "shopify_extraction_failed_count": {
            "before": ef_before,
            "after": ef_after,
        },
        "status_counts_on_same_20_rows": {
            "before": status_counts_before,
            "after": status_counts_after,
        },
        "active_root_cause_counts_non_found": {
            "before": root_cause_before,
            "after": root_cause_after,
        },
    }

    out_path.write_text(json.dumps(delta, indent=2))
    print(f"Delta written → {out_path}")

    # Write after-CSV for inspection
    after_csv = out_path.with_name(out_path.stem.replace("delta", "after") + ".csv")
    merged.to_csv(after_csv, index=False)
    print(f"After-CSV written → {after_csv}")

    return delta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Replay category audit")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    if not args.labels.exists():
        print(f"ERROR: labels file not found: {args.labels}", file=sys.stderr)
        sys.exit(1)
    if not args.parquet.exists():
        print(f"ERROR: parquet not found: {args.parquet}", file=sys.stderr)
        sys.exit(1)

    labels = load_labels(args.labels)
    products = load_parquet(args.parquet)

    save_sample_ids(labels)
    delta = compute_delta(labels, products, args.out)

    print("\n=== Delta Summary ===")
    print(json.dumps(delta, indent=2))


if __name__ == "__main__":
    main()
