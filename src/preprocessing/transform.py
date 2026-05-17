"""
Transformations for feature prep: category harmonization, fill missing, etc.
"""

import pandas as pd


def harmonize_categories(df: pd.DataFrame, category_col: str = "category") -> pd.DataFrame:
    """Normalize category names: lowercase, strip, consolidate similar names."""
    if category_col not in df.columns:
        return df
    out = df.copy()
    cat = out[category_col].fillna("").astype(str).str.strip().str.lower()
    cat = cat.replace({"": pd.NA, "none": pd.NA, "nan": pd.NA, "unknown": pd.NA})
    out[category_col] = cat
    return out


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Sensible defaults for missing values (avoid breaking downstream)."""
    out = df.copy()
    if "category" in out.columns:
        out["category"] = out["category"].fillna("uncategorized")
    if "brand" in out.columns:
        out["brand"] = out["brand"].fillna("unknown")
    if "availability" in out.columns:
        out["availability"] = out["availability"].fillna("unknown")
    if "geography" in out.columns:
        out["geography"] = out["geography"].fillna("unknown")
    return out
