"""
Deterministic cleaning: dedupe, standardize prices, normalize text,
harmonize categories, handle missing values, numeric ratings/reviews,
strip residual HTML from descriptions.
"""

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd
from bs4 import BeautifulSoup

_NAVIGATION_LABELS = {
    "",
    "none",
    "nan",
    "unknown",
    "shop all",
    "new arrivals",
    "sale",
    "home",
    "shop",
    "all products",
    "products",
    "catalog",
}


def _normalize_category_token(value: str | None) -> str | None:
    if value is None:
        return None
    token = str(value).strip().lower()
    if not token:
        return None
    token = token.replace("&", " and ").replace("'", " ")
    token = pd.Series([token]).str.replace(r"[^a-z0-9\s]", " ", regex=True).iloc[0]
    token = pd.Series([token]).str.replace(r"\s+", " ", regex=True).str.strip().iloc[0]
    if not token:
        return None
    token = pd.Series([token]).str.replace(r"^shop all\s+", "", regex=True).str.strip().iloc[0]
    return token if token else None


def _looks_like_product_title(category: str | None, title: str | None) -> bool:
    if not category or not title:
        return False
    cat_words = set(category.split())
    title_words = set((_normalize_category_token(title) or "").split())
    if len(cat_words) < 3 or not title_words:
        return False
    overlap = len(cat_words & title_words) / max(len(cat_words), 1)
    return overlap >= 0.8


def _path_category_candidate(path_raw: str | None, title: str | None) -> str | None:
    if not isinstance(path_raw, str) or not path_raw.strip():
        return None
    title_norm = _normalize_category_token(title)
    path_tokens = [_normalize_category_token(tok) for tok in path_raw.split(">")]
    path_tokens = [tok for tok in path_tokens if tok and tok not in _NAVIGATION_LABELS]
    if not path_tokens:
        return None

    # Prefer the deepest meaningful taxonomy token that is not product-title-like.
    for token in reversed(path_tokens):
        if title_norm and token == title_norm:
            continue
        if _looks_like_product_title(token, title):
            continue
        return token
    return None


def remove_duplicates(df: pd.DataFrame, subset: list[str] | None = None) -> pd.DataFrame:
    """Remove duplicate rows. Default: key on source_platform, shop_name, product_id."""
    key = subset or ["source_platform", "shop_name", "product_id"]
    return df.drop_duplicates(subset=key, keep="first").reset_index(drop=True)


def canonicalize_url(url: str) -> str:
    """Normalize URL to a canonical comparable form for dedup and joins."""
    if not isinstance(url, str):
        return ""
    trimmed = url.strip()
    if not trimmed:
        return ""

    parts = urlsplit(trimmed)
    scheme = (parts.scheme or "https").lower()
    netloc = parts.netloc.lower()
    path = parts.path or "/"
    path = "/".join(seg for seg in path.split("/") if seg)
    path = f"/{path}" if path else "/"

    query_items = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=False)
        if not k.lower().startswith("utm_")
        and k.lower() not in {"gclid", "fbclid", "ref", "source"}
    ]
    query = urlencode(sorted(query_items), doseq=True)
    return urlunsplit((scheme, netloc, path, query, ""))


def standardize_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure price columns are numeric; invalid/negative -> NaN."""
    out = df.copy()
    for col in ["price", "old_price"]:
        if col in out.columns:
            original = out[col]
            non_empty_original = original.notna() & original.astype(str).str.strip().ne("")
            converted = pd.to_numeric(out[col], errors="coerce")
            out[f"{col}_parse_failed"] = (non_empty_original & converted.isna()).astype(bool)
            out[col] = converted
            out.loc[out[col] < 0, col] = pd.NA
    return out


def strip_html(text: str) -> str:
    """Remove any residual HTML tags from a string."""
    if not isinstance(text, str) or not text:
        return ""
    if "<" in text and ">" in text:
        return BeautifulSoup(text, "html.parser").get_text(separator=" ").strip()
    return text.strip()


def normalize_text_columns(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Strip whitespace and residual HTML from text columns."""
    cols = columns or [
        c for c in ["title", "description", "category", "brand", "availability"] if c in df.columns
    ]
    out = df.copy()
    for c in cols:
        if out[c].dtype == object:
            out[c] = out[c].fillna("").astype(str).apply(strip_html)
            out[c] = out[c].str.replace(r"\s+", " ", regex=True).str.strip()
            out[c] = out[c].replace({"nan": "", "None": "", "none": ""})
    return out


def canonicalize_product_urls(df: pd.DataFrame, url_col: str = "product_url") -> pd.DataFrame:
    """Attach canonical URL column and normalize URL field for consistency."""
    out = df.copy()
    if url_col not in out.columns:
        return out
    canonical_col = f"{url_col}_canonical"
    out[canonical_col] = out[url_col].fillna("").astype(str).map(canonicalize_url)
    out[url_col] = out[canonical_col].replace({"": pd.NA})
    return out


def numeric_ratings_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Convert rating and review_count to numeric."""
    out = df.copy()
    if "rating" in out.columns:
        rating_orig = out["rating"]
        rating_non_empty = rating_orig.notna() & rating_orig.astype(str).str.strip().ne("")
        rating_num = pd.to_numeric(out["rating"], errors="coerce")
        out["rating_parse_failed"] = (rating_non_empty & rating_num.isna()).astype(bool)
        out["rating"] = rating_num
    if "review_count" in out.columns:
        review_orig = out["review_count"]
        review_non_empty = review_orig.notna() & review_orig.astype(str).str.strip().ne("")
        review_num = pd.to_numeric(out["review_count"], errors="coerce")
        out["review_count_parse_failed"] = (review_non_empty & review_num.isna()).astype(bool)
        out["review_count"] = review_num.fillna(0).astype(int)
    return out


def clean_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize categories: lowercase, strip, replace empty/none with NaN."""
    out = df.copy()
    if "category" in out.columns:
        original = out["category"].fillna("").astype(str)
        had_raw_value = original.str.strip().ne("")
        out["category"] = original.map(_normalize_category_token)
        aliases = {
            "uncategorised": "uncategorized",
            "uncategorized products": "uncategorized",
            "gift card": "gift cards",
            "indoor rugs": "rugs",
            "outdoor rugs": "rugs",
            "women": "womens clothing",
            "woman": "womens clothing",
            "men": "mens clothing",
            "mens": "mens clothing",
        }
        out["category"] = out["category"].replace(aliases)
        out.loc[out["category"].isin(_NAVIGATION_LABELS), "category"] = pd.NA

        has_path = "category_path_raw" in out.columns
        has_title = "title" in out.columns
        if has_path:
            path_candidate = [
                _path_category_candidate(
                    path_raw=path,
                    title=(out.at[i, "title"] if has_title else None),
                )
                for i, path in out["category_path_raw"].items()
            ]
            out["_path_candidate"] = pd.Series(path_candidate, index=out.index)
            out["_path_candidate"] = out["_path_candidate"].replace(aliases)

            # Demote title-like category values when path has better taxonomy signal.
            if has_title:
                title_like_mask = out.apply(
                    lambda row: _looks_like_product_title(row.get("category"), row.get("title")),
                    axis=1,
                )
            else:
                title_like_mask = pd.Series(False, index=out.index)

            department_only = {"womens clothing", "mens clothing"}
            department_mask = out["category"].isin(department_only)
            better_path_mask = out["_path_candidate"].notna()
            replace_mask = (
                title_like_mask | department_mask | out["category"].isna()
            ) & better_path_mask
            out.loc[replace_mask, "category"] = out.loc[replace_mask, "_path_candidate"]
            out = out.drop(columns=["_path_candidate"])

        out["category_parse_failed"] = (had_raw_value & out["category"].isna()).astype(bool)
    return out


def deduplicate_core_records(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic dedup by strongest available core key."""
    subset: list[str] = []
    for col in ["source_platform", "shop_name", "product_id", "product_url_canonical"]:
        if col in df.columns:
            subset.append(col)
    if len(subset) < 3:
        subset = [c for c in ["source_platform", "shop_name", "product_id"] if c in df.columns]
    if not subset:
        return df.reset_index(drop=True)
    return remove_duplicates(df, subset=subset)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline."""
    df = standardize_prices(df)
    df = normalize_text_columns(df)
    df = canonicalize_product_urls(df)
    df = numeric_ratings_reviews(df)
    df = clean_categories(df)
    df = deduplicate_core_records(df)
    return df
