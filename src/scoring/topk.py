"""
Top-K scoring engine. Explainable formula:
  score = 0.35*rating_norm + 0.30*review_count_norm + 0.20*availability_norm + 0.15*discount_norm
Compute Top-K overall, per category, per shop.
"""

from collections import Counter

import pandas as pd

from src.config import analytics_dir, get_logger, processed_dir

logger = get_logger(__name__)


# Weights (document for oral defense)
WEIGHTS = {
    "rating": 0.35,
    "review_count": 0.30,
    "availability": 0.20,
    "discount": 0.15,
}


def normalize(series: pd.Series) -> pd.Series:
    """Min-max to [0, 1]."""
    s = series.astype(float)
    lo, hi = s.min(), s.max()
    if hi <= lo:
        return pd.Series(0.5, index=series.index)
    return (s - lo) / (hi - lo)


def compute_score(df: pd.DataFrame) -> pd.Series:
    """Single explainable score per product with sparse-signal fallback.

    Missing rating/review signals are imputed to a neutral value (0.5) rather
    than zero, then adjusted by an evidence-confidence factor so rows with no
    user feedback do not dominate the leaderboard.
    """
    rating = df.get("rating", pd.Series(0.0, index=df.index)).fillna(0)
    rating_norm = normalize(rating / 5.0)
    review_count = df.get("review_count", pd.Series(0.0, index=df.index)).fillna(0).astype(float)
    review_norm = normalize(review_count)
    availability = (
        df.get("is_in_stock", True).astype(float)
        if "is_in_stock" in df.columns
        else pd.Series(1.0, index=df.index)
    )
    availability_norm = (
        availability if isinstance(availability, pd.Series) else pd.Series(1.0, index=df.index)
    )
    discount = df.get("discount_pct", pd.Series(0.0, index=df.index)).fillna(0)
    discount_norm = normalize(discount)

    # Row-level signal availability masks.
    rating_active = rating > 0
    review_active = review_count > 0
    if "availability" in df.columns:
        availability_active = df["availability"].fillna("").astype(str).str.strip().ne("")
    else:
        availability_active = pd.Series(True, index=df.index)
    discount_active = pd.Series(True, index=df.index)

    rating_signal = rating_norm.where(rating_active, 0.5)
    review_signal = review_norm.where(review_active, 0.5)
    availability_signal = availability_norm.where(availability_active, 0.5)
    discount_signal = discount_norm.where(discount_active, 0.5)

    raw_score = (
        WEIGHTS["rating"] * rating_signal
        + WEIGHTS["review_count"] * review_signal
        + WEIGHTS["availability"] * availability_signal
        + WEIGHTS["discount"] * discount_signal
    )

    # Penalize rows that carry no rating/review evidence.
    evidence = 0.5 * rating_active.astype(float) + 0.5 * review_active.astype(float)
    confidence = 0.7 + 0.3 * evidence
    return raw_score * confidence


def topk_overall(df: pd.DataFrame, k: int = 50, max_per_shop_ratio: float = 0.4) -> pd.DataFrame:
    """Return a diversified Top-K list.

    Strategy:
    - Pass 1: keep the best product from each shop (representation floor).
    - Pass 2: fill by score while capping each shop contribution.
    - Pass 3: if cap is too strict to reach K, backfill by score.
    """
    if "shop_name" not in df.columns:
        return df.nlargest(k, "score").reset_index(drop=True)

    ranked = df.sort_values("score", ascending=False)
    if ranked.empty:
        return ranked.reset_index(drop=True)

    cap = max(1, int(k * max_per_shop_ratio))
    selected_idx: list[object] = []
    counts: Counter = Counter()

    # Representation floor: keep the best row per shop.
    for _, group in ranked.groupby("shop_name", sort=False):
        idx = group.index[0]
        selected_idx.append(idx)
        counts[group.iloc[0]["shop_name"]] += 1

    if len(selected_idx) > k:
        top_repr = ranked.loc[selected_idx].sort_values("score", ascending=False).head(k)
        return top_repr.reset_index(drop=True)

    # Fill by score while respecting cap.
    selected_set = set(selected_idx)
    for idx, row in ranked.iterrows():
        if len(selected_idx) >= k:
            break
        if idx in selected_set:
            continue
        shop = row["shop_name"]
        if counts[shop] >= cap:
            continue
        selected_idx.append(idx)
        selected_set.add(idx)
        counts[shop] += 1

    # Backfill if cap blocked completion.
    if len(selected_idx) < k:
        for idx in ranked.index:
            if len(selected_idx) >= k:
                break
            if idx in selected_set:
                continue
            selected_idx.append(idx)
            selected_set.add(idx)

    return (
        ranked.loc[selected_idx]
        .sort_values("score", ascending=False)
        .head(k)
        .reset_index(drop=True)
    )


def topk_per_category(df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    if "category" not in df.columns:
        return pd.DataFrame()
    chunks = [group.nlargest(k, "score") for _, group in df.groupby("category", sort=False)]
    if not chunks:
        return pd.DataFrame(columns=df.columns)
    return pd.concat(chunks, ignore_index=True)


def topk_per_shop(df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    key = ["source_platform", "shop_name"] if "shop_name" in df.columns else ["source_platform"]
    chunks = [group.nlargest(k, "score") for _, group in df.groupby(key, sort=False)]
    if not chunks:
        return pd.DataFrame(columns=df.columns)
    return pd.concat(chunks, ignore_index=True)


def run(k_overall: int = 50, k_per: int = 10):
    p_dir = processed_dir()
    a_dir = analytics_dir()
    a_dir.mkdir(parents=True, exist_ok=True)

    in_path = p_dir / "features.parquet"
    if not in_path.exists():
        logger.warning("No features.parquet. Run features step first.")
        return
    df = pd.read_parquet(in_path)
    if df.empty:
        logger.warning("Empty features. Skipping scoring.")
        return
    df = df.copy()
    df["score"] = compute_score(df)

    topk_overall(df, k_overall).to_csv(a_dir / "topk_products.csv", index=False)
    topk_per_category(df, k_per).to_csv(a_dir / "topk_per_category.csv", index=False)
    topk_per_shop(df, k_per).to_csv(a_dir / "topk_per_shop.csv", index=False)
    logger.info("Scoring done: topk_products.csv, topk_per_category.csv, topk_per_shop.csv")
    return df


if __name__ == "__main__":
    run()
