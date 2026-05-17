"""
Association rules using mlxtend (Apriori algorithm).
Dossier: discover patterns like {category_A} -> {category_B} or {price_bucket} -> {availability}.
Evaluated with support, confidence, and lift.
"""

import pandas as pd

from src.config import analytics_dir, get_logger
from src.ml.utils import load_features

logger = get_logger(__name__)

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder

    HAS_MLXTEND = True
except ImportError:
    HAS_MLXTEND = False


def run(min_support: float = 0.05, min_confidence: float = 0.3):
    if not HAS_MLXTEND:
        logger.warning("mlxtend not installed. pip install mlxtend to enable association rules.")
        return

    out_dir = analytics_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_features()
    if df.empty:
        logger.warning("Empty features. Skipping association rules.")
        return

    def _build_items(row: "pd.Series") -> list:
        items = []
        cat = row.get("category", None)
        if pd.notna(cat) and str(cat) not in ("unknown", "", "nan"):
            items.append(f"cat:{cat}")
        pb = row.get("price_bucket", "mid")
        if pd.notna(pb) and str(pb) != "mid":
            items.append(f"price:{pb}")
        items.append("in_stock" if row.get("is_in_stock") else "out_of_stock")
        if float(row.get("discount_pct", 0) or 0) > 0.1:
            items.append("has_discount")
        return items

    use_cols = [c for c in ["category", "price_bucket", "is_in_stock", "discount_pct"] if c in df.columns]
    transactions = [
        items for items in df[use_cols].apply(_build_items, axis=1)
        if items
    ]

    if len(transactions) < 10:
        logger.warning("Not enough transactions for association rules.")
        return

    te = TransactionEncoder()
    te_ary = te.fit_transform(transactions)
    basket = pd.DataFrame(te_ary, columns=te.columns_)

    freq = apriori(basket, min_support=min_support, use_colnames=True)
    if freq.empty:
        logger.warning("No frequent itemsets found. Try lowering min_support.")
        return

    rules = association_rules(freq, metric="confidence", min_threshold=min_confidence)
    if rules.empty:
        logger.warning("No association rules found above confidence threshold.")
        return

    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))

    out_path = out_dir / "association_rules.csv"
    rules.to_csv(out_path, index=False)
    logger.info(
        "Association rules: %d rules found (support >= %.2f, confidence >= %.2f) -> %s",
        len(rules),
        min_support,
        min_confidence,
        out_path,
    )
    return rules


if __name__ == "__main__":
    run()
