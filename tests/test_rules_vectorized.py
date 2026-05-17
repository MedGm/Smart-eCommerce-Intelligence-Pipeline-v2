import pandas as pd
import numpy as np
from unittest.mock import patch


def _make_df(n=50):
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "category": rng.choice(["electronics", "clothing", "home", None], n),
        "price_bucket": rng.choice(["low", "mid", "high"], n),
        "is_in_stock": rng.integers(0, 2, n).astype(bool),
        "discount_pct": rng.uniform(0, 0.3, n),
        "source_platform": rng.choice(["shopify", "woocommerce"], n),
        "brand": rng.choice(["BrandA", "BrandB", "unknown"], n),
        "shop_name": rng.choice(["shopX", "shopY"], n),
    })


def test_no_brand_or_platform_in_transactions(tmp_path):
    try:
        from mlxtend.preprocessing import TransactionEncoder
    except ImportError:
        import pytest; pytest.skip("mlxtend not installed")

    test_df = _make_df(n=50)
    captured = []

    class CapTE(TransactionEncoder):
        def fit_transform(self, transactions, **kw):
            captured.extend(transactions)
            return super().fit_transform(transactions, **kw)

    with patch("src.config.data_dir", return_value=tmp_path), \
         patch("src.ml.rules.load_features", return_value=test_df), \
         patch("src.ml.rules.TransactionEncoder", CapTE):
        from src.ml.rules import run
        run(min_support=0.05, min_confidence=0.2)

    assert captured, "No transactions captured"
    for tx in captured:
        for item in tx:
            assert not item.startswith("brand:"), f"brand item found: {item}"
            assert not item.startswith("platform:"), f"platform item found: {item}"
