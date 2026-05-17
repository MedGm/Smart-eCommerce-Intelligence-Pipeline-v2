import importlib
import json
from unittest.mock import patch

import numpy as np
import pandas as pd


def test_cluster_metrics_json_written(tmp_path):
    with patch("src.config.data_dir", return_value=tmp_path):
        from src.ml import cluster_products

        importlib.reload(cluster_products)
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "price": rng.uniform(10, 500, 40),
                "dq_score": rng.uniform(0, 1, 40),
                "is_in_stock": rng.integers(0, 2, 40).astype(float),
            }
        )
        with patch("src.ml.cluster_products.load_features", return_value=df):
            cluster_products.run(n_clusters=2)
    metrics_path = tmp_path / "analytics" / "cluster_metrics.json"
    assert metrics_path.exists(), "cluster_metrics.json not written"
    metrics = json.loads(metrics_path.read_text())
    assert "silhouette_score" in metrics
    assert isinstance(metrics["silhouette_score"], float)
    assert "n_clusters" in metrics
