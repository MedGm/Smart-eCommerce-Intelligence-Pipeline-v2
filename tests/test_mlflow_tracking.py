import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch


def _features(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    # Produce a mix of positive and negative labels for build_high_potential_target:
    # positives: rating>=4.3, review_count>=10, is_in_stock=True
    # negatives: lower rating or not in stock
    half = n // 2
    ratings = np.concatenate([rng.uniform(4.3, 5.0, half), rng.uniform(3.0, 4.2, n - half)])
    reviews = np.concatenate([rng.uniform(50, 500, half), rng.uniform(1, 9, n - half)])
    in_stock = [True] * half + [False] * (n - half)
    return pd.DataFrame({
        "product_id": [str(i) for i in range(n)],
        "shop_name": [f"shop_{i % 3}" for i in range(n)],
        "price": rng.uniform(10, 200, n),
        "rating": ratings,
        "review_count": reviews,
        "discount_pct": rng.uniform(0, 40, n),
        "availability": ["instock"] * n,
        "is_in_stock": in_stock,
    })


def _mock_run():
    ctx = MagicMock()
    ctx.__enter__ = lambda s: s
    ctx.__exit__ = lambda s, *a: None
    return ctx


def test_rf_mlflow_called_when_uri_set(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    (tmp_path / "analytics").mkdir()
    (tmp_path / "models").mkdir()

    with patch("src.ml.train_classifier.load_features", return_value=_features()), \
         patch("src.ml.train_classifier.mlflow") as m:
        m.start_run.return_value = _mock_run()
        from src.ml import train_classifier
        train_classifier.run()

    m.set_tracking_uri.assert_called_once_with("http://localhost:5000")
    assert m.start_run.called
    assert m.log_params.called
    assert m.log_metrics.called


def test_rf_mlflow_noop_without_uri(monkeypatch, tmp_path):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    (tmp_path / "analytics").mkdir()
    (tmp_path / "models").mkdir()

    with patch("src.ml.train_classifier.load_features", return_value=_features()), \
         patch("src.ml.train_classifier.mlflow") as m:
        from src.ml import train_classifier
        train_classifier.run()

    m.set_tracking_uri.assert_not_called()
    m.start_run.assert_not_called()


def test_xgb_mlflow_called_when_uri_set(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    (tmp_path / "analytics").mkdir()
    (tmp_path / "models").mkdir()

    n = 30
    fake_proba = np.column_stack([np.full(n, 0.3), np.full(n, 0.7)])
    mock_diagnostics = {
        "majority_baseline_accuracy": 0.5,
        "shuffled_label_accuracy": 0.45,
        "direct_leakage_check": "pass",
        "notes": "",
    }
    mock_honesty = {"status": "ok", "trust_score": 80}

    mock_clf = MagicMock()
    mock_clf.feature_importances_ = np.ones(5)
    mock_clf.predict_proba.return_value = fake_proba

    with patch("src.ml.train_xgboost.load_features", return_value=_features(n)), \
         patch("src.ml.train_xgboost.cross_val_predict", return_value=fake_proba), \
         patch("src.ml.train_xgboost.label_integrity_diagnostics", return_value=mock_diagnostics), \
         patch("src.ml.train_xgboost.honesty_gate", return_value=mock_honesty), \
         patch("src.ml.train_xgboost.XGBClassifier", return_value=mock_clf), \
         patch("src.ml.train_xgboost.joblib"), \
         patch("src.ml.train_xgboost.mlflow_xgb"), \
         patch("src.ml.train_xgboost.mlflow") as m:
        m.start_run.return_value = _mock_run()
        from src.ml import train_xgboost
        train_xgboost.run()

    m.set_tracking_uri.assert_called_once_with("http://localhost:5000")
    assert m.start_run.called


def test_kmeans_mlflow_called_when_uri_set(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    (tmp_path / "analytics").mkdir()

    with patch("src.ml.cluster_products.load_features", return_value=_features(30)), \
         patch("src.ml.cluster_products.mlflow") as m:
        m.start_run.return_value = _mock_run()
        from src.ml import cluster_products
        cluster_products.run()

    m.set_tracking_uri.assert_called_once_with("http://localhost:5000")
    assert m.start_run.called
    assert m.log_param.called or m.log_params.called
