"""
Central place for ML evaluation (classification + clustering metrics).
Aggregates all model outputs for dashboard and report.
"""

import json
from pathlib import Path

from src.config import analytics_dir, get_logger

logger = get_logger(__name__)


def aggregate_metrics(out_dir: Path | None = None) -> dict:
    """Load all model metrics and clustering outputs for dashboard/report."""
    p = Path(out_dir) if out_dir else analytics_dir()
    results: dict = {}

    # Classification: RandomForest
    rf_path = p / "model_metrics.json"
    if rf_path.exists():
        with open(rf_path) as f:
            results["random_forest"] = json.load(f)

    # Classification: XGBoost
    xgb_path = p / "model_metrics_xgboost.json"
    if xgb_path.exists():
        with open(xgb_path) as f:
            results["xgboost"] = json.load(f)

    # Clustering: KMeans
    clusters_path = p / "clusters.csv"
    if clusters_path.exists():
        import pandas as pd

        clusters = pd.read_csv(clusters_path)
        results["kmeans"] = {
            "n_clusters": int(clusters["cluster"].nunique())
            if "cluster" in clusters.columns
            else 0,
            "cluster_sizes": clusters["cluster"].value_counts().to_dict()
            if "cluster" in clusters.columns
            else {},
        }

    # Clustering: DBSCAN
    dbscan_path = p / "dbscan_clusters.csv"
    if dbscan_path.exists():
        import pandas as pd

        dbscan = pd.read_csv(dbscan_path)
        if "dbscan_cluster" in dbscan.columns:
            labels = dbscan["dbscan_cluster"]
            results["dbscan"] = {
                "n_clusters": int(labels[labels != -1].nunique()),
                "n_outliers": int((labels == -1).sum()),
                "cluster_sizes": labels.value_counts().to_dict(),
            }

    # Association rules
    rules_path = p / "association_rules.csv"
    if rules_path.exists():
        import pandas as pd

        rules = pd.read_csv(rules_path)
        results["association_rules"] = {
            "n_rules": len(rules),
            "avg_confidence": float(rules["confidence"].mean())
            if "confidence" in rules.columns
            else 0,
            "avg_lift": float(rules["lift"].mean()) if "lift" in rules.columns else 0,
        }

    logger.info("Aggregated metrics from %d model outputs.", len(results))
    return results


if __name__ == "__main__":
    metrics = aggregate_metrics()
    print(json.dumps(metrics, indent=2, default=str))
