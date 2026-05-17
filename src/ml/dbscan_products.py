"""
DBSCAN clustering for anomaly/outlier detection in the product catalog.
Dossier: DBSCAN for detecting atypical products and price anomalies.
"""

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from src.config import analytics_dir, get_logger
from src.ml.utils import get_feature_columns, load_features

logger = get_logger(__name__)


def run(eps: float = 1.5, min_samples: int = 5):
    out_dir = analytics_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_features()
    if df.empty or len(df) < min_samples:
        logger.warning("Not enough data for DBSCAN (%d rows).", len(df))
        return

    features = get_feature_columns(df, exclude_score=True)
    if not features:
        logger.warning("No numeric features found.")
        return

    X = df[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)

    df = df.copy()
    df["dbscan_cluster"] = labels

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())

    cols = ["product_id", "title", "category", "shop_name", "dbscan_cluster"]
    df[[c for c in cols if c in df.columns]].to_csv(out_dir / "dbscan_clusters.csv", index=False)

    logger.info(
        "DBSCAN done: %d clusters, %d noise/outlier points -> analytics/dbscan_clusters.csv",
        n_clusters,
        n_noise,
    )
    return df


if __name__ == "__main__":
    run()
