"""
Clustering: KMeans for product segments. Export cluster labels and PCA viz data.
"""

import json

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import analytics_dir, get_logger
from src.ml.utils import get_feature_columns, load_features

logger = get_logger(__name__)


def run(n_clusters: int = 4):
    out_dir = analytics_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_features()
    if df.empty or len(df) < n_clusters:
        logger.warning("Not enough data for clustering (%d rows).", len(df))
        return

    features = get_feature_columns(df, exclude_score=True)
    if not features:
        logger.warning("No numeric features found.")
        return

    X = df[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df = df.copy()
    df["cluster"] = km.fit_predict(X_scaled)

    if "score" not in df.columns:
        from src.scoring.topk import compute_score

        df["score"] = compute_score(df)

    cols = ["product_id", "title", "category", "shop_name", "cluster", "score"]
    df[[c for c in cols if c in df.columns]].to_csv(out_dir / "clusters.csv", index=False)

    sil = silhouette_score(X_scaled, df["cluster"])

    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X_scaled)
    viz = pd.DataFrame({"pc1": X2[:, 0], "pc2": X2[:, 1], "cluster": df["cluster"]})
    viz.to_csv(out_dir / "pca_viz.csv", index=False)

    cluster_metrics = {
        "n_clusters": n_clusters,
        "silhouette_score": float(sil),
        "n_samples": len(df),
    }
    with open(out_dir / "cluster_metrics.json", "w") as f:
        json.dump(cluster_metrics, f, indent=2)

    logger.info("Clustering done: clusters.csv, pca_viz.csv (silhouette=%.3f)", sil)
    return df


if __name__ == "__main__":
    run()
