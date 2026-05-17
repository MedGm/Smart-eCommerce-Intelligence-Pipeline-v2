"""
Kubeflow-compatible pipeline: wrap local steps as KFP components.

This pipeline mirrors the local Python pipeline:
    preprocess -> features -> score -> train_classifier -> train_xgboost
    -> cluster (KMeans) -> cluster (DBSCAN) -> association rules

It is meant to be compiled with kfp and run on a Kubeflow Pipelines
installation (e.g. on Minikube or a managed KFP cluster).

NOTE: Scraping and LLM summary are excluded from KFP because:
- Scraping requires browser automation (Playwright) which needs a different
  container image with system deps; it runs as a pre-pipeline step.
- LLM summary depends on API keys that should not be baked into pipeline
  containers; it runs as a post-pipeline step or via the dashboard.
"""

from __future__ import annotations

from kfp import dsl


@dsl.component(base_image="smart-ecommerce-pipeline:local")
def preprocess_op():
    """Run preprocessing step (from raw JSON to cleaned parquet)."""
    import sys

    sys.path.append("/app")
    from src.preprocessing.run import run

    run()


@dsl.component(base_image="smart-ecommerce-pipeline:local")
def features_op():
    """Run feature engineering step (from cleaned to features parquet)."""
    import sys

    sys.path.append("/app")
    from src.features.build_features import run

    run()


@dsl.component(base_image="smart-ecommerce-pipeline:local")
def score_op():
    """Run Top-K scoring and export analytics CSVs."""
    import sys

    sys.path.append("/app")
    from src.scoring.topk import run

    run()


@dsl.component(base_image="smart-ecommerce-pipeline:local")
def train_classifier_op():
    """Train RandomForest classifier and export metrics."""
    import sys

    sys.path.append("/app")
    from src.ml.train_classifier import run

    run()


@dsl.component(base_image="smart-ecommerce-pipeline:local")
def train_xgboost_op():
    """Train XGBoost classifier and export metrics."""
    import sys

    sys.path.append("/app")
    from src.ml.train_xgboost import run

    run()


@dsl.component(base_image="smart-ecommerce-pipeline:local")
def cluster_kmeans_op():
    """Run KMeans clustering with PCA visualization."""
    import sys

    sys.path.append("/app")
    from src.ml.cluster_products import run

    run()


@dsl.component(base_image="smart-ecommerce-pipeline:local")
def cluster_dbscan_op():
    """Run DBSCAN clustering for anomaly detection."""
    import sys

    sys.path.append("/app")
    from src.ml.dbscan_products import run

    run()


@dsl.component(base_image="smart-ecommerce-pipeline:local")
def association_rules_op():
    """Run Apriori association rules mining."""
    import sys

    sys.path.append("/app")
    from src.ml.rules import run

    run()


@dsl.pipeline(
    name="smart-ecommerce-intelligence-pipeline",
    description=(
        "Full ML/DM pipeline for eCommerce product analysis. "
        "Scraping runs as a pre-step; LLM summary via dashboard."
    ),
)
def smart_ecommerce_pipeline():
    """Kubeflow pipeline DAG definition (8 components)."""
    p = preprocess_op()
    f = features_op().after(p)
    s = score_op().after(f)
    # Classifiers can run in parallel after scoring
    train_classifier_op().after(s)
    train_xgboost_op().after(s)
    # Clustering runs after scoring (needs features)
    cluster_kmeans_op().after(f)
    cluster_dbscan_op().after(f)
    # Association rules run after features
    association_rules_op().after(f)


def run() -> None:
    """Entry point used when calling this module as a script."""
    print(
        "Kubeflow pipeline is defined as `smart_ecommerce_pipeline` (8 components).\\n"
        "Compile it with kfp, for example:\\n"
        "  from kfp import compiler\\n"
        "  from src.pipeline.kubeflow_pipeline import smart_ecommerce_pipeline\\n"
        "  compiler.Compiler().compile(\\n"
        "      pipeline_func=smart_ecommerce_pipeline,\\n"
        "      package_path='kubeflow_smart_ecommerce_pipeline.yaml',\\n"
        "  )"
    )
