"""
Kubeflow Pipelines v2 — typed components with caching, retry, and data_dir param.

Changes from v1:
- base_image: smart-ecommerce-pipeline-v2-app:latest (built by `docker compose build app`)
- sys.path.append removed — PYTHONPATH=/app is set in Dockerfile
- data_dir pipeline parameter passed to all components (overrides DATA_DIR env var)
- Caching enabled on stable steps; retry=2 on all steps

Scraping and LLM summary excluded from KFP (browser automation / API keys).

NOTE: Data flows via DATA_DIR (MinIO-backed via Phase 1). Artifacts carry metadata +
caching key only. On single-node Minikube this is equivalent to shared PVC.
"""

from kfp import dsl

_IMAGE = "smart-ecommerce-pipeline-v2-app:latest"


@dsl.component(base_image=_IMAGE)
def preprocess_op(
    data_dir: str,
    processed: dsl.Output[dsl.Dataset],
):
    import os

    os.environ["DATA_DIR"] = data_dir
    from src.preprocessing.run import run

    run()
    processed.metadata["data_dir"] = data_dir


@dsl.component(base_image=_IMAGE)
def dq_op(data_dir: str, processed: dsl.Input[dsl.Dataset]):
    """Great Expectations DQ gate — hard-stop if cleaned_products.parquet is invalid."""
    import os

    os.environ["DATA_DIR"] = data_dir
    from src.pipeline.dq_step import run_or_raise

    run_or_raise()


@dsl.component(base_image=_IMAGE)
def features_op(
    data_dir: str,
    processed: dsl.Input[dsl.Dataset],
    features: dsl.Output[dsl.Dataset],
):
    import os

    os.environ["DATA_DIR"] = data_dir
    from src.features.build_features import run

    run()
    features.metadata["data_dir"] = data_dir


@dsl.component(base_image=_IMAGE)
def score_op(data_dir: str, features: dsl.Input[dsl.Dataset]):
    import os

    os.environ["DATA_DIR"] = data_dir
    from src.scoring.topk import run

    run()


@dsl.component(base_image=_IMAGE)
def train_classifier_op(data_dir: str, features: dsl.Input[dsl.Dataset]):
    import os

    os.environ["DATA_DIR"] = data_dir
    from src.ml.train_classifier import run

    run()


@dsl.component(base_image=_IMAGE)
def train_xgboost_op(data_dir: str, features: dsl.Input[dsl.Dataset]):
    import os

    os.environ["DATA_DIR"] = data_dir
    from src.ml.train_xgboost import run

    run()


@dsl.component(base_image=_IMAGE)
def cluster_kmeans_op(data_dir: str, features: dsl.Input[dsl.Dataset]):
    import os

    os.environ["DATA_DIR"] = data_dir
    from src.ml.cluster_products import run

    run()


@dsl.component(base_image=_IMAGE)
def cluster_dbscan_op(data_dir: str, features: dsl.Input[dsl.Dataset]):
    import os

    os.environ["DATA_DIR"] = data_dir
    from src.ml.dbscan_products import run

    run()


@dsl.component(base_image=_IMAGE)
def association_rules_op(data_dir: str, features: dsl.Input[dsl.Dataset]):
    import os

    os.environ["DATA_DIR"] = data_dir
    from src.ml.rules import run

    run()


@dsl.pipeline(
    name="smart-ecommerce-intelligence-pipeline",
    description=(
        "Full ML/DM pipeline for eCommerce product analysis. "
        "Scraping runs as a pre-step; LLM summary via dashboard."
    ),
)
def smart_ecommerce_pipeline(data_dir: str = "/app/data"):
    """KFP v2 DAG with caching and retry."""
    p = (
        preprocess_op(data_dir=data_dir)
        .set_caching_options(enable_caching=False)
        .set_retry(num_retries=2)
    )

    dq = (
        dq_op(data_dir=data_dir, processed=p.outputs["processed"])
        .set_caching_options(enable_caching=False)
        .set_retry(num_retries=1)
    )

    f = (
        features_op(data_dir=data_dir, processed=p.outputs["processed"])
        .after(dq)
        .set_caching_options(enable_caching=True)
        .set_retry(num_retries=2)
    )

    s = score_op(data_dir=data_dir, features=f.outputs["features"]).set_caching_options(enable_caching=True).set_retry(num_retries=2)

    train_classifier_op(data_dir=data_dir, features=f.outputs["features"]).after(s).set_caching_options(enable_caching=True).set_retry(num_retries=2)
    train_xgboost_op(data_dir=data_dir, features=f.outputs["features"]).after(s).set_caching_options(enable_caching=True).set_retry(num_retries=2)
    cluster_kmeans_op(data_dir=data_dir, features=f.outputs["features"]).set_caching_options(enable_caching=True).set_retry(num_retries=2)
    cluster_dbscan_op(data_dir=data_dir, features=f.outputs["features"]).set_caching_options(enable_caching=True).set_retry(num_retries=2)
    association_rules_op(data_dir=data_dir, features=f.outputs["features"]).set_caching_options(enable_caching=True).set_retry(num_retries=2)


def run() -> None:
    """Entry point used when calling this module as a script."""
    print(
        "Kubeflow pipeline defined as `smart_ecommerce_pipeline`.\n"
        "Compile: make compile-kfp\n"
        "Run:     make kfp-operator-deploy"
    )
