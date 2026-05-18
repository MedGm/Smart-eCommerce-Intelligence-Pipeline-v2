"""
Kubeflow Pipelines v2 — typed components with caching, retry, and data_dir param.

Changes from v1:
- base_image: prism-app:latest (built by `docker compose build app`)
- sys.path.append removed — PYTHONPATH=/app is set in Dockerfile
- data_dir pipeline parameter passed to all components (overrides DATA_DIR env var)
- Caching enabled on stable steps; retry=2 on all steps

Scraping and LLM summary excluded from KFP (browser automation / API keys).

NOTE: Data flows via DATA_DIR (MinIO-backed via Phase 1). Artifacts carry metadata +
caching key only. On single-node Minikube this is equivalent to shared PVC.
"""

from kfp import dsl

_IMAGE = "prism-app:local"


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
def dq_op(
    data_dir: str,
    processed: dsl.Input[dsl.Dataset],
    minio_endpoint: str = "http://192.168.49.1:9000",
    minio_access_key: str = "minioadmin",
    minio_secret_key: str = "minioadmin",
):
    """Great Expectations DQ gate — downloads parquet from MinIO then validates."""
    import os
    import pathlib

    os.environ["DATA_DIR"] = data_dir
    os.environ["MINIO_ENDPOINT"] = minio_endpoint
    os.environ["MINIO_ACCESS_KEY"] = minio_access_key
    os.environ["MINIO_SECRET_KEY"] = minio_secret_key

    parquet_local = pathlib.Path(data_dir) / "processed" / "cleaned_products.parquet"
    if not parquet_local.exists():
        parquet_local.parent.mkdir(parents=True, exist_ok=True)
        from src.storage.minio_client import _client
        _client().download_file("processed", "cleaned_products.parquet", str(parquet_local))

    from src.pipeline.dq_step import run_or_raise
    run_or_raise(parquet_path=str(parquet_local))


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
    name="prism-pipeline",
    description=(
        "Full ML/DM pipeline for eCommerce product analysis. "
        "Scraping runs as a pre-step; LLM summary via dashboard."
    ),
)
def prism_pipeline(data_dir: str = "/app/data"):
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

    s = (
        score_op(data_dir=data_dir, features=f.outputs["features"])
        .set_caching_options(enable_caching=True)
        .set_retry(num_retries=2)
    )

    train_classifier_op(data_dir=data_dir, features=f.outputs["features"]).after(s).set_caching_options(enable_caching=True).set_retry(num_retries=2)
    train_xgboost_op(data_dir=data_dir, features=f.outputs["features"]).after(s).set_caching_options(enable_caching=True).set_retry(num_retries=2)
    cluster_kmeans_op(data_dir=data_dir, features=f.outputs["features"]).set_caching_options(enable_caching=True).set_retry(num_retries=2)
    cluster_dbscan_op(data_dir=data_dir, features=f.outputs["features"]).set_caching_options(enable_caching=True).set_retry(num_retries=2)
    association_rules_op(data_dir=data_dir, features=f.outputs["features"]).set_caching_options(enable_caching=True).set_retry(num_retries=2)

def run() -> None:
    """Entry point used when calling this module as a script."""
    print(
        "Kubeflow pipeline defined as `prism_pipeline`.\n"
        "Compile: make compile-kfp\n"
        "Run:     make kfp-operator-deploy"
    )
