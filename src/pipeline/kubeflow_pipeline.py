"""
Kubeflow Pipelines v2 — typed components with MinIO-backed I/O.

Each pod is ephemeral with an empty filesystem. Every component must:
  1. Download required inputs from host MinIO at the start
  2. Run its logic
  3. Upload outputs to host MinIO at the end

Host MinIO is reachable at 192.168.49.1:9000 (minikube gateway).
Host MLflow is reachable at 192.168.49.1:5000.
"""

from kfp import dsl

_IMAGE = "prism-app:local"


def _dl(client, bucket: str, key: str, local_path) -> None:
    """Download a single file from MinIO, creating parent dirs."""
    import pathlib

    p = pathlib.Path(local_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(p))


def _ul(client, local_path, bucket: str, key: str) -> None:
    """Upload a file to MinIO if it exists."""
    import pathlib

    p = pathlib.Path(local_path)
    if p.exists():
        client.upload_file(str(p), bucket, key)


@dsl.component(base_image=_IMAGE)
def preprocess_op(data_dir: str, processed: dsl.Output[dsl.Dataset]):
    """Sync raw from MinIO → preprocess → upload cleaned parquet."""
    import os

    os.environ["DATA_DIR"] = data_dir
    from src.preprocessing.run import run

    run()  # internally syncs raw from MinIO + uploads processed
    processed.metadata["data_dir"] = data_dir


@dsl.component(base_image=_IMAGE)
def dq_op(data_dir: str, processed: dsl.Input[dsl.Dataset]):
    """Download cleaned parquet from MinIO → run Great Expectations."""
    import os
    import pathlib

    os.environ["DATA_DIR"] = data_dir
    parquet = pathlib.Path(data_dir) / "processed" / "cleaned_products.parquet"
    if not parquet.exists():
        parquet.parent.mkdir(parents=True, exist_ok=True)
        from src.storage.minio_client import _client

        _client().download_file("processed", "cleaned_products.parquet", str(parquet))
    from src.pipeline.dq_step import run_or_raise

    run_or_raise(parquet_path=str(parquet))


@dsl.component(base_image=_IMAGE)
def features_op(
    data_dir: str,
    processed: dsl.Input[dsl.Dataset],
    features: dsl.Output[dsl.Dataset],
):
    """Download cleaned parquet → build features → upload features parquet."""
    import os
    import pathlib

    os.environ["DATA_DIR"] = data_dir
    p_dir = pathlib.Path(data_dir) / "processed"
    p_dir.mkdir(parents=True, exist_ok=True)

    from src.storage.minio_client import _client

    c = _client()
    c.download_file(
        "processed", "cleaned_products.parquet", str(p_dir / "cleaned_products.parquet")
    )

    from src.features.build_features import run

    run()

    c.upload_file(str(p_dir / "features.parquet"), "processed", "features.parquet")
    features.metadata["data_dir"] = data_dir


@dsl.component(base_image=_IMAGE)
def score_op(data_dir: str, features: dsl.Input[dsl.Dataset]):
    """Download features → score → upload topk CSVs."""
    import os
    import pathlib

    os.environ["DATA_DIR"] = data_dir
    p_dir = pathlib.Path(data_dir) / "processed"
    p_dir.mkdir(parents=True, exist_ok=True)
    a_dir = pathlib.Path(data_dir) / "analytics"
    a_dir.mkdir(parents=True, exist_ok=True)

    from src.storage.minio_client import _client

    c = _client()
    c.download_file("processed", "features.parquet", str(p_dir / "features.parquet"))

    from src.scoring.topk import run

    run()

    for fname in ["topk_products.csv", "topk_per_category.csv", "topk_per_shop.csv"]:
        if (a_dir / fname).exists():
            c.upload_file(str(a_dir / fname), "processed", f"analytics/{fname}")


@dsl.component(base_image=_IMAGE)
def train_classifier_op(data_dir: str, features: dsl.Input[dsl.Dataset]):
    """Download features → train RF → log to MLflow → upload model to MinIO."""
    import os
    import pathlib

    os.environ["DATA_DIR"] = data_dir
    p_dir = pathlib.Path(data_dir) / "processed"
    p_dir.mkdir(parents=True, exist_ok=True)
    pathlib.Path(data_dir, "models").mkdir(parents=True, exist_ok=True)
    pathlib.Path(data_dir, "analytics").mkdir(parents=True, exist_ok=True)

    from src.storage.minio_client import _client

    _client().download_file("processed", "features.parquet", str(p_dir / "features.parquet"))

    from src.ml.train_classifier import run

    run()


@dsl.component(base_image=_IMAGE)
def train_xgboost_op(data_dir: str, features: dsl.Input[dsl.Dataset]):
    """Download features → train XGBoost → log to MLflow → upload model to MinIO."""
    import os
    import pathlib

    os.environ["DATA_DIR"] = data_dir
    p_dir = pathlib.Path(data_dir) / "processed"
    p_dir.mkdir(parents=True, exist_ok=True)
    pathlib.Path(data_dir, "models").mkdir(parents=True, exist_ok=True)
    pathlib.Path(data_dir, "analytics").mkdir(parents=True, exist_ok=True)

    from src.storage.minio_client import _client

    _client().download_file("processed", "features.parquet", str(p_dir / "features.parquet"))

    from src.ml.train_xgboost import run

    run()


@dsl.component(base_image=_IMAGE)
def cluster_kmeans_op(data_dir: str, features: dsl.Input[dsl.Dataset]):
    """Download features → KMeans clustering → upload clusters CSV."""
    import os
    import pathlib

    os.environ["DATA_DIR"] = data_dir
    p_dir = pathlib.Path(data_dir) / "processed"
    p_dir.mkdir(parents=True, exist_ok=True)
    a_dir = pathlib.Path(data_dir) / "analytics"
    a_dir.mkdir(parents=True, exist_ok=True)

    from src.storage.minio_client import _client

    c = _client()
    c.download_file("processed", "features.parquet", str(p_dir / "features.parquet"))

    from src.ml.cluster_products import run

    run()

    for fname in ["clusters.csv", "pca_viz.csv", "cluster_metrics.json"]:
        if (a_dir / fname).exists():
            c.upload_file(str(a_dir / fname), "processed", f"analytics/{fname}")


@dsl.component(base_image=_IMAGE)
def cluster_dbscan_op(data_dir: str, features: dsl.Input[dsl.Dataset]):
    """Download features → DBSCAN clustering → upload results."""
    import os
    import pathlib

    os.environ["DATA_DIR"] = data_dir
    p_dir = pathlib.Path(data_dir) / "processed"
    p_dir.mkdir(parents=True, exist_ok=True)
    a_dir = pathlib.Path(data_dir) / "analytics"
    a_dir.mkdir(parents=True, exist_ok=True)

    from src.storage.minio_client import _client

    c = _client()
    c.download_file("processed", "features.parquet", str(p_dir / "features.parquet"))

    from src.ml.dbscan_products import run

    run()

    if (a_dir / "dbscan_clusters.csv").exists():
        c.upload_file(
            str(a_dir / "dbscan_clusters.csv"), "processed", "analytics/dbscan_clusters.csv"
        )


@dsl.component(base_image=_IMAGE)
def association_rules_op(data_dir: str, features: dsl.Input[dsl.Dataset]):
    """Download features → mine association rules → upload CSV."""
    import os
    import pathlib

    os.environ["DATA_DIR"] = data_dir
    p_dir = pathlib.Path(data_dir) / "processed"
    p_dir.mkdir(parents=True, exist_ok=True)
    a_dir = pathlib.Path(data_dir) / "analytics"
    a_dir.mkdir(parents=True, exist_ok=True)

    from src.storage.minio_client import _client

    c = _client()
    c.download_file("processed", "features.parquet", str(p_dir / "features.parquet"))

    from src.ml.rules import run

    run()

    if (a_dir / "association_rules.csv").exists():
        c.upload_file(
            str(a_dir / "association_rules.csv"), "processed", "analytics/association_rules.csv"
        )


@dsl.pipeline(
    name="prism-pipeline",
    description="PRISM: preprocess → DQ → features → score + ML → MinIO + MLflow",
)
def prism_pipeline(data_dir: str = "/app/data"):
    """KFP v2 DAG. All pods wired to host MinIO (192.168.49.1:9000) + MLflow."""
    _MINIO = "http://192.168.49.1:9000"
    _MLFLOW = "http://192.168.49.1:5000"
    _KEY = "minioadmin"

    def _wire(task):
        return (
            task.set_env_variable("MINIO_ENDPOINT", _MINIO)
            .set_env_variable("MINIO_ACCESS_KEY", _KEY)
            .set_env_variable("MINIO_SECRET_KEY", _KEY)
            .set_env_variable("AWS_ACCESS_KEY_ID", _KEY)
            .set_env_variable("AWS_SECRET_ACCESS_KEY", _KEY)
            .set_env_variable("MLFLOW_S3_ENDPOINT_URL", _MINIO)
            .set_env_variable("MLFLOW_TRACKING_URI", _MLFLOW)
            .set_env_variable("GIT_PYTHON_REFRESH", "quiet")
        )

    p = _wire(
        preprocess_op(data_dir=data_dir)
        .set_caching_options(enable_caching=False)
        .set_retry(num_retries=2)
    )

    dq = _wire(
        dq_op(data_dir=data_dir, processed=p.outputs["processed"])
        .set_caching_options(enable_caching=False)
        .set_retry(num_retries=1)
    )

    f = _wire(
        features_op(data_dir=data_dir, processed=p.outputs["processed"])
        .after(dq)
        .set_caching_options(enable_caching=False)
        .set_retry(num_retries=2)
    )

    s = _wire(
        score_op(data_dir=data_dir, features=f.outputs["features"])
        .set_caching_options(enable_caching=False)
        .set_retry(num_retries=2)
    )

    _wire(
        train_classifier_op(data_dir=data_dir, features=f.outputs["features"])
        .after(s)
        .set_caching_options(enable_caching=False)
        .set_retry(num_retries=2)
    )
    _wire(
        train_xgboost_op(data_dir=data_dir, features=f.outputs["features"])
        .after(s)
        .set_caching_options(enable_caching=False)
        .set_retry(num_retries=2)
    )
    _wire(
        cluster_kmeans_op(data_dir=data_dir, features=f.outputs["features"])
        .set_caching_options(enable_caching=False)
        .set_retry(num_retries=2)
    )
    _wire(
        cluster_dbscan_op(data_dir=data_dir, features=f.outputs["features"])
        .set_caching_options(enable_caching=False)
        .set_retry(num_retries=2)
    )
    _wire(
        association_rules_op(data_dir=data_dir, features=f.outputs["features"])
        .set_caching_options(enable_caching=False)
        .set_retry(num_retries=2)
    )


def run() -> None:
    print(
        "Kubeflow pipeline defined as `prism_pipeline`.\n"
        "Compile: make compile-kfp\n"
        "Run:     upload kubeflow_prism_pipeline.yaml to http://localhost:8080"
    )
