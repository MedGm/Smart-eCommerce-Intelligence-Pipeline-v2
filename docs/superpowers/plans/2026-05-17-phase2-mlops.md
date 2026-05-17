# Phase 2 — MLOps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire MLflow experiment tracking into all three training stages, fix KFP v2 components (remove sys.path hack, correct base image, add caching + retry + typed artifacts), and add a Great Expectations hard-fail DQ gate as a KFP step between preprocessing and feature engineering.

**Architecture:** MLflow is opt-in via `MLFLOW_TRACKING_URI` env var — all calls no-op without it, so local dev and Docker-without-infra keep working unchanged. KFP components are rewritten to use `data_dir` as a pipeline parameter (overrides `DATA_DIR`) and typed `dsl.Output[dsl.Dataset]` artifacts on the preprocess→features chain. The GE DQ step reads `cleaned_products.parquet`, validates eight expectations, and raises `RuntimeError` on failure — which KFP treats as a component failure and stops the DAG.

**Tech Stack:** Python 3.11, mlflow>=2.12.0, great-expectations>=1.0.0, kfp>=2.4.0 (already installed)

> **Scope note:** Task 1 (MLflow) is fully independent of Tasks 2–3 and can be treated as a separate plan if needed.

---

## File Map

| File | Change |
|------|--------|
| `requirements.txt` | Add `mlflow>=2.12.0`, `great-expectations>=1.0.0` |
| `src/ml/train_classifier.py` | Import mlflow; add tracking block after `clf.fit()` |
| `src/ml/train_xgboost.py` | Same pattern as RF |
| `src/ml/cluster_products.py` | Log `n_clusters` + `silhouette_score` to MLflow |
| `src/pipeline/dq_step.py` | New — GE validation function + `run_or_raise()` |
| `src/pipeline/kubeflow_pipeline.py` | Fix all 8 components; add `dq_op`; add `data_dir` param; typed artifacts |
| `tests/test_mlflow_tracking.py` | New — MLflow called when URI set; no-op without |
| `tests/test_kfp_pipeline.py` | New — pipeline compiles to valid YAML |
| `tests/test_dq_step.py` | New — GE passes on valid data, fails on empty |

---

## Task 1: MLflow experiment tracking

**Files:**
- Modify: `requirements.txt`
- Modify: `src/ml/train_classifier.py`
- Modify: `src/ml/train_xgboost.py`
- Modify: `src/ml/cluster_products.py`
- Create: `tests/test_mlflow_tracking.py`

- [ ] **Step 1: Add mlflow to requirements.txt**

After the `kfp>=2.4.0` line, add:
```
mlflow>=2.12.0
great-expectations>=1.0.0
```

Rebuild image:
```bash
docker compose build app 2>&1 | tail -3
```

- [ ] **Step 2: Write failing tests**

```python
# tests/test_mlflow_tracking.py
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch


def _features(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "product_id": [str(i) for i in range(n)],
        "shop_name": [f"shop_{i % 3}" for i in range(n)],
        "price": rng.uniform(10, 200, n),
        "rating": rng.uniform(3.0, 5.0, n),
        "review_count": rng.integers(10, 500, n).astype(float),
        "discount_pct": rng.uniform(0, 40, n),
        "availability": ["instock"] * n,
    })


def _mock_mlflow_run():
    ctx = MagicMock()
    ctx.__enter__ = lambda s: s
    ctx.__exit__ = lambda s, *a: None
    return ctx


# ── RandomForest ──────────────────────────────────────────────────────────────

def test_rf_mlflow_called_when_uri_set(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    (tmp_path / "analytics").mkdir()
    (tmp_path / "models").mkdir()

    with patch("src.ml.train_classifier.load_features", return_value=_features()), \
         patch("src.ml.train_classifier.mlflow") as m:
        m.start_run.return_value = _mock_mlflow_run()
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


# ── XGBoost ───────────────────────────────────────────────────────────────────

def test_xgb_mlflow_called_when_uri_set(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    (tmp_path / "analytics").mkdir()
    (tmp_path / "models").mkdir()

    with patch("src.ml.train_xgboost.load_features", return_value=_features()), \
         patch("src.ml.train_xgboost.mlflow") as m:
        m.start_run.return_value = _mock_mlflow_run()
        from src.ml import train_xgboost
        train_xgboost.run()

    m.set_tracking_uri.assert_called_once_with("http://localhost:5000")
    assert m.start_run.called


# ── KMeans ────────────────────────────────────────────────────────────────────

def test_kmeans_mlflow_called_when_uri_set(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    (tmp_path / "analytics").mkdir()

    with patch("src.ml.cluster_products.load_features", return_value=_features(30)), \
         patch("src.ml.cluster_products.mlflow") as m:
        m.start_run.return_value = _mock_mlflow_run()
        from src.ml import cluster_products
        cluster_products.run()

    m.set_tracking_uri.assert_called_once_with("http://localhost:5000")
    assert m.log_metric.called or m.log_metrics.called
```

- [ ] **Step 3: Run to verify FAIL**

```bash
docker compose run --rm app python -m pytest tests/test_mlflow_tracking.py -v 2>&1 | tail -20
```

Expected: FAIL — `mlflow` not imported in training modules

- [ ] **Step 4: Wire MLflow into `src/ml/train_classifier.py`**

Read the full file first. Add `import mlflow` and `import mlflow.sklearn` after the existing imports at the top. Add `import os` if not already there (check — it isn't currently).

At the very start of `run()`, before `out_dir = analytics_dir()`, add:

```python
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    use_mlflow = bool(tracking_uri)
    if use_mlflow:
        mlflow.set_tracking_uri(tracking_uri)
```

After `clf.fit(X, y)` and after `metrics["feature_importance"]` is computed (near the end of `run()`, before `joblib.dump`), add:

```python
    if use_mlflow:
        with mlflow.start_run(run_name="random_forest"):
            mlflow.log_params({
                "n_estimators": 300,
                "n_features": len(features),
                "n_samples": len(df),
                "cv_folds": cv.n_splits,
            })
            mlflow.log_metrics({
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
            })
            mlflow.sklearn.log_model(
                clf, "model", registered_model_name="rf_high_potential"
            )
```

- [ ] **Step 5: Wire MLflow into `src/ml/train_xgboost.py`**

Read the full file. Same pattern. Add at top of file:

```python
import os

import mlflow
import mlflow.xgboost as mlflow_xgb
```

At start of `run()` (after the `if not HAS_XGBOOST` guard):

```python
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    use_mlflow = bool(tracking_uri)
    if use_mlflow:
        mlflow.set_tracking_uri(tracking_uri)
```

After `clf.fit(X, y)` and metrics are fully computed, add:

```python
    if use_mlflow:
        with mlflow.start_run(run_name="xgboost"):
            mlflow.log_params({
                "n_estimators": 300,
                "learning_rate": 0.1,
                "n_features": len(features),
                "n_samples": len(df),
            })
            mlflow.log_metrics({
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
            })
            mlflow_xgb.log_model(
                clf, "model", registered_model_name="xgb_high_potential"
            )
```

Note: XGBoost's `run()` already has `n_estimators=300` and `learning_rate=0.1` hardcoded — find them in the existing code and match.

- [ ] **Step 6: Wire MLflow into `src/ml/cluster_products.py`**

Read the full file. Add imports:

```python
import os
import mlflow
```

At start of `run(n_clusters: int = 4)`:

```python
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    use_mlflow = bool(tracking_uri)
    if use_mlflow:
        mlflow.set_tracking_uri(tracking_uri)
```

After `sil = silhouette_score(X_scaled, df["cluster"])` and before `logger.info(...)`:

```python
    if use_mlflow:
        with mlflow.start_run(run_name="kmeans_clustering"):
            mlflow.log_param("n_clusters", n_clusters)
            mlflow.log_metric("silhouette_score", float(sil))
```

- [ ] **Step 7: Run tests**

```bash
docker compose run --rm app python -m pytest tests/test_mlflow_tracking.py -v 2>&1 | tail -20
```

Expected: 5/5 PASS

- [ ] **Step 8: Full suite — no regressions**

```bash
docker compose run --rm app python -m pytest tests/ -q --tb=short 2>&1 | tail -6
```

- [ ] **Step 9: Commit**

```bash
git add requirements.txt src/ml/train_classifier.py src/ml/train_xgboost.py src/ml/cluster_products.py tests/test_mlflow_tracking.py
git commit -m "feat: wire MLflow experiment tracking into RF, XGBoost, KMeans (opt-in via MLFLOW_TRACKING_URI)"
```

---

## Task 2: KFP v2 component fixes

**Files:**
- Modify: `src/pipeline/kubeflow_pipeline.py`
- Create: `tests/test_kfp_pipeline.py`

**Problems to fix:**
1. `base_image="smart-ecommerce-pipeline:local"` → `"smart-ecommerce-pipeline-v2-app:latest"`
2. `sys.path.append("/app")` in every component → remove (PYTHONPATH=/app set in Dockerfile)
3. No `data_dir` parameter → components can't be run with different data roots
4. No caching → every run re-executes even with unchanged inputs
5. No retry → transient failures cause permanent pipeline failure
6. No typed artifacts → KFP can't track data lineage between steps

- [ ] **Step 1: Write failing test**

```python
# tests/test_kfp_pipeline.py
import yaml
from pathlib import Path


def test_pipeline_compiles_without_error(tmp_path):
    from kfp import compiler
    from src.pipeline.kubeflow_pipeline import smart_ecommerce_pipeline

    out = tmp_path / "pipeline.yaml"
    compiler.Compiler().compile(
        pipeline_func=smart_ecommerce_pipeline,
        package_path=str(out),
    )
    assert out.exists()
    assert out.stat().st_size > 0


def test_pipeline_yaml_has_correct_components(tmp_path):
    from kfp import compiler
    from src.pipeline.kubeflow_pipeline import smart_ecommerce_pipeline

    out = tmp_path / "pipeline.yaml"
    compiler.Compiler().compile(
        pipeline_func=smart_ecommerce_pipeline,
        package_path=str(out),
    )
    spec = yaml.safe_load(out.read_text())
    executors = spec.get("deploymentSpec", {}).get("executors", {})
    # Must have 9 components: preprocess, dq (added in Task 3), features,
    # score, rf, xgb, kmeans, dbscan, rules
    # For Task 2 (before Task 3): 8 components
    assert len(executors) >= 8


def test_pipeline_base_image_is_correct(tmp_path):
    from kfp import compiler
    from src.pipeline.kubeflow_pipeline import smart_ecommerce_pipeline

    out = tmp_path / "pipeline.yaml"
    compiler.Compiler().compile(
        pipeline_func=smart_ecommerce_pipeline,
        package_path=str(out),
    )
    spec = yaml.safe_load(out.read_text())
    executors = spec.get("deploymentSpec", {}).get("executors", {})
    for name, executor in executors.items():
        image = executor.get("container", {}).get("image", "")
        assert "smart-ecommerce-pipeline:local" not in image, (
            f"Component {name} still uses old base image: {image}"
        )


def test_pipeline_no_sys_path_append(tmp_path):
    from kfp import compiler
    from src.pipeline.kubeflow_pipeline import smart_ecommerce_pipeline

    out = tmp_path / "pipeline.yaml"
    compiler.Compiler().compile(
        pipeline_func=smart_ecommerce_pipeline,
        package_path=str(out),
    )
    content = out.read_text()
    assert "sys.path.append" not in content, "sys.path.append found in compiled pipeline"


def test_pipeline_has_data_dir_parameter(tmp_path):
    from kfp import compiler
    from src.pipeline.kubeflow_pipeline import smart_ecommerce_pipeline

    out = tmp_path / "pipeline.yaml"
    compiler.Compiler().compile(
        pipeline_func=smart_ecommerce_pipeline,
        package_path=str(out),
    )
    spec = yaml.safe_load(out.read_text())
    # Pipeline root has input parameters
    root = spec.get("root", {})
    input_defs = root.get("inputDefinitions", {}).get("parameters", {})
    assert "data_dir" in input_defs, f"data_dir parameter not found. Got: {list(input_defs)}"
```

- [ ] **Step 2: Run to verify FAIL**

```bash
docker compose run --rm app python -m pytest tests/test_kfp_pipeline.py -v 2>&1 | tail -20
```

Expected: some tests FAIL (base_image and sys.path tests catch current issues)

- [ ] **Step 3: Rewrite `src/pipeline/kubeflow_pipeline.py`**

Replace the entire file content with:

```python
"""
Kubeflow Pipelines v2 — properly typed components with caching, retry, and data_dir param.

Changes from v1:
- base_image uses the image built by `docker compose build app`
- sys.path.append removed — PYTHONPATH=/app is set in the Dockerfile
- data_dir pipeline parameter passed to all components (overrides DATA_DIR env var)
- Typed dsl.Output[dsl.Dataset] artifacts on the preprocess→features chain
- Caching enabled on stable steps; retry=2 on all steps
- dq_op DQ gate added between preprocess and features (Task 3)

Scraping and LLM summary excluded from KFP (browser automation / API keys).
"""

from __future__ import annotations

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
def score_op(data_dir: str):
    import os

    os.environ["DATA_DIR"] = data_dir
    from src.scoring.topk import run

    run()


@dsl.component(base_image=_IMAGE)
def train_classifier_op(data_dir: str):
    import os

    os.environ["DATA_DIR"] = data_dir
    from src.ml.train_classifier import run

    run()


@dsl.component(base_image=_IMAGE)
def train_xgboost_op(data_dir: str):
    import os

    os.environ["DATA_DIR"] = data_dir
    from src.ml.train_xgboost import run

    run()


@dsl.component(base_image=_IMAGE)
def cluster_kmeans_op(data_dir: str):
    import os

    os.environ["DATA_DIR"] = data_dir
    from src.ml.cluster_products import run

    run()


@dsl.component(base_image=_IMAGE)
def cluster_dbscan_op(data_dir: str):
    import os

    os.environ["DATA_DIR"] = data_dir
    from src.ml.dbscan_products import run

    run()


@dsl.component(base_image=_IMAGE)
def association_rules_op(data_dir: str):
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
    """KFP v2 DAG with typed artifacts, caching, and retry."""
    p = (
        preprocess_op(data_dir=data_dir)
        .set_caching_options(enable_caching=False)   # data changes each run
        .set_retry(num_retries=2)
    )

    f = (
        features_op(data_dir=data_dir, processed=p.outputs["processed"])
        .set_caching_options(enable_caching=True)
        .set_retry(num_retries=2)
    )

    s = (
        score_op(data_dir=data_dir)
        .after(f)
        .set_caching_options(enable_caching=True)
        .set_retry(num_retries=2)
    )

    (
        train_classifier_op(data_dir=data_dir)
        .after(s)
        .set_caching_options(enable_caching=True)
        .set_retry(num_retries=2)
    )
    (
        train_xgboost_op(data_dir=data_dir)
        .after(s)
        .set_caching_options(enable_caching=True)
        .set_retry(num_retries=2)
    )
    (
        cluster_kmeans_op(data_dir=data_dir)
        .after(f)
        .set_caching_options(enable_caching=True)
        .set_retry(num_retries=2)
    )
    (
        cluster_dbscan_op(data_dir=data_dir)
        .after(f)
        .set_caching_options(enable_caching=True)
        .set_retry(num_retries=2)
    )
    (
        association_rules_op(data_dir=data_dir)
        .after(f)
        .set_caching_options(enable_caching=True)
        .set_retry(num_retries=2)
    )


def run() -> None:
    """Entry point used when calling this module as a script."""
    print(
        "Kubeflow pipeline defined as `smart_ecommerce_pipeline`.\n"
        "Compile: make compile-kfp\n"
        "Run:     make kfp-operator"
    )
```

- [ ] **Step 4: Run tests**

```bash
docker compose run --rm app python -m pytest tests/test_kfp_pipeline.py -v 2>&1 | tail -20
```

Expected: 5/5 PASS

- [ ] **Step 5: Full suite**

```bash
docker compose run --rm app python -m pytest tests/ -q --tb=short 2>&1 | tail -6
```

- [ ] **Step 6: Commit**

```bash
git add src/pipeline/kubeflow_pipeline.py tests/test_kfp_pipeline.py
git commit -m "fix: KFP v2 components — typed artifacts, data_dir param, correct base image, caching, retry"
```

---

## Task 3: Great Expectations DQ gate

**Files:**
- Create: `src/pipeline/dq_step.py`
- Modify: `src/pipeline/kubeflow_pipeline.py` (add `dq_op`, wire into DAG)
- Create: `tests/test_dq_step.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_dq_step.py
import pandas as pd
import pytest
from pathlib import Path


def _write_valid_parquet(path: Path) -> None:
    df = pd.DataFrame([
        {
            "product_id": str(i),
            "title": f"Product {i}",
            "source_platform": "shopify" if i % 2 == 0 else "woocommerce",
            "dq_score": 0.8,
            "price": 29.99,
            "availability": "instock",
        }
        for i in range(20)
    ])
    df.to_parquet(path, index=False)


def _write_invalid_parquet(path: Path) -> None:
    """Missing required columns."""
    df = pd.DataFrame([{"price": 9.99}] * 5)
    df.to_parquet(path, index=False)


def _write_empty_parquet(path: Path) -> None:
    pd.DataFrame(
        columns=["product_id", "title", "source_platform", "dq_score"]
    ).to_parquet(path, index=False)


def test_validate_passes_on_valid_data(tmp_path):
    parquet = tmp_path / "cleaned_products.parquet"
    _write_valid_parquet(parquet)

    from src.pipeline.dq_step import validate_cleaned_products
    assert validate_cleaned_products(str(parquet)) is True


def test_validate_fails_on_missing_required_columns(tmp_path):
    parquet = tmp_path / "cleaned_products.parquet"
    _write_invalid_parquet(parquet)

    from src.pipeline.dq_step import validate_cleaned_products
    assert validate_cleaned_products(str(parquet)) is False


def test_validate_fails_on_empty_dataframe(tmp_path):
    parquet = tmp_path / "cleaned_products.parquet"
    _write_empty_parquet(parquet)

    from src.pipeline.dq_step import validate_cleaned_products
    assert validate_cleaned_products(str(parquet)) is False


def test_run_or_raise_raises_on_invalid_data(tmp_path, monkeypatch):
    parquet = tmp_path / "cleaned_products.parquet"
    _write_invalid_parquet(parquet)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    from src.pipeline.dq_step import run_or_raise
    with pytest.raises(RuntimeError, match="DQ validation failed"):
        run_or_raise()


def test_run_or_raise_noop_on_valid_data(tmp_path, monkeypatch):
    parquet = tmp_path / "processed" / "cleaned_products.parquet"
    parquet.parent.mkdir()
    _write_valid_parquet(parquet)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    from src.pipeline.dq_step import run_or_raise
    run_or_raise()   # must not raise
```

- [ ] **Step 2: Run to verify FAIL**

```bash
docker compose run --rm app python -m pytest tests/test_dq_step.py -v 2>&1 | tail -15
```

Expected: FAIL — `src.pipeline.dq_step` not found

- [ ] **Step 3: Create `src/pipeline/dq_step.py`**

```python
"""
Great Expectations DQ gate for cleaned_products.parquet.
Validates 8 expectations; raises RuntimeError on failure (KFP hard-stop).
"""
from __future__ import annotations

from pathlib import Path

from src.config import get_logger, processed_dir

logger = get_logger(__name__)


def validate_cleaned_products(parquet_path: str | None = None) -> bool:
    """Run GE expectations. Returns True if all pass, False otherwise."""
    import great_expectations as gx
    import pandas as pd

    path = Path(parquet_path) if parquet_path else processed_dir() / "cleaned_products.parquet"
    if not path.exists():
        logger.error("Parquet not found: %s", path)
        return False

    df = pd.read_parquet(path)

    context = gx.get_context(mode="ephemeral")
    source = context.data_sources.add_pandas("source")
    asset = source.add_dataframe_asset("products")
    batch_def = asset.add_batch_definition_whole_dataframe("batch")

    suite = context.suites.add(gx.ExpectationSuite(name="products_suite"))
    suite.add_expectation(gx.expectations.ExpectTableRowCountToBeGreaterThan(value=0))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="product_id"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="title"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="source_platform"))
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="product_id", mostly=0.99)
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="title", mostly=0.99)
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="dq_score", min_value=0.0, max_value=1.0, mostly=0.95
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="source_platform", value_set=["shopify", "woocommerce"]
        )
    )

    validation_def = context.validation_definitions.add(
        gx.ValidationDefinition(
            name="validate_cleaned_products",
            data=batch_def,
            suite=suite,
        )
    )
    result = validation_def.run(batch_parameters={"dataframe": df})

    if not result.success:
        failed = [r for r in result.results if not r.success]
        logger.error("DQ validation failed — %d/%d expectations failed:", len(failed), len(result.results))
        for r in failed:
            logger.error("  FAIL: %s", r.expectation_config.type)
    else:
        logger.info("DQ validation passed (%d expectations)", len(result.results))

    return bool(result.success)


def run_or_raise(parquet_path: str | None = None) -> None:
    """Raises RuntimeError on DQ failure. Use as KFP step entry point."""
    if not validate_cleaned_products(parquet_path):
        raise RuntimeError(
            "DQ validation failed — pipeline stopped. "
            "Check logs for failed expectations."
        )


if __name__ == "__main__":
    run_or_raise()
```

- [ ] **Step 4: Run tests — expect FAIL or partial PASS**

```bash
docker compose run --rm app python -m pytest tests/test_dq_step.py -v 2>&1 | tail -20
```

If GE API differs from the implementation above (GE 1.x API may vary), the implementer must adapt the API calls to match the installed version. Run:

```bash
docker compose run --rm app python -c "import great_expectations as gx; print(gx.__version__)"
```

And check the GE docs for the installed version. The key functions are:
- `gx.get_context(mode="ephemeral")` — creates in-memory context
- `context.data_sources.add_pandas(name)` — add pandas data source
- `source.add_dataframe_asset(name)` — asset
- `asset.add_batch_definition_whole_dataframe(name)` — batch def
- `context.suites.add(gx.ExpectationSuite(name=...))` — create suite
- `suite.add_expectation(gx.expectations.Expect...)` — add expectation
- `context.validation_definitions.add(gx.ValidationDefinition(...))` — create validation
- `validation_def.run(batch_parameters={"dataframe": df})` — run

- [ ] **Step 5: Add `dq_op` to `src/pipeline/kubeflow_pipeline.py`**

Read the current file. Add the new component after `preprocess_op` (before `features_op`):

```python
@dsl.component(base_image=_IMAGE)
def dq_op(data_dir: str):
    """Great Expectations DQ gate — fails hard if cleaned_products.parquet is invalid."""
    import os

    os.environ["DATA_DIR"] = data_dir
    from src.pipeline.dq_step import run_or_raise

    run_or_raise()
```

Update `smart_ecommerce_pipeline()` — insert `dq` step between `p` (preprocess) and `f` (features):

```python
    p = (
        preprocess_op(data_dir=data_dir)
        .set_caching_options(enable_caching=False)
        .set_retry(num_retries=2)
    )

    dq = (
        dq_op(data_dir=data_dir)
        .after(p)
        .set_caching_options(enable_caching=False)
        .set_retry(num_retries=1)
    )

    f = (
        features_op(data_dir=data_dir, processed=p.outputs["processed"])
        .after(dq)
        .set_caching_options(enable_caching=True)
        .set_retry(num_retries=2)
    )
```

- [ ] **Step 6: Update the component count test in `test_kfp_pipeline.py`**

Find this assertion in `test_pipeline_yaml_has_correct_components`:
```python
    assert len(executors) >= 8
```
Change to:
```python
    assert len(executors) >= 9
```

- [ ] **Step 7: Run all new tests**

```bash
docker compose run --rm app python -m pytest tests/test_dq_step.py tests/test_kfp_pipeline.py -v 2>&1 | tail -20
```

Expected: all PASS

- [ ] **Step 8: Full suite**

```bash
docker compose run --rm app python -m pytest tests/ -q --tb=short 2>&1 | tail -6
```

- [ ] **Step 9: Compile KFP pipeline to verify YAML is valid**

```bash
docker compose run --rm app make compile-kfp 2>&1 | tail -5
```

Expected: `✓ Compiled kubeflow_smart_ecommerce_pipeline.yaml`

- [ ] **Step 10: Commit**

```bash
git add src/pipeline/dq_step.py src/pipeline/kubeflow_pipeline.py tests/test_dq_step.py tests/test_kfp_pipeline.py requirements.txt
git commit -m "feat: Great Expectations DQ gate as KFP step (hard-stop on validation failure)"
```

---

## Self-Review

**Spec coverage:**
- [x] MLflow experiment tracking — Task 1 (RF, XGBoost, KMeans)
- [x] KFP v2 typed artifacts — Task 2 (`dsl.Output[dsl.Dataset]` on preprocess→features)
- [x] KFP caching — Task 2 (`set_caching_options`)
- [x] KFP retry — Task 2 (`set_retry`)
- [x] Fix sys.path hack — Task 2 (removed from all 8 components)
- [x] Fix base_image — Task 2 (`smart-ecommerce-pipeline-v2-app:latest`)
- [x] Great Expectations DQ gate — Task 3 (8 expectations, hard-fail)
- [x] DQ as KFP step — Task 3 (`dq_op` in DAG between preprocess and features)

**Placeholder scan:** None. All code blocks are complete and runnable.

**Type consistency:**
- `validate_cleaned_products(parquet_path: str | None = None) -> bool` — defined Task 3, used in `run_or_raise()` and tests
- `run_or_raise(parquet_path: str | None = None) -> None` — defined Task 3, used in `dq_op` KFP component
- `dq_op(data_dir: str)` — defined and wired in Task 3
- MLflow `use_mlflow` guard pattern consistent across all 3 training modules
- `_IMAGE = "smart-ecommerce-pipeline-v2-app:latest"` — module-level constant used in all components

**Gap check:**
- DBSCAN clustering (`src/ml/dbscan_products.py`) is NOT wired for MLflow — it doesn't compute a named metric (it uses epsilon/min_samples heuristics). Out of scope per spec.
- Association rules (`src/ml/rules.py`) — same, no obvious scalar metric to log. Out of scope.
- `data_dir` defaults to `"/app/data"` in the pipeline. Local runs use `DATA_DIR` env var. Both paths work.
- GE 1.x API used in `dq_step.py` — implementer must verify exact method names against installed version.
