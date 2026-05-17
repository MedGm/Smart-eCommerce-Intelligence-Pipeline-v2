# Phase 1 — Data Lake Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire MinIO as primary object storage for raw scraping output and processed artifacts, add DuckDB as the analytical query layer, and bootstrap a dbt project with three SQL models (staging + two marts) so all pipeline data flows through a proper data lake.

**Architecture:** `BaseScraper.save()` dual-writes: local filesystem first (existing tests keep working), then uploads to MinIO `raw-data` bucket when `MINIO_ENDPOINT` is set. Preprocessing syncs raw data down from MinIO before loading, then uploads artifacts back to `processed` bucket. A `duckdb_client.py` module reads cleaned Parquet from MinIO (or local fallback) and registers it as a DuckDB table in `data/warehouse.duckdb`. dbt models run against that file to produce `stg_products`, `mart_products_clean`, and `mart_top_products`.

**Tech Stack:** Python 3.11, boto3 (S3/MinIO client), duckdb>=0.10.0, dbt-core>=1.7.0, dbt-duckdb>=1.7.0, MinIO (running via `docker compose --profile infra up -d`)

---

## File Map

| File | Change |
|------|--------|
| `src/storage/__init__.py` | New — package marker |
| `src/storage/minio_client.py` | New — boto3 wrapper (upload, download, list, sync) |
| `src/storage/duckdb_client.py` | New — DuckDB over MinIO/local Parquet |
| `src/scraping/base.py` | Modify `save()` — upload to MinIO after local write |
| `src/preprocessing/run.py` | Add `_sync_raw_from_minio()` at start of `run()`, `_upload_processed()` at end |
| `dbt/dbt_project.yml` | New |
| `dbt/profiles.yml` | New |
| `dbt/models/staging/sources.yml` | New |
| `dbt/models/staging/stg_products.sql` | New |
| `dbt/models/staging/schema.yml` | New |
| `dbt/models/marts/mart_products_clean.sql` | New |
| `dbt/models/marts/mart_top_products.sql` | New |
| `dbt/models/marts/schema.yml` | New |
| `requirements.txt` | Add boto3>=1.34.0, duckdb>=0.10.0, dbt-core>=1.7.0, dbt-duckdb>=1.7.0 |
| `Makefile` | Add `warehouse` and `dbt-run` targets |
| `tests/test_minio_client.py` | New — upload/download/list/sync unit tests |
| `tests/test_minio_upload.py` | New — BaseScraper uploads to MinIO when env set |
| `tests/test_duckdb_client.py` | New — DuckDB load + query |

---

## Task 1: MinIO storage client

**Files:**
- Create: `src/storage/__init__.py`
- Create: `src/storage/minio_client.py`
- Modify: `requirements.txt`
- Create: `tests/test_minio_client.py`

- [ ] **Step 1: Add dependencies to requirements.txt**

Open `requirements.txt`. After the `pyyaml>=6.0` line, add:

```
boto3>=1.34.0
duckdb>=0.10.0
dbt-core>=1.7.0
dbt-duckdb>=1.7.0
```

Install them:

```bash
pip install boto3 duckdb dbt-core dbt-duckdb
python3 -c "import boto3; import duckdb; print('OK')"
```

Expected: `OK`

- [ ] **Step 2: Create storage package**

```bash
touch src/storage/__init__.py
```

- [ ] **Step 3: Write failing tests**

```python
# tests/test_minio_client.py
import os
from pathlib import Path
from unittest.mock import MagicMock, call, patch


def test_is_minio_configured_false_when_no_env(monkeypatch):
    monkeypatch.delenv("MINIO_ENDPOINT", raising=False)
    from src.storage.minio_client import is_minio_configured
    assert not is_minio_configured()


def test_is_minio_configured_true_when_env_set(monkeypatch):
    monkeypatch.setenv("MINIO_ENDPOINT", "http://localhost:9000")
    from src.storage.minio_client import is_minio_configured
    assert is_minio_configured()


def test_upload_file_calls_s3_upload(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIO_ENDPOINT", "http://localhost:9000")
    local_file = tmp_path / "test.json"
    local_file.write_text('{"x": 1}')

    mock_client = MagicMock()
    with patch("src.storage.minio_client._client", return_value=mock_client):
        from src.storage.minio_client import upload_file
        upload_file(local_file, bucket="raw-data", key="raw/test.json")

    mock_client.upload_file.assert_called_once_with(
        str(local_file), "raw-data", "raw/test.json"
    )


def test_upload_file_noop_when_not_configured(tmp_path, monkeypatch):
    monkeypatch.delenv("MINIO_ENDPOINT", raising=False)
    local_file = tmp_path / "test.json"
    local_file.write_text("{}")

    mock_client = MagicMock()
    with patch("src.storage.minio_client._client", return_value=mock_client):
        from src.storage.minio_client import upload_file
        upload_file(local_file, bucket="raw-data", key="test.json")

    mock_client.upload_file.assert_not_called()


def test_list_objects_returns_keys(monkeypatch):
    monkeypatch.setenv("MINIO_ENDPOINT", "http://localhost:9000")
    mock_paginator = MagicMock()
    mock_paginator.paginate.return_value = [
        {"Contents": [{"Key": "raw/a.json"}, {"Key": "raw/b.json"}]},
        {"Contents": [{"Key": "raw/c.json"}]},
    ]
    mock_client = MagicMock()
    mock_client.get_paginator.return_value = mock_paginator

    with patch("src.storage.minio_client._client", return_value=mock_client):
        from src.storage.minio_client import list_objects
        keys = list_objects("raw-data", prefix="raw/")

    assert keys == ["raw/a.json", "raw/b.json", "raw/c.json"]


def test_list_objects_returns_empty_when_not_configured(monkeypatch):
    monkeypatch.delenv("MINIO_ENDPOINT", raising=False)
    from src.storage.minio_client import list_objects
    assert list_objects("raw-data") == []


def test_sync_to_local_downloads_objects(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIO_ENDPOINT", "http://localhost:9000")
    mock_client = MagicMock()
    mock_paginator = MagicMock()
    mock_paginator.paginate.return_value = [
        {"Contents": [{"Key": "raw/shopify/ruggable/run1.json"}]},
    ]
    mock_client.get_paginator.return_value = mock_paginator

    with patch("src.storage.minio_client._client", return_value=mock_client):
        from src.storage.minio_client import sync_to_local
        downloaded = sync_to_local(
            bucket="raw-data", prefix="raw/", local_dir=tmp_path
        )

    assert len(downloaded) == 1
    assert downloaded[0] == tmp_path / "shopify/ruggable/run1.json"
    mock_client.download_file.assert_called_once_with(
        "raw-data",
        "raw/shopify/ruggable/run1.json",
        str(tmp_path / "shopify/ruggable/run1.json"),
    )
```

- [ ] **Step 4: Run to verify FAIL**

```bash
python3 -m pytest tests/test_minio_client.py -v 2>&1 | tail -15
```

Expected: FAIL — `src.storage.minio_client` not found

- [ ] **Step 5: Create `src/storage/minio_client.py`**

```python
"""
MinIO / S3-compatible storage client.
All operations are no-ops when MINIO_ENDPOINT is not set — local dev works unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path

import boto3

from src.config import get_logger

logger = get_logger(__name__)


def is_minio_configured() -> bool:
    return bool(os.environ.get("MINIO_ENDPOINT"))


def _client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("MINIO_ENDPOINT"),
        aws_access_key_id=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
        aws_secret_access_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
    )


def upload_file(local_path: Path, bucket: str, key: str) -> None:
    if not is_minio_configured():
        return
    try:
        _client().upload_file(str(local_path), bucket, key)
        logger.debug("Uploaded %s → s3://%s/%s", local_path, bucket, key)
    except Exception as exc:
        logger.warning("MinIO upload failed for %s: %s", local_path, exc)


def download_file(bucket: str, key: str, local_path: Path) -> None:
    if not is_minio_configured():
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    _client().download_file(bucket, key, str(local_path))
    logger.debug("Downloaded s3://%s/%s → %s", bucket, key, local_path)


def list_objects(bucket: str, prefix: str = "") -> list[str]:
    if not is_minio_configured():
        return []
    paginator = _client().get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def sync_to_local(bucket: str, prefix: str, local_dir: Path) -> list[Path]:
    """Download all objects matching prefix into local_dir, preserving key sub-path."""
    if not is_minio_configured():
        return []
    local_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for key in list_objects(bucket, prefix):
        rel = key[len(prefix):].lstrip("/")
        dest = local_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        _client().download_file(bucket, key, str(dest))
        downloaded.append(dest)
    return downloaded
```

- [ ] **Step 6: Run tests**

```bash
python3 -m pytest tests/test_minio_client.py -v 2>&1 | tail -15
```

Expected: 7/7 PASS

- [ ] **Step 7: Run full suite — no regressions**

```bash
python3 -m pytest tests/ -q --tb=short --ignore=tests/test_cleaning.py --ignore=tests/test_cluster_metrics.py --ignore=tests/test_features.py --ignore=tests/test_ml.py --ignore=tests/test_model_persistence.py --ignore=tests/test_rules_vectorized.py --ignore=tests/test_scoring.py --ignore=tests/test_validate.py 2>&1 | tail -6
```

Expected: all pass, 0 regressions

- [ ] **Step 8: Commit**

```bash
git add requirements.txt src/storage/__init__.py src/storage/minio_client.py tests/test_minio_client.py
git commit -m "feat: add MinIO storage client with upload/download/list/sync (no-op when MINIO_ENDPOINT unset)"
```

---

## Task 2: Wire scraping output to MinIO

**Files:**
- Modify: `src/scraping/base.py`
- Create: `tests/test_minio_upload.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_minio_upload.py
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile

from src.scraping.base import ProductRecord


def _record():
    return ProductRecord(
        source_platform="shopify",
        shop_name="TestShop",
        product_id="1",
        product_url="https://example.com/p/1",
        title="Test",
        description="Desc",
        category="Cat",
        brand="Brand",
        price=9.99,
        old_price=None,
        availability="instock",
        rating=4.5,
        review_count=10,
        geography="US",
        scraped_at="2026-05-17T00:00:00Z",
    )


def test_save_uploads_to_minio_when_configured(monkeypatch):
    monkeypatch.setenv("MINIO_ENDPOINT", "http://localhost:9000")

    with tempfile.TemporaryDirectory() as tmp:
        from src.scraping.base import BaseScraper
        scraper = BaseScraper.__new__(BaseScraper)
        scraper.output_dir = Path(tmp)
        scraper.run_id = "20260517T130000Z"

        with patch("src.storage.minio_client.upload_file") as mock_upload, \
             patch("src.storage.minio_client.is_minio_configured", return_value=True):
            path = scraper.save([_record()], filename="ruggable.json")

        assert mock_upload.called
        _, kwargs = mock_upload.call_args_list[0][0], mock_upload.call_args_list[0][1]
        # positional: (local_path, bucket, key)
        call_args = mock_upload.call_args_list[0][0]
        assert call_args[0] == path       # local path
        assert call_args[1] == "raw-data" # bucket
        assert "ruggable" in call_args[2] # key contains store name


def test_save_no_upload_when_not_configured(monkeypatch):
    monkeypatch.delenv("MINIO_ENDPOINT", raising=False)

    with tempfile.TemporaryDirectory() as tmp:
        from src.scraping.base import BaseScraper
        scraper = BaseScraper.__new__(BaseScraper)
        scraper.output_dir = Path(tmp)
        scraper.run_id = "20260517T130000Z"

        with patch("src.storage.minio_client.upload_file") as mock_upload, \
             patch("src.storage.minio_client.is_minio_configured", return_value=False):
            scraper.save([_record()], filename="ruggable.json")

        mock_upload.assert_not_called()


def test_save_local_file_always_written(monkeypatch):
    monkeypatch.delenv("MINIO_ENDPOINT", raising=False)

    with tempfile.TemporaryDirectory() as tmp:
        from src.scraping.base import BaseScraper
        scraper = BaseScraper.__new__(BaseScraper)
        scraper.output_dir = Path(tmp)
        scraper.run_id = None

        path = scraper.save([_record()], filename="ruggable.json")

        assert path.exists()
        data = json.loads(path.read_text())
        assert data[0]["title"] == "Test"
```

- [ ] **Step 2: Run to verify FAIL**

```bash
python3 -m pytest tests/test_minio_upload.py -v 2>&1 | tail -15
```

Expected: FAIL — `save()` doesn't call `upload_file`

- [ ] **Step 3: Read current `src/scraping/base.py`**

Read the file. The `save()` method ends with `return path`. Add the MinIO upload immediately before that return.

- [ ] **Step 4: Modify `BaseScraper.save()` in `src/scraping/base.py`**

Add these two imports at the top of the file (after existing imports):

```python
from src.config import data_dir as _data_dir
```

Modify the `save()` method — append the upload block before `return path`:

```python
    def save(self, records: list[ProductRecord], filename: str = "products.json") -> Path:
        if self.run_id:
            stem = Path(filename).stem
            suffix = Path(filename).suffix
            dest_dir = self.output_dir / stem
            dest_dir.mkdir(parents=True, exist_ok=True)
            path = dest_dir / f"{self.run_id}{suffix}"
        else:
            path = self.output_dir / filename
        data = [r.to_dict() for r in records]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Upload to MinIO when configured — falls back silently if unavailable
        from src.storage.minio_client import is_minio_configured, upload_file
        if is_minio_configured():
            try:
                key = str(path.relative_to(_data_dir()))
            except ValueError:
                key = path.name
            upload_file(path, bucket="raw-data", key=key)

        return path
```

- [ ] **Step 5: Run tests**

```bash
python3 -m pytest tests/test_minio_upload.py tests/test_partitioned_output.py -v 2>&1 | tail -15
```

Expected: all PASS (including existing partitioned output tests — local write unchanged)

- [ ] **Step 6: Run full suite**

```bash
python3 -m pytest tests/ -q --tb=short --ignore=tests/test_cleaning.py --ignore=tests/test_cluster_metrics.py --ignore=tests/test_features.py --ignore=tests/test_ml.py --ignore=tests/test_model_persistence.py --ignore=tests/test_rules_vectorized.py --ignore=tests/test_scoring.py --ignore=tests/test_validate.py 2>&1 | tail -6
```

- [ ] **Step 7: Commit**

```bash
git add src/scraping/base.py tests/test_minio_upload.py
git commit -m "feat: dual-write scraping output to MinIO raw-data bucket when MINIO_ENDPOINT is set"
```

---

## Task 3: Wire preprocessing to MinIO

**Files:**
- Modify: `src/preprocessing/run.py`

No new test file needed — the MinIO calls are guarded by `is_minio_configured()`. The existing preprocessing tests remain valid. We add two private helpers called inside `run()`.

- [ ] **Step 1: Read `src/preprocessing/run.py` fully** (it is long — understand `run()` entry point and `load_raw()`)

- [ ] **Step 2: Add two private helpers at the bottom of `src/preprocessing/run.py`**

Add after all existing function definitions, before `if __name__ == "__main__":`:

```python
def _sync_raw_from_minio(root=None) -> None:
    """Download raw JSONs from MinIO raw-data/raw/ into data/raw/ before preprocessing."""
    from src.storage.minio_client import is_minio_configured, sync_to_local
    if not is_minio_configured():
        return
    root = root or data_dir()
    count = sync_to_local(bucket="raw-data", prefix="raw/", local_dir=root / "raw")
    logger.info("Synced %d raw files from MinIO to %s/raw/", len(count), root)


def _upload_processed(p_dir: Path) -> None:
    """Upload processed artifacts to MinIO processed/ bucket."""
    from src.storage.minio_client import is_minio_configured, upload_file
    if not is_minio_configured():
        return
    for fname in (
        "cleaned_products.parquet",
        "dq_counters.json",
        "run_metadata.json",
        "field_failure_samples.json",
    ):
        path = p_dir / fname
        if path.exists():
            upload_file(path, bucket="processed", key=fname)
    logger.info("Uploaded processed artifacts to MinIO processed/ bucket")
```

- [ ] **Step 3: Call helpers inside `run()`**

At the very start of `run()`, before `root = data_dir()`, add:

```python
    _sync_raw_from_minio()
```

At the very end of `run()`, after `logger.info("Preprocessing done: %d rows -> %s", len(df), out_path)` and before `return df`, add:

```python
    _upload_processed(p_dir)
```

- [ ] **Step 4: Run full suite — no regressions**

```bash
python3 -m pytest tests/ -q --tb=short --ignore=tests/test_cleaning.py --ignore=tests/test_cluster_metrics.py --ignore=tests/test_features.py --ignore=tests/test_ml.py --ignore=tests/test_model_persistence.py --ignore=tests/test_rules_vectorized.py --ignore=tests/test_scoring.py --ignore=tests/test_validate.py 2>&1 | tail -6
```

Expected: no regressions (helpers are no-ops when MINIO_ENDPOINT not set)

- [ ] **Step 5: Commit**

```bash
git add src/preprocessing/run.py
git commit -m "feat: sync raw data from MinIO before preprocessing, upload artifacts after"
```

---

## Task 4: DuckDB analytics client

**Files:**
- Create: `src/storage/duckdb_client.py`
- Create: `tests/test_duckdb_client.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_duckdb_client.py
import json
import os
from pathlib import Path
import tempfile

import pandas as pd
import pytest


def _make_parquet(tmp_dir: Path) -> Path:
    """Create a minimal cleaned_products.parquet for testing."""
    df = pd.DataFrame([
        {
            "source_platform": "shopify",
            "shop_name": "Ruggable",
            "product_id": "1",
            "product_url": "https://ruggable.com/p/1",
            "title": "Blue Rug",
            "description": "Nice rug",
            "category": "Rugs",
            "brand": "Ruggable",
            "price": 89.0,
            "old_price": 120.0,
            "availability": "instock",
            "rating": 4.8,
            "review_count": 200,
            "geography": "US",
            "scraped_at": "2026-05-17T00:00:00Z",
            "dq_score": 0.9,
        }
    ])
    path = tmp_dir / "cleaned_products.parquet"
    df.to_parquet(path, index=False)
    return path


def test_load_products_local_returns_dataframe(tmp_path, monkeypatch):
    monkeypatch.delenv("MINIO_ENDPOINT", raising=False)
    parquet = _make_parquet(tmp_path)
    monkeypatch.setenv("DUCKDB_PATH", str(tmp_path / "test.duckdb"))
    # Point processed_dir to tmp_path
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    # Write to expected location
    (tmp_path / "processed").mkdir()
    import shutil
    shutil.copy(parquet, tmp_path / "processed" / "cleaned_products.parquet")

    from src.storage.duckdb_client import load_products
    df = load_products(source="local")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["title"] == "Blue Rug"


def test_query_runs_sql(tmp_path, monkeypatch):
    monkeypatch.delenv("MINIO_ENDPOINT", raising=False)
    monkeypatch.setenv("DUCKDB_PATH", str(tmp_path / "test.duckdb"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    (tmp_path / "processed").mkdir()
    import shutil
    shutil.copy(_make_parquet(tmp_path), tmp_path / "processed" / "cleaned_products.parquet")

    from src.storage.duckdb_client import load_products, query
    load_products(source="local")
    result = query("SELECT COUNT(*) AS n FROM products")
    assert result.iloc[0]["n"] == 1


def test_load_products_skips_minio_when_not_configured(tmp_path, monkeypatch):
    monkeypatch.delenv("MINIO_ENDPOINT", raising=False)
    monkeypatch.setenv("DUCKDB_PATH", str(tmp_path / "test.duckdb"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    (tmp_path / "processed").mkdir()
    import shutil
    shutil.copy(_make_parquet(tmp_path), tmp_path / "processed" / "cleaned_products.parquet")

    # Should use local path, not raise
    from src.storage.duckdb_client import load_products
    df = load_products(source="auto")
    assert len(df) == 1
```

- [ ] **Step 2: Run to verify FAIL**

```bash
python3 -m pytest tests/test_duckdb_client.py -v 2>&1 | tail -15
```

Expected: FAIL — `src.storage.duckdb_client` not found

- [ ] **Step 3: Create `src/storage/duckdb_client.py`**

```python
"""
DuckDB analytics layer.
Reads cleaned Parquet from MinIO (s3://) or local filesystem,
exposes a SQL query interface via a persistent warehouse.duckdb file.
"""
from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd

from src.config import data_dir, get_logger, processed_dir

logger = get_logger(__name__)


def _warehouse_path() -> Path:
    env = os.environ.get("DUCKDB_PATH", "")
    return Path(env) if env else data_dir() / "warehouse.duckdb"


def _conn() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(str(_warehouse_path()))
    endpoint = os.environ.get("MINIO_ENDPOINT", "")
    if endpoint:
        host = endpoint.replace("http://", "").replace("https://", "")
        access_key = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
        secret_key = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
        conn.execute(f"""
            INSTALL httpfs; LOAD httpfs;
            SET s3_endpoint='{host}';
            SET s3_access_key_id='{access_key}';
            SET s3_secret_access_key='{secret_key}';
            SET s3_use_ssl=false;
            SET s3_url_style='path';
        """)
    return conn


def load_products(source: str = "auto") -> pd.DataFrame:
    """
    Load cleaned products into DuckDB `products` table and return as DataFrame.

    source="auto"  — MinIO if MINIO_ENDPOINT set, else local Parquet
    source="minio" — s3://processed/cleaned_products.parquet
    source="local" — data/processed/cleaned_products.parquet
    """
    minio_configured = bool(os.environ.get("MINIO_ENDPOINT"))

    if source == "minio" or (source == "auto" and minio_configured):
        parquet_path = "s3://processed/cleaned_products.parquet"
    else:
        parquet_path = str(processed_dir() / "cleaned_products.parquet")

    conn = _conn()
    conn.execute(
        f"CREATE OR REPLACE TABLE products AS SELECT * FROM read_parquet('{parquet_path}')"
    )
    logger.info("Loaded products table from %s into %s", parquet_path, _warehouse_path())
    return conn.execute("SELECT * FROM products").df()


def query(sql: str) -> pd.DataFrame:
    """Run arbitrary SQL against warehouse.duckdb and return as DataFrame."""
    return _conn().execute(sql).df()
```

- [ ] **Step 4: Run tests**

```bash
python3 -m pytest tests/test_duckdb_client.py -v 2>&1 | tail -15
```

Expected: 3/3 PASS

- [ ] **Step 5: Add Makefile targets**

Read `Makefile`. After the `dashboard:` target, add:

```makefile
warehouse:
	$(PYTHON) -c "from src.storage.duckdb_client import load_products; load_products(); print('warehouse.duckdb ready')"

dbt-run: warehouse
	cd dbt && dbt run --profiles-dir .

dbt-test: warehouse
	cd dbt && dbt test --profiles-dir .
```

- [ ] **Step 6: Run full suite — no regressions**

```bash
python3 -m pytest tests/ -q --tb=short --ignore=tests/test_cleaning.py --ignore=tests/test_cluster_metrics.py --ignore=tests/test_features.py --ignore=tests/test_ml.py --ignore=tests/test_model_persistence.py --ignore=tests/test_rules_vectorized.py --ignore=tests/test_scoring.py --ignore=tests/test_validate.py 2>&1 | tail -6
```

- [ ] **Step 7: Commit**

```bash
git add src/storage/duckdb_client.py tests/test_duckdb_client.py Makefile
git commit -m "feat: add DuckDB analytics client reading from MinIO or local Parquet"
```

---

## Task 5: dbt project

**Files:**
- Create: `dbt/dbt_project.yml`
- Create: `dbt/profiles.yml`
- Create: `dbt/models/staging/sources.yml`
- Create: `dbt/models/staging/stg_products.sql`
- Create: `dbt/models/staging/schema.yml`
- Create: `dbt/models/marts/mart_products_clean.sql`
- Create: `dbt/models/marts/mart_top_products.sql`
- Create: `dbt/models/marts/schema.yml`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p dbt/models/staging dbt/models/marts
```

- [ ] **Step 2: Create `dbt/dbt_project.yml`**

```yaml
name: ecommerce
version: "1.0.0"
config-version: 2

profile: ecommerce

model-paths: ["models"]
analysis-paths: ["analyses"]
test-paths: ["tests"]
seed-paths: ["seeds"]
macro-paths: ["macros"]

target-path: "target"
clean-targets: ["target", "dbt_packages"]

models:
  ecommerce:
    staging:
      +materialized: view
    marts:
      +materialized: table
```

- [ ] **Step 3: Create `dbt/profiles.yml`**

```yaml
ecommerce:
  target: dev
  outputs:
    dev:
      type: duckdb
      path: "{{ env_var('DUCKDB_PATH', '../data/warehouse.duckdb') }}"
      schema: main
      threads: 1
```

- [ ] **Step 4: Create `dbt/models/staging/sources.yml`**

```yaml
version: 2

sources:
  - name: warehouse
    schema: main
    description: "DuckDB warehouse populated by duckdb_client.load_products()"
    tables:
      - name: products
        description: "Cleaned product records from preprocessing pipeline"
        columns:
          - name: product_id
            description: "Unique product identifier (platform-scoped)"
          - name: source_platform
            description: "shopify or woocommerce"
          - name: shop_name
            description: "Store name from stores.yaml"
          - name: price
            description: "Current price in local currency"
          - name: dq_score
            description: "Data quality score 0.0–1.0 from preprocessing"
```

- [ ] **Step 5: Create `dbt/models/staging/stg_products.sql`**

```sql
-- Staging: cast types, trim strings, drop hard nulls.
-- Downstream models reference {{ ref('stg_products') }}, not the raw source.
SELECT
    source_platform,
    shop_name,
    product_id,
    product_url,
    TRIM(title)                          AS title,
    TRIM(description)                    AS description,
    category,
    brand,
    CAST(price          AS DOUBLE)       AS price,
    CAST(old_price      AS DOUBLE)       AS old_price,
    availability,
    CAST(rating         AS DOUBLE)       AS rating,
    CAST(review_count   AS INTEGER)      AS review_count,
    geography,
    scraped_at,
    COALESCE(CAST(dq_score AS DOUBLE), 0.0) AS dq_score
FROM {{ source('warehouse', 'products') }}
WHERE title     IS NOT NULL
  AND product_id IS NOT NULL
```

- [ ] **Step 6: Create `dbt/models/staging/schema.yml`**

```yaml
version: 2

models:
  - name: stg_products
    description: "Typed, trimmed products from raw warehouse table"
    columns:
      - name: product_id
        tests:
          - not_null
      - name: source_platform
        tests:
          - not_null
          - accepted_values:
              values: [shopify, woocommerce]
      - name: dq_score
        tests:
          - not_null
```

- [ ] **Step 7: Create `dbt/models/marts/mart_products_clean.sql`**

```sql
-- Mart: apply business rules — fill unknown labels, compute discount %, filter low-DQ rows.
SELECT
    source_platform,
    shop_name,
    product_id,
    product_url,
    title,
    description,
    COALESCE(category,     'Unknown') AS category,
    COALESCE(brand,        'Unknown') AS brand,
    price,
    old_price,
    CASE
        WHEN old_price IS NOT NULL
         AND old_price > price
         AND price     > 0
        THEN ROUND((old_price - price) / old_price * 100.0, 2)
        ELSE NULL
    END                               AS discount_pct,
    COALESCE(availability, 'unknown') AS availability,
    rating,
    COALESCE(review_count, 0)         AS review_count,
    geography,
    scraped_at,
    dq_score
FROM {{ ref('stg_products') }}
WHERE dq_score >= 0.5
```

- [ ] **Step 8: Create `dbt/models/marts/mart_top_products.sql`**

Scoring weights match `src/scoring/topk.py`: rating=0.35, review_count=0.30, availability=0.20, discount=0.15.

```sql
-- Mart: explainable Top-K score matching Python topk.py weights.
-- rating=0.35, review_count=0.30, availability=0.20, discount=0.15
WITH scored AS (
    SELECT
        *,
        ROUND(
              COALESCE(rating / 5.0, 0.0)
            * 0.35

            + COALESCE(LN(review_count + 1) / LN(1001.0), 0.0)
            * 0.30

            + CASE WHEN availability = 'instock' THEN 1.0 ELSE 0.0 END
            * 0.20

            + COALESCE(discount_pct / 100.0, 0.0)
            * 0.15
        , 4) AS top_k_score
    FROM {{ ref('mart_products_clean') }}
),
ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY shop_name
            ORDER BY top_k_score DESC
        ) AS shop_rank,
        ROW_NUMBER() OVER (
            ORDER BY top_k_score DESC
        ) AS global_rank
    FROM scored
)
SELECT * FROM ranked
ORDER BY global_rank
```

- [ ] **Step 9: Create `dbt/models/marts/schema.yml`**

```yaml
version: 2

models:
  - name: mart_products_clean
    description: "Cleaned products with discount_pct, unknown-filled labels, dq_score >= 0.5"
    columns:
      - name: product_id
        tests:
          - not_null
      - name: dq_score
        tests:
          - not_null

  - name: mart_top_products
    description: "Products scored by top_k_score with shop_rank and global_rank"
    columns:
      - name: top_k_score
        tests:
          - not_null
      - name: global_rank
        tests:
          - not_null
```

- [ ] **Step 10: Validate dbt project parses cleanly**

```bash
cd dbt && dbt parse --profiles-dir . 2>&1 | tail -10
```

Expected: `Done` with no errors (note: `dbt parse` validates YAML and SQL syntax without executing against the database)

- [ ] **Step 11: Run full Python suite — no regressions**

```bash
cd /home/medgm/vsc/smart-ecommerce-pipeline-v2
python3 -m pytest tests/ -q --tb=short --ignore=tests/test_cleaning.py --ignore=tests/test_cluster_metrics.py --ignore=tests/test_features.py --ignore=tests/test_ml.py --ignore=tests/test_model_persistence.py --ignore=tests/test_rules_vectorized.py --ignore=tests/test_scoring.py --ignore=tests/test_validate.py 2>&1 | tail -6
```

- [ ] **Step 12: Commit**

```bash
git add dbt/
git commit -m "feat: bootstrap dbt project with stg_products, mart_products_clean, mart_top_products (DuckDB backend)"
```

---

## Self-Review

**Spec coverage:**
- [x] MinIO object storage — Task 1 (client) + Task 2 (scraping upload) + Task 3 (preprocessing sync + upload)
- [x] DuckDB analytical layer — Task 4 (client + warehouse target in Makefile)
- [x] dbt SQL transforms + lineage — Task 5 (dbt project with 3 models)
- [x] No environment problems — all MinIO calls guarded by `is_minio_configured()`, no breakage without infra

**Placeholder scan:** None found. Every step has runnable commands and complete code.

**Type consistency:**
- `upload_file(local_path: Path, bucket: str, key: str)` — used consistently in Task 1, Task 2, Task 3
- `sync_to_local(bucket, prefix, local_dir) -> list[Path]` — defined Task 1, called Task 3
- `load_products(source="auto") -> pd.DataFrame` — defined Task 4, used in `make warehouse` target
- `query(sql: str) -> pd.DataFrame` — defined Task 4, tested Task 4
- dbt model chain: `stg_products` ← `mart_products_clean` ← `mart_top_products` — refs consistent throughout Task 5

**Gap check:**
- `dbt parse` in Step 10 validates SQL and YAML but does not execute against the database. To run models end-to-end, `make warehouse && make dbt-run` is required (needs `data/processed/cleaned_products.parquet` to exist). This is expected — dbt runs are integration-level, not unit tests.
- `tests/test_duckdb_client.py` imports `pandas` — these tests will fail if pandas is not installed (same as the existing pandas-dependent test suite). Not a regression.
