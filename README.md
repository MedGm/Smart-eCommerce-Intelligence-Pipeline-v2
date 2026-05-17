# Smart eCommerce Intelligence Pipeline v2

**Author:** Mohamed El Gorrim  
**Repo:** [MedGm/Smart-eCommerce-Intelligence-Pipeline-v2](https://github.com/MedGm/Smart-eCommerce-Intelligence-Pipeline-v2)

End-to-end ML pipeline: scrapes products from 13 Shopify + 3 WooCommerce stores → cleans and validates → engineers features → scores by potential → trains RF + XGBoost classifiers + KMeans + DBSCAN clustering + Apriori association rules → Streamlit dashboard + Gemini LLM synthesis. Orchestrated via Kubeflow Pipelines v2 on Minikube.

---

## Architecture

```
Stores (16 targets)
    │
    ▼
A2A Scraping Layer          ← Shopify JSON API + WooCommerce Store API
(CoordinatorAgent → WorkerAgents)
    │  run_id-partitioned raw JSON
    ▼
Preprocessing               ← clean, validate, DQ counters
    │  Parquet + CSV
    ▼
Feature Engineering         ← scoring features, model features
    │
    ├──► Top-K Scoring      ← explainable weighted formula
    │
    └──► ML / Data Mining   ← RF, XGBoost, KMeans, DBSCAN, Apriori
             │
             ▼
         MLflow             ← experiment tracking + model registry   (Phase 2)
             │
             ▼
         MinIO              ← partitioned object storage             (Phase 1)
             │
             ▼
    Streamlit Dashboard     ← BI pages + Gemini LLM synthesis
    (+ Apache Superset)                                              (Phase 3)
             │
             ▼
    Kubeflow Pipelines v2   ← orchestration on Minikube
```

---

## Stack

| Layer | Current | Planned |
|---|---|---|
| Scraping | Playwright, requests, BeautifulSoup, A2A agents | — |
| Store config | `stores.yaml` (16 targets) | — |
| Raw storage | JSON (partitioned by `run_id`) | MinIO on Minikube (Phase 1) |
| Data lake / transforms | — | DuckDB + dbt (Phase 1) |
| Preprocessing | pandas, pyarrow | — |
| ML / Data mining | scikit-learn, XGBoost, mlxtend | — |
| Experiment tracking | — | MLflow (Phase 2) |
| Data quality | — | Great Expectations as KFP step (Phase 2) |
| Dashboard | Streamlit + Plotly + Altair | Apache Superset (Phase 3) |
| LLM | Google Gemini (`google-genai`) | — |
| Analytics gate | MCP allowlist (`src/mcp/`) | — |
| Orchestration | Local pipeline + KFP v2 (Minikube) | Fix KFP typed artifacts + caching (Phase 2) |
| Object storage | local `data/` | MinIO on Minikube (Phase 1) |
| Infra | Docker Compose, Minikube, Kustomize | — |
| CI / Lint | GitHub Actions, Ruff, pytest | — |

---

## Running the project

Everything runs inside Docker — no local Python install, no pip, no venv needed.

**Prerequisite:** Docker + Docker Compose v2. Optionally a `.env` file with `GEMINI_API_KEY=...` for LLM features.

### Build once

```bash
make build          # builds the app image (installs all deps inside)
```

### Day-to-day commands

```bash
make test           # run pytest inside container
make lint           # ruff check + format
make scrape         # scrape all 16 stores
make preprocess     # clean, validate, DQ counters
make features       # feature engineering
make score          # Top-K scoring
make train          # RF, XGBoost, KMeans, DBSCAN, Apriori
make pipeline       # full end-to-end run (no infra needed)
make dashboard      # Streamlit on http://localhost:8501
```

### With infrastructure (MinIO + MLflow)

```bash
make infra-up       # start MinIO + MLflow in background
make pipeline-full  # full pipeline wired to MinIO + MLflow
make infra-down     # stop infrastructure
```

### One-off commands inside the container

```bash
docker compose run --rm app <any command>
# examples:
docker compose run --rm app python -m src.scraping.run_scrapers
docker compose run --rm app python -m pytest tests/test_minio_client.py -v
docker compose run --rm app ruff check src
```

### Service URLs (when infra is up)

| Service | URL | Purpose |
|---|---|---|
| Streamlit | http://localhost:8501 | LLM chat interface (Synthesis + Chat) |
| Superset | http://localhost:8088 | BI charts, rankings, product tables, clustering |
| MinIO console | http://localhost:9001 | Browse raw/processed/model artifacts |
| MinIO S3 API | http://localhost:9000 | S3-compatible endpoint |
| MLflow | http://localhost:5000 | Experiment runs + model registry |

MinIO dev credentials: `minioadmin` / `minioadmin` (override with `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD`).

---

## Makefile targets

| Target | Description |
|---|---|
| `make build` | Build Docker app image |
| `make test` | pytest inside container |
| `make test-v` | pytest verbose inside container |
| `make lint` | Ruff check + format inside container |
| `make scrape` | A2A scraping (all 16 stores) |
| `make preprocess` | Preprocessing + DQ counters |
| `make features` | Feature engineering |
| `make score` | Top-K scoring artifacts |
| `make train` | All ML/DM models |
| `make pipeline` | Full end-to-end run (local, no infra) |
| `make pipeline-full` | Full pipeline wired to MinIO + MLflow |
| `make dashboard` | Launch Streamlit dashboard |
| `make infra-up` | Start MinIO + MLflow in background |
| `make infra-down` | Stop infrastructure |
| `make warehouse` | Load DuckDB warehouse from Parquet |
| `make dbt-run` | Run dbt models |
| `make dbt-test` | Run dbt tests |
| `make compile-kfp` | Compile Kubeflow pipeline YAML |
| `make clean` | Remove pycache / pytest cache |

---

## Roadmap

### Phase 0 — Scraping layer fixes ✅ done
- WooCommerce retry/backoff (exponential, 429/503)
- Store config moved from Python to `stores.yaml`
- Raw output partitioned by `run_id` timestamp (`raw/shopify/ruggable/20260517T130000Z.json`)
- Checkpoint/resume: completed stores written to `checkpoint.json`, skipped on restart

### Phase 1 — Data lake
- MinIO on Minikube as primary object storage (raw, processed, models)
- DuckDB as analytical query layer (replaces pandas for large-scale queries)
- dbt for SQL-based data transforms + lineage

### Phase 2 — ML infrastructure
- Kubeflow Pipelines v2: fix typed artifacts and step caching
- MLflow alongside KFP: experiment tracking + model registry
- Great Expectations: data quality as a KFP pipeline step

### Phase 3 — BI layer
- Apache Superset on Minikube: replaces most Streamlit reporting pages
- Streamlit retained for LLM chat interface only

---

## Repository structure

```
smart-ecommerce-pipeline-v2/
├── data/
│   ├── raw/              # scraped JSON — raw/<platform>/<shop>/<run_id>.json
│   ├── processed/        # cleaned Parquet, DQ counters
│   └── analytics/        # scoring CSVs, model metrics, cluster outputs
├── docs/
│   └── diagrams/         # Mermaid: platform_architecture.mmd, pipeline_workflow.mmd
├── manifests/            # Kustomize overlays (Kubeflow / Minikube)
├── notebooks/            # EDA
├── scripts/              # helpers: KFP operator, audit replay, target validation
├── src/
│   ├── scraping/         # A2A agents, Shopify + WooCommerce adapters, base, stores
│   ├── preprocessing/    # clean, transform, validate, DQ run
│   ├── features/         # feature engineering
│   ├── scoring/          # explainable Top-K formula
│   ├── ml/               # RF, XGBoost, KMeans, DBSCAN, Apriori, PCA
│   ├── llm/              # Gemini summariser, prompts
│   ├── mcp/              # MCP read-only analytics gate
│   ├── pipeline/         # local runner + KFP v2 pipeline definition
│   └── dashboard/        # Streamlit multi-page BI app
├── tests/                # 48+ unit + integration tests
├── stores.yaml           # store catalog — edit here to add/remove targets
├── Makefile
├── Dockerfile            # app + dashboard image
├── Dockerfile.mlflow     # MLflow tracking server image
├── docker-compose.yml    # infra (MinIO + MLflow) + pipeline + dashboard
├── kubeflow_smart_ecommerce_pipeline.yaml
└── requirements.txt
```

---

## Key design decisions

**Explainable scoring** — each score is a weighted sum of normalised signals; weights documented in `src/scoring/topk.py`.

**Run-id partitioned output** — every scrape run writes to `raw/<platform>/<shop>/<run_id>.json`. No overwriting, full history, maps directly to MinIO object paths in Phase 1.

**Checkpoint/resume** — `data/raw/<run_id>/checkpoint.json` records completed stores. A crashed run restarts from where it left off.

**Store config in YAML** — `stores.yaml` is the single place to add/remove scraping targets. No Python changes needed.

**MCP read-only gate** — LLM analytics access goes through an allowlist in `src/mcp/architecture.py`. The LLM layer cannot write to artifacts.

**Kubeflow parity** — pipeline stages are plain Python functions in `src/`. The KFP definition in `src/pipeline/kubeflow_pipeline.py` wraps the same functions, so local and KFP runs are equivalent.

**Minikube-only deployment** — no cloud dependency. Everything (KFP, MinIO, Superset) runs on a local Minikube cluster.

---

## License

Academic project.
