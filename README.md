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

## Running with Docker Compose

Docker Compose is the canonical way to run this project. It removes all environment problems — Python version, system deps, Playwright browsers, and infrastructure services are all containerised.

### Prerequisites

- Docker + Docker Compose v2
- (Optional) a `.env` file with `GEMINI_API_KEY=...` for LLM features

### Services and profiles

| Profile | Services | Command |
|---|---|---|
| `infra` | MinIO (S3) + MLflow + bucket init | `docker compose --profile infra up -d` |
| `pipeline` | Pipeline runner (needs infra running) | `docker compose --profile pipeline up` |
| `dashboard` | Streamlit on :8501 (needs infra running) | `docker compose --profile dashboard up -d` |

### Typical workflow

```bash
# 1. Start infrastructure (MinIO + MLflow)
docker compose --profile infra up -d

# 2. Run the full pipeline
docker compose --profile pipeline up

# 3. Launch the dashboard
docker compose --profile dashboard up -d
```

### Service URLs

| Service | URL | Purpose |
|---|---|---|
| Streamlit | http://localhost:8501 | BI dashboard + LLM chat |
| MinIO console | http://localhost:9001 | Browse raw/processed/model artifacts |
| MinIO S3 API | http://localhost:9000 | S3-compatible endpoint |
| MLflow | http://localhost:5000 | Experiment runs + model registry |

### Credentials (dev defaults)

MinIO default credentials: `minioadmin` / `minioadmin`.  
Override via environment: `MINIO_ROOT_USER` and `MINIO_ROOT_PASSWORD`.

---

## Running locally (without Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
cp .env.example .env   # set GEMINI_API_KEY

# Full pipeline
make pipeline

# Or stage by stage
make scrape        # A2A scraping (16 stores)
make preprocess    # clean, validate, DQ
make features      # feature engineering
make score         # Top-K ranking
make train         # RF, XGBoost, KMeans, DBSCAN, Apriori

make dashboard     # Streamlit on http://localhost:8501
make test          # pytest
make lint          # Ruff
```

---

## Makefile targets

| Target | Description |
|---|---|
| `make pipeline` | Full end-to-end local run |
| `make scrape` | A2A scraping (all 16 stores) |
| `make preprocess` | Preprocessing + DQ counters |
| `make features` | Feature engineering |
| `make score` | Top-K scoring artifacts |
| `make train` | All ML/DM models |
| `make dashboard` | Launch Streamlit dashboard |
| `make compile-kfp` | Compile Kubeflow pipeline YAML |
| `make lint` | Ruff check + format |
| `make test` | pytest |
| `make docker-build` | Build Docker image |

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
