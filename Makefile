# PRISM — Product Ranking Intelligence & Signal Mining Pipeline — Makefile
#
# Everything runs inside Docker. No local Python install needed.
# Prerequisites: Docker + Docker Compose v2.
#
# Quick start:
#   make build          # build the app image
#   make test           # run pytest inside container
#   make scrape         # scrape all 16 stores
#   make pipeline       # full end-to-end run (local, no infra)
#
# With infrastructure (MinIO + MLflow):
#   make infra-up       # start MinIO + MLflow
#   make infra-down     # stop and remove infra containers
#   make pipeline-full  # full pipeline wired to MinIO + MLflow

DOCKER_RUN = docker compose run --rm app

.PHONY: build test lint \
        scrape preprocess features score train pipeline \
        dashboard \
        infra-up infra-down \
        superset-up superset-down \
        pipeline-full \
        warehouse refresh dbt-run dbt-test \
        compile-kfp \
        clean

# ── Image ─────────────────────────────────────────────────────────────────────
build:
	docker compose build app

# ── Tests & lint ──────────────────────────────────────────────────────────────
test:
	$(DOCKER_RUN) python -m pytest tests/ -q --tb=short

test-v:
	$(DOCKER_RUN) python -m pytest tests/ -v

lint:
	$(DOCKER_RUN) ruff check src tests
	$(DOCKER_RUN) ruff format --check src tests

# ── Pipeline stages ───────────────────────────────────────────────────────────
scrape:
	$(DOCKER_RUN) python -m src.scraping.run_scrapers

preprocess:
	$(DOCKER_RUN) python -m src.preprocessing.run

features:
	$(DOCKER_RUN) python -m src.features.build_features

score:
	$(DOCKER_RUN) python -m src.scoring.topk

train:
	$(DOCKER_RUN) python -m src.ml.train_classifier
	$(DOCKER_RUN) python -m src.ml.cluster_products

pipeline:
	$(DOCKER_RUN) python -m src.pipeline.local_pipeline

# ── Dashboard ─────────────────────────────────────────────────────────────────
dashboard:
	docker compose --profile dashboard up

# ── Infrastructure ────────────────────────────────────────────────────────────
infra-up:
	docker compose --profile infra up -d

infra-down:
	docker compose --profile infra down

superset-up:
	docker compose --profile superset up -d

superset-down:
	docker compose --profile superset down

# ── Full pipeline wired to infra ──────────────────────────────────────────────
pipeline-full:
	docker compose --profile pipeline up

# ── Data lake (Phase 1) ───────────────────────────────────────────────────────
warehouse:
	$(DOCKER_RUN) python -c "from src.storage.duckdb_client import rebuild_warehouse; rebuild_warehouse(); print('warehouse.duckdb ready')"

# Sync MinIO analytics → local → rebuild warehouse (run after KFP finishes)
refresh:
	$(DOCKER_RUN) python -c "\
from src.storage.duckdb_client import rebuild_warehouse; \
rebuild_warehouse(); \
print('Done. Restart Superset to pick up new data.')"

dbt-run: warehouse
	$(DOCKER_RUN) sh -c "cd dbt && dbt run --profiles-dir ."

dbt-test: warehouse
	$(DOCKER_RUN) sh -c "cd dbt && dbt test --profiles-dir ."

# ── Kubeflow ──────────────────────────────────────────────────────────────────
compile-kfp:
	$(DOCKER_RUN) python -c "from kfp import compiler; from src.pipeline.kubeflow_pipeline import prism_pipeline; compiler.Compiler().compile(pipeline_func=prism_pipeline, package_path='kubeflow_prism_pipeline.yaml')"

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	$(DOCKER_RUN) sh -c "find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; rm -rf .pytest_cache .ruff_cache; true"
