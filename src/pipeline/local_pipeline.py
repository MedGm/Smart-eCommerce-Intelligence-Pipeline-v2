"""
Local end-to-end pipeline: scrape -> preprocess -> features -> score -> train -> LLM summary.
Run with: python -m src.pipeline.local_pipeline  or  make pipeline
"""

import sys
from pathlib import Path

from src.config import get_logger, processed_dir

logger = get_logger(__name__)

# (name, module, artifact_check_fn)
# artifact_check_fn: callable returning Path to check after the step runs.
# If the path doesn't exist after the step, all downstream steps are skipped.
STEPS = [
    ("Scraping", "src.scraping.run_scrapers", None),
    (
        "Preprocessing",
        "src.preprocessing.run",
        lambda: processed_dir() / "cleaned_products.parquet",
    ),
    ("Features", "src.features.build_features", lambda: processed_dir() / "features.parquet"),
    ("Scoring", "src.scoring.topk", None),
    ("Train classifier (RF)", "src.ml.train_classifier", None),
    ("Train classifier (XGBoost)", "src.ml.train_xgboost", None),
    ("Clustering (KMeans)", "src.ml.cluster_products", None),
    ("Clustering (DBSCAN)", "src.ml.dbscan_products", None),
    ("Association rules", "src.ml.rules", None),
    ("LLM summary", "src.llm.summarizer", None),
]


def _run_step(name: str, mod: str) -> None:
    run_module = __import__(mod, fromlist=["run"])
    getattr(run_module, "run")()


def run() -> None:
    skip_from: str | None = None

    for name, mod, artifact_fn in STEPS:
        if skip_from is not None:
            logger.warning(
                "Skipping %s — upstream step '%s' did not produce expected artifact.",
                name,
                skip_from,
            )
            continue

        logger.info("--- %s ---", name)
        step_failed = False
        try:
            _run_step(name, mod)
        except Exception as e:
            logger.error("Step %s failed: %s", name, e)
            step_failed = True

        if artifact_fn is not None:
            artifact = Path(artifact_fn())
            if step_failed or not artifact.exists():
                logger.error(
                    "Step '%s' did not produce expected artifact %s — skipping downstream steps.",
                    name,
                    artifact,
                )
                skip_from = name

    logger.info("--- Pipeline finished ---")


if __name__ == "__main__":
    run()
    sys.exit(0)
