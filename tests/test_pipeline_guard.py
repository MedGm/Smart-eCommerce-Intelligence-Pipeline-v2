from unittest.mock import patch
from pathlib import Path
import logging


def test_features_step_skipped_when_preprocessing_fails(tmp_path, caplog):
    executed_steps = []

    def fake_run_step(name, mod):
        if name == "Preprocessing":
            raise RuntimeError("preprocessing failed")
        executed_steps.append(name)

    from src.pipeline import local_pipeline

    with patch("src.pipeline.local_pipeline._run_step", side_effect=fake_run_step), \
         patch("src.pipeline.local_pipeline.processed_dir", return_value=tmp_path / "processed"), \
         caplog.at_level(logging.WARNING):
        local_pipeline.run()

    assert "Features" not in executed_steps, \
        f"Features ran despite missing preprocessing artifact. Steps run: {executed_steps}"


def test_steps_run_normally_when_artifacts_present(tmp_path):
    executed_steps = []

    def fake_run_step(name, mod):
        # Create expected artifacts
        if name == "Preprocessing":
            proc = tmp_path / "processed"
            proc.mkdir(parents=True, exist_ok=True)
            (proc / "products.parquet").write_bytes(b"fake")
        if name == "Features":
            proc = tmp_path / "processed"
            proc.mkdir(parents=True, exist_ok=True)
            (proc / "features.parquet").write_bytes(b"fake")
        executed_steps.append(name)

    from src.pipeline import local_pipeline

    with patch("src.pipeline.local_pipeline._run_step", side_effect=fake_run_step), \
         patch("src.pipeline.local_pipeline.processed_dir", return_value=tmp_path / "processed"):
        local_pipeline.run()

    assert "Preprocessing" in executed_steps
    assert "Features" in executed_steps
