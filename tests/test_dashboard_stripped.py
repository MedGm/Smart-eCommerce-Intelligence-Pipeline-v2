from pathlib import Path


def _app_content() -> str:
    return Path("src/dashboard/app.py").read_text()


def test_reporting_pages_removed():
    content = _app_content()
    removed = [
        "Product Rankings",
        "Shop Analysis",
        "ML Models",
        "Segmentation",
        "Association Rules",
    ]
    for page in removed:
        assert page not in content, f"Reporting page '{page}' still in stripped dashboard"


def test_llm_chat_functions_present():
    content = _app_content()
    assert "chat_with_data" in content, "chat_with_data not in new dashboard"
    assert "generate_summary" in content or "generate_strategy" in content, (
        "LLM synthesis function missing"
    )


def test_superset_link_in_sidebar():
    content = _app_content()
    assert "8088" in content or "superset" in content.lower(), "No Superset link in new dashboard"


def test_app_is_much_smaller():
    lines = len(_app_content().splitlines())
    assert lines < 300, (
        f"New dashboard should be < 300 lines (got {lines}). "
        "Reporting pages were not fully removed."
    )


def test_load_context_returns_empty_without_data(tmp_path, monkeypatch):
    """_load_context returns {} when analytics/ has no topk_products.csv."""
    import sys
    from unittest.mock import MagicMock

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    (tmp_path / "analytics").mkdir()

    mock_st = MagicMock()
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "streamlit", mock_st)
        if "src.dashboard.app" in sys.modules:
            del sys.modules["src.dashboard.app"]
        from src.dashboard.app import _load_context

    assert _load_context() == {}


def test_load_context_parses_topk(tmp_path, monkeypatch):
    import sys
    from unittest.mock import MagicMock

    import pandas as pd

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    analytics = tmp_path / "analytics"
    analytics.mkdir()
    pd.DataFrame(
        [
            {"product_id": "1", "category": "Rugs", "shop_name": "Ruggable", "score": 0.9},
            {"product_id": "2", "category": "Rugs", "shop_name": "NoBull", "score": 0.5},
        ]
    ).to_csv(analytics / "topk_products.csv", index=False)

    mock_st = MagicMock()
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "streamlit", mock_st)
        if "src.dashboard.app" in sys.modules:
            del sys.modules["src.dashboard.app"]
        from src.dashboard.app import _load_context

    ctx = _load_context()
    assert ctx["n_products"] == 2
    assert "Rugs" in ctx["top_categories"]
    assert ctx["best_shop"] == "Ruggable"
