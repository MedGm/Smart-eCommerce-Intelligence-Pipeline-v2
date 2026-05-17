"""
Store catalog loader. Edit stores.yaml to add/remove stores — no Python changes needed.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_DEFAULT_YAML = Path(__file__).parent.parent.parent / "stores.yaml"


def load_stores(path: Path | None = None) -> tuple[list[dict], list[dict]]:
    """Return (shopify_stores, woocommerce_stores) from YAML config."""
    yaml_path = Path(path) if path else _DEFAULT_YAML
    with open(yaml_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    shopify = config.get("shopify") or []
    woocommerce = config.get("woocommerce") or []
    return shopify, woocommerce


# Backwards-compatible exports for any code that imports these directly.
def _lazy_load():
    s, w = load_stores()
    return s, w

SHOPIFY_STORES, WOOCOMMERCE_STORES = _lazy_load()
