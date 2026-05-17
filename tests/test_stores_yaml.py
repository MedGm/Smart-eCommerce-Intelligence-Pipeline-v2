def test_load_stores_returns_shopify_and_woocommerce():
    from src.scraping.stores import load_stores

    shopify, wc = load_stores()
    assert isinstance(shopify, list)
    assert isinstance(wc, list)
    assert len(shopify) > 0
    assert len(wc) > 0


def test_shopify_store_has_required_fields():
    from src.scraping.stores import load_stores

    shopify, _ = load_stores()
    for store in shopify:
        assert "url" in store, f"Missing 'url' in {store}"
        assert "name" in store, f"Missing 'name' in {store}"


def test_woocommerce_store_has_required_fields():
    from src.scraping.stores import load_stores

    _, wc = load_stores()
    for store in wc:
        assert "url" in store, f"Missing 'url' in {store}"
        assert "name" in store, f"Missing 'name' in {store}"


def test_load_stores_accepts_custom_path(tmp_path):
    from src.scraping.stores import load_stores

    yaml_content = """
shopify:
  - url: https://test.myshopify.com
    name: TestShop
    geography: US
    collections: [all]
woocommerce:
  - url: https://testwc.com
    name: TestWC
    geography: US
"""
    custom = tmp_path / "custom_stores.yaml"
    custom.write_text(yaml_content)
    shopify, wc = load_stores(path=custom)
    assert shopify[0]["name"] == "TestShop"
    assert wc[0]["name"] == "TestWC"
