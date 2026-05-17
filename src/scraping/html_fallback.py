from __future__ import annotations

import html as _html_module
import json
import re
from urllib.parse import unquote, urlparse

from bs4 import BeautifulSoup

_BREADCRUMB_JUNK = {
    "home",
    "catalog",
    "products",
    "all products",
    "shop",
}

_WOO_SOURCE_PRIORITY = [
    "woo_meta_links",
    "woo_breadcrumb_html",
    "jsonld_breadcrumb",
    "woo_body_class",
    "woo_taxonomy_blocks",
    "url_hint",
]


def _parse_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    cleaned = value.strip().replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    try:
        return float(match.group(0))
    except (ValueError, TypeError):
        return None


def _parse_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if not isinstance(value, str):
        return None
    match = re.search(r"\d+", value.replace(",", ""))
    if not match:
        return None
    try:
        return int(match.group(0))
    except (ValueError, TypeError):
        return None


def _extract_text(soup: BeautifulSoup, selectors: list[str]) -> str | None:
    for selector in selectors:
        node = soup.select_one(selector)
        if not node:
            continue
        content = node.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        text = node.get_text(" ", strip=True)
        if text:
            return text
    return None


def _jsonld_objects(soup: BeautifulSoup) -> list[dict]:
    objects: list[dict] = []
    for script in soup.find_all("script", type="application/ld+json"):
        raw = script.string or script.get_text() or ""
        if not raw.strip():
            continue
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

        candidates: list[object]
        if isinstance(parsed, list):
            candidates = parsed
        else:
            candidates = [parsed]

        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            graph = candidate.get("@graph")
            if isinstance(graph, list):
                for item in graph:
                    if isinstance(item, dict):
                        objects.append(item)
            objects.append(candidate)
    return objects


def _is_product_jsonld(obj: dict) -> bool:
    atype = obj.get("@type")
    if isinstance(atype, str):
        return atype.lower() == "product"
    if isinstance(atype, list):
        return any(isinstance(x, str) and x.lower() == "product" for x in atype)
    return False


def _is_breadcrumb_jsonld(obj: dict) -> bool:
    atype = obj.get("@type")
    if isinstance(atype, str):
        return atype.lower() == "breadcrumblist"
    if isinstance(atype, list):
        return any(isinstance(x, str) and x.lower() == "breadcrumblist" for x in atype)
    return False


def _join_sources(sources: list[str]) -> str | None:
    uniq = sorted(set(sources))
    return "|".join(uniq) if uniq else None


def _taxonomy_strength(*, high: bool, medium: bool, low: bool) -> str:
    if high:
        return "high"
    if medium:
        return "medium"
    if low:
        return "low"
    return "none"


def _slug_label_from_url(value: str) -> str | None:
    parsed = urlparse(value)
    path = parsed.path or ""
    if not path.strip("/"):
        return None
    leaf = unquote(path.rstrip("/").split("/")[-1]).strip()
    if not leaf:
        return None
    leaf = re.sub(r"[-_]+", " ", leaf)
    leaf = re.sub(r"\s+", " ", leaf).strip()
    return leaf or None


def _normalize_breadcrumb_token(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    token = value.strip()
    if not token:
        return None
    token = re.sub(r"\s+", " ", token).strip()
    return token or None


def _item_url(item: object) -> str | None:
    """Extract the canonical URL from a JSON-LD BreadcrumbList item."""
    if isinstance(item, str):
        return item.rstrip("/")
    if not isinstance(item, dict):
        return None
    nested = item.get("item")
    if isinstance(nested, dict):
        url = nested.get("@id") or nested.get("url") or nested.get("id")
        if isinstance(url, str):
            return url.rstrip("/")
    if isinstance(nested, str):
        return nested.rstrip("/")
    url = item.get("@id") or item.get("url") or item.get("id")
    if isinstance(url, str):
        return url.rstrip("/")
    return None


def _breadcrumb_item_name(item: object) -> str | None:
    if isinstance(item, str):
        return _slug_label_from_url(item)
    if not isinstance(item, dict):
        return None

    direct_name = _normalize_breadcrumb_token(item.get("name"))
    nested_item = item.get("item")

    if isinstance(nested_item, dict):
        nested_name = _normalize_breadcrumb_token(nested_item.get("name"))
        if nested_name:
            return _html_module.unescape(nested_name)
        nested_id = nested_item.get("@id") or nested_item.get("id") or nested_item.get("url")
        if isinstance(nested_id, str):
            url_label = _slug_label_from_url(nested_id)
            if url_label:
                return url_label

    if isinstance(nested_item, str):
        if direct_name:
            return _html_module.unescape(direct_name)
        return _slug_label_from_url(nested_item)

    if direct_name:
        return _html_module.unescape(direct_name)

    item_id = item.get("@id") or item.get("id") or item.get("url")
    if isinstance(item_id, str):
        return _slug_label_from_url(item_id)
    return None


def _extract_breadcrumb_path_from_jsonld(obj: dict, product_url: str | None = None) -> list[str]:
    """Extract ordered breadcrumb names, skipping items that resolve to the product URL itself."""
    item_list = obj.get("itemListElement")
    if not isinstance(item_list, list):
        return []

    product_url_key = product_url.rstrip("/").lower() if product_url else None

    ordered: list[tuple[int, str]] = []
    fallback: list[str] = []
    for entry in item_list:
        # Skip items whose URL is the product page itself — they carry the product title, not a category
        if product_url_key:
            entry_url = _item_url(entry)
            if entry_url and entry_url.lower() == product_url_key:
                continue

        name = _breadcrumb_item_name(entry)
        if not name:
            continue
        if isinstance(entry, dict):
            position = entry.get("position")
            try:
                if position is not None:
                    ordered.append((int(position), name))
                    continue
            except (TypeError, ValueError):
                pass
        fallback.append(name)

    if ordered:
        ordered_names = [name for _, name in sorted(ordered, key=lambda x: x[0])]
        if fallback:
            ordered_names.extend(fallback)
        return ordered_names
    return fallback


def _last_meaningful_breadcrumb(path: list[str], product_url: str | None = None) -> str | None:
    product_slug_label = _slug_label_from_url(product_url) if isinstance(product_url, str) else None
    product_slug_norm = (
        re.sub(r"\s+", " ", product_slug_label.strip().lower()) if product_slug_label else None
    )

    for token in reversed(path):
        norm = token.strip().lower()
        if norm in _BREADCRUMB_JUNK:
            continue
        norm = re.sub(r"\s+", " ", norm)
        if product_slug_norm and norm == product_slug_norm:
            continue
        return token
    return None


def _normalize_category_candidate(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    token = _html_module.unescape(value)
    token = re.sub(r"\s+", " ", token).strip()
    if not token:
        return None
    return token


def _is_garbage_category(value: str) -> bool:
    norm = value.strip().lower()
    return norm in {
        "home",
        "shop",
        "product",
        "products",
        "all",
        "catalog",
        "uncategorized",
        "uncategorised",
    }


def _looks_like_title(value: str, product_title: str | None) -> bool:
    if not product_title:
        return False
    norm_value = re.sub(r"\s+", " ", value.strip().lower())
    norm_title = re.sub(r"\s+", " ", product_title.strip().lower())
    return bool(norm_value and norm_title and norm_value == norm_title)


def _add_category_candidate(
    candidates: list[dict[str, str]],
    seen: set[tuple[str, str]],
    *,
    value: str | None,
    source: str,
    confidence: str,
    product_title: str | None,
) -> None:
    token = _normalize_category_candidate(value)
    if not token or _is_garbage_category(token) or _looks_like_title(token, product_title):
        return
    key = (token.lower(), source)
    if key in seen:
        return
    seen.add(key)
    candidates.append({"value": token, "source": source, "confidence": confidence})


def _pick_candidate_by_priority(
    candidates: list[dict[str, str]], source_priority: list[str]
) -> dict[str, str] | None:
    by_source: dict[str, list[dict[str, str]]] = {}
    for candidate in candidates:
        by_source.setdefault(candidate["source"], []).append(candidate)

    for source in source_priority:
        options = by_source.get(source)
        if options:
            return options[0]
    return candidates[0] if candidates else None


def _extract_woo_meta_link_candidates(soup: BeautifulSoup) -> list[str]:
    selectors = [
        ".product_meta .posted_in a",
        ".posted_in a",
        ".product_meta a[href*='/product-category/']",
        ".summary a[href*='/product-category/']",
        ".entry-summary a[href*='/product-category/']",
    ]
    values: list[str] = []
    for selector in selectors:
        for node in soup.select(selector):
            values.append(node.get_text(" ", strip=True))
    return values


def _extract_woo_body_class_candidates(soup: BeautifulSoup) -> list[str]:
    body = soup.find("body")
    if body is None:
        return []
    classes = body.get("class") or []
    values: list[str] = []
    for cls in classes:
        if not isinstance(cls, str):
            continue
        if not cls.startswith("product_cat-"):
            continue
        slug = cls.replace("product_cat-", "").strip("-")
        if not slug:
            continue
        label = re.sub(r"[-_]+", " ", slug).strip()
        if label:
            values.append(label)
    return values


def _extract_woo_taxonomy_block_candidates(soup: BeautifulSoup) -> list[str]:
    values: list[str] = []
    blocks = soup.select(".single-product, .product_meta, .summary, .entry-summary")
    for block in blocks:
        for node in block.select("a[href*='/product-category/']"):
            values.append(node.get_text(" ", strip=True))
    return values


def extract_woocommerce_taxonomy_from_html(
    html: str, product_url: str | None = None, product_title: str | None = None
) -> dict:
    """WooCommerce-specific taxonomy extraction with ranked source cascade."""
    soup = BeautifulSoup(html, "html.parser")

    candidates: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    taxonomy_sources: list[str] = []

    # 1) Explicit category links in product meta/summary
    for value in _extract_woo_meta_link_candidates(soup):
        _add_category_candidate(
            candidates,
            seen,
            value=value,
            source="woo_meta_links",
            confidence="high",
            product_title=product_title,
        )
    if any(c["source"] == "woo_meta_links" for c in candidates):
        taxonomy_sources.append("woo_meta_links")

    # 2) HTML breadcrumb leaf
    breadcrumb_nodes = soup.select("nav.woocommerce-breadcrumb a, .woocommerce-breadcrumb a")
    html_breadcrumb_path = [
        _normalize_category_candidate(node.get_text(" ", strip=True)) for node in breadcrumb_nodes
    ]
    html_breadcrumb_path = [x for x in html_breadcrumb_path if x]
    if html_breadcrumb_path:
        taxonomy_sources.append("woo_breadcrumb_html")
        html_leaf = _last_meaningful_breadcrumb(html_breadcrumb_path, product_url=product_url)
        _add_category_candidate(
            candidates,
            seen,
            value=html_leaf,
            source="woo_breadcrumb_html",
            confidence="high",
            product_title=product_title,
        )

    # 3) JSON-LD BreadcrumbList leaf
    jsonld_paths: list[list[str]] = []
    jsonld_breadcrumb_present = False
    for obj in _jsonld_objects(soup):
        if not _is_breadcrumb_jsonld(obj):
            continue
        jsonld_breadcrumb_present = True
        path = _extract_breadcrumb_path_from_jsonld(obj, product_url=product_url)
        if not path:
            continue
        jsonld_paths.append(path)
        leaf = _last_meaningful_breadcrumb(path, product_url=product_url)
        _add_category_candidate(
            candidates,
            seen,
            value=leaf,
            source="jsonld_breadcrumb",
            confidence="medium",
            product_title=product_title,
        )
    if jsonld_breadcrumb_present:
        taxonomy_sources.append("jsonld_breadcrumb")

    # 4) body classes like product_cat-*
    for value in _extract_woo_body_class_candidates(soup):
        _add_category_candidate(
            candidates,
            seen,
            value=value,
            source="woo_body_class",
            confidence="low",
            product_title=product_title,
        )
    if any(c["source"] == "woo_body_class" for c in candidates):
        taxonomy_sources.append("woo_body_class")

    # 5) taxonomy/meta blocks fallback
    for value in _extract_woo_taxonomy_block_candidates(soup):
        _add_category_candidate(
            candidates,
            seen,
            value=value,
            source="woo_taxonomy_blocks",
            confidence="low",
            product_title=product_title,
        )
    if any(c["source"] == "woo_taxonomy_blocks" for c in candidates):
        taxonomy_sources.append("woo_taxonomy_blocks")

    # 6) URL hint (weak)
    url_hint_present = False
    if product_url:
        url_path = urlparse(product_url).path.lower()
        url_hint_present = any(token in url_path for token in ["/category/", "/product-category/"])
        if url_hint_present:
            taxonomy_sources.append("url_hint")
            url_leaf = _slug_label_from_url(product_url)
            _add_category_candidate(
                candidates,
                seen,
                value=url_leaf,
                source="url_hint",
                confidence="low",
                product_title=product_title,
            )

    winner = _pick_candidate_by_priority(candidates, _WOO_SOURCE_PRIORITY)

    breadcrumb_path_raw: list[str] = []
    if html_breadcrumb_path:
        breadcrumb_path_raw = html_breadcrumb_path
    elif jsonld_paths:
        breadcrumb_path_raw = max(jsonld_paths, key=len)

    high_sources = {"woo_meta_links", "woo_breadcrumb_html", "jsonld_breadcrumb"}
    medium_sources = {"woo_body_class", "woo_taxonomy_blocks"}
    winner_source = winner["source"] if winner else None

    fields: dict = {}
    if winner:
        fields["category"] = winner["value"]
        fields["category_leaf_raw"] = winner["value"]
    if breadcrumb_path_raw:
        fields["category_path_raw"] = " > ".join(breadcrumb_path_raw)

    fields["taxonomy_breadcrumb_present"] = bool(html_breadcrumb_path)
    fields["taxonomy_breadcrumb_count"] = len(breadcrumb_path_raw) if breadcrumb_path_raw else None
    fields["taxonomy_jsonld_category_present"] = False
    fields["taxonomy_jsonld_breadcrumb_present"] = jsonld_breadcrumb_present
    fields["taxonomy_product_type_present"] = False
    fields["taxonomy_tags_present"] = False
    fields["taxonomy_url_hint_present"] = url_hint_present
    fields["taxonomy_sources_detected"] = _join_sources(taxonomy_sources)
    fields["taxonomy_evidence_strength"] = _taxonomy_strength(
        high=winner_source in high_sources,
        medium=winner_source in medium_sources,
        low=winner_source == "url_hint" or (winner is None and url_hint_present),
    )

    return fields


def extract_product_fields_from_html(html: str, product_url: str | None = None) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    fields: dict = {}
    taxonomy_sources: list[str] = []

    description = _extract_text(
        soup,
        [
            "meta[name='description']",
            "meta[property='og:description']",
            "meta[itemprop='description']",
            "[itemprop='description']",
            "[data-product-description]",
        ],
    )
    if description:
        fields["description"] = description[:500]

    price_text = _extract_text(
        soup,
        [
            "meta[property='og:price:amount']",
            "meta[itemprop='price']",
            "[data-product-price]",
            ".price",
        ],
    )
    parsed_price = _parse_float(price_text)
    if parsed_price is not None:
        fields["price"] = parsed_price

    availability = _extract_text(
        soup,
        [
            "meta[property='og:availability']",
            "meta[itemprop='availability']",
            "link[itemprop='availability']",
        ],
    )
    if availability:
        fields["availability"] = availability

    category = _extract_text(
        soup,
        [
            "meta[property='product:category']",
            "meta[name='product:category']",
            "[itemprop='category']",
            ".breadcrumb a:last-child",
        ],
    )
    if category:
        fields["category"] = category

    breadcrumb_nodes = soup.select(
        "nav.woocommerce-breadcrumb a, .woocommerce-breadcrumb a, "
        "nav.breadcrumb a, .breadcrumb a, [aria-label='breadcrumb'] a"
    )
    html_breadcrumb_path = [
        node.get_text(" ", strip=True)
        for node in breadcrumb_nodes
        if node.get_text(" ", strip=True)
    ]
    breadcrumb_count = len(html_breadcrumb_path)
    breadcrumb_present = breadcrumb_count > 0
    if breadcrumb_present:
        taxonomy_sources.append("breadcrumb_html")

    jsonld_category_present = False
    jsonld_breadcrumb_present = False
    jsonld_breadcrumb_paths: list[list[str]] = []

    for obj in _jsonld_objects(soup):
        if _is_breadcrumb_jsonld(obj):
            jsonld_breadcrumb_present = True
            taxonomy_sources.append("jsonld_breadcrumb")
            path = _extract_breadcrumb_path_from_jsonld(obj, product_url=product_url)
            if path:
                jsonld_breadcrumb_paths.append(path)
        if not _is_product_jsonld(obj):
            continue

        if "description" not in fields and obj.get("description"):
            fields["description"] = str(obj["description"]).strip()[:500]

        if "category" not in fields and obj.get("category"):
            fields["category"] = str(obj["category"]).strip()
        if obj.get("category"):
            jsonld_category_present = True
            taxonomy_sources.append("jsonld_category")

        agg = obj.get("aggregateRating")
        if isinstance(agg, dict):
            if "rating" not in fields:
                rating = _parse_float(agg.get("ratingValue"))
                if rating is not None:
                    fields["rating"] = rating
            if "review_count" not in fields:
                review_count = _parse_int(agg.get("reviewCount"))
                if review_count is not None:
                    fields["review_count"] = review_count

        offers = obj.get("offers")
        offer: dict | None = None
        if isinstance(offers, dict):
            offer = offers
        elif isinstance(offers, list):
            offer = next((item for item in offers if isinstance(item, dict)), None)

        if offer:
            if "price" not in fields:
                offer_price = _parse_float(offer.get("price"))
                if offer_price is not None:
                    fields["price"] = offer_price
            if "availability" not in fields and offer.get("availability"):
                fields["availability"] = str(offer["availability"]).strip()

    breadcrumb_path_raw: list[str] = []
    if jsonld_breadcrumb_paths:
        breadcrumb_path_raw = max(jsonld_breadcrumb_paths, key=len)
    elif html_breadcrumb_path:
        breadcrumb_path_raw = html_breadcrumb_path

    category_leaf_raw = _last_meaningful_breadcrumb(breadcrumb_path_raw, product_url=product_url)
    if breadcrumb_path_raw:
        fields["category_path_raw"] = " > ".join(breadcrumb_path_raw)
    if category_leaf_raw:
        fields["category_leaf_raw"] = category_leaf_raw

    if "category" not in fields and category_leaf_raw:
        fields["category"] = category_leaf_raw

    merged_breadcrumb_count = max(
        breadcrumb_count,
        len(breadcrumb_path_raw) if breadcrumb_path_raw else 0,
    )

    url_hint_present = False
    if product_url:
        path = urlparse(product_url).path.lower()
        url_hint_present = any(
            token in path for token in ["/collections/", "/category/", "/product-category/"]
        )
        if url_hint_present:
            taxonomy_sources.append("url_hint")

    high_evidence = breadcrumb_present or jsonld_category_present or jsonld_breadcrumb_present
    low_evidence = url_hint_present

    fields["taxonomy_breadcrumb_present"] = breadcrumb_present
    fields["taxonomy_breadcrumb_count"] = (
        merged_breadcrumb_count if merged_breadcrumb_count else None
    )
    fields["taxonomy_jsonld_category_present"] = jsonld_category_present
    fields["taxonomy_jsonld_breadcrumb_present"] = jsonld_breadcrumb_present
    fields["taxonomy_product_type_present"] = False
    fields["taxonomy_tags_present"] = False
    fields["taxonomy_url_hint_present"] = url_hint_present
    fields["taxonomy_sources_detected"] = _join_sources(taxonomy_sources)
    fields["taxonomy_evidence_strength"] = _taxonomy_strength(
        high=high_evidence,
        medium=False,
        low=low_evidence,
    )

    return fields
