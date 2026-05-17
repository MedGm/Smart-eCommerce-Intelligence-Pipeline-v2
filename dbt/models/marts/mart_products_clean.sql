-- Mart: fill unknown labels, compute discount_pct, filter low-DQ rows.
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
