-- Staging: cast types, trim strings, drop hard nulls.
-- Downstream models reference {{ ref('stg_products') }}, not the raw source.
SELECT
    source_platform,
    shop_name,
    product_id,
    product_url,
    TRIM(title)                          AS title,
    TRIM(description)                    AS description,
    category,
    brand,
    CAST(price          AS DOUBLE)       AS price,
    CAST(old_price      AS DOUBLE)       AS old_price,
    availability,
    CAST(rating         AS DOUBLE)       AS rating,
    CAST(review_count   AS INTEGER)      AS review_count,
    geography,
    scraped_at,
    COALESCE(CAST(dq_score AS DOUBLE), 0.0) AS dq_score
FROM {{ source('warehouse', 'products') }}
WHERE title      IS NOT NULL
  AND product_id IS NOT NULL
