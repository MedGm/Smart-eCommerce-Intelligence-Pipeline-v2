-- Mart: explainable Top-K score matching Python topk.py weights.
-- rating=0.35, review_count=0.30, availability=0.20, discount=0.15
WITH scored AS (
    SELECT
        *,
        ROUND(
              COALESCE(rating / 5.0, 0.0)
            * 0.35

            + COALESCE(LN(review_count + 1) / LN(1001.0), 0.0)
            * 0.30

            + CASE WHEN availability = 'instock' THEN 1.0 ELSE 0.0 END
            * 0.20

            + COALESCE(discount_pct / 100.0, 0.0)
            * 0.15
        , 4) AS top_k_score
    FROM {{ ref('mart_products_clean') }}
),
ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY shop_name
            ORDER BY top_k_score DESC
        ) AS shop_rank,
        ROW_NUMBER() OVER (
            ORDER BY top_k_score DESC
        ) AS global_rank
    FROM scored
)
SELECT * FROM ranked
ORDER BY global_rank
