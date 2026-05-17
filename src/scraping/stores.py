"""
Multi-store configuration for scraping.
Each store defines platform, URL, name, geography, and platform-specific settings.
"""

SHOPIFY_STORES = [
    {
        "url": "https://ruggable.com",
        "name": "Ruggable",
        "geography": "US",
        "collections": ["all", "area-rugs", "runner-rugs"],
    },
    {
        "url": "https://www.turtlebeach.com",
        "name": "Turtle Beach",
        "geography": "US",
        "collections": ["all"],
    },
    {
        "url": "https://hiutdenim.co.uk",
        "name": "Hiut Denim",
        "geography": "UK",
        "collections": ["all"],
    },
    {
        "url": "https://www.fashionnova.com",
        "name": "Fashion Nova",
        "geography": "US",
        "collections": ["all"],
    },
    {
        "url": "https://www.deathwishcoffee.com",
        "name": "Death Wish Coffee",
        "geography": "US",
        "collections": ["all"],
    },
    {
        "url": "https://www.allbirds.com",
        "name": "Allbirds",
        "geography": "US",
        "collections": ["all"],
        "max_collection_pages": 6,
    },
    {
        "url": "https://representclo.com",
        "name": "Represent",
        "geography": "US",
        "collections": ["all"],
        "max_collection_pages": 8,
    },
    {
        "url": "https://bornprimitive.com",
        "name": "Born Primitive",
        "geography": "US",
        "collections": ["all"],
        "max_collection_pages": 8,
    },
    {
        "url": "https://nobullproject.com",
        "name": "NoBull",
        "geography": "US",
        "collections": ["all"],
        "max_collection_pages": 6,
    },
    {
        "url": "https://www.goattape.com",
        "name": "Goat Tape",
        "geography": "US",
        "collections": ["all"],
        "max_collection_pages": 4,
    },
    {
        "url": "https://www.tenthousand.cc",
        "name": "Ten Thousand",
        "geography": "US",
        "collections": ["all"],
        "max_collection_pages": 5,
    },
    {
        "url": "https://cutsclothing.com",
        "name": "Cuts Clothing",
        "geography": "US",
        "collections": ["all"],
        "max_collection_pages": 4,
    },
    {
        "url": "https://setactive.co",
        "name": "Set Active",
        "geography": "US",
        "collections": ["all"],
        "max_collection_pages": 6,
    },
]

WOOCOMMERCE_STORES = [
    {
        "url": "https://danosseasoning.com",
        "name": "Dan-O's Seasoning",
        "geography": "US",
    },
    {
        "url": "https://nalgene.com",
        "name": "Nalgene",
        "geography": "US",
    },
    {
        "url": "https://www.nutribullet.com",
        "name": "NutriBullet",
        "geography": "US",
    },
]
