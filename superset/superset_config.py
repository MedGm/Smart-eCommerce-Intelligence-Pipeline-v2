import os

# Required — change in production
SECRET_KEY = os.environ.get("SUPERSET_SECRET_KEY", "change-me-in-production-32-chars!")

# Metadata store — SQLite for dev, PostgreSQL for prod
SQLALCHEMY_DATABASE_URI = "sqlite:////app/superset_home/superset.db"

# Allow DuckDB local files (disabled by default for safety)
PREVENT_UNSAFE_DB_CONNECTIONS = False

# Disable CSRF for development convenience
WTF_CSRF_ENABLED = False

# Simple in-process cache for dev
CACHE_CONFIG = {"CACHE_TYPE": "SimpleCache", "CACHE_DEFAULT_TIMEOUT": 300}

# Timeout
SUPERSET_WEBSERVER_TIMEOUT = 300

# Feature flags
FEATURE_FLAGS = {
    "ENABLE_TEMPLATE_PROCESSING": True,
}
