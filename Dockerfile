# Smart eCommerce Intelligence Pipeline — app and dashboard
FROM python:3.11-slim

WORKDIR /app

# Playwright system deps (required for Shopify dynamic scraping)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
    libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
    libgbm1 libasound2 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN playwright install chromium

COPY . .
ENV PYTHONPATH=/app

# Default: run pipeline; override for dashboard
CMD ["python", "-m", "src.pipeline.local_pipeline"]
