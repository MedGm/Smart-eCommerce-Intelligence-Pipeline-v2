#!/usr/bin/env python3
"""
Minimal Scrapy spider for Ruggable collections.

Demonstrates the Scrapy approach described in the dossier (structured scraping).
Can be run standalone:

    pip install scrapy
    scrapy runspider scripts/scrapy_spider.py -o scrapy_output.json

Or from Python:

    python scripts/scrapy_spider.py
"""

import sys

try:
    import scrapy
    from scrapy.crawler import CrawlerProcess
except ImportError:
    print(
        "Scrapy not installed. Install with: pip install scrapy\n"
        "Main pipeline uses Playwright instead."
    )
    sys.exit(0)


class RuggableSpider(scrapy.Spider):
    """Crawl Ruggable collection pages and extract product links + titles."""

    name = "ruggable"
    allowed_domains = ["ruggable.com"]
    start_urls = ["https://ruggable.com/collections/all"]

    def parse(self, response):
        for a in response.css("a[href*='/products/']"):
            href = a.attrib.get("href", "")
            title = a.css("::text").get("").strip()
            if href and title and len(title) > 2:
                yield {
                    "product_url": response.urljoin(href),
                    "title": title[:200],
                    "source": "scrapy",
                }


def main():
    output_file = "scrapy_output.json"
    process = CrawlerProcess(
        settings={
            "FEEDS": {output_file: {"format": "json", "overwrite": True}},
            "LOG_LEVEL": "WARNING",
            "USER_AGENT": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            "ROBOTSTXT_OBEY": True,
        }
    )
    process.crawl(RuggableSpider)
    process.start()
    print(f"Scrapy spider done. Output: {output_file}")


if __name__ == "__main__":
    main()
