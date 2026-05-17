"""
MCP-inspired responsible architecture for the Smart eCommerce Pipeline.

Implements the Model Context Protocol concepts from the dossier:
- MCP Host: the main application environment (Streamlit dashboard)
- MCP Client: the component that interacts with MCP Servers (LLM module)
- MCP Servers: expose specific tools/data with controlled access
- Permissions: enforce read-only access to analytics
- Logs: record all LLM interactions for accountability

Reference: https://modelcontextprotocol.io/specification/2025-03-26
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from src.config import analytics_dir, get_logger

logger = get_logger(__name__)


# ── MCP Servers (tools exposed to the LLM client) ────────────


class AnalyticsReaderServer:
    """MCP Server: read-only access to pipeline analytics outputs."""

    ALLOWED_FILES = {
        "topk_products.csv",
        "topk_per_category.csv",
        "topk_per_shop.csv",
        "clusters.csv",
        "pca_viz.csv",
        "model_metrics.json",
        "model_metrics_xgboost.json",
        "association_rules.csv",
        "dbscan_clusters.csv",
    }

    def __init__(self):
        self.analytics_dir = analytics_dir()

    def list_tools(self) -> list[str]:
        """Declare available tools (MCP: tool declaration)."""
        return ["read_analytics_file", "list_available_files", "get_top_products"]

    def list_available_files(self) -> list[str]:
        """List analytics files the client is allowed to read."""
        if not self.analytics_dir.exists():
            return []
        return [f.name for f in self.analytics_dir.iterdir() if f.name in self.ALLOWED_FILES]

    def read_analytics_file(self, filename: str) -> str | None:
        """Read an analytics file if it is in the allowed list (permission check)."""
        if filename not in self.ALLOWED_FILES:
            _log_access("DENIED", filename, "File not in allowed list")
            return None
        path = self.analytics_dir / filename
        if not path.exists():
            return None
        _log_access("READ", filename, "OK")
        return path.read_text(encoding="utf-8")

    def get_top_products(self, limit: int = 5) -> str | None:
        """Securely extract only the Top N products for LLM profiling."""
        path = self.analytics_dir / "topk_products.csv"
        if not path.exists():
            return None
        import pandas as pd

        try:
            df = pd.read_csv(path).head(limit)
            _log_access("READ", "topk_products.csv", f"Extracted top {limit} products")
            return df.to_json(orient="records", indent=2)
        except Exception as e:
            _log_access("ERROR", "topk_products.csv", str(e))
            return None


class SummaryGeneratorServer:
    """MCP Server: generate LLM summaries from structured analytics data only."""

    def list_tools(self) -> list[str]:
        return ["generate_summary"]

    def generate_summary(self, structured_data: dict) -> str:
        """Generate a summary using the LLM module (with logging)."""
        from src.llm.summarizer import generate_summary

        result = generate_summary(structured_data)
        _log_access("GENERATE_SUMMARY", "llm_call", f"length={len(result)}")
        return result


# ── MCP Client (interfaces with servers on behalf of the host) ──


class MCPClient:
    """MCP Client: routes requests from the Host to the appropriate Server."""

    def __init__(self):
        self.analytics_server = AnalyticsReaderServer()
        self.summary_server = SummaryGeneratorServer()

    def get_analytics(self, filename: str) -> str | None:
        return self.analytics_server.read_analytics_file(filename)

    def list_analytics(self) -> list[str]:
        return self.analytics_server.list_available_files()

    def get_top_products(self, limit: int = 5) -> str | None:
        return self.analytics_server.get_top_products(limit)

    def generate_summary(self, data: dict) -> str:
        return self.summary_server.generate_summary(data)


# ── MCP Host (the Streamlit app orchestrates everything) ────────
# The Streamlit dashboard (src/dashboard/app.py) acts as the MCP Host.
# It creates an MCPClient and uses it to:
#   - read analytics data (via AnalyticsReaderServer)
#   - generate LLM summaries (via SummaryGeneratorServer)
# The host never gives the LLM direct access to raw data or code execution.


# ── Logging / Permissions ───────────────────────────────────────


def _log_access(action: str, resource: str, detail: str) -> None:
    """Append an access log entry for MCP accountability."""
    log_dir = analytics_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "resource": resource,
        "detail": detail[:200],
    }
    log_path = log_dir / "mcp_access_log.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Permissions summary (for documentation / report) ───────────

PERMISSIONS = {
    "AnalyticsReaderServer": {
        "access": "read-only",
        "scope": "data/analytics/ (allowed files only)",
        "write": False,
        "execute": False,
    },
    "SummaryGeneratorServer": {
        "access": "read analytics + call LLM",
        "scope": "structured aggregates only, no raw product rows",
        "write": "append-only logs",
        "execute": False,
    },
}
