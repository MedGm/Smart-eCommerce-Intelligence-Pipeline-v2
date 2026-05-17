"""
LLM summarizer: consumes aggregated metrics only (top products, cluster summaries, etc.).
Logs prompt inputs and data source for responsible design.

Phase 5 expects a Gemini (or Gemini-compatible) client:
- we read a generic LLM_API_KEY or GEMINI_API_KEY from the environment
- the actual HTTP call remains a TODO so that the pipeline runs without secrets.
"""

import json
import os

from dotenv import load_dotenv

from src.config import analytics_dir, get_logger

# Charge les variables depuis .env (GEMINI_API_KEY, etc.) si présent
load_dotenv()

logger = get_logger(__name__)


def _log_usage(source: str, prompt_preview: str, response_preview: str) -> None:
    """Record prompt and response for MCP-inspired accountability."""
    log_dir = analytics_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "source": source,
        "prompt_preview": prompt_preview[:200],
        "response_preview": (response_preview or "")[:200],
    }
    log_path = log_dir / "llm_usage_log.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.debug("LLM usage logged: source=%s", source)


def _call_llm(prompt: str, source_tag: str = "gemini") -> str:
    """Core function to call Gemini API and log usage."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("LLM_API_KEY")
    if not api_key:
        out = "(LLM disabled: set GEMINI_API_KEY in your environment to enable.)"
        _log_usage("none", prompt[:200], out)
        return out

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        out = getattr(response, "text", "") or str(response)
        _log_usage(source_tag, prompt[:200], out)
        return out
    except Exception as exc:  # pragma: no cover
        out = f"(LLM error: {exc})"
        _log_usage(f"{source_tag}_error", prompt[:200], out)
        return out


def generate_summary(structured_data: dict) -> str:
    """Generate 3-sentence executive summary."""
    from src.llm.prompts import EXECUTIVE_SUMMARY_PROMPT

    prompt = EXECUTIVE_SUMMARY_PROMPT.format(data=json.dumps(structured_data, indent=2))
    return _call_llm(prompt, "gemini_summary")


def generate_strategy_report(structured_data: dict) -> str:
    """Generate a Chain-of-Thought strategic report."""
    from src.llm.prompts import CHAIN_OF_THOUGHT_PROMPT

    prompt = CHAIN_OF_THOUGHT_PROMPT.format(data=json.dumps(structured_data, indent=2))
    return _call_llm(prompt, "gemini_strategy")


def generate_product_profile(top_products_json: str) -> str:
    """Generate a cohesive customer profile & competitive analysis of top 5 products."""
    from src.llm.prompts import PRODUCT_COMPARISON_PROMPT

    prompt = PRODUCT_COMPARISON_PROMPT.format(data=top_products_json)
    return _call_llm(prompt, "gemini_profiling")


def chat_with_data(query: str, context: dict, history: list[dict]) -> str:
    """Handle interactive chat using current BI context."""
    history_str = "\\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in history[-3:]]
    )
    context_str = json.dumps(context, indent=2)

    prompt = f"""You are an expert eCommerce AI Assistant embedded in a BI Dashboard.
You answer user questions accurately based ONLY on the current context provided below.
Do not hallucinate products or numbers.

Current Dashboard Context:
{context_str}

Recent Chat History:
{history_str}

User Question: {query}
Answer concisely and professionally:"""

    return _call_llm(prompt, "gemini_chat")


def run():
    """Load analytics outputs and generate summary."""
    a_dir = analytics_dir()
    if not (a_dir / "topk_products.csv").exists():
        return "(Run pipeline first to generate analytics.)"
    import pandas as pd

    topk = pd.read_csv(a_dir / "topk_products.csv")
    top_categories = (
        topk["category"].value_counts().head(5).index.tolist() if "category" in topk.columns else []
    )
    best_shop = ""
    if "shop_name" in topk.columns and not topk.empty:
        best_shop = topk.groupby("shop_name")["score"].mean().idxmax()
    data = {
        "top_categories": top_categories,
        "best_shop": best_shop,
        "n_top_products": len(topk),
    }
    if (a_dir / "clusters.csv").exists():
        clusters = pd.read_csv(a_dir / "clusters.csv")
        data["cluster_summary"] = clusters.groupby("cluster").size().to_dict()
    return generate_summary(data)


if __name__ == "__main__":
    print(run())
