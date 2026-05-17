"""
PRISM — LLM Chat Interface
Focused surface for Gemini-powered synthesis and interactive BI chat.
BI exploration (charts, rankings, tables) → Apache Superset at http://localhost:8088
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import streamlit as st

from src.config import analytics_dir
from src.mcp.architecture import MCPClient

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PRISM — LLM Intelligence",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal dark theme ────────────────────────────────────────────────────────
st.markdown(
    """
<style>
body, .stApp { background-color: #0C1117; color: #F2EAD9; }
.stButton > button {
    background: rgba(214,168,95,0.10);
    border: 1px solid rgba(214,168,95,0.22);
    color: #F2EAD9;
    border-radius: 12px;
}
.stButton > button:hover { background: rgba(214,168,95,0.20); }
</style>
""",
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## PRISM")
    st.markdown("---")
    superset_url = os.environ.get("SUPERSET_URL", "http://localhost:8088")
    st.markdown(f"### [BI Dashboard (Superset)]({superset_url})")
    st.caption("Charts · Rankings · Clustering · Rules")
    st.markdown("---")
    page = st.radio(
        "LLM Interface",
        ["Synthesis Reports", "Chat Assistant"],
    )

mcp = MCPClient()


def _load_context() -> dict:
    """Load top-K analytics summary for LLM prompts. Returns {} if not available."""
    topk_path = analytics_dir() / "topk_products.csv"
    if not topk_path.exists():
        return {}
    topk = pd.read_csv(topk_path)
    ctx: dict = {"n_products": len(topk)}
    if "category" in topk.columns:
        ctx["top_categories"] = topk["category"].value_counts().head(5).index.tolist()
    if "shop_name" in topk.columns and "score" in topk.columns and not topk.empty:
        ctx["best_shop"] = str(topk.groupby("shop_name")["score"].mean().idxmax())
    return ctx


# ── Synthesis Reports page ────────────────────────────────────────────────────
if page == "Synthesis Reports":
    st.title("LLM Synthesis Reports")
    st.caption(
        "Narrative intelligence over pipeline analytics — grounded in curated Top-K artifacts. "
        f"For charts and data exploration, open **[Superset]({superset_url})**."
    )

    ctx = _load_context()
    if not ctx:
        st.warning("No analytics data found. Run `make pipeline` first.")
    else:
        st.info(
            f"Loaded {ctx['n_products']} Top-K products across "
            f"{len(ctx.get('top_categories', []))} categories."
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Executive Summary", use_container_width=True):
            with st.spinner("Calling Gemini…"):
                from src.llm.summarizer import generate_summary

                result = generate_summary(ctx)
            st.markdown("### Executive Summary")
            st.info(result)

    with col2:
        if st.button("Strategic Recommendations", use_container_width=True):
            with st.spinner("Generating strategy…"):
                from src.llm.summarizer import generate_strategy_report

                result = generate_strategy_report(ctx)
            st.markdown("### Strategic Report")
            st.success(result)

    with col3:
        if st.button("Competitive Profiling", use_container_width=True):
            with st.spinner("Profiling top products…"):
                from src.llm.summarizer import generate_product_profile

                result = generate_product_profile(mcp.get_top_products(5))
            st.markdown("### Competitive Profile")
            st.warning(result)

# ── Chat Assistant page ───────────────────────────────────────────────────────
elif page == "Chat Assistant":
    st.title("BI Chat Assistant")
    st.caption(
        "Ask questions about rankings, categories, shops, and model behaviour. "
        f"For visual exploration, open **[Superset]({superset_url})**."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about your data…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                from src.llm.summarizer import chat_with_data

                response = chat_with_data(
                    query=prompt,
                    context=_load_context(),
                    history=st.session_state.messages[:-1],
                )
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
