"""LLM REST endpoints — chat and synthesis reports."""

from __future__ import annotations

import json
import re

import pandas as pd
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.config import analytics_dir

router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    history: list[dict] = []


class SummaryRequest(BaseModel):
    type: str  # "executive" | "strategy" | "profile"


def _load_context() -> dict:
    a_dir = analytics_dir()
    topk_path = a_dir / "topk_products.csv"
    if not topk_path.exists():
        return {}
    topk = pd.read_csv(topk_path)
    ctx: dict = {"n_products": len(topk)}

    cols = topk.columns.tolist()
    if "category" in cols:
        ctx["top_categories"] = topk["category"].value_counts().head(5).to_dict()
    if "shop_name" in cols and "score" in cols and not topk.empty:
        shop_scores = topk.groupby("shop_name")["score"].mean()
        ctx["best_shop"] = str(shop_scores.idxmax())
        ctx["shop_rankings"] = shop_scores.sort_values(ascending=False).round(4).to_dict()

    keep = [c for c in ["title", "shop_name", "category", "price", "score"] if c in cols]
    if keep:
        ctx["top_products"] = topk[keep].head(20).to_dict("records")

    per_cat = a_dir / "topk_per_category.csv"
    if per_cat.exists():
        df = pd.read_csv(per_cat)
        keep2 = [c for c in ["category", "title", "shop_name", "score"] if c in df.columns]
        if keep2:
            ctx["top_per_category"] = df[keep2].head(15).to_dict("records")

    per_shop = a_dir / "topk_per_shop.csv"
    if per_shop.exists():
        df = pd.read_csv(per_shop)
        keep3 = [c for c in ["shop_name", "title", "category", "score"] if c in df.columns]
        if keep3:
            ctx["top_per_shop"] = df[keep3].head(15).to_dict("records")

    return ctx


@router.post("/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    from src.llm.summarizer import chat_with_data

    ctx = _load_context()
    response = chat_with_data(query=req.query, context=ctx, history=req.history)

    def generate():
        for token in re.split(r"(\s+)", response):
            if token:
                yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/summary")
def summary(req: SummaryRequest) -> dict:
    ctx = _load_context()
    if req.type == "executive":
        from src.llm.summarizer import generate_summary

        result = generate_summary(ctx)
    elif req.type == "strategy":
        from src.llm.summarizer import generate_strategy_report

        result = generate_strategy_report(ctx)
    else:
        from src.llm.summarizer import generate_product_profile
        from src.mcp.architecture import MCPClient

        result = generate_product_profile(MCPClient().get_top_products(5))
    return {"result": result}
