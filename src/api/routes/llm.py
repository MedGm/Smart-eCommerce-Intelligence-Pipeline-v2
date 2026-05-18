"""LLM REST endpoints — chat and synthesis reports."""

from __future__ import annotations

import json

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


@router.post("/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    from src.llm.summarizer import chat_with_data

    ctx = _load_context()
    response = chat_with_data(query=req.query, context=ctx, history=req.history)

    def generate():
        words = response.split()
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            yield f"data: {json.dumps({'token': chunk})}\n\n"
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
