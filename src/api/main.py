"""PRISM FastAPI application — serves REST API + static SPA."""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.routes import analytics, llm

app = FastAPI(title="PRISM API", version="1.0.0")

app.include_router(analytics.router, prefix="/api")
app.include_router(llm.router, prefix="/api")


@app.get("/api/health")
def health():
    return {"status": "ok"}


_STATIC = Path(__file__).parent.parent / "dashboard" / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


@app.get("/", include_in_schema=False)
@app.get("/{path:path}", include_in_schema=False)
def spa(path: str = ""):
    return FileResponse(str(_STATIC / "index.html"))
