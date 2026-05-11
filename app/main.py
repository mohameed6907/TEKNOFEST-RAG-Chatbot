from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import os
from dotenv import load_dotenv
load_dotenv()

# Verify tracing is active on startup
import langsmith
print(f"[LangSmith] Tracing active: {os.getenv('LANGCHAIN_TRACING_V2')}")
print(f"[LangSmith] Project: {os.getenv('LANGCHAIN_PROJECT')}")

from .config import get_settings
from .rag.graph import build_teknofest_graph, run_graph
from .tracing import init_langsmith, is_tracing_enabled
from .database import engine, Base

# Import routers
from .routers import auth, chat, admin

# Initialize DB tables
Base.metadata.create_all(bind=engine)


def create_app() -> FastAPI:
    settings = get_settings()

    # Initialise LangSmith tracing BEFORE building the graph so every
    # LangChain / LangGraph call is captured from the very first request.
    init_langsmith(settings)

    app = FastAPI(title=settings.app_name)

    # Static & templates
    base_dir = Path(__file__).resolve().parent
    static_dir = base_dir / "static"
    templates_dir = base_dir / "templates"

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    templates = Jinja2Templates(directory=str(templates_dir))

    # Build LangGraph workflow once at startup
    # Note: we are not storing `graph` globally anymore because `app/routers/chat.py` builds it independently,
    # but we can leave it here if other parts use it.

    # Include routers
    app.include_router(auth.router)
    app.include_router(chat.router)
    app.include_router(admin.router)

    # A.3 — Vector DB startup health check
    @app.on_event("startup")
    async def startup_vectordb_check():
        import logging as _logging
        from app.rag.retrievers import verify_collections
        _log = _logging.getLogger("startup")
        health = verify_collections(settings)
        for name, info in health.items():
            status = info.get("status", "?")
            if status == "ok":
                cosine_flag = "cosine=YES" if info.get("cosine") else "cosine=NO (WARNING: L2 distance!)"
                _log.info(
                    "[VectorDB] '%s': %d chunks | %s | model=%s",
                    name, info["chunks"], cosine_flag, info.get("embedding_model", "?")
                )
                if not info.get("cosine"):
                    _log.warning(
                        "[VectorDB] '%s' cosine metric eksik! "
                        "Lutfen recreate_cosine_collections.py calistirin.",
                        name
                    )
                if info["chunks"] == 0:
                    _log.warning(
                        "[VectorDB] '%s' BOS! "
                        "Ingestion scripti calistirilmamis olabilir.",
                        name
                    )
            elif status == "missing":
                _log.error(
                    "[VectorDB] '%s': collection bulunamadi. "
                    "Ingestion scripti calistirilmamis.",
                    name
                )
            else:
                _log.error("[VectorDB] '%s': HATA -- %s", name, info.get("error"))

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        from app.rag.retrievers import verify_collections
        vdb_health = verify_collections(settings)
        return {
            "status": "ok",
            "tracing_enabled": is_tracing_enabled(),
            "langsmith_project": settings.langsmith_project if is_tracing_enabled() else None,
            "vector_db": vdb_health,
        }

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request=request, name="index.html", context={"app_name": settings.app_name})

    @app.get("/{full_path:path}", response_class=HTMLResponse)
    async def index_fallback(request: Request, full_path: str) -> HTMLResponse:
        # Avoid intercepting API calls or static files
        if full_path.startswith("api/") or full_path.startswith("static/"):
            raise HTTPException(status_code=404)
        return templates.TemplateResponse(request=request, name="index.html", context={"app_name": settings.app_name})

    return app


app = create_app()

